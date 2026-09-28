"""Offline collection, provenance, policy, and local RPC contract tests.

No checkpoint is loaded and no network/API request is made. The RPC test starts
the real JSON-lines server over local subprocess pipes, without SSH.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import unified_game_pipeline as pipeline
import scaled_maze


ROOT = Path(__file__).resolve().parents[1]


def case(identity="episode-a", split="train", seed=17, task="maze"):
    spec = {"task": task, "size": 8, "max_steps": 4}
    if task == "maze":
        spec["initial_state"] = {"game": "scaled_maze", "size": 8,
                                 "position": [1, 1], "goal": [1, 3], "walls": []}
    return {"id": identity, "split": split, "seed": seed, "variant": task + "8", "spec": spec}


def write_rows(path, rows):
    Path(path).write_text("".join(pipeline.encode(row) + "\n" for row in rows))


def collection(root, episode_mutator=None, receipt_mutator=None):
    """Two real successful Maze actions plus detached mock API receipts."""
    environment_case = case()
    declaration = {"engine": "jev", "controller": "greedy", "epsilon": 0.0}
    policy_id = pipeline.digest(declaration)
    env = pipeline.factory(environment_case["spec"])
    obs, _ = env.reset(environment_case["seed"])
    steps, receipts = [], []
    for index in range(2):
        request = pipeline.policy_request(obs, environment_case["id"])
        api_input = {"model": "typesafe-ai/jev", "state": request["state"], "questions": request["questions"]}
        input_hash = pipeline.digest(api_input)
        probabilities = {a: float(a == "east") for a in obs["candidates"]}
        receipt = {"id": f"mock-api-{index}", "status": "succeeded", "model": "typesafe-ai/jev",
                   "input": api_input, "input_sha256": input_hash,
                   "native_probs": {"action": probabilities}, "rounding": {"decimals": 2}}
        answer = {"type": "choice", "probabilities": probabilities,
                  "native_probabilities": probabilities, "source_api_call_id": receipt["id"],
                  "source_input_sha256": input_hash}
        following, reward, terminated, truncated, info = env.step("east")
        steps.append({"observation": obs, "action": "east", "scores": probabilities,
                      "behavior_probs": probabilities, "answers": {"action": answer},
                      "reward": reward, "terminated": terminated, "truncated": truncated, "info": info})
        receipts.append(receipt)
        obs = following
    env.close()
    episode = {"case": environment_case, "continuation_policy_id": policy_id,
               "steps": steps, "complete": True, "success": True, "final_info": info}
    if episode_mutator:
        episode_mutator(episode)
    if receipt_mutator:
        receipt_mutator(receipts)
    source = root / "episodes.jsonl"
    write_rows(source, [episode])
    pipeline.write_json(source.with_suffix(".manifest.json"), {
        "finished": True, "episode_sha256": pipeline.file_digest(source),
        "continuation_policy_id": policy_id, "policy": declaration,
        "selected_cases": [environment_case["id"]],
    })
    journal = root / "journal"
    journal.mkdir()
    write_rows(journal / "calls.jsonl", receipts)
    return source, journal, episode


def dataset_args(root, source, journal=None, role="outcome", **options):
    return SimpleNamespace(episodes=str(source), output=str(root / "dataset"),
                           journal_dir=str(journal) if journal else None, role=role,
                           retention=options.get("retention"), max_states_per_episode=options.get("max_states_per_episode", 0))


class PolicyTests(unittest.TestCase):
    def test_greedy_ties_epsilon_and_sampling(self):
        self.assertEqual(pipeline.behavior_distribution({"z": .7, "a": .7, "b": .1}, "greedy", 0),
                         {"a": 1, "b": 0, "z": 0})
        self.assertEqual(pipeline.behavior_distribution({"b": 0, "a": 1}, "greedy", .2), {"a": .9, "b": .1})
        self.assertEqual(pipeline.behavior_distribution({"b": 3, "a": 1}, "sample", 0), {"a": .25, "b": .75})
        a, b = random.Random(4), random.Random(4)
        draws_a = [pipeline.choose({"z": .7, "a": .3}, a) for _ in range(30)]
        draws_b = [pipeline.choose({"a": .3, "z": .7}, b) for _ in range(30)]
        self.assertEqual(draws_a, draws_b)

    def test_invalid_probability_inputs(self):
        for scores, mode, epsilon in (({}, "sample", 0), ({"a": float("nan")}, "greedy", 0),
                                      ({"a": 0}, "sample", 0), ({"a": -1}, "sample", 0),
                                      ({"a": 1}, "sample", 1.1)):
            with self.assertRaises(ValueError):
                pipeline.behavior_distribution(scores, mode, epsilon)

    def test_observed_request_conditions_only_named_executed_action(self):
        obs = {"state": "Visible state", "candidates": {"north": "Attempt north", "east": "Attempt east"}}
        req = pipeline.outcome_request(obs, "request-a", ["north"])
        self.assertEqual(set(req["questions"]), {"success_north"})
        self.assertIn("'north'", req["questions"]["success_north"]["instructions"])
        self.assertIn("frozen behavior policy", req["questions"]["success_north"]["instructions"])
        self.assertEqual(set(req), {"id", "state", "questions"})

    def test_duplicate_id_and_same_seed_cross_split_rejected(self):
        for cases in ([case(), case()], [case(), case("other", "test")]):
            with self.assertRaises(ValueError):
                pipeline.validate_cases(cases)

    def test_snake_alias_seed_cannot_cross_splits(self):
        with self.assertRaises(ValueError):
            pipeline.validate_cases([case("a", "train", 17, "snake"),
                                     case("b", "test", 17 + 2**64, "snake")])

    def test_d4_equivalent_maze_maps_cannot_cross_splits(self):
        a, b = case("a", "train", 17), case("b", "test", 91)
        a["spec"]["initial_state"]["walls"] = [[0, 1], [2, 3], [4, 6]]
        b["spec"]["initial_state"] = scaled_maze.transform_state(a["spec"]["initial_state"], 5)
        self.assertEqual(pipeline.environment_group(a), pipeline.environment_group(b))
        with self.assertRaises(ValueError):
            pipeline.validate_cases([a, b])

    def test_doom_frame_skip_does_not_create_an_independent_seed_group(self):
        a = {"id": "a", "split": "train", "seed": 17, "spec": {
            "task": "shooting", "scenario": "basic", "frame_skip": 4, "max_steps": 75}}
        b = copy.deepcopy(a)
        b.update(id="b", split="ood")
        b["spec"].update(frame_skip=8, max_steps=38)
        with self.assertRaises(ValueError):
            pipeline.validate_cases([a, b])

    def test_partial_observation_aliases_are_not_latent_environment_aliases(self):
        a, b = case("a", "train", 17), case("b", "test", 91)
        b["spec"]["initial_state"]["walls"] = [[7, 0]]
        environments = [pipeline.factory(c["spec"]) for c in (a, b)]
        try:
            observations = [env.reset(c["seed"])[0] for env, c in zip(environments, (a, b))]
            self.assertEqual(observations[0], observations[1])
            self.assertNotEqual(pipeline.environment_group(a), pipeline.environment_group(b))
            pipeline.validate_cases([a, b])
        finally:
            for env in environments:
                env.close()


class DatasetTests(unittest.TestCase):
    def test_state_sampling_is_role_specific_and_outcomes_keep_every_transition(self):
        # Inclusion of an outcome row must not depend on the future episode length.
        # Policy supervision may still use a fixed, reproducible per-episode cap.
        for role, requested_limit, effective_limit, expected_indices in (
                ("outcome", 1, None, None),
                ("outcome", 24, None, None),
                ("outcome", 0, 0, [0, 1]),
                ("outcome", None, 0, [0, 1]),
                ("policy", 1, 1, "one"),
                ("policy", None, 24, [0, 1]),
                ("policy", 0, 0, [0, 1])):
            with self.subTest(role=role, limit=requested_limit):
                with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                    root = Path(directory)
                    source, journal, _ = collection(root)
                    args = dataset_args(root, source, journal if role == "policy" else None,
                                        role, max_states_per_episode=requested_limit)
                    if expected_indices is None:
                        with self.assertRaisesRegex(ValueError, "Outcome data must retain every executed transition"):
                            pipeline.make_dataset(args)
                        self.assertFalse((root / "dataset").exists())
                        continue
                    pipeline.make_dataset(args)
                    rows = pipeline.read_rows(root / "dataset/train.jsonl")
                    indices = [row["metadata"]["decision_index"] for row in rows]
                    if expected_indices == "one":
                        self.assertEqual(len(indices), 1)
                        self.assertIn(indices[0], (0, 1))
                    else:
                        self.assertEqual(indices, expected_indices)
                    manifest = json.loads((root / "dataset/manifest.json").read_text())
                    self.assertEqual(manifest["max_states_per_episode"], effective_limit)

    def test_labels_are_eventual_success_only_on_actual_action(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, _, episode = collection(root)
            pipeline.make_dataset(dataset_args(root, source))
            rows = pipeline.read_rows(root / "dataset/train.jsonl")
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["gold"], {"success_east": True})
                self.assertEqual(row["gold_label_kind"], {"success_east": "observed_outcome"})
                self.assertEqual(set(row["questions"]), {"success_east"})
                self.assertEqual(row["metadata"]["executed_action"], "east")
                self.assertEqual(row["metadata"]["conditioned_action"], "east")
                self.assertEqual(row["metadata"]["continuation_policy_id"], episode["continuation_policy_id"])
                self.assertNotIn("teacher", row)
            self.assertEqual(episode["steps"][0]["reward"], 0)
            self.assertEqual(rows[0]["gold"]["success_east"], True)
            for split in pipeline.SPLITS[1:]:
                self.assertEqual(pipeline.read_rows(root / f"dataset/{split}.jsonl"), [])

    def test_native_policy_labels_preserved_with_matching_receipt(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, journal, _ = collection(root)
            pipeline.make_dataset(dataset_args(root, source, journal, "policy"))
            rows = pipeline.read_rows(root / "dataset/train.jsonl")
            self.assertEqual(len(rows), 2)
            for index, row in enumerate(rows):
                native = pipeline.read_rows(journal / "calls.jsonl")[index]["native_probs"]
                self.assertEqual(row["teacher"]["native_probs"], native)
                self.assertNotIn("gold", row)

    def test_receipt_from_different_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            def wrong_receipt(receipts):
                receipts[0]["input"]["state"] = "A different observation with the same candidate IDs."
            source, journal, _ = collection(root, receipt_mutator=wrong_receipt)
            with self.assertRaises(ValueError):
                pipeline.make_dataset(dataset_args(root, source, journal, "policy"))

    def test_native_receipt_probabilities_must_match_recorded_answer(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            def different_distribution(receipts):
                receipts[0]["native_probs"] = {"action": {"north": 1, "east": 0, "south": 0, "west": 0}}
            source, journal, _ = collection(root, receipt_mutator=different_distribution)
            with self.assertRaises(ValueError):
                pipeline.make_dataset(dataset_args(root, source, journal, "policy"))

    def test_complete_marker_cannot_replace_a_terminal_transition(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            def nonterminal(episode):
                episode["steps"][-1]["terminated"] = False
            source, _, _ = collection(root, episode_mutator=nonterminal)
            with self.assertRaises(ValueError):
                pipeline.make_dataset(dataset_args(root, source))

    def test_final_success_must_match_transition_and_final_info(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, _, _ = collection(root, episode_mutator=lambda e: e.update(success=False))
            with self.assertRaises(ValueError):
                pipeline.make_dataset(dataset_args(root, source))

    def test_incomplete_mixed_policy_truncation_and_hash_corruption_rejected(self):
        mutations = [lambda e: e.update(complete=False), lambda e: e.update(continuation_policy_id="other"),
                     lambda e: e["steps"][0].update(truncated=True)]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                source, _, _ = collection(root, episode_mutator=mutate)
                with self.assertRaises(ValueError):
                    pipeline.make_dataset(dataset_args(root, source))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = collection(root)
            source.write_text(source.read_text() + "\n")
            with self.assertRaises(ValueError):
                pipeline.make_dataset(dataset_args(root, source))


class RolloutTests(unittest.TestCase):
    def rollout(self, root, name, batch):
        source = root / "cases.jsonl"
        if not source.exists():
            write_rows(source, [case("a", seed=1), case("b", seed=2, task="snake")])
        args = SimpleNamespace(cases=str(source), output=str(root / f"{name}.jsonl"), engine="random",
            checkpoint=None, controller="greedy", epsilon=.2, seed=7, splits="train", limit_per_variant=0,
            env_batch=batch, batch_questions=16, max_length=8192, remote_host=None, remote_command=None,
            env_file=None, journal_dir=None, budget_usd=.1)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.rollout(args)
        return pipeline.read_rows(args.output)

    def test_random_episodes_independent_of_batch_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two = self.rollout(root, "single", 1), self.rollout(root, "batched", 2)
            self.assertEqual(one, two)
            self.assertTrue(all(e["complete"] and e["steps"][-1]["terminated"] for e in one))
            for episode in one:
                for step in episode["steps"]:
                    self.assertEqual(set(step["behavior_probs"]), set(step["observation"]["candidates"]))
                    self.assertAlmostEqual(sum(step["behavior_probs"].values()), 1)
                    self.assertIn(step["action"], step["observation"]["candidates"])

    def test_existing_collection_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.rollout(root, "existing", 1)
            before = (root / "existing.jsonl").read_bytes()
            with self.assertRaises(ValueError):
                self.rollout(root, "existing", 1)
            self.assertEqual(before, (root / "existing.jsonl").read_bytes())

    def test_json_rpc_matches_local_environment(self):
        remote = pipeline.RemoteEnvironments.__new__(pipeline.RemoteEnvironments)
        remote.process = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/unified_game_pipeline.py"), "env-server"],
                                         cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1)
        local = pipeline.LocalEnvironments()
        try:
            self.assertEqual(remote.call("describe", None), pipeline.source_hashes())
            cases = [case()]
            self.assertEqual(remote.reset(cases), local.reset(cases))
            for action in ("east", "east"):
                self.assertEqual(remote.step({"episode-a": action}), local.step({"episode-a": action}))
        finally:
            remote.close()
            local.close()
            remote.process.stdout.close()
            remote.process.stderr.close()
        self.assertEqual(remote.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
