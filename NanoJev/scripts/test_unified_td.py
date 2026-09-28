"""Synthetic finite decision branches test TD indexing, not game performance."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import unified_game_pipeline as pipeline
from unified_td import load_td_index, weighted_bootstrap


def fixture(root, success=True, length=4, forced_at=None, split="train", identity="toy-branch", seed=7):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    initial = {"game": "scaled_maze", "size": 8, "position": [1, 1], "goal": [7, 7],
               "walls": [[0, seed % 7]]}
    case = {"id": identity, "split": split, "variant": "toy", "seed": seed,
            "spec": {"task": "maze", "size": 8, "max_steps": length * 3, "initial_state": initial}}
    declaration = {"engine": "checkpoint", "controller": "q_greedy", "epsilon": .4,
                   "sampling_seed": 17, "temperature": 1.0, "tie_break": "lexicographic_first",
                   "environment_contract": "finite_task_deadline_v1",
                   "source_sha256": {"synthetic_contract": "0" * 64},
                   "checkpoint_sha256": {"best.safetensors": "1" * 64}}
    pid = pipeline.digest(declaration)
    rng = random.Random(int(pipeline.digest([identity, declaration["sampling_seed"]])[:16], 16))
    steps = []
    for index in range(length):
        candidates = {"east": "Move east into the visible branch.", "north": "Move north into the visible branch."}
        if index == forced_at:
            candidates.pop("north")
        scores = {action: (1.0 if len(candidates) == 1 else .8 if action == "east" else .2) for action in candidates}
        probabilities = pipeline.behavior_distribution(scores, "greedy", declaration["epsilon"])
        action = pipeline.choose(probabilities, rng)
        terminated = index == length - 1
        info = {"success": success if terminated else False, "terminated": terminated, "truncated": False,
                "episode_metrics": {"physical_steps": 3 * (index + 1)}}
        observation = {"task": "maze", "state": f"Visible decision branch {index}. Full observed memory: node-{index}.",
                       "step": index * 3, "remaining_steps": (length - index) * 3, "candidates": candidates}
        answers = {} if len(candidates) == 1 else {
            "success_" + action: {"type": "boolean", "probabilities": {"false": 1-value, "true": value}}
            for action, value in scores.items()}
        steps.append({"observation": observation, "action": action, "scores": scores,
                      "behavior_probs": probabilities, "answers": answers, "forced": len(candidates) == 1,
                      "terminated": terminated, "truncated": False, "info": info,
                      "reward": 999.0})  # Native score deliberately differs from task success.
    episode = {"case": case, "steps": steps, "complete": True, "success": success,
               "final_info": copy.deepcopy(steps[-1]["info"]), "continuation_policy_id": pid}
    path = root / "episodes.jsonl"
    manifest = {"schema_version": "nanojev-unified-episodes-v1", "finished": True,
                "policy": declaration, "continuation_policy_id": pid, "selected_cases": [identity]}
    dataset = {"collection_policy": copy.deepcopy(declaration), "continuation_policy_id": pid,
               "max_states_per_episode": 0}
    rewrite(path, [episode], manifest, dataset)
    group = pipeline.environment_group(case)
    records = []
    for index, step in enumerate(steps):
        obs, action = step["observation"], step["action"]
        rid = pipeline.digest([pid, identity, index, "outcome"])
        records.append({**pipeline.outcome_request(obs, rid, [action]),
            "state_id": pipeline.digest([group, seed, index, obs["state"]]), "split": split,
            "gold": {"success_" + action: success}, "gold_label_kind": {"success_" + action: "observed_outcome"},
            "metadata": {"record_role": "outcome", "task": "maze", "episode_id": identity,
                "decision_index": index, "source_group_id": group, "continuation_policy_id": pid,
                "executed_action": action, "conditioned_action": action, "spec": case["spec"],
                "remaining_steps": obs["remaining_steps"], "episode_success": success,
                "public_observation_sha256": pipeline.digest(obs["state"])}})
    return path, records, dataset, episode, manifest


def rewrite(path, episodes, manifest, dataset):
    path.write_text("".join(pipeline.encode(episode) + "\n" for episode in episodes))
    manifest["episode_sha256"] = dataset["episode_sha256"] = pipeline.file_digest(path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))


def entry(index, row):
    return index[(row["id"], next(iter(row["questions"])))]


class TDIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_one_step_uses_next_decision_not_next_physical_move(self):
        path, rows, dataset, episode, _ = fixture(self.root)
        index, audit = load_td_index(path, rows, dataset, 1)
        first = entry(index, rows[0])
        self.assertEqual(first["bootstrap_decision_index"], 1)
        self.assertEqual(first["request"]["state"], episode["steps"][1]["observation"]["state"])
        self.assertEqual(first["behavior_probs"], episode["steps"][1]["behavior_probs"])
        self.assertEqual(first["actual_decision_steps"], 1)
        self.assertEqual(entry(index, rows[-1])["terminal_target"], 1.)
        self.assertEqual(audit["by_split"]["train"], {"terminal": 1, "bootstrap": 3})
        self.assertEqual(audit["gamma"], 1.)
        self.assertFalse(audit["native_game_rewards_used"])

    def test_n3_cuts_terminal_and_never_uses_native_score_reward(self):
        for success in (True, False):
            path, rows, dataset, _, _ = fixture(self.root / str(success), success=success)
            index, audit = load_td_index(path, rows, dataset, 3)
            self.assertEqual(entry(index, rows[0])["bootstrap_decision_index"], 3)
            for row, span in zip(rows[1:], (3, 2, 1)):
                value = entry(index, row)
                self.assertEqual(value["terminal_target"], float(success))
                self.assertEqual(value["actual_decision_steps"], span)
                self.assertNotIn("request", value)
            self.assertEqual(audit["by_split"]["train"], {"terminal": 3, "bootstrap": 1})

    def test_forced_singleton_still_has_a_boolean_bootstrap(self):
        path, rows, dataset, _, _ = fixture(self.root, forced_at=1)
        index, _ = load_td_index(path, rows, dataset, 1)
        value = entry(index, rows[0])
        self.assertEqual(value["behavior_probs"], {"east": 1.})
        self.assertEqual(value["action_to_qid"], {"east": "success_east"})
        self.assertEqual(value["request"]["questions"]["success_east"]["type"], "boolean")
        self.assertAlmostEqual(weighted_bootstrap({"east": .37}, value["behavior_probs"]), .37)

    def test_recorded_policy_weights_not_max_or_uniform(self):
        path, rows, dataset, episode, _ = fixture(self.root)
        index, _ = load_td_index(path, rows, dataset, 1)
        value = entry(index, rows[0])
        self.assertAlmostEqual(weighted_bootstrap({"east": .1, "north": .9}, value["behavior_probs"]), .26)
        value["behavior_probs"]["east"] = 0
        self.assertEqual(episode["steps"][1]["behavior_probs"]["east"], .8)

    def test_unobserved_future_outcome_never_enters_bootstrap_request(self):
        a = fixture(self.root / "success", success=True)
        b = fixture(self.root / "failure", success=False)
        ai, _ = load_td_index(a[0], a[1], a[2], 1)
        bi, _ = load_td_index(b[0], b[1], b[2], 1)
        self.assertEqual(entry(ai, a[1][0]), entry(bi, b[1][0]))
        self.assertEqual(set(entry(ai, a[1][0])["request"]), {"id", "state", "questions"})
        self.assertNotIn("terminal_target", entry(ai, a[1][0]))
        self.assertNotEqual(entry(ai, a[1][-1])["terminal_target"], entry(bi, b[1][-1])["terminal_target"])

    def test_nstep_does_not_cross_into_another_episode_or_split(self):
        a = fixture(self.root / "a", identity="train-episode", split="train", seed=7)
        b = fixture(self.root / "b", identity="test-episode", split="test", seed=9)
        manifest = copy.deepcopy(a[4]); manifest["selected_cases"] += b[4]["selected_cases"]
        rewrite(a[0], [a[3], b[3]], manifest, a[2])
        index, audit = load_td_index(a[0], a[1] + b[1], a[2], 3)
        self.assertEqual(entry(index, a[1][-1])["terminal_target"], 1.)
        self.assertNotIn("request", entry(index, a[1][-1]))
        self.assertEqual(audit["by_split"]["test"], {"terminal": 3, "bootstrap": 1})

    def test_missing_or_duplicate_transition_rejected(self):
        path, rows, dataset, _, _ = fixture(self.root)
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                load_td_index(path, bad, dataset, 1)

    def test_hash_finished_or_policy_mismatch_rejected(self):
        for name in ("hash", "finished", "policy"):
            path, rows, dataset, _, manifest = fixture(self.root / name)
            if name == "hash":
                path.write_text(path.read_text() + "\n")
            else:
                if name == "finished": manifest["finished"] = False
                else: manifest["policy"]["epsilon"] = .2
                path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_td_index(path, rows, dataset, 1)

    def test_source_snapshot_attested_and_modified_source_rejected(self):
        path, rows, dataset, episode, manifest = fixture(self.root)
        snapshot = self.root / "episodes.sources"
        snapshot.mkdir()
        source = snapshot / "synthetic_contract"
        source.write_text("# Frozen synthetic decision contract.\n")
        manifest["source_snapshot"] = snapshot.name
        declaration = manifest["policy"]
        declaration["source_sha256"][source.name] = pipeline.file_digest(source)
        pid = pipeline.digest(declaration)
        manifest["continuation_policy_id"] = episode["continuation_policy_id"] = pid
        dataset["continuation_policy_id"] = pid
        dataset["collection_policy"] = copy.deepcopy(declaration)
        for row in rows:
            row["metadata"]["continuation_policy_id"] = pid
        rewrite(path, [episode], manifest, dataset)
        _, audit = load_td_index(path, rows, dataset, 1)
        self.assertTrue(audit["source_snapshot_verified"])
        source.write_text("# Changed after collection.\n")
        with self.assertRaisesRegex(ValueError, "Source snapshot hash mismatch"):
            load_td_index(path, rows, dataset, 1)

    def test_weighted_probability_endpoint_roundoff(self):
        values = {"a": 1., "b": 1., "c": 1.}
        probabilities = {"a": .1, "b": .2, "c": .7000000000000001}
        self.assertEqual(weighted_bootstrap(values, probabilities), 1.)
        self.assertEqual(weighted_bootstrap(dict.fromkeys(values, 0.), probabilities), 0.)
        probabilities["c"] = .70001
        with self.assertRaises(ValueError):
            weighted_bootstrap(values, probabilities)

    def test_malformed_probabilities_rng_and_truncation_rejected(self):
        for name in ("missing", "sum", "rng", "truncation"):
            path, rows, dataset, episode, manifest = fixture(self.root / name)
            if name == "missing": del episode["steps"][1]["behavior_probs"]["north"]
            if name == "sum": episode["steps"][1]["behavior_probs"]["north"] = .1
            if name == "rng": episode["steps"][1]["action"] = "east" if episode["steps"][1]["action"] == "north" else "north"
            if name == "truncation": episode["steps"][1]["truncated"] = True
            rewrite(path, [episode], manifest, dataset)
            with self.assertRaises(ValueError):
                load_td_index(path, rows, dataset, 1)

    def test_split_state_action_question_and_remaining_mismatch_rejected(self):
        for name in ("split", "state", "action", "question", "remaining"):
            path, rows, dataset, _, _ = fixture(self.root / name)
            if name == "split": rows[0]["split"] = "test"
            if name == "state": rows[0]["state"] = "different observation"
            if name == "action": rows[0]["metadata"]["conditioned_action"] = "unexecuted"
            if name == "question": next(iter(rows[0]["questions"].values()))["instructions"] = "Ask about a different action."
            if name == "remaining": rows[0]["metadata"]["remaining_steps"] -= 1
            with self.assertRaises(ValueError):
                load_td_index(path, rows, dataset, 1)

    def test_invalid_n_and_bootstrap_values(self):
        path, rows, dataset, _, _ = fixture(self.root)
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                load_td_index(path, rows, dataset, n)
        for values, probs in (({"east": .2}, {"north": 1}), ({"east": float("nan")}, {"east": 1}),
                              ({"east": .2}, {"east": .8})):
            with self.assertRaises(ValueError):
                weighted_bootstrap(values, probs)


if __name__ == "__main__":
    unittest.main()
