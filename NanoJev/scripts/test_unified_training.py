#!/usr/bin/env python3
"""Contract tests for mixed policy/event training; no API, downloads or GPU.

Run: python -m unittest discover -s scripts -p test_unified_training.py -v
Numerical tests require the project's torch dependency and otherwise skip clearly.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import train_unified_games as unified


class FakeTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(char) for char in text]


def record(task="maze", role="policy", split="train", suffix="0", policy_id="frozen-policy-v1"):
    rid = f"{task}:{role}:{split}:{suffix}"
    meta = {"task": task, "record_role": role, "source_group_id": f"{task}:{split}:{suffix}",
            "episode_id": f"{task}:{split}:{suffix}"}
    row = {"id": rid, "state_id": rid, "family_id": task, "split": split,
           "state": "Agent at an intersection with a visible target.", "metadata": meta}
    if role == "policy":
        row["questions"] = {"action": {"type": "choice", "instructions": "Select the next physical action.",
                                        "criteria": {"left": "Move left.", "right": "Move right.", "wait": "Wait."}}}
        row["teacher"] = {"native_probs": {"action": {"left": .2, "right": .7, "wait": .1}},
                          "rounding": {"probabilityDecimals": 2}}
    else:
        row["questions"] = {"win": {"type": "boolean", "instructions": "After moving left, does this episode end in success under the frozen continuation policy?",
                                     "criteria": {"true": "The episode succeeds.", "false": "The episode does not succeed."}}}
        row["gold"] = {"win": False}
        row["gold_label_kind"] = {"win": "observed_outcome"}
        meta.update(continuation_policy_id=policy_id, executed_action="left", conditioned_action="left")
    return row


MANIFEST = {"continuation_policy_id": "frozen-policy-v1"}


def examples(rows):
    unified.validate_unified_records(rows, MANIFEST)
    return unified.prepare_unified_examples(rows, FakeTokenizer(), 4096)[0]


def full_rows(split="train"):
    return [record(task, role, split) for task in unified.TASKS for role in unified.ROLES]


class ProvenanceTests(unittest.TestCase):
    def test_mixed_policy_version_rejected(self):
        with self.assertRaisesRegex(ValueError, "continuation_policy_id"):
            unified.validate_unified_records([record(role="outcome", policy_id="another-policy")], MANIFEST)
        with self.assertRaisesRegex(ValueError, "continuation_policy_id"):
            unified.validate_unified_records([record(role="outcome")], {})

    def test_api_target_cannot_replace_event_truth(self):
        row = record(role="outcome")
        row.pop("gold")
        row["teacher"] = {"native_probs": {"win": {"false": .01, "true": .99}}}
        with self.assertRaisesRegex(ValueError, "actual Boolean gold"):
            unified.validate_unified_records([row], MANIFEST)

    def test_conflicting_api_does_not_change_observed_target(self):
        row = record(role="outcome")
        row["teacher"] = {"native_probs": {"win": {"false": 0.0, "true": 1.0}}}
        ex = examples([row])[0]
        self.assertEqual(unified.target_for(ex, unified.objective_for(ex)), [1.0, 0.0])
        self.assertEqual(ex["teacher_probs"], [0.0, 1.0])

    def test_unexecuted_action_cannot_receive_observation(self):
        row = record(role="outcome")
        row["metadata"]["conditioned_action"] = "right"
        with self.assertRaisesRegex(ValueError, "unexecuted"):
            unified.validate_unified_records([row], MANIFEST)
        row["metadata"]["conditioned_action_by_question"] = {"win": "right"}
        with self.assertRaisesRegex(ValueError, "unexecuted"):
            unified.validate_unified_records([row], MANIFEST)

    def test_missing_action_provenance_rejected(self):
        row = record(role="outcome")
        row["metadata"].pop("executed_action")
        with self.assertRaisesRegex(ValueError, "executed action"):
            unified.validate_unified_records([row], MANIFEST)

    def test_structured_joint_action_supported(self):
        row = record(task="shooting", role="outcome")
        row["metadata"].update(executed_action={"turn": 1, "fire": True}, conditioned_action={"fire": True, "turn": 1})
        unified.validate_unified_records([row], MANIFEST)

    def test_soft_gold_cannot_be_mislabeled_real_observation(self):
        row = record(role="outcome")
        row.update(gold_probs={"win": {"false": .5, "true": .5}}, gold_probs_kind="programmatic_conditional_distribution")
        with self.assertRaisesRegex(ValueError, "substituted soft"):
            unified.validate_unified_records([row], MANIFEST)

    def test_rounded_policy_quarantine_does_not_invent_target(self):
        row = record()
        row["teacher"]["native_probs"]["action"] = {"left": .33, "right": .33, "wait": .33}
        audit = unified.validate_unified_records([row], MANIFEST)
        self.assertEqual(audit["quarantined_policy_questions"], 1)
        self.assertIsNone(examples([row])[0]["teacher_probs"])

    def test_episode_cannot_cross_splits(self):
        train, dev = record(role="outcome"), record(role="outcome", split="dev")
        dev["metadata"]["episode_id"] = train["metadata"]["episode_id"]
        with self.assertRaisesRegex(ValueError, "episode crosses"):
            unified.validate_unified_records([train, dev], MANIFEST)

    def test_episode_audit_counts_complete_trials_not_repeated_states(self):
        first = record(role="outcome", suffix="first")
        repeated = record(role="outcome", suffix="second")
        repeated["metadata"]["episode_id"] = first["metadata"]["episode_id"]
        success = record(role="outcome", suffix="third")
        success["gold"]["win"] = True
        audit = unified.validate_unified_records([first, repeated, success], MANIFEST)
        self.assertEqual(audit["unique_episodes_by_split_task_role"]["train/maze/outcome"], 2)
        self.assertEqual(audit["outcome_episode_counts_by_split_task"]["train/maze"], {"successes": 1, "failures": 1})
        self.assertEqual(audit["outcome_state_label_counts_by_split_task"]["train/maze"], {"true": 1, "false": 2})

    def test_conflicting_eventual_success_for_one_episode_is_rejected(self):
        first, second = record(role="outcome"), record(role="outcome", suffix="later")
        second["metadata"]["episode_id"] = first["metadata"]["episode_id"]
        second["gold"]["win"] = True
        with self.assertRaisesRegex(ValueError, "Conflicting eventual-success"):
            unified.validate_unified_records([first, second], MANIFEST)

    def test_one_class_outcomes_are_audited_without_forced_rebalancing(self):
        rows = [record(role="outcome", suffix=str(i)) for i in range(3)]
        audit = unified.validate_unified_records(rows, MANIFEST)
        self.assertEqual(audit["outcome_episode_counts_by_split_task"]["train/maze"], {"successes": 0, "failures": 3})
        self.assertEqual(audit["outcome_question_label_counts_by_split_task"]["train/maze"], {"true": 0, "false": 3})

    def test_missing_episode_identity_is_not_invented(self):
        row = record(role="outcome")
        row["metadata"].pop("episode_id")
        audit = unified.validate_unified_records([row], MANIFEST)
        self.assertEqual(audit["unique_episodes_by_split_task_role"]["train/maze/outcome"], 0)
        self.assertEqual(audit["rows_missing_episode_id_by_split_task_role"]["train/maze/outcome"], 1)

    def test_metadata_never_enters_tokens(self):
        row = record(role="outcome")
        baseline = examples([row])[0]["leaf_tokens"]
        row["metadata"].update(oracle_action="right", future_win=True, policy_private_notes="never encode me")
        row["gold"]["win"] = True
        self.assertEqual(examples([row])[0]["leaf_tokens"], baseline)

    def test_five_split_file_placement_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps(MANIFEST))
            for split in unified.SPLITS:
                (root / f"{split}.jsonl").write_text("")
            (root / "train.jsonl").write_text(json.dumps(record(split="dev")) + "\n")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                unified.read_unified_dataset(root)


class BalancedSamplingTests(unittest.TestCase):
    def test_unequal_corpus_does_not_change_task_mass_or_drop_retention(self):
        rows = full_rows()
        rows.extend(record("maze", "outcome", suffix=str(i)) for i in range(1, 80))
        sampler = unified.BalancedQuestionSampler(examples(rows), "critic", retention_fraction=.25)
        for _ in range(12):
            batch = sampler.sample(11)  # Deliberately not divisible by task or role count.
            for task in unified.TASKS:
                mass = math.fsum(ex["loss_weight"] for ex in batch if ex["task"] == task)
                self.assertAlmostEqual(mass, 1 / 3)
                for role, fraction in (("policy", .25), ("outcome", .75)):
                    mass = math.fsum(ex["loss_weight"] for ex in batch if ex["task"] == task and ex["record_role"] == role)
                    self.assertAlmostEqual(mass, fraction / 3)

    def test_task_role_balancing_assigns_six_equal_cells(self):
        sampler = unified.BalancedQuestionSampler(examples(full_rows()), "critic", balance="task_role")
        batch = sampler.sample(13)
        for task in unified.TASKS:
            for role in unified.ROLES:
                mass = math.fsum(ex["loss_weight"] for ex in batch if (ex["task"], ex["record_role"]) == (task, role))
                self.assertAlmostEqual(mass, 1 / 6)

    def test_no_heldout_or_outcome_training_in_sft(self):
        exs = examples(full_rows() + full_rows("dev") + full_rows("test"))
        sampler = unified.BalancedQuestionSampler(exs, "sft")
        batch = sampler.sample(12)
        self.assertEqual({ex["split"] for ex in batch}, {"train"})
        self.assertEqual({ex["record_role"] for ex in batch}, {"policy"})

    def test_reproducible_sampling_and_missing_task_rejection(self):
        exs = examples(full_rows())
        a = unified.BalancedQuestionSampler(exs, "critic", seed=3)
        b = unified.BalancedQuestionSampler(exs, "critic", seed=3)
        self.assertEqual(a.sample(15), b.sample(15))
        with self.assertRaisesRegex(ValueError, "Missing eligible"):
            unified.BalancedQuestionSampler([ex for ex in exs if ex["task"] != "shooting"], "critic")
        with self.assertRaisesRegex(ValueError, "at least 6"):
            a.sample(5)

    def test_every_loss_keeps_policy_retention_as_ce(self):
        policy, outcome = examples([record(), record(role="outcome")])
        for kind in unified.LOSSES:
            self.assertEqual(unified.loss_spec(policy, "critic", kind), ("teacher", "ce"))
            self.assertEqual(unified.loss_spec(outcome, "critic", kind), ("observed_outcome", kind))
        with self.assertRaisesRegex(ValueError, "cannot enter SFT"):
            unified.loss_spec(outcome, "sft", "ce")

    def test_dev_selection_is_task_macro_not_row_micro(self):
        rows = []
        for task, count, p in (("maze", 100, .9), ("snake", 2, .8), ("shooting", 1, .2)):
            for _ in range(count):
                rows.append({"task": task, "record_role": "policy", "training_target": [1., 0.],
                             "student_logits": [math.log(p), math.log(1 - p)], "student_probs": [p, 1 - p]})
        summary = unified.summarize_predictions(rows, unified.population_weights("sft"), require_all=True)
        self.assertAlmostEqual(summary["selection_ce"], sum(-math.log(p) for p in (.9, .8, .2)) / 3)
        with self.assertRaisesRegex(ValueError, "Dev selection lacks"):
            unified.summarize_predictions(rows[:-1], unified.population_weights("sft"), require_all=True)

    def test_defaults_and_cli_loss_validation(self):
        self.assertEqual(unified.parse_args(["--input", "unused", "--stage", "sft", "--validate-only"]).loss, "ce")
        self.assertEqual(unified.parse_args(["--input", "unused", "--stage", "critic", "--validate-only"]).loss, "paired_brier_pg")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed; numerical checks require project dependencies")
class NumericalTrainingTests(unittest.TestCase):
    def test_padding_and_microbatch_partition_leave_ce_and_brier_gradients_unchanged(self):
        import torch
        batch = unified.BalancedQuestionSampler(examples(full_rows()), "critic").sample(11)
        for loss_kind in ("ce", "brier"):
            z = torch.randn(len(batch), 5, generator=torch.Generator().manual_seed(7), requires_grad=True)
            loss = unified.mixed_question_loss(z, batch, "critic", loss_kind)
            loss.backward()
            split = z.detach().clone().requires_grad_()
            for start, end in ((0, 1), (1, 5), (5, len(batch))):
                unified.mixed_question_loss(split[start:end], batch[start:end], "critic", loss_kind).backward()
            self.assertTrue(torch.allclose(z.grad, split.grad, atol=1e-7, rtol=1e-6))
            for i, ex in enumerate(batch):
                self.assertEqual(float(z.grad[i, len(ex["candidate_ids"]):].abs().sum()), 0.)

    def test_paired_updates_are_independent_of_conflicting_api_event_distribution(self):
        import torch
        original = examples([record(role="outcome")])[0]
        original["loss_weight"] = 1.
        conflicting = copy.deepcopy(original)
        conflicting["teacher_probs"] = [0., 1.]
        gradients = []
        for ex in (original, conflicting):
            z = torch.tensor([[0., .6]], requires_grad=True)
            unified.mixed_question_loss(z, [ex], "critic", "paired_brier_pg", 32,
                                       torch.Generator().manual_seed(9)).backward()
            gradients.append(z.grad.clone())
        self.assertTrue(torch.equal(*gradients))

    def test_policy_gradient_is_identical_across_event_loss_variants(self):
        import torch
        ex = examples([record()])[0]
        ex["loss_weight"] = .25
        gradients = []
        for kind in unified.LOSSES:
            z = torch.tensor([[.1, .2, -.4, 1000.]], requires_grad=True)
            unified.mixed_question_loss(z, [ex], "critic", kind, generator=torch.Generator().manual_seed(0)).backward()
            gradients.append(z.grad.clone())
        self.assertTrue(all(torch.equal(gradients[0], value) for value in gradients[1:]))
        self.assertEqual(float(gradients[0][0, -1]), 0.)

    def test_paired_mean_gradient_matches_direct_brier_for_observed_boolean(self):
        import torch
        ex = examples([record(role="outcome")])[0]
        ex["loss_weight"] = 1.
        z = torch.tensor([[0., .4]], requires_grad=True)
        unified.mixed_question_loss(z, [ex], "critic", "brier").backward()
        reference = z.grad.detach()
        averaged = torch.zeros_like(reference)
        rng = torch.Generator().manual_seed(654)
        for _ in range(600):
            sample = z.detach().clone().requires_grad_()
            unified.mixed_question_loss(sample, [ex], "critic", "paired_brier_pg", 32, rng).backward()
            averaged += sample.grad / 600
        self.assertTrue(torch.allclose(averaged, reference, atol=.015, rtol=.05), (averaged, reference))


if __name__ == "__main__":
    unittest.main()
