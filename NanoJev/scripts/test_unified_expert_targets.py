#!/usr/bin/env python3
"""Expert target and fixed policy-pool contracts; no GPU, API or model download."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import train_unified_games as training
from test_unified_training import FakeTokenizer, MANIFEST, examples, full_rows, record


POOL_WEIGHTS = {"maze/policy": 1 / 3, "snake/policy": 1 / 3,
                "shooting/policy/basic": 453 / 2910,
                "shooting/policy/predict_position": 517 / 2910}


def expert(kind="expert_action", split="train", suffix="expert"):
    row = record("shooting", split=split, suffix=suffix)
    row["metadata"].update(policy_target_kind=kind, executed_action="left",
                           spec={"scenario": "basic"})
    row["gold"] = {"action": "right"}
    row["gold_label_kind"] = "reference_argmax_compatibility"
    # The API favors left, while the expert label is right. Behavior also took left.
    row["teacher"]["native_probs"]["action"] = {"left": .9, "right": .05, "wait": .05}
    if kind == "expert_distribution":
        row["gold_probs"] = {"action": {"left": .1, "right": .2, "wait": .7}}
        row["gold_probs_kind"] = "expert_policy_distribution"
    return row


def four_pools(split="train"):
    pp = record("shooting", split=split, suffix="pp")
    pp["metadata"]["spec"] = {"scenario": "predict_position"}
    return [record("maze", split=split), record("snake", split=split), expert(split=split), pp]


class ExpertTargetTests(unittest.TestCase):
    def test_hard_expert_action_is_not_exploration_action_or_api_target(self):
        row = expert()
        ex = examples([row])[0]
        self.assertNotEqual(row["gold"]["action"], row["metadata"]["executed_action"])
        self.assertEqual(training.target_for(ex, training.objective_for(ex)), [0., 1., 0.])
        self.assertEqual(ex["teacher_probs"], [.9, .05, .05])
        for loss in training.LOSSES:
            self.assertEqual(training.loss_spec(ex, "critic", loss), ("gold_distribution", "ce"))

    def test_soft_expert_uses_complete_distribution_and_specific_provenance(self):
        row = expert("expert_distribution")
        ex = examples([row])[0]
        self.assertEqual(training.target_for(ex, training.objective_for(ex)), [.1, .2, .7])
        self.assertEqual(training.target_transform(ex), "identity_expert_distribution")
        row["gold_probs_kind"] = "programmatic_conditional_distribution"
        with self.assertRaisesRegex(ValueError, "expert_policy_distribution"):
            examples([row])

    def test_declared_expert_never_falls_back_to_api(self):
        row = expert()
        row.pop("gold")
        with self.assertRaisesRegex(ValueError, "hard gold"):
            examples([row])
        row = expert("expert_distribution")
        row.pop("gold_probs")
        with self.assertRaisesRegex(ValueError, "expert_policy_distribution"):
            examples([row])

    def test_hard_target_forbids_soft_override_even_when_one_hot(self):
        row = expert()
        row["gold_probs"] = {"action": {"left": 0., "right": 1., "wait": 0.}}
        row["gold_probs_kind"] = "expert_policy_distribution"
        with self.assertRaisesRegex(ValueError, "soft gold override"):
            examples([row])

    def test_undeclared_old_gold_does_not_change_legacy_api_supervision(self):
        row = record()
        row["gold"] = {"action": "left"}
        ex = examples([row])[0]
        self.assertEqual(training.objective_for(ex), "teacher")
        self.assertEqual(training.target_for(ex, "teacher"), [.2, .7, .1])

    def test_policy_target_cannot_be_an_observed_event_or_unknown_kind(self):
        row = expert()
        row["gold_label_kind"] = "observed_outcome"
        with self.assertRaisesRegex(ValueError, "observed event"):
            examples([row])
        row = expert("invented_expert")
        with self.assertRaisesRegex(ValueError, "policy_target_kind"):
            examples([row])
        row = record(role="outcome")
        row["metadata"]["policy_target_kind"] = "expert_action"
        with self.assertRaisesRegex(ValueError, "Outcome rows"):
            examples([row])

    def test_expert_provenance_and_labels_do_not_enter_input_tokens(self):
        row = expert()
        initial = examples([row])[0]["leaf_tokens"]
        row["gold"]["action"] = "wait"
        row["metadata"].update(expert_rnn_digest="private-rnn", executed_action="right",
                               expert_pixel_digest="private-pixels", expert_checkpoint="another-model")
        self.assertEqual(initial, examples([row])[0]["leaf_tokens"])

    def test_audit_distinguishes_all_target_transforms(self):
        rows = [record(), expert(), expert("expert_distribution", suffix="soft"), record(role="outcome")]
        audit = training.validate_unified_records(rows, MANIFEST)
        self.assertEqual(audit["quarantined_policy_questions"], 0)
        self.assertEqual(audit["policy_questions_by_split_task_target_kind"]["train/shooting/expert_action"], 1)
        _, targets = training.prepare_unified_examples(rows, FakeTokenizer(), 4096)
        self.assertEqual([row["target_transform"] for row in targets],
                         ["identity_rounded_proxy", "expert_action_one_hot",
                          "identity_expert_distribution", "observed_boolean_one_hot"])


class PoolContractTests(unittest.TestCase):
    def test_hard_and_soft_arms_share_exact_question_sampling_sequence(self):
        hard = four_pools() + [expert(suffix=f"extra-{i}") for i in range(12)]
        soft = copy.deepcopy(hard)
        for row in soft:
            if row["metadata"].get("policy_target_kind") == "expert_action":
                row["metadata"]["policy_target_kind"] = "expert_distribution"
                row["gold_probs"] = {"action": {"left": .2, "right": .7, "wait": .1}}
                row["gold_probs_kind"] = "expert_policy_distribution"
        samplers = [training.BalancedQuestionSampler(examples(rows), "sft", seed=17,
                                                     policy_pool_weights=POOL_WEIGHTS) for rows in (hard, soft)]
        for _ in range(200):
            batches = [sampler.sample(24) for sampler in samplers]
            self.assertEqual([(ex["id"], ex["loss_weight"]) for ex in batches[0]],
                             [(ex["id"], ex["loss_weight"]) for ex in batches[1]])

    def test_each_update_preserves_task_and_scenario_mass_despite_corpus_sizes(self):
        rows = four_pools()
        rows.extend(expert(suffix=f"extra-{i}") for i in range(140))
        sampler = training.BalancedQuestionSampler(examples(rows), "sft", seed=17,
                                                   policy_pool_weights=POOL_WEIGHTS)
        for _ in range(20):
            batch = sampler.sample(23)  # Not divisible by four or three.
            for key, mass in POOL_WEIGHTS.items():
                actual = math.fsum(ex["loss_weight"] for ex in batch
                                   if "/".join(training.sampling_cell(ex, sampler.weights)) == key)
                self.assertAlmostEqual(actual, mass, places=14)
            self.assertAlmostEqual(sum(ex["loss_weight"] for ex in batch), 1.)
            self.assertEqual({ex["split"] for ex in batch}, {"train"})

    def test_pool_samples_remain_uniform_within_pool(self):
        rows = four_pools() + [expert(suffix="second")]
        exs = examples(rows)
        for ex in exs:
            ex["probe_value"] = float(ex["source"]["metadata"]["episode_id"].endswith("second"))
        sampler = training.BalancedQuestionSampler(exs, "sft", seed=47, policy_pool_weights=POOL_WEIGHTS)
        means = [sum(ex["loss_weight"] * ex["probe_value"] for ex in sampler.sample(24)) for _ in range(500)]
        self.assertAlmostEqual(sum(means) / len(means), POOL_WEIGHTS["shooting/policy/basic"] / 2, delta=.006)

    def test_configuration_missing_extra_or_reweighted_task_is_rejected(self):
        bad = copy.deepcopy(POOL_WEIGHTS)
        bad.pop("shooting/policy/basic")
        cases = [bad, {**POOL_WEIGHTS, "shooting/policy/other": .1},
                 {**POOL_WEIGHTS, "maze/policy": .4},
                 {**POOL_WEIGHTS, "maze/policy": True},
                 {**POOL_WEIGHTS, "maze/policy": float("nan")}]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                training.population_weights("sft", policy_pool_weights=config)

    def test_missing_eligible_pool_or_unknown_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing eligible.*predict_position"):
            training.BalancedQuestionSampler(examples(four_pools()[:-1]), "sft", policy_pool_weights=POOL_WEIGHTS)
        rows = four_pools()
        rows[2]["metadata"]["spec"]["scenario"] = "other_scenario"
        with self.assertRaisesRegex(ValueError, "Unconfigured policy pool"):
            training.BalancedQuestionSampler(examples(rows), "sft", policy_pool_weights=POOL_WEIGHTS)

    def test_pools_are_checked_before_model_load_and_dev_cannot_be_missing(self):
        weights = training.population_weights("sft", policy_pool_weights=POOL_WEIGHTS)
        rows = four_pools() + four_pools("dev")
        counts = training.validate_sampling_pools(rows, weights)
        self.assertEqual(counts["dev/shooting/policy/basic"], 1)
        with self.assertRaisesRegex(ValueError, "Missing eligible dev/shooting/policy/predict_position"):
            training.validate_sampling_pools(rows[:-1], weights)

    def test_critic_and_td_cannot_silently_accept_custom_policy_pools(self):
        with self.assertRaisesRegex(ValueError, "only for SFT"):
            training.population_weights("critic", policy_pool_weights=POOL_WEIGHTS)
        exs = examples(full_rows())
        default = training.BalancedQuestionSampler(exs, "critic", seed=29)
        explicit_none = training.BalancedQuestionSampler(exs, "critic", seed=29, policy_pool_weights=None)
        self.assertEqual(default.sample(24), explicit_none.sample(24))

    def test_cli_reads_json_and_enforces_four_pool_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pools.json"
            path.write_text(json.dumps(POOL_WEIGHTS))
            argv = ["--input", "unused", "--stage", "sft", "--validate-only", "--policy-pool-weights", str(path)]
            args = training.parse_args(argv)
            self.assertEqual(args.resolved_policy_pool_weights, POOL_WEIGHTS)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(argv + ["--batch-questions", "3"])

    def test_dev_selection_uses_frozen_pools_not_new_basic_row_count(self):
        rows = []
        probabilities = {"maze/policy": .9, "snake/policy": .8,
                         "shooting/policy/basic": .1, "shooting/policy/predict_position": .7}
        for key, p in probabilities.items():
            task, role, *scenario = key.split("/")
            row = {"task": task, "record_role": role, "scenario": scenario[0] if scenario else None,
                   "policy_target_kind": "expert_action" if scenario == ["basic"] else "api_policy_distribution",
                   "training_target": [1., 0.], "student_logits": [math.log(p), math.log(1 - p)],
                   "student_probs": [p, 1 - p]}
            rows.extend([row] * (500 if scenario == ["basic"] else 1))
        weights = training.population_weights("sft", policy_pool_weights=POOL_WEIGHTS)
        result = training.summarize_predictions(rows, weights, require_all=True)
        expected = sum(POOL_WEIGHTS[key] * -math.log(p) for key, p in probabilities.items())
        self.assertAlmostEqual(result["selection_ce"], expected, places=12)
        self.assertEqual(result["by_task_role"]["shooting/policy"]["objective"], "mixed_policy_supervision")
        self.assertEqual(result["by_task_role_target_source"]["shooting/policy/expert_action"]["questions"], 500)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable; CPU numerical checks need project dependencies")
class ExpertNumericalTests(unittest.TestCase):
    def test_hard_expert_ce_gradient_ignores_behavior_and_conflicting_api(self):
        import torch
        ex = examples([expert()])[0]
        ex["loss_weight"] = 1.
        z = torch.tensor([[.3, -.7, .1, 999.]], requires_grad=True)
        training.mixed_question_loss(z, [ex], "sft", "ce").backward()
        expected = z.detach()[0, :3].softmax(-1) - torch.tensor([0., 1., 0.])
        self.assertTrue(torch.allclose(z.grad[0, :3], expected, atol=1e-7))
        self.assertEqual(z.grad[0, 3].item(), 0.)


if __name__ == "__main__":
    unittest.main()
