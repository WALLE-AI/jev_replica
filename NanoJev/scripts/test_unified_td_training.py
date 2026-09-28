#!/usr/bin/env python3
"""TD training contracts and CPU numerical checks; no simulator, API or GPU."""
import contextlib
import copy
import importlib.util
import io
import math
from types import SimpleNamespace
import unittest

import train_unified_games as training
from test_unified_training import FakeTokenizer, examples, full_rows, record


class Tokenizer(FakeTokenizer):
    pad_token_id = 0


def arguments(**updates):
    values = dict(precision="fp32", microbatch_questions=8,
                  max_microbatch_tokens=16384, max_length=8192)
    values.update(updates)
    return SimpleNamespace(**values)


def bootstrap_entry(action_count=2):
    actions = [f"a{i}" for i in range(action_count)]
    mapping = {action: "success_" + action for action in actions}
    questions = {qid: {"type": "boolean", "instructions": f"Take {action} now and follow the frozen policy. Will the task succeed before its deadline?",
                       "criteria": {"false": "The task fails.", "true": "The task succeeds."}}
                 for action, qid in mapping.items()}
    return {"episode_id": "frozen-training-episode", "bootstrap_decision_index": 3,
            "request": {"id": "future-observation", "state": "Visible local state, remaining time 8.",
                        "questions": questions},
            "behavior_probs": {action: 1 / action_count for action in actions},
            "action_to_qid": mapping}


def outcome_example(success=False):
    row = record(role="outcome")
    row["gold"]["win"] = success
    result = examples([row])[0]
    result["loss_weight"] = 1.
    return result


def index_for(example, entry):
    return {(example["source"]["id"], example["qid"]): entry}


class PlainModel:
    """Only the model-copy contract; no neural inference is simulated here."""
    def __init__(self):
        self.backbone = SimpleNamespace()
        self.value, self.training, self.requires_grad = 1., True, True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, values, strict=True):
        self.value = values["value"]


class TDContractTests(unittest.TestCase):
    def parse(self, extra):
        return training.parse_args(["--input", "unused", "--stage", "critic", "--loss", "brier",
                                    "--validate-only"] + extra)

    def test_td_is_opt_in_and_endpoint_weights_are_valid(self):
        args = self.parse([])
        self.assertEqual(args.td_weight, 0.)
        self.assertIsNone(args.td_episodes)
        self.assertEqual(args.target_update_every, 25)
        for weight in ("0", "0.5", "1"):
            self.assertEqual(self.parse(["--td-episodes", "episodes.jsonl", "--td-weight", weight]).td_weight,
                             float(weight))

    def test_cli_rejects_wrong_objective_warmup_and_invalid_sizes(self):
        bad = [["--td-weight", ".5"], ["--td-weight", "nan"], ["--td-weight", "1.01"],
               ["--td-n-step", "0"], ["--target-update-every", "0"],
               ["--td-episodes", "e", "--loss", "paired_brier_pg"],
               ["--td-episodes", "e", "--loss", "ce"],
               ["--td-episodes", "e", "--head-steps", "1"],
               ["--td-episodes", "e", "--stage", "sft", "--loss", "ce"]]
        for args in bad:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(args)

    def test_target_snapshot_is_independent_and_refresh_is_explicit(self):
        ex = outcome_example()
        online = PlainModel()
        provider = training.TDTargetProvider(online, index_for(ex, {"terminal_target": 0.}),
                                              [ex], Tokenizer(), arguments())
        online.value = 7.
        self.assertEqual(provider.target_model.value, 1.)
        self.assertTrue(online.training)
        self.assertFalse(provider.target_model.training)
        self.assertFalse(provider.target_model.requires_grad)
        provider.refresh(online, 25)
        self.assertEqual(provider.target_model.value, 7.)
        self.assertEqual(provider.refresh_steps, [25])
        with self.assertRaises(ValueError):
            provider.refresh(online, 25)

    def test_heldout_and_mismatched_terminal_truth_are_rejected(self):
        ex = outcome_example()
        with self.assertRaisesRegex(ValueError, "actual final Boolean"):
            training.TDTargetProvider(PlainModel(), index_for(ex, {"terminal_target": 1.}),
                                      [ex], Tokenizer(), arguments())
        ex["split"] = "test"
        with self.assertRaisesRegex(ValueError, "only receive training"):
            training.TDTargetProvider(PlainModel(), index_for(ex, {"terminal_target": 0.}),
                                      [ex], Tokenizer(), arguments())

    def test_behavior_and_question_mapping_must_be_complete(self):
        ex = outcome_example()
        for mutation in ("missing_action", "missing_question", "nonunit", "nonfinite"):
            entry = bootstrap_entry()
            if mutation == "missing_action":
                entry["behavior_probs"].pop("a1")
            elif mutation == "missing_question":
                entry["request"]["questions"].pop("success_a1")
            elif mutation == "nonunit":
                entry["behavior_probs"]["a0"] = .7
            else:
                entry["behavior_probs"]["a0"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                training.TDTargetProvider(PlainModel(), index_for(ex, entry), [ex], Tokenizer(), arguments())

    def test_complete_boolean_inputs_enforce_budget_without_truncation(self):
        ex = outcome_example()
        with self.assertRaisesRegex(ValueError, "over budget"):
            training.TDTargetProvider(PlainModel(), index_for(ex, bootstrap_entry()), [ex], Tokenizer(),
                                      arguments(max_microbatch_tokens=1))
        provider = training.TDTargetProvider(PlainModel(), index_for(ex, bootstrap_entry(7)), [ex],
                                              Tokenizer(), arguments())
        self.assertEqual(provider.max_questions, 4)
        self.assertEqual(provider.cache_stats["tokenized_boolean_questions"], 7)

    def test_retention_and_question_sampling_do_not_depend_on_td_settings(self):
        exs = examples(full_rows())
        first = training.BalancedQuestionSampler(exs, "critic", retention_fraction=.25, seed=17)
        second = training.BalancedQuestionSampler(exs, "critic", retention_fraction=.25, seed=17)
        for _ in range(4):
            a, b = first.sample(24), second.sample(24)
            self.assertEqual([x["id"] for x in a], [x["id"] for x in b])
            self.assertAlmostEqual(sum(x["loss_weight"] for x in a if x["record_role"] == "policy"), .25)

    def test_empty_target_batch_packs_without_a_forward(self):
        self.assertEqual(training.pack_complete_questions([], 4, 16384), [])

    def test_heldout_metrics_use_real_outcome_not_td_pseudotarget(self):
        row = {"task": "maze", "record_role": "outcome", "training_target": [1., 0.],
               "student_logits": [math.log(.8), math.log(.2)], "student_probs": [.8, .2],
               "gold_index": 0, "td_target": .99}
        result = training.summarize_predictions([row], {("maze", "outcome"): 1.})
        self.assertAlmostEqual(result["by_task_role"]["maze/outcome"]["brier"], .08)
        self.assertAlmostEqual(result["by_task_role"]["maze/outcome"]["ce"], -math.log(.8))


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU numerical TD checks require torch")
class TDNumericalTests(unittest.TestCase):
    def model(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.backbone.weight.fill_(1.)
                self.calls = []

            def forward(self, group, pad_token):
                self.calls.append({"questions": len(group), "training": self.training,
                                   "gradient_enabled": torch.is_grad_enabled()})
                factors = torch.tensor([1. if q["qid"].endswith("a0") else -1. for q in group])
                z = self.backbone.weight.reshape(()) * factors
                return torch.stack((torch.zeros_like(z), z), dim=-1), None

        return TinyModel()

    def test_expected_sarsa_uses_recorded_pi_and_never_max_q(self):
        ex, entry = outcome_example(), bootstrap_entry()
        entry["behavior_probs"] = {"a0": .1, "a1": .9}
        provider = training.TDTargetProvider(self.model(), index_for(ex, entry), [ex], Tokenizer(), arguments())
        result, stats = provider.targets([ex, copy.deepcopy(ex)])
        high = 1 / (1 + math.exp(-1))
        self.assertAlmostEqual(result[ex["id"]], .1 * high + .9 * (1 - high), places=6)
        self.assertLess(result[ex["id"]], high)
        self.assertEqual(stats["bootstrap_target_instances"], 2)
        self.assertEqual(stats["target_question_predictions"], 2)
        self.assertTrue(all(not c["training"] and not c["gradient_enabled"] for c in provider.target_model.calls))

    def test_refresh_changes_target_values_and_predictions_are_never_cached(self):
        import torch
        ex, online = outcome_example(), self.model()
        entry = bootstrap_entry(1)
        provider = training.TDTargetProvider(online, index_for(ex, entry), [ex], Tokenizer(), arguments())
        first = provider.targets([ex])[0][ex["id"]]
        with torch.no_grad():
            online.backbone.weight.fill_(-2.)
        self.assertAlmostEqual(provider.targets([ex])[0][ex["id"]], first)
        provider.refresh(online, 25)
        after = provider.targets([ex])[0][ex["id"]]
        self.assertAlmostEqual(after, 1 / (1 + math.exp(2)), places=6)
        self.assertNotEqual(first, after)
        self.assertEqual(provider.totals["target_forward_calls"], 3)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in provider.target_model.parameters()))

    def test_all_one_and_all_zero_target_outputs_tolerate_only_roundoff(self):
        import torch

        class ConstantModel(torch.nn.Module):
            def __init__(self, logit):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.backbone.weight.fill_(logit)

            def forward(self, group, pad_token):
                z = self.backbone.weight.reshape(()).expand(len(group))
                return torch.stack((torch.zeros_like(z), z), dim=-1), None

        ex, entry = outcome_example(), bootstrap_entry()
        # A valid floating-point simplex can sum to just above one.
        entry["behavior_probs"] = {"a0": .5000000000000002, "a1": .5}
        for logit, expected in ((1000., 1.), (-1000., 0.)):
            provider = training.TDTargetProvider(ConstantModel(logit), index_for(ex, entry), [ex],
                                                  Tokenizer(), arguments())
            self.assertEqual(provider.targets([ex])[0][ex["id"]], expected)
        entry["behavior_probs"]["a0"] = .501
        with self.assertRaisesRegex(ValueError, "simplex"):
            training.TDTargetProvider(ConstantModel(1000.), index_for(ex, entry), [ex], Tokenizer(), arguments())

    def test_terminals_need_no_target_inference_and_large_action_sets_are_split(self):
        ex = outcome_example(True)
        terminal = training.TDTargetProvider(self.model(), index_for(ex, {"terminal_target": 1.}),
                                             [ex], Tokenizer(), arguments())
        values, stats = terminal.targets([ex])
        self.assertEqual(values[ex["id"]], 1.)
        self.assertEqual(stats["target_forward_calls"], 0)
        bootstrap = training.TDTargetProvider(self.model(), index_for(ex, bootstrap_entry(7)),
                                              [ex], Tokenizer(), arguments())
        _, stats = bootstrap.targets([ex])
        self.assertEqual(stats["target_forward_calls"], 2)
        self.assertEqual(stats["target_question_predictions"], 7)
        self.assertEqual([c["questions"] for c in bootstrap.target_model.calls], [4, 3])

    def test_soft_target_is_detached_and_loss_endpoints_are_correct(self):
        import torch
        ex = outcome_example(True)
        for w in (0., .5, 1.):
            z = torch.tensor([[.2, -.6, 99.]], requires_grad=True)
            target = torch.tensor(.3, requires_grad=True)
            loss = training.mixed_question_loss(z, [ex], "critic", "brier",
                                                td_targets={ex["id"]: target}, td_weight=w)
            p = z[0, :2].softmax(-1)
            expected = (1 - w) * ((p[0]) ** 2 + (p[1] - 1) ** 2) + w * ((p[0] - .7) ** 2 + (p[1] - .3) ** 2)
            self.assertAlmostEqual(float(loss), float(expected), places=7)
            loss.backward()
            self.assertIsNone(target.grad)
            self.assertEqual(float(z.grad[0, 2]), 0.)

    def test_td_mixture_preserves_microbatch_and_retention_gradients(self):
        import torch
        batch = training.BalancedQuestionSampler(examples(full_rows()), "critic").sample(24)
        targets = {ex["id"]: .3 for ex in batch if ex["record_role"] == "outcome"}
        z = torch.randn(24, 5, generator=torch.Generator().manual_seed(7), requires_grad=True)
        training.mixed_question_loss(z, batch, "critic", "brier", td_targets=targets, td_weight=.5).backward()
        reference = z.detach().clone().requires_grad_()
        for start in range(0, len(batch), 4):
            training.mixed_question_loss(reference[start:start+4], batch[start:start+4], "critic", "brier",
                                        td_targets=targets, td_weight=.5).backward()
        self.assertTrue(torch.allclose(z.grad, reference.grad, atol=1e-7, rtol=1e-6))
        mc = z.detach().clone().requires_grad_()
        training.mixed_question_loss(mc, batch, "critic", "brier").backward()
        for i, ex in enumerate(batch):
            if ex["record_role"] == "policy":
                self.assertTrue(torch.equal(z.grad[i], mc.grad[i]))

    def test_missing_nonfinite_or_noncategorical_targets_fail(self):
        import torch
        ex = outcome_example()
        for mapping in ({}, {ex["id"]: float("nan")}, {ex["id"]: 1.1}):
            with self.assertRaises(ValueError):
                training.mixed_question_loss(torch.tensor([[0., 1.]]), [ex], "critic", "brier",
                                            td_targets=mapping, td_weight=.5)
        with self.assertRaisesRegex(ValueError, "critic Brier"):
            training.mixed_question_loss(torch.tensor([[0., 1.]]), [ex], "critic", "paired_brier_pg",
                                        td_targets={ex["id"]: .5}, td_weight=.5)


def run_real_model_smoke(dataset, episodes_path, checkpoint, precision="bf16", disable_native_triton=True):
    """Explicit manual GPU smoke, never called by unittest discovery.

    Uses six real training questions (one eligible policy and one bootstrap
    outcome per task), performs two updates, and never evaluates heldout data
    or saves a modified checkpoint. The caller controls GPU assignment.
    """
    import torch
    from predict_toy_decisions import DecisionPredictor
    from unified_td import load_td_index

    records, manifest, _, _ = training.read_unified_dataset(dataset)
    index, audit = load_td_index(episodes_path, records, manifest, 3)
    selected = []
    for task in training.TASKS:
        policy = next(row for row in records if row["split"] == "train" and
                      row["metadata"]["task"] == task and row["metadata"]["record_role"] == "policy" and
                      all(t["teacher_probs"] is not None for t in training.validate_training_row(row).values()))
        outcomes = [row for row in records if row["split"] == "train" and
                    row["metadata"]["task"] == task and row["metadata"]["record_role"] == "outcome" and
                    all("request" in index[(row["id"], qid)] for qid in row["questions"])]
        if not outcomes:
            raise ValueError("Smoke needs at least one nonterminal bootstrap training row per task")
        outcomes.sort(key=lambda row: not any(row["gold"].values()))
        selected.extend((policy, outcomes[0]))
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    runtime = DecisionPredictor(checkpoint, max_length=8192, precision=precision,
                                disable_native_triton=disable_native_triton)
    model = runtime.model
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batch, _ = training.prepare_unified_examples(selected, runtime.tokenizer, 8192)
    assert len(batch) == 6
    for ex in batch:
        ex["loss_weight"] = (.25 if ex["record_role"] == "policy" else .75) / 3
    args = arguments(precision=precision, microbatch_questions=8, max_microbatch_tokens=32768)
    provider = training.TDTargetProvider(model, index, batch, runtime.tokenizer, args)
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": 2e-5},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")], "lr": 2e-4}
    ], weight_decay=.01)
    initial_scalar = model.scalar.weight.detach().clone()
    backbone_probe_name, backbone_probe = next(
        (name, value) for name, value in model.backbone.named_parameters()
        if "norm" in name.lower() and value.numel() <= 8192)
    initial_backbone_probe = backbone_probe.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    checks = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        targets, stats = provider.targets(batch)
        differences = [abs(targets[ex["id"]] - ex["gold_index"]) for ex in batch if ex["record_role"] == "outcome"]
        assert any(delta > 1e-6 for delta in differences), "Smoke TD target unexpectedly equals every MC label"
        model.train()
        total = 0.
        for group in training.pack_complete_questions(batch, 8, 32768):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits, _ = model(group, runtime.tokenizer.pad_token_id)
            loss = training.mixed_question_loss(logits, group, "critic", "brier", td_targets=targets, td_weight=.5)
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        assert all(p.grad is None and not p.requires_grad for p in provider.target_model.parameters())
        assert backbone_probe.grad is not None and torch.isfinite(backbone_probe.grad).all()
        backbone_grad_absmax = float(backbone_probe.grad.detach().abs().max())
        assert backbone_grad_absmax > 0, "Smoke observed no backbone gradient"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        provider.refresh(model, step)
        assert all(torch.equal(p.detach(), q.detach()) for p, q in
                   zip(model.parameters(), provider.target_model.parameters()))
        assert all(p.dtype == torch.float32 for p in model.parameters())
        checks.append({"step": step, "loss": total, "td_mc_target_max_difference": max(differences),
                       "backbone_probe_gradient_absmax": backbone_grad_absmax,
                       "target_refreshed_to_step": provider.version_step, **stats})
    assert not torch.equal(initial_scalar, model.scalar.weight.detach())
    assert not torch.equal(initial_backbone_probe, backbone_probe.detach())
    torch.cuda.synchronize()
    return {"status": "passed", "updates": 2, "training_questions": len(batch),
            "task_count": 3, "target_gradients_none": True, "target_hardcopies_exact": True,
            "fp32_parameter_update_observed": True, "heldout_forward_calls": 0,
            "head_parameter_update_observed": True, "backbone_parameter_update_observed": True,
            "backbone_probe_parameter": backbone_probe_name,
            "online_microbatch_questions": 8, "target_microbatch_questions": 4,
            "max_microbatch_tokens": 32768,
            "max_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "checkpoint_saved": False, "continuation_policy_id": audit["continuation_policy_id"],
            "steps": checks}


if __name__ == "__main__":
    unittest.main()
