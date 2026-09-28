#!/usr/bin/env python3
"""Verify and summarize fixed-data MC versus TD controls without model execution.

Example experiment JSON (path_root resolves relative to the experiment file):
  {"path_root":"..", "cases":"configs/unified_games_v1_cases.jsonl",
   "frozen_dataset":"data/unified_v2/outcomes_v2", "expected_case_count":228,
   "training_seeds":[17,29], "mc_condition":"mc",
   "conditions":{"mc":{"config":{"td_n_step":3,"td_weight":0}},
                 "mix_n3":{"config":{"td_n_step":3,"td_weight":0.5}}},
   "shared_training":{"steps":200,"target_update_every":25},
   "arms":[{"name":"mc_seed17","condition":"mc","seed":17,
            "run_dir":"runs/unified_td_v1/mc_seed17",
            "episodes":"data/unified_td_v1/q_mc_seed17.jsonl"}, ...]}

All condition/seed cells must be declared. Missing or unfinished artifacts are
listed, never silently dropped from full-seed aggregates. Available artifacts
with invalid hashes, targets, controls, or cohorts fail validation. Baselines
are game-only episode files, supplied with --baseline NAME=FILE. Standard library
only; writes summary.json and summary.md into a fresh output directory.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import summarize_unified_games as games

TD_PARAMETERS = {"td_n_step", "td_weight", "target_update_every", "steps"}
DERIVED_FIELDS = {"td_audit", "td_enabled", "objective"}
PATH_FIELDS = {"output_dir"}
DEFAULT_CONTROLS = {"stage": "critic", "loss": "brier", "head_steps": 0,
                    "retention_fraction": 0.25, "balance": "task"}
DEFAULT_CONTROLLER = {"engine": "checkpoint", "controller": "q_greedy", "epsilon": 0.15,
                      "temperature": 1.0, "tie_break": "lexicographic_first",
                      "environment_contract": "finite_task_deadline_v1"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(value):
    return hashlib.sha256(games.canon(value).encode()).hexdigest()


def finite(value, where):
    require(games.is_num(value), f"{where}: expected finite number")
    return value


def average(values):
    return math.fsum(values) / len(values)


def mean_range(values, expected):
    present = {str(k): v for k, v in values.items() if v is not None}
    complete = set(present) == {str(s) for s in expected}
    return {"complete": complete, "expected_seeds": expected, "values_by_seed": present,
            "mean": average(list(present.values())) if complete else None,
            "min": min(present.values()) if complete else None,
            "max": max(present.values()) if complete else None}


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict) and value:
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    else:
        result[prefix] = value
    return result


def differences(a, b):
    aa, bb = flatten(a), flatten(b)
    return {key: {"left_present": key in aa, "right_present": key in bb,
                  "left": aa.get(key), "right": bb.get(key)}
            for key in sorted(aa.keys() | bb.keys())
            if key not in aa or key not in bb or aa[key] != bb[key]}


def dataset_reference(path):
    manifest = read_json(path / "manifest.json")
    pid = manifest.get("continuation_policy_id")
    require(isinstance(pid, str) and pid, "Frozen dataset needs a continuation policy ID")
    require(manifest.get("max_states_per_episode") == 0, "Frozen outcomes must retain all decisions")
    hashes, targets = {"manifest.json": games.sha256_file(path / "manifest.json")}, {}
    for split in games.SPLITS:
        source = path / f"{split}.jsonl"
        hashes[source.name] = games.sha256_file(source)
        require(manifest["split_sha256"][split] == hashes[source.name], f"Dataset hash mismatch: {split}")
        expected = {}
        for row in read_rows(source):
            meta = row["metadata"]
            require(row["split"] == split, f"Dataset split mismatch: {row['id']}")
            for qid, question in row["questions"].items():
                key = row["id"] + ":" + qid
                require(key not in expected, f"Duplicate dataset question: {key}")
                target = {"task": meta["task"], "record_role": meta["record_role"]}
                if meta["record_role"] == "outcome":
                    require(meta["continuation_policy_id"] == pid, "Mixed outcome policies in frozen dataset")
                    require(question["type"] == "boolean" and row["gold_label_kind"][qid] == "observed_outcome",
                            f"Expected observed Boolean outcome: {key}")
                    require(type(row["gold"][qid]) is bool, f"Outcome is not Boolean: {key}")
                    target["outcome"] = int(row["gold"][qid])
                expected[key] = target
        targets[split] = expected
    return {"path": str(path), "continuation_policy_id": pid, "hashes": hashes,
            "episode_sha256": manifest.get("episode_sha256"), "targets": targets}


def td_source_reference(path, cases, reference, n_steps):
    run = games.load_run_sources('frozen_td_source', path)
    require(run['cases'] == cases, 'Frozen TD source uses different cases')
    require(run['episodes_sha256'] == reference['episode_sha256'], 'Frozen TD source/dataset episode hashes differ')
    require(run['continuation_policy_id'] == reference['continuation_policy_id'] == digest(run['policy']),
            'Frozen TD source/dataset policy differs')
    stats = {n: {'by_split': {s: {'terminal': 0, 'bootstrap': 0} for s in games.SPLITS},
                 'train_bootstrap_questions': 0} for n in n_steps}
    for episode in read_rows(path):
        split, steps = episode['case']['split'], episode['steps']
        for n in n_steps:
            stats[n]['by_split'][split]['terminal'] += min(n, len(steps))
            stats[n]['by_split'][split]['bootstrap'] += max(0, len(steps) - n)
            if split == 'train':
                stats[n]['train_bootstrap_questions'] += sum(len(step['observation']['candidates']) for step in steps[n:])
    return {'episodes_sha256': run['episodes_sha256'], 'manifest_sha256': run['manifest_sha256'],
            'episodes': len(cases), 'by_n': stats}


def verify_td_metadata(name, config, summary, reference, source):
    active, n = config['td_weight'] > 0, config['td_n_step']
    require(config.get('td_enabled') is active, f'{name}: td_enabled contradicts weight')
    stats = summary.get('td')
    require(isinstance(stats, dict), f'{name}: missing TD summary counters')
    for key, expected in [('enabled', active), ('n_step', n), ('weight', config['td_weight']),
                          ('target_update_every', config['target_update_every'])]:
        require(stats.get(key) == expected, f'{name}: summary TD {key} mismatch')
    refreshes = list(range(config['target_update_every'], summary['completed_steps'] + 1,
                           config['target_update_every'])) if active else []
    require(stats.get('target_refresh_steps') == refreshes, f'{name}: target refresh schedule mismatch')
    if not active:
        require(all(value == 0 for key, value in stats.items()
                    if key.startswith('target_') and key != 'target_update_every' and isinstance(value, (int, float))),
                f'{name}: MC unexpectedly has target forwards')
    if source is None:
        return
    audit, contract = config.get('td_audit'), config.get('td_contract')
    require(isinstance(audit, dict) and isinstance(contract, dict), f'{name}: TD provenance missing')
    expected_audit = {key: source[key] for key in ('episodes_sha256', 'manifest_sha256', 'episodes')}
    expected_audit.update(n_step=n, gamma=1, continuation_policy_id=reference['continuation_policy_id'],
                          outcome_rows=sum(t['record_role'] == 'outcome' for split in reference['targets'].values() for t in split.values()),
                          by_split=source['by_n'][n]['by_split'], complete_transition_coverage=True,
                          cross_split_environment_groups=0, native_game_rewards_used=False,
                          behavior_distribution_and_rng_verified=True)
    for key, expected in expected_audit.items():
        require(audit.get(key) == expected, f'{name}: TD audit {key} mismatch')
    expected_contract = {'active': active, 'gamma': 1, 'continuation_policy_id': reference['continuation_policy_id'],
                         'heldout_target': 'observed terminal outcome only',
                         'bootstrap_policy': 'recorded frozen behavior_probs; never online greedy',
                         'target_microbatch_questions': min(4, config['microbatch_questions']),
                         'max_microbatch_tokens': config['max_microbatch_tokens']}
    for key, expected in expected_contract.items():
        require(contract.get(key) == expected, f'{name}: TD contract {key} mismatch')
    expected_objective = ('Choice full-distribution CE plus observed Boolean MC Brier and frozen-policy soft TD Brier'
                          if active else 'Choice API full-distribution CE plus observed Boolean event loss')
    require(config.get('objective') == expected_objective, f'{name}: objective differs from declared MC/TD mode')
    cache = contract.get('bootstrap_cache')
    if not active:
        require(cache is None, f'{name}: MC unexpectedly builds a target cache')
    else:
        counts = source['by_n'][n]['by_split']['train']
        require(isinstance(cache, dict) and cache.get('outcome_questions') == sum(counts.values()) and
                cache.get('bootstrap_states') == counts['bootstrap'] and
                cache.get('tokenized_boolean_questions') == source['by_n'][n]['train_bootstrap_questions'],
                f'{name}: bootstrap cache counts disagree with recorded episodes')
        require(type(cache.get('max_path_tokens')) is int and 0 <= cache['max_path_tokens'] <= config['max_length'],
                f'{name}: invalid bootstrap token limit')


def normalize_data_hashes(values):
    normalized = {}
    for path, value in values.items():
        name = Path(path).name
        require(name not in normalized, f"Duplicate dataset hash basename: {name}")
        normalized[name] = value
    return normalized


def probability_metrics(path, expected, split, pid, summary):
    seen, cells = set(), defaultdict(list)
    for row in read_rows(path):
        key = row["id"]
        require(key not in seen and key in expected, f"Unexpected/duplicate prediction ID: {key}")
        seen.add(key)
        target = expected[key]
        require(row["split"] == split and row["task"] == target["task"] and
                row["record_role"] == target["record_role"], f"Prediction identity mismatch: {key}")
        if target["record_role"] != "outcome":
            continue
        y = target["outcome"]
        onehot = [float(y == 0), float(y == 1)]
        require(row.get("gold_label_kind") == "observed_outcome" and row.get("gold_index") == y and
                row.get("training_target") == onehot and row.get("target_objective") == "observed_outcome",
                f"Held-out outcome must use actual terminal Y, not a TD target: {key}")
        require(row.get("continuation_policy_id") == pid and row.get("candidate_ids") == ["false", "true"],
                f"Outcome policy/candidates mismatch: {key}")
        logits, probs = row["student_logits"], row["student_probs"]
        require(len(logits) == len(probs) == 2, f"Boolean output shape mismatch: {key}")
        for value in logits + probs:
            finite(value, key)
        require(all(0 <= p <= 1 for p in probs) and abs(sum(probs) - 1) < 2e-6,
                f"Invalid probability distribution: {key}")
        maximum = max(logits)
        partition = math.fsum(math.exp(v - maximum) for v in logits)
        softmax = [math.exp(v - maximum) / partition for v in logits]
        require(max(abs(p - q) for p, q in zip(probs, softmax)) < 2e-6,
                f"Probability/logit mismatch: {key}")
        nll = math.log(partition) + (maximum - logits[y])
        brier = math.fsum((p - t) ** 2 for p, t in zip(probs, onehot))
        cells[target["task"]].append((nll, brier))
    require(seen == set(expected), f"Missing prediction questions in {path}: {len(set(expected) - seen)}")
    by_task = {}
    for task, values in cells.items():
        metrics = {"questions": len(values), "nll": average([v[0] for v in values]),
                   "vector_brier": average([v[1] for v in values])}
        stored = summary["by_task_role"][task + "/outcome"]
        require(stored["questions"] == len(values) and stored.get("excluded_questions", 0) == 0,
                f"Outcome summary count mismatch: {split}/{task}")
        require(abs(stored["ce"] - metrics["nll"]) < 2e-6 and
                abs(stored["brier"] - metrics["vector_brier"]) < 2e-6,
                f"Outcome summary metrics mismatch: {split}/{task}")
        by_task[task] = metrics
    require(by_task, f"No observed outcomes in {path}")
    macro = {"questions": sum(v["questions"] for v in by_task.values()), "tasks": sorted(by_task),
             **{key: average([v[key] for v in by_task.values()]) for key in ("nll", "vector_brier")}}
    stored = summary["by_role_macro_task"]["outcome"]
    require(stored["questions"] == macro["questions"] and abs(stored["ce"] - macro["nll"]) < 2e-6 and
            abs(stored["brier"] - macro["vector_brier"]) < 2e-6, f"Outcome macro mismatch: {split}")
    return {"prediction_sha256": games.sha256_file(path), "by_task": by_task, "macro_equal_task": macro}


def completed_rollout(name, path, cases, pending):
    manifest = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest.is_file():
        pending.append({"name": name, "artifact": "episodes", "status": "missing", "path": str(path)})
        return None
    if read_json(manifest).get("finished") is not True:
        pending.append({"name": name, "artifact": "episodes", "status": "unfinished", "path": str(path)})
        return None
    run = games.load_run_sources(name, path)
    require(run["cases"] == cases, f"{name}: rollout cohort differs from frozen case definitions")
    for source in run["sources"]:
        require(source["continuation_policy_id"] == digest(source["policy"]), f"{name}: invalid policy identity hash")
    return run


def cost_record(run_dir, summary):
    log_path = run_dir / "train_log.json"
    log = read_json(log_path) if log_path.is_file() else None
    batch_hashes = [row.get("batch_question_ids_sha256") for row in log] if log is not None else None
    if batch_hashes is not None:
        require(len(log) == summary["completed_steps"], f"{run_dir}: training log update count mismatch")
        require(all(isinstance(h, str) and h for h in batch_hashes), f"{run_dir}: incomplete batch hashes")
    td_version_verified = log is not None and all(isinstance(row.get('td'), dict) for row in log)
    if td_version_verified:
        active, interval = summary['td']['enabled'], summary['td']['target_update_every']
        for index, row in enumerate(log, 1):
            require(row['step'] == index, f'{run_dir}: nonsequential log updates')
            expected_version = ((index - 1) // interval) * interval if active else None
            expected_refresh = index if active and index % interval == 0 else None
            require(row['td'].get('target_version_step') == expected_version and
                    row['td'].get('target_refreshed_to_step') == expected_refresh,
                    f'{run_dir}: target version/log schedule mismatch')
    initial_metrics_path = run_dir / 'initial_dev_metrics.json'
    selection_verified = log is not None and initial_metrics_path.is_file()
    if selection_verified:
        initial = read_json(initial_metrics_path)
        candidates = [(0, finite(initial['selection_ce'], 'initial dev CE'))]
        candidates.extend((row['step'], finite(row['dev']['selection_ce'], 'logged dev CE'))
                          for row in log if 'dev' in row)
        selected = min(candidates, key=lambda item: item[1])
        require(selected[0] == summary['best_step'] and abs(selected[1] - summary['best_dev_selection_ce']) < 1e-9,
                f'{run_dir}: selected checkpoint differs from minimum recorded dev CE')
    return {"training_seconds": summary.get("training_seconds"),
            "max_gpu_allocated_gb": summary.get("max_gpu_allocated_gb"),
            "td": summary.get("td"), "online_compute": summary.get("online_compute"),
            "training_token_counts": {k: v for k, v in summary.items()
                                      if any(s in k for s in ("token", "leaf_paths", "forward_calls"))},
            "training_token_scope": "Only explicitly recorded counters; missing totals are not estimated.",
            "train_log_sha256": games.sha256_file(log_path) if log is not None else None,
            "target_version_log_verified": td_version_verified,
            "dev_selection_verified_from_initial_and_log": selection_verified,
            "initial_dev_metrics_sha256": games.sha256_file(initial_metrics_path) if initial_metrics_path.is_file() else None,
            "batch_sequence_sha256": digest(batch_hashes) if batch_hashes is not None else None}


def paired_game(left, right):
    def cell(a, b):
        aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
        require(aa.keys() == bb.keys(), "Paired episode IDs disagree")
        return {"episodes": len(aa), "mc_successes": sum(r["success"] for r in a),
                "other_successes": sum(r["success"] for r in b),
                "other_only_success": sum(bb[k]["success"] and not aa[k]["success"] for k in aa),
                "mc_only_success": sum(aa[k]["success"] and not bb[k]["success"] for k in aa),
                "success_rate_delta": average([int(bb[k]["success"]) - int(aa[k]["success"]) for k in aa])}
    out = {}
    for split in games.SPLITS:
        a = [r for r in left["per_episode"] if r["split"] == split]
        b = [r for r in right["per_episode"] if r["split"] == split]
        if not a:
            continue
        tasks = sorted({r["task"] for r in a})
        task = {t: cell([r for r in a if r["task"] == t], [r for r in b if r["task"] == t]) for t in tasks}
        variant = {v: cell([r for r in a if r["variant"] == v], [r for r in b if r["variant"] == v])
                   for v in sorted({r["variant"] for r in a})}
        out[split] = {"task_macro_success_delta": average([v["success_rate_delta"] for v in task.values()]),
                      "by_task": task, "by_variant": variant}
    return out


def build_summary(experiment_path, baselines=None):
    experiment_path = Path(experiment_path).resolve()
    exp = read_json(experiment_path)
    root = (experiment_path.parent / exp.get("path_root", ".")).resolve()
    resolve = lambda p: (root / p).resolve()
    seeds, conditions, arms = exp.get("training_seeds", exp.get("seeds")), exp["conditions"], exp["arms"]
    require(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds), "Invalid experiment seeds")
    require(isinstance(conditions, dict) and conditions, "conditions must be an object")
    mc = exp.get("mc_condition", "mc")
    require(mc in conditions, "Missing MC condition")
    require(conditions[mc]["config"].get("td_weight") == 0, "MC condition must declare td_weight=0")
    for name, condition in conditions.items():
        require(set(condition["config"]) <= TD_PARAMETERS, f"{name}: undeclared type of condition control")
        require({"td_n_step", "td_weight"} <= condition["config"].keys(), f"{name}: TD n and weight required")
        require(type(condition['config']['td_n_step']) is int and condition['config']['td_n_step'] > 0 and
                games.is_num(condition['config']['td_weight']) and 0 <= condition['config']['td_weight'] <= 1,
                f'{name}: invalid TD n or weight')
    by_cell = {(a["condition"], a["seed"]): a for a in arms}
    require(len(by_cell) == len(arms) and len({a["name"] for a in arms}) == len(arms), "Duplicate experiment arm")
    require(set(by_cell) == {(c, s) for c in conditions for s in seeds}, "Declare every condition/seed arm")
    cases_path = resolve(exp["cases"])
    case_rows = list(read_rows(cases_path))
    cases = {r["id"]: r for r in case_rows}
    require(len(cases) == len(case_rows) == exp.get("expected_case_count", 228), "Frozen case count mismatch")
    if 'case_sha256' in exp:
        require(games.sha256_file(cases_path) == exp['case_sha256'], 'Registered case hash mismatch')
    reference = dataset_reference(resolve(exp["frozen_dataset"]))
    if 'continuation_policy_id' in exp:
        require(reference['continuation_policy_id'] == exp['continuation_policy_id'], 'Registered continuation policy mismatch')
    td_source = (td_source_reference(resolve(exp['td_episodes']), cases, reference,
                 {c['config']['td_n_step'] for c in conditions.values()}) if exp.get('td_episodes') else None)
    controls = {**DEFAULT_CONTROLS, **exp.get("shared_training", exp.get("controls", {}))}
    pending, configs, training, rollouts = [], {}, {}, {}
    for arm in arms:
        name, run_dir = arm["name"], resolve(arm["run_dir"])
        config_path, summary_path = run_dir / "config.json", run_dir / "summary.json"
        if not config_path.is_file() or not summary_path.is_file():
            pending.append({"name": name, "artifact": "training", "status": "missing", "path": str(run_dir)})
        else:
            config, summary = read_json(config_path), read_json(summary_path)
            expected = {**controls, **conditions[arm["condition"]]["config"], "seed": arm["seed"]}
            for key, value in expected.items():
                require(config.get(key) == value, f"{name}: control {key} differs from experiment declaration")
            require(normalize_data_hashes(config["data_sha256"]) == reference["hashes"], f"{name}: frozen dataset hashes differ")
            require(config.get("continuation_policy_id") == reference["continuation_policy_id"] == summary.get("continuation_policy_id"),
                    f"{name}: frozen outcome policy differs")
            require(summary["completed_steps"] == config["steps"] + config.get("head_steps", 0), f"{name}: unfinished updates")
            require(summary.get("selected_on") == "dev only", f"{name}: checkpoint selection is not dev only")
            require(summary.get("stage") == config["stage"] and summary.get("loss") == config["loss"], f"{name}: summary/config objective mismatch")
            require(config.get("init_weights_sha256"), f"{name}: missing warm-start weights hash")
            if 'initial_checkpoint_sha256' in exp:
                require(config['init_weights_sha256'] == exp['initial_checkpoint_sha256'], f'{name}: registered warm start mismatch')
            require(summary.get("weights_sha256"), f"{name}: missing selected weights hash")
            verify_td_metadata(name, config, summary, reference, td_source)
            weights_verified_locally = False
            if (run_dir / "best.safetensors").is_file():
                require(games.sha256_file(run_dir / "best.safetensors") == summary["weights_sha256"], f"{name}: local weights hash mismatch")
                weights_verified_locally = True
            configs[name] = config
            probabilities = {}
            for split in ("dev", "calibration", "test", "ood"):
                pred = run_dir / f"predictions_{split}.jsonl"
                if not pred.is_file():
                    pending.append({"name": name, "artifact": f"predictions_{split}", "status": "missing", "path": str(pred)})
                    continue
                probabilities[split] = probability_metrics(pred, reference["targets"][split], split,
                        reference["continuation_policy_id"], summary["metrics_by_split"][split])
            training[name] = {"config_sha256": games.sha256_file(config_path), "summary_sha256": games.sha256_file(summary_path),
                              "weights_sha256": summary["weights_sha256"], "init_weights_sha256": config["init_weights_sha256"],
                              "best_step": summary["best_step"], "completed_steps": summary["completed_steps"],
                              "best_dev_selection_ce": summary["best_dev_selection_ce"], "probabilities": probabilities,
                              "weights_rehashed_locally": weights_verified_locally,
                              "weight_binding": "Selected-weight hash and exact config-byte hash must match the rollout manifest; local weight bytes are optional.",
                              "policy_metrics_by_split": {s: m["by_role_macro_task"].get("policy")
                                  for s, m in summary["metrics_by_split"].items()}, "cost": cost_record(run_dir, summary)}
        run = completed_rollout(name, resolve(arm["episodes"]), cases, pending)
        if run is not None:
            require(name in training, f"{name}: completed rollout lacks completed training summary")
            policy = run["policy"]
            declared_controller = exp.get('controller', {})
            expected_controller = {**DEFAULT_CONTROLLER,
                                  **{k: v for k, v in declared_controller.items() if k in DEFAULT_CONTROLLER}}
            rollout_seed = declared_controller.get('seed', exp.get('rollout_seed'))
            if rollout_seed is not None:
                expected_controller['sampling_seed'] = rollout_seed
            for key, value in expected_controller.items():
                require(policy.get(key) == value, f"{name}: unexpected rollout controller {key}")
            weights = policy.get("checkpoint_sha256", {})
            require(weights.get("best.safetensors") == training[name]["weights_sha256"], f"{name}: rollout checkpoint mismatch")
            require(weights.get("config.json") == training[name]["config_sha256"], f"{name}: rollout config hash mismatch")
            rollouts[name] = run
    config_comparison = []
    if configs:
        first = next(iter(configs))
        for name in configs:
            diff = differences(configs[first], configs[name])
            derived_contract = ('td_contract.active', 'td_contract.bootstrap_cache')
            disallowed = {key: value for key, value in diff.items()
                          if key.split('.')[0] not in TD_PARAMETERS | DERIVED_FIELDS | PATH_FIELDS | {"seed"}
                          and not any(key == field or key.startswith(field + '.') for field in derived_contract)}
            require(not disallowed, f"Uncontrolled training config differences {first} vs {name}: {list(disallowed)}")
            for key in TD_PARAMETERS:
                if configs[first].get(key) != configs[name].get(key):
                    require(all(key in conditions[a["condition"]]["config"] or key in controls
                                for a in arms if a["name"] in (first, name)), f"Undeclared varying control: {key}")
            config_comparison.append({"runs": [first, name], "differences": diff,
                                      "derived_fields_reported": sorted(DERIVED_FIELDS), "uncontrolled_differences": []})
    for name, path in (baselines or {}).items():
        require(name not in {a['name'] for a in arms}, f"Baseline collides with arm name: {name}")
        run = completed_rollout(name, Path(path).resolve(), cases, pending)
        if run is not None:
            rollouts[name] = run
    game_summaries = {n: games.summarize_run(r) for n, r in rollouts.items()}
    policy_diff = games.policy_differences(list(rollouts.values())) if rollouts else []
    paired = {}
    for condition in conditions:
        if condition == mc:
            continue
        paired[condition] = {}
        for seed in seeds:
            a, b = by_cell[mc, seed]['name'], by_cell[condition, seed]['name']
            item = {"mc": a, "other": b, "seed": seed, "games": None, "probability_deltas": {}}
            if a in training and b in training:
                ca, cb = training[a]['cost'], training[b]['cost']
                item['same_training_batch_sequence'] = (ca['batch_sequence_sha256'] == cb['batch_sequence_sha256']
                    if ca['batch_sequence_sha256'] and cb['batch_sequence_sha256'] else None)
                require(item['same_training_batch_sequence'] is not False,
                        f'{a} vs {b}: same-seed online question sequence differs')
                for split in training[a]['probabilities'].keys() & training[b]['probabilities'].keys():
                    pa, pb = training[a]['probabilities'][split], training[b]['probabilities'][split]
                    item['probability_deltas'][split] = {key: pb['macro_equal_task'][key] - pa['macro_equal_task'][key]
                                                         for key in ('nll', 'vector_brier')}
            if a in rollouts and b in rollouts:
                matched = next(d for d in policy_diff if set(d['runs']) == {a, b})
                require(matched['same_controller'], f"{a} vs {b}: paired controllers differ")
                require(rollouts[a]['policy'].get('source_sha256') == rollouts[b]['policy'].get('source_sha256'),
                        f"{a} vs {b}: rollout implementations differ")
                item['games'] = paired_game(game_summaries[a], game_summaries[b])
            paired[condition][str(seed)] = item
    aggregates = {}
    for condition in conditions:
        aggregate = {"game_task_macro": {}, "game_variants": {}, "probabilities": {}}
        names = {seed: by_cell[condition, seed]['name'] for seed in seeds}
        for split in games.SPLITS:
            vals = {seed: game_summaries[n]['by_split_task_macro'].get(split, {}).get('task_macro_success_rate')
                    for seed, n in names.items() if n in game_summaries}
            aggregate['game_task_macro'][split] = mean_range(vals, seeds)
            for variant in sorted({c['variant'] for c in cases.values() if c['split'] == split}):
                key = f"{split}/{next(c['spec']['task'] for c in cases.values() if c['variant'] == variant)}/{variant}"
                vals = {seed: game_summaries[n]['by_split_task_variant'][key]['success_rate']
                        for seed, n in names.items() if n in game_summaries}
                aggregate['game_variants'][key] = mean_range(vals, seeds)
            for metric in ('nll', 'vector_brier'):
                vals = {seed: training[n]['probabilities'][split]['macro_equal_task'][metric]
                        for seed, n in names.items() if n in training and split in training[n]['probabilities']}
                if split != 'train':
                    aggregate['probabilities'][f'{split}/{metric}'] = mean_range(vals, seeds)
        aggregates[condition] = aggregate
    paired_aggregates = {condition: {split: mean_range({seed: item['games'][split]['task_macro_success_delta']
            for seed, item in pairs.items() if item['games'] and split in item['games']}, seeds)
            for split in games.SPLITS} for condition, pairs in paired.items()}
    return {"schema_version": "nanojev-unified-td-comparison-v1", "complete": not pending,
            "generated_at": datetime.now(timezone.utc).isoformat(), "experiment": exp,
            "experiment_sha256": games.sha256_file(experiment_path), "missing": pending,
            "cohort": {"case_count": len(cases), "cases_file_sha256": games.sha256_file(cases_path),
                       "canonical_sha256": digest([cases[k] for k in sorted(cases)])},
            "frozen_probability_dataset": {k: v for k, v in reference.items() if k != 'targets'},
            "td_source": td_source,
            "rollout_execution_parameters": {k: exp.get('controller', {}).get(k) for k in ('env_batch', 'batch_questions', 'max_length')},
            "rollout_execution_verification": "Batch sizes and max_length are registered CLI settings, absent from legacy policy manifests; they are not independently attested here.",
            "training": training, "config_comparison": config_comparison,
            "policies": {n: r['sources'] for n, r in rollouts.items()}, "policy_differences": policy_diff,
            "runs": game_summaries, "paired_vs_mc": paired, "condition_aggregates": aggregates,
            "paired_game_aggregates": paired_aggregates,
            "notes": ["Probability metrics are recomputed against fixed observed terminal outcomes under the dataset policy.",
                      "NLL uses finite logits and stable log-sum-exp without probability clipping; Brier sums both Boolean classes.",
                      "Game metrics come from new completed Q-controller episodes, with every success and failure retained.",
                      "Deltas are other minus MC, paired by seed and exact case ID; task macros equally weight tasks.",
                      "Across-seed summaries give mean and observed range only, without a significance claim.",
                      "All declared seeds are required for a condition aggregate; incomplete cells remain null.",
                      "Training time and recorded token/target counters are computational costs, not probability or game metrics."]}


def render_markdown(result):
    registered = result['experiment'].get('primary_comparison', {})
    lines = ['# Unified MC versus TD comparison', '', f"Complete: **{result['complete']}**. Frozen cases: {result['cohort']['case_count']}.", '',
             'All probability metrics use the fixed observed-outcome dataset. Game scores use new Q-controller trajectories. '
             'Across-seed summaries describe mean and observed range; no significance test is inferred.', '']
    if registered:
        lines.extend([f"Predeclared primary comparison: **{registered['treatment']} versus {registered['control']}**.", ''])
    def table(headers, rows):
        lines.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        lines.append('')
    def fmt(value):
        return 'pending' if value is None else f'{value:.6f}'
    def spread(value):
        return 'pending' if not value['complete'] else f"{value['mean']:.4f} [{value['min']:.4f}, {value['max']:.4f}]"
    if result['missing']:
        lines.extend(['## Pending artifacts', ''])
        table(['Arm', 'Artifact', 'Status'], [[v['name'], v['artifact'], v['status']] for v in result['missing']])
    lines.extend(['## Fixed-policy probability scores by arm', ''])
    table(['Arm', 'Split', 'Questions', 'NLL', 'Vector Brier'], [[name, split, m['macro_equal_task']['questions'],
            fmt(m['macro_equal_task']['nll']), fmt(m['macro_equal_task']['vector_brier'])]
            for name, run in result['training'].items() for split, m in run['probabilities'].items()])
    lines.extend(['## Probability scores by task', ''])
    table(['Arm', 'Split', 'Task', 'Questions', 'NLL', 'Vector Brier'], [[name, split, task, m['questions'], fmt(m['nll']), fmt(m['vector_brier'])]
            for name, run in result['training'].items() for split, metrics in run['probabilities'].items() for task, m in metrics['by_task'].items()])
    lines.extend(['## Game success by variant', ''])
    table(['Arm', 'Split/task/variant', 'Successes / episodes'], [[name, key, f"{m['successes']}/{m['n']}"]
            for name, run in result['runs'].items() for key, m in run['by_split_task_variant'].items()])
    lines.extend(['## Condition task-macro success: seed mean [min, max]', ''])
    table(['Condition', 'Split', 'Success rate'], [[name, split, spread(value)] for name, agg in result['condition_aggregates'].items()
            for split, value in agg['game_task_macro'].items()])
    lines.extend(['## Fixed-policy probability scores: seed mean [min, max]', ''])
    table(['Condition', 'Split/metric', 'Score'], [[name, key, spread(value)]
            for name, agg in result['condition_aggregates'].items() for key, value in agg['probabilities'].items()])
    lines.extend(['## Paired difference from MC: seed mean [min, max]', '', 'Positive game deltas favor the named condition.', ''])
    table(['Condition', 'Split', 'Task-macro success delta'], [[name, split, spread(value)]
            for name, splits in result['paired_game_aggregates'].items() for split, value in splits.items()])
    lines.extend(['## Training and target-computation costs', ''])
    table(['Arm', 'Selected / completed updates', 'Training seconds', 'Peak GPU GB', 'Online padded tokens', 'Target forwards', 'Target padded tokens', 'Target forward seconds'],
          [[name, f"{run['best_step']}/{run['completed_steps']}", fmt(run['cost']['training_seconds']),
            fmt(run['cost']['max_gpu_allocated_gb']), (run['cost']['online_compute'] or {}).get('padded_tokens', 'not recorded'),
            (run['cost']['td'] or {}).get('target_forward_calls', 'not recorded'),
            (run['cost']['td'] or {}).get('target_padded_tokens', 'not recorded'),
            (run['cost']['td'] or {}).get('target_forward_seconds', 'not recorded')] for name, run in result['training'].items()])
    lines.extend(['## Verification', '', 'The JSON report retains source hashes, checkpoint/config matches, every declared configuration difference, '
                  'controller differences, paired case counts, and policy-retention metrics. Missing results remain explicit.', '', *['- ' + n for n in result['notes']], ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--baseline', action='append', default=[], metavar='NAME=EPISODES_JSONL')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output_dir.exists(), '--output-dir must be fresh')
        baselines = games.parse_named(args.baseline, '--baseline')
        result = build_summary(args.experiment, baselines)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        (args.output_dir / 'summary.md').write_text(render_markdown(result))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(f"Wrote {args.output_dir}; complete={result['complete']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
