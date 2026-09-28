#!/usr/bin/env python3
"""Compare saved models on identical APPO Basic states and common action targets.

No inference, fitting or checkpoint selection is performed. Expert argmax labels
and original APPO distributions come exclusively from the frozen expert exports,
never a run's training_target. All computations use Python float64 arithmetic.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path


SPLITS = ("calibration", "test", "ood")
PROBABILITY_ATOL = 2e-6
PROBABILITY_RTOL = 1e-5
ECE_BINS = 15


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def decode(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def load(path):
    return decode(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield decode(line)


def valid_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_distribution(mapping, candidates):
    if not isinstance(mapping, dict) or set(mapping) != set(candidates):
        raise ValueError("Expert distribution must cover exactly the offered candidates")
    if any(not valid_number(value) or not 0 <= value <= 1 for value in mapping.values()):
        raise ValueError("Invalid expert probability")
    if abs(math.fsum(mapping.values()) - 1) > 1e-8:
        raise ValueError("Expert target is not a unit distribution; no imputation is performed")


def argmax(mapping):
    maximum = max(mapping.values())
    return min(key for key, value in mapping.items() if value == maximum)


def load_expert(root):
    root = Path(root)
    manifests = {arm: load(root / arm / "manifest.json") for arm in ("hard", "soft")}
    if (manifests["hard"].get("episode_sha256") != manifests["soft"].get("episode_sha256")
            or not manifests["hard"].get("episode_sha256")
            or not isinstance(manifests["hard"].get("collection_policy"), dict)
            or manifests["hard"].get("collection_policy") != manifests["soft"].get("collection_policy")):
        raise ValueError("Hard and soft exports do not identify the same expert collection")
    hashes = {arm: {"manifest_sha256": file_hash(root / arm / "manifest.json"), "splits": {}}
              for arm in manifests}
    targets, all_ids = {}, set()
    for split in SPLITS:
        variants = {}
        for arm in manifests:
            path = root / arm / (split + ".jsonl")
            actual = file_hash(path)
            if manifests[arm].get("split_sha256", {}).get(split) != actual:
                raise ValueError("Expert split hash mismatch: " + arm + "/" + split)
            hashes[arm]["splits"][split] = actual
            variants[arm] = {}
            for row in rows(path):
                rid = row.get("id")
                if not isinstance(rid, str) or not rid or rid in variants[arm]:
                    raise ValueError("Duplicate or invalid expert record ID")
                meta = row.get("metadata", {})
                expected_kind = "expert_action" if arm == "hard" else "expert_distribution"
                if (row.get("split") != split or meta.get("task") != "shooting"
                        or meta.get("spec", {}).get("scenario") != "basic"
                        or meta.get("record_role") != "policy" or meta.get("policy_target_kind") != expected_kind):
                    raise ValueError("Unexpected expert split/task/scenario/target contract")
                if set(row.get("questions", {})) != {"action"} or row["questions"]["action"].get("type") != "choice":
                    raise ValueError("Expected one Basic action Choice per expert state")
                variants[arm][rid] = row
        if not variants["hard"] or set(variants["hard"]) != set(variants["soft"]):
            raise ValueError("Missing or unaligned expert Basic states")
        targets[split] = {}
        for rid, hard in variants["hard"].items():
            soft = variants["soft"][rid]
            if any(hard.get(key) != soft.get(key) for key in ("state", "questions", "state_id", "gold", "family_id", "split")):
                raise ValueError("Hard and soft expert views disagree on inputs/labels")
            candidates = list(hard["questions"]["action"]["criteria"])
            if len(candidates) < 2 or len(candidates) != len(set(candidates)):
                raise ValueError("Invalid expert candidate set")
            target = soft.get("gold_probs", {}).get("action")
            kind = soft.get("gold_probs_kind")
            if (kind.get("action") if isinstance(kind, dict) else kind) != "expert_policy_distribution":
                raise ValueError("Soft target must be the original expert action distribution")
            validate_distribution(target, candidates)
            if any(row.get("expert", {}).get("policy_probs") != target for row in (hard, soft)):
                raise ValueError("Stored gold distribution differs from original APPO probabilities")
            label = hard.get("gold", {}).get("action")
            if label != argmax(target) or hard.get("gold_probs"):
                raise ValueError("Hard target differs from expert argmax or contains a soft override")
            prediction_id = rid + ":action"
            if prediction_id in all_ids:
                raise ValueError("Expert ID reused across splits")
            all_ids.add(prediction_id)
            targets[split][prediction_id] = {"candidate_ids": candidates, "soft": target, "hard": label,
                "state_id": hard["state_id"], "episode_id": hard["metadata"].get("episode_id"),
                "input_sha256": digest({key: hard[key] for key in ("state", "questions")})}
    return targets, {"exports": hashes, "episode_sha256": manifests["hard"]["episode_sha256"],
                     "expert_collection_policy": manifests["hard"]["collection_policy"]}


def probabilities_from_saved(row, target, split):
    if row.get("split") != split or row.get("qid") != "action" or row.get("type") != "choice":
        raise ValueError("Prediction question identity/type/split differs from expert input")
    if row.get("state_id") != target["state_id"]:
        raise ValueError("Prediction state_id differs from the aligned expert state")
    if row.get("task") not in (None, "shooting") or row.get("scenario") not in (None, "basic"):
        raise ValueError("Known Basic prediction is mislabeled as another task/scenario")
    ids, logits, saved = row.get("candidate_ids"), row.get("student_logits"), row.get("student_probs")
    if (not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(target["candidate_ids"])
            or not isinstance(logits, list) or not isinstance(saved, list) or len(logits) != len(ids) or len(saved) != len(ids)):
        raise ValueError("Prediction candidates/logits/probabilities do not align")
    if any(not valid_number(value) for value in logits):
        raise ValueError("Saved logits must be finite")
    if any(not valid_number(value) or not 0 <= value <= 1 for value in saved) or abs(math.fsum(saved) - 1) > 1e-5:
        raise ValueError("Saved probabilities are not a valid float32 simplex")
    maximum = max(logits)
    shifted = [value - maximum for value in logits]
    if any(not math.isfinite(value) for value in shifted):
        raise ValueError("Logit differences overflow float64")
    log_normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    logp = [value - log_normalizer for value in shifted]
    probabilities = [math.exp(value) for value in logp]
    if any(not math.isclose(a, b, rel_tol=PROBABILITY_RTOL, abs_tol=PROBABILITY_ATOL)
           for a, b in zip(probabilities, saved)):
        raise ValueError("Saved probabilities do not match stable softmax of saved logits")
    return dict(zip(ids, probabilities)), dict(zip(ids, logp)), max(abs(a - b) for a, b in zip(probabilities, saved))


def new_metrics():
    return {"values": [], "bins": [{"count": 0, "confidence_sum": 0., "correct": 0} for _ in range(ECE_BINS)]}


def add_metrics(accumulator, p, logp, target):
    q, label = target["soft"], target["hard"]
    selected = argmax(p)
    correct, confidence = int(selected == label), p[selected]
    soft_ce = -math.fsum(q[key] * logp[key] for key in q if q[key] > 0)
    entropy = -math.fsum(value * math.log(value) for value in q.values() if value > 0)
    kl = soft_ce - entropy
    if kl < -1e-8:
        raise ValueError("Negative forward KL beyond roundoff")
    result = {"expert_argmax_agreement": correct, "hard_nll": -logp[label], "expert_soft_ce": soft_ce,
              "expert_soft_kl": max(0., kl), "expert_entropy_nats": entropy,
              "vector_brier_soft": math.fsum((p[key] - q[key]) ** 2 for key in q),
              "vector_brier_hard": math.fsum((p[key] - float(key == label)) ** 2 for key in q)}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Metric cannot be represented as a finite float64 value")
    accumulator["values"].append(result)
    index = min(ECE_BINS - 1, int(confidence * ECE_BINS))
    bucket = accumulator["bins"][index]
    bucket["count"] += 1
    bucket["confidence_sum"] += confidence
    bucket["correct"] += correct


def finish_metrics(accumulator):
    values = accumulator["values"]
    if not values:
        raise ValueError("No aligned Basic predictions")
    n = len(values)
    metrics = {key: math.fsum(value[key] for value in values) / n for key in values[0]}
    bins, ece = [], 0.
    for index, bucket in enumerate(accumulator["bins"]):
        count = bucket["count"]
        confidence = bucket["confidence_sum"] / count if count else None
        agreement = bucket["correct"] / count if count else None
        gap = abs(confidence - agreement) if count else None
        if count:
            ece += count / n * gap
        bins.append({"lower": index / ECE_BINS, "upper": (index + 1) / ECE_BINS,
                     "upper_inclusive": index == ECE_BINS - 1, "n": count,
                     "mean_confidence": confidence, "argmax_agreement": agreement, "absolute_gap": gap})
    return {"n": n, **metrics, "top_label_ece_15": ece, "ece_bins": bins}


def load_run(name, directory, targets):
    directory = Path(directory)
    summary_path, config_path = directory / "summary.json", directory / "config.json"
    summary, config = load(summary_path), load(config_path)
    if summary.get("stage") == "evaluation":
        if summary.get("finished") is not True or summary.get("completed_steps") != 0:
            raise ValueError("Offline evaluation is not complete")
    elif summary.get("completed_steps") != config.get("steps", 0) + config.get("head_steps", 0) or summary.get("completed_steps", 0) <= 0:
        raise ValueError("Training run has not completed its declared updates")
    if summary.get("temperature", 1) != 1 or summary.get("temperature_fitted", False):
        raise ValueError("Use unchanged T=1 predictions; this comparison does not fit a temperature")
    if summary.get("config_sha256") and summary["config_sha256"] != file_hash(config_path):
        raise ValueError("Run config SHA256 mismatch")
    weights = summary.get("weights_sha256")
    if not isinstance(weights, str) or len(weights) != 64 or any(c not in "0123456789abcdef" for c in weights):
        raise ValueError("Run must identify its selected checkpoint SHA256")
    by_split, prediction_hashes = {}, {}
    all_target_ids = set().union(*(set(targets[split]) for split in SPLITS))
    seen_all = set()
    for split in SPLITS:
        path = directory / ("predictions_" + split + ".jsonl")
        actual_hash = file_hash(path)
        if summary.get("prediction_sha256", {}).get(split, actual_hash) != actual_hash:
            raise ValueError("Prediction SHA256 mismatch")
        prediction_hashes[split] = actual_hash
        matched, ignored = set(), Counter()
        accumulator, maximum_error = new_metrics(), 0.
        for row in rows(path):
            rid = row.get("id")
            if not isinstance(rid, str) or not rid or rid in seen_all:
                raise ValueError("Invalid or duplicate prediction ID")
            seen_all.add(rid)
            if row.get("split") != split:
                raise ValueError("Prediction stored under the wrong split")
            if rid not in targets[split]:
                if rid in all_target_ids:
                    raise ValueError("Basic prediction belongs to another split")
                task, scenario = row.get("task"), row.get("scenario")
                if task in ("maze", "snake"):
                    ignored[task] += 1
                elif task == "shooting" and scenario == "predict_position":
                    ignored["shooting/predict_position"] += 1
                else:
                    raise ValueError("Unknown Basic ID or insufficient provenance to classify a non-Basic prediction")
                continue
            target = targets[split][rid]
            p, logp, error = probabilities_from_saved(row, target, split)
            maximum_error = max(maximum_error, error)
            add_metrics(accumulator, p, logp, target)
            matched.add(rid)
        if matched != set(targets[split]):
            raise ValueError(f"{name}/{split}: missing {len(set(targets[split]) - matched)} Basic predictions")
        by_split[split] = {**finish_metrics(accumulator), "ignored_non_basic_predictions": dict(sorted(ignored.items())),
                           "max_saved_probability_absolute_error": maximum_error,
                           "expert_episode_count": len({target["episode_id"] for target in targets[split].values()}),
                           "aligned_ids_sha256": digest(sorted(matched))}
    return {"weights_sha256_as_recorded": weights, "stage_as_recorded": summary.get("stage"),
            "source_hashes": {"summary_sha256": file_hash(summary_path), "config_sha256": file_hash(config_path),
                              "predictions": prediction_hashes}, "by_split": by_split}


def build_report(run_specs, expert):
    targets, provenance = load_expert(expert)
    runs = {}
    for name, directory in run_specs:
        if not name or name in runs:
            raise ValueError("Run names must be nonempty and unique")
        runs[name] = load_run(name, directory, targets)
    if not runs:
        raise ValueError("Provide at least one saved prediction run")
    return {"schema_version": "nanojev-appo-action-fit-v1", "complete": True,
            "implementation_sha256": file_hash(__file__), "expert_source": provenance,
            "common_targets": {split: {"n": len(targets[split]), "aligned_ids_sha256": digest(sorted(targets[split])),
                                         "inputs_and_targets_sha256": digest(targets[split])} for split in SPLITS},
            "runs": runs,
            "definitions": {"weighting": "equal weight per recorded Basic decision; correlated states within episodes",
                "probabilities": "float64 stable log-softmax of saved T=1 logits; no probability floor",
                "saved_probability_check": {"atol": PROBABILITY_ATOL, "rtol": PROBABILITY_RTOL, "simplex_sum_atol": 1e-5},
                "hard_target": "original APPO argmax, with lexicographic tie breaking; not epsilon exploration action",
                "soft_target": "original normalized APPO policy distribution from the common soft export",
                "soft_kl": "KL(expert || student); only tiny negative roundoff is clamped to zero",
                "vector_brier": "sum over every offered action, without dividing by candidate count",
                "ece": "15 equal-width top-label confidence bins on [0,1], final bin includes 1; correctness means expert-argmax agreement",
                "scope": "Offline action-policy fit on common expert-visited states, not probability-of-winning calibration or closed-loop game performance",
                "selection": "none; no fitting, temperature calibration or checkpoint selection by this script",
                "ignored_fields": "Each run's training_target, gold_index, teacher_probs and gold_probs are not used as evaluation targets"}}


def markdown(report):
    lines = ["# APPO action-policy fit", "", "Every model is evaluated on the same frozen expert states and targets.", "",
             "| Model | Split | Decisions | Argmax agreement | Hard NLL | Soft CE | Soft KL | Brier hard | Brier soft | ECE 15 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, run in report["runs"].items():
        safe_name = name.replace("|", "\\|").replace("\n", " ")
        for split, metrics in run["by_split"].items():
            values = [metrics[key] for key in ("hard_nll", "expert_soft_ce", "expert_soft_kl", "vector_brier_hard", "vector_brier_soft", "top_label_ece_15")]
            lines.append(f"| {safe_name} | {split} | {metrics['n']} | {100 * metrics['expert_argmax_agreement']:.2f}% | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines.extend(["", "ECE measures agreement with expert action labels; it is not a calibration metric for winning the game.",
                  "CE, KL and NLL use natural logarithms. Vector Brier sums all action components. Decisions from one episode are correlated.",
                  "No run-specific training targets, temperature fitting or test-based checkpoint selection are used.",
                  "The JSON report includes per-bin ECE counts, ignored non-Basic counts, input IDs and source/checkpoint hashes.", ""])
    return "\n".join(lines)


def self_check():
    import tempfile
    target = {"candidate_ids": ["a", "b"], "soft": {"a": .8, "b": .2}, "hard": "a", "state_id": "state"}
    row = {"split": "test", "qid": "action", "type": "choice", "state_id": "state",
           "candidate_ids": ["b", "a"], "student_logits": [0., 0.], "student_probs": [.5, .5]}
    p, logp, _ = probabilities_from_saved(row, target, "test")
    accumulator = new_metrics()
    add_metrics(accumulator, p, logp, target)
    result = finish_metrics(accumulator)
    assert math.isclose(result["hard_nll"], math.log(2))
    assert math.isclose(result["expert_soft_ce"], math.log(2))
    assert math.isclose(result["expert_soft_kl"], .8 * math.log(1.6) + .2 * math.log(.4))
    assert math.isclose(result["vector_brier_hard"], .5) and math.isclose(result["vector_brier_soft"], .18)
    assert result["expert_argmax_agreement"] == 1 and result["top_label_ece_15"] == .5
    wrong = copy.deepcopy(row)
    wrong["student_probs"] = [.2, .8]
    try:
        probabilities_from_saved(wrong, target, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent saved probabilities accepted")
    endpoint = {**row, "candidate_ids": ["a", "b"], "student_logits": [0., -1000.], "student_probs": [1., 0.]}
    p, logp, _ = probabilities_from_saved(endpoint, target, "test")
    accumulator = new_metrics()
    add_metrics(accumulator, p, logp, target)
    result = finish_metrics(accumulator)
    assert result["ece_bins"][-1]["n"] == 1 and result["ece_bins"][-1]["upper_inclusive"]
    assert math.isclose(result["expert_soft_ce"], 200.)  # log-softmax remains finite despite exp underflow.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        expert_root, run_root = root / "expert", root / "run"
        run_root.mkdir()
        for arm in ("hard", "soft"):
            directory = expert_root / arm
            directory.mkdir(parents=True)
            hashes = {}
            for split in SPLITS:
                example = {"id": split, "state": "A synthetic metric fixture.", "state_id": "state-" + split,
                    "family_id": "unified_shooting", "split": split,
                    "questions": {"action": {"type": "choice", "instructions": "Choose an action.", "criteria": {"a": "A", "b": "B"}}},
                    "gold": {"action": "a"}, "expert": {"policy_probs": {"a": .8, "b": .2}},
                    "metadata": {"task": "shooting", "record_role": "policy", "spec": {"scenario": "basic"},
                                 "episode_id": "episode-" + split,
                                 "policy_target_kind": "expert_action" if arm == "hard" else "expert_distribution"}}
                if arm == "soft":
                    example.update(gold_probs={"action": {"a": .8, "b": .2}}, gold_probs_kind="expert_policy_distribution")
                path = directory / (split + ".jsonl")
                path.write_text(json.dumps(example) + "\n")
                hashes[split] = file_hash(path)
            (directory / "manifest.json").write_text(json.dumps({"episode_sha256": "0" * 64,
                "collection_policy": {"engine": "synthetic_test_fixture"}, "split_sha256": hashes}))
        (run_root / "summary.json").write_text(json.dumps({"stage": "evaluation", "finished": True,
            "completed_steps": 0, "weights_sha256": "1" * 64, "temperature": 1., "temperature_fitted": False}))
        (run_root / "config.json").write_text("{}")
        prediction_rows = {}
        for split in SPLITS:
            prediction = {"id": split + ":action", "state_id": "state-" + split, "split": split, "qid": "action",
                          "type": "choice", "candidate_ids": ["a", "b"], "student_logits": [0., 0.],
                          "student_probs": [.5, .5], "task": "shooting", "scenario": "basic",
                          "training_target": [0., 1.], "gold_index": 1}  # Deliberately contradict the authoritative expert.
            non_basic = {"id": "retained-" + split, "split": split, "task": "snake"}
            prediction_rows[split] = [prediction, non_basic]
            (run_root / ("predictions_" + split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in prediction_rows[split]))
        report = build_report([("fixture", run_root)], expert_root)
        assert report["runs"]["fixture"]["by_split"]["test"]["expert_argmax_agreement"] == 1.
        assert report["runs"]["fixture"]["by_split"]["test"]["ignored_non_basic_predictions"] == {"snake": 1}
        cases, _ = load_expert(expert_root)
        good, retained = prediction_rows["test"]
        unknown = {**good, "id": "unknown-basic:action"}
        for invalid in ([retained], [good, good, retained], [good, retained, unknown]):
            (run_root / "predictions_test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in invalid))
            try:
                load_run("fixture", run_root, cases)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing/duplicate/unknown Basic prediction accepted")
    return {"self_check": "passed", "checks": ["candidate permutation", "hard/soft CE and forward KL", "vector Brier", "ECE endpoint", "no probability floor", "saved-probability rejection", "common targets override run labels", "full coverage and duplicate/unknown rejection", "non-Basic accounting"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--expert", type=Path, help="Expert root containing hard/ and soft/ exports")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
        return
    if not args.expert or not args.output_dir or not args.run:
        parser.error("--expert, --output-dir and at least one --run are required")
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    if any("=" not in value for value in args.run):
        parser.error("Each --run must be NAME=DIRECTORY")
    report = build_report([value.split("=", 1) for value in args.run], args.expert)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.md").write_text(markdown(report))
    print(json.dumps({"complete": True, "models": list(report["runs"]), "n_by_split": {key: value["n"] for key, value in report["common_targets"].items()}}))


if __name__ == "__main__":
    main()
