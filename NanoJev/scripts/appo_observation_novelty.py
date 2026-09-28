#!/usr/bin/env python3
"""Freeze initial-input overlap groups, then describe matched rollout results.

With no --summary, write the immutable --novelty JSON from expert initial
standard observations only. With --summary, read that partition and a completed
APPO-supervision summary and write --output. Never select cases by outcomes.
This is exact initial-input overlap, not spatial or latent-environment novelty.
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from summarize_appo_supervision import digest, paired, require, rows
from summarize_unified_games import SPLITS, group_stats, is_num, sha256_file
from unified_game_pipeline import policy_request

SCHEMA = "nanojev-appo-initial-observation-novelty-v1"
GROUPS = ("initial_seen_in_train", "initial_unseen_in_train")
NOTES = [
    "The reference is the set of initial state-plus-question requests of training episodes, not every later training state.",
    "Request IDs are excluded. State text is preserved exactly; outer state/questions JSON is serialized with sorted keys and compact separators before SHA256.",
    "The partition uses no action labels, probabilities, rewards, outcomes, or model evaluation results. Every registered test and OOD case remains included.",
    "This supplemental input-overlap analysis does not replace or filter the full primary benchmark.",
    "Initial-unseen does not mean spatial novelty, a new latent environment, or absence from later training states.",
    "Eight-tick OOD requests change action descriptions and remaining-time fields relative to four-tick training requests. This alone can make their full input hashes different.",
    "Duplicate visible inputs remain separate seeded episodes. Episode-level intervals are descriptive and do not measure uncertainty over training runs or over unique initial positions.",
]


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f"Refuse to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def membership_digest(frozen):
    return digest({key: frozen[key] for key in (
        "case_file_sha256", "expert_episodes_sha256", "expert_manifest_sha256",
        "train_case_ids", "training_initial_request_hashes", "selected_cases")})


def freeze(episodes_path, case_path, protocol_path):
    episodes_path, case_path, protocol_path = map(Path, (episodes_path, case_path, protocol_path))
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == "nanojev-appo-supervision-v1", "Unexpected experiment protocol")
    case_sha, episode_sha = sha256_file(case_path), sha256_file(episodes_path)
    require(case_sha == protocol["case_sha256"], "Registered case-file SHA mismatch")
    manifest_path = episodes_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == "nanojev-unified-episodes-v1" and manifest.get("finished") is True,
            "Expert source must be a complete unified episode file")
    require(manifest.get("episode_sha256") == episode_sha and manifest.get("cases_sha256") == case_sha,
            "Expert source SHA or case-file identity differs")
    cases = list(rows(case_path))
    registry = {c["id"]: c for c in cases}
    require(registry and len(registry) == len(cases), "Duplicate or empty case registry")
    require(dict(Counter(c["split"] for c in cases)) == protocol["case_counts"], "Registered split counts differ")
    require(len(manifest["selected_cases"]) == len(registry) and set(manifest["selected_cases"]) == set(registry),
            "Expert source must include the entire registered case file")
    require(all(c["split"] in SPLITS and c["spec"].get("task") == "shooting" and
                c["spec"].get("scenario") == "basic" for c in cases), "Unexpected task, scenario or split")
    initial, by_split = {}, defaultdict(list)
    for episode in rows(episodes_path):
        case = episode["case"]
        cid = case["id"]
        require(cid in registry and case == registry[cid] and cid not in initial, "Unknown, changed or duplicate case")
        require(episode.get("complete") is True and
                episode.get("continuation_policy_id") == manifest["continuation_policy_id"],
                f"{cid}: incomplete or mixed-policy source")
        # Inspect only initial model inputs. Later actions and terminal outcomes
        # are intentionally not consulted when defining overlap membership.
        require(episode.get("steps"), f"{cid}: missing initial pre-action request")
        first = episode["steps"][0]
        obs = first["observation"]
        require(episode.get("initial_observation") == obs and obs["step"] == 0 and
                obs["task"] == "shooting" and obs["candidates"], f"{cid}: initial observation differs")
        public_state = json.loads(obs["state"])
        require(public_state.get("screen_size") == [320, 240] and public_state.get("terminal") is False,
                f"{cid}: expected the standard live renderer")
        expected_request = policy_request(obs, cid)
        require(first["request"] == expected_request, f"{cid}: initial request is not the standard policy request")
        key = digest({"state": expected_request["state"], "questions": expected_request["questions"]})
        initial[cid] = key
        by_split[case["split"]].append(key)
    require(set(initial) == set(registry), "Expert source has missing cases")
    require(sha256_file(episodes_path) == episode_sha, "Expert source changed while reading")
    train_hashes = set(by_split["train"])
    require(train_hashes, "Training reference is empty")
    selected = [{"case": registry[cid], "initial_request_sha256": initial[cid],
                 "membership": GROUPS[0] if initial[cid] in train_hashes else GROUPS[1]}
                for cid in sorted(registry) if registry[cid]["split"] in ("test", "ood")]
    stats = {split: {"episodes": len(keys), "unique_initial_requests": len(set(keys)),
                    "initial_seen_in_train": sum(k in train_hashes for k in keys),
                    "initial_unseen_in_train": sum(k not in train_hashes for k in keys),
                    "shared_unique_initial_requests_with_train": len(set(keys) & train_hashes)}
             for split, keys in by_split.items()}
    frozen = {"schema_version": SCHEMA, "frozen_at": datetime.now(timezone.utc).isoformat(),
              "selection_inputs": "Initial expert standard state plus questions only; no outcome-based filtering",
              "protocol_path": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
              "case_file": str(case_path), "case_file_sha256": case_sha,
              "expert_episodes_path": str(episodes_path), "expert_episodes_sha256": episode_sha,
              "expert_manifest_path": str(manifest_path), "expert_manifest_sha256": sha256_file(manifest_path),
              "registered_models": protocol["evaluation"]["models"],
              "train_case_ids": sorted(cid for cid, c in registry.items() if c["split"] == "train"),
              "training_initial_request_hashes": sorted(train_hashes), "source_initial_stats": stats,
              "selected_cases": selected, "selected_episode_count": len(selected),
              "implementation_sha256": {Path(__file__).name: sha256_file(__file__),
                  "unified_game_pipeline.py": sha256_file(Path(__file__).with_name("unified_game_pipeline.py"))},
              "notes": NOTES}
    frozen["membership_sha256"] = membership_digest(frozen)
    return frozen


def validate_partition(frozen):
    require(frozen.get("schema_version") == SCHEMA and frozen.get("membership_sha256") == membership_digest(frozen),
            "Frozen partition schema or membership hash differs")
    training = frozen["training_initial_request_hashes"]
    require(training and len(training) == len(set(training)), "Invalid training hash reference")
    items = frozen["selected_cases"]
    selected = {v["case"]["id"]: v for v in items}
    require(len(selected) == len(items) == frozen["selected_episode_count"], "Duplicate or missing partition cases")
    require(not set(selected) & set(frozen["train_case_ids"]), "Training case enters evaluation partition")
    for item in items:
        require(item["case"]["split"] in ("test", "ood"), "Partition contains a non-heldout split")
        expected = GROUPS[0] if item["initial_request_sha256"] in training else GROUPS[1]
        require(item["membership"] == expected, "Partition membership contradicts its input hash")
    return selected


def apply_summary(novelty_path, summary_path):
    novelty_path, summary_path = Path(novelty_path), Path(summary_path)
    frozen, summary = json.loads(novelty_path.read_text()), json.loads(summary_path.read_text())
    selected = validate_partition(frozen)
    require(summary.get("schema_version") == "nanojev-appo-supervision-summary-v1" and summary.get("complete") is True,
            "Supply a completed APPO-supervision summary")
    require(summary["case_file_sha256"] == frozen["case_file_sha256"] and
            summary["protocol_sha256"] == frozen["protocol_sha256"], "Summary belongs to a different registered experiment")
    require(len(summary["cohort_case_ids"]) == len(selected) and set(summary["cohort_case_ids"]) == set(selected),
            "Summary must contain the complete frozen test/OOD cohort")
    require(set(summary["runs"]) == set(frozen["registered_models"]), "Summary has missing or extra registered models")
    reference = summary["reference"]
    require(reference in summary["runs"], "Reference model is missing")
    buckets, output = {}, {}
    for name, run in summary["runs"].items():
        eps = run["per_episode"]
        index = {e["id"]: e for e in eps}
        require(len(index) == len(eps) == len(selected) and set(index) == set(selected), f"{name}: incomplete or duplicate cohort")
        for cid, ep in index.items():
            case = selected[cid]["case"]
            require(type(ep["success"]) is bool and all(ep[k] == case[k] for k in ("id", "seed", "split", "variant")) and
                    ep["task"] == case["spec"]["task"] and ep["scenario"] == case["spec"]["scenario"],
                    f"{name}/{cid}: changed episode identity or invalid outcome")
            require(all(is_num(ep["metrics"].get(k)) for k in ("physical_ticks", "ammo_consumed", "native_reward")),
                    f"{name}/{cid}: missing metric")
        buckets[name] = {f"{split}/{membership}": [index[cid] for cid, item in selected.items()
                           if item["case"]["split"] == split and item["membership"] == membership]
                         for split in ("test", "ood") for membership in GROUPS}
        output[name] = {"source": run["source"], "by_split_initial_overlap": {
            key: group_stats(values) if values else {"n": 0, "successes": 0, "success_rate": None,
                "wilson95": None, "mean_steps": None, "mean_decisions": None, "metrics": {}}
            for key, values in buckets[name].items()}}
    contrasts = {name: {key: paired(buckets[reference][key], values) if values else {
        "n": 0, "success_rate_delta": None, "exact_mcnemar_two_sided_p": None, "reason": "Empty frozen subgroup"}
        for key, values in by_group.items()} for name, by_group in buckets.items() if name != reference}
    return {"schema_version": "nanojev-appo-initial-overlap-summary-v1", "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "novelty_path": str(novelty_path), "novelty_sha256": sha256_file(novelty_path),
            "membership_sha256": frozen["membership_sha256"],
            "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
            "case_file_sha256": frozen["case_file_sha256"], "expert_episodes_sha256": frozen["expert_episodes_sha256"],
            "reference": reference, "runs": output, "paired_vs_reference": contrasts,
            "validation_scope": "The existing completed summary supplies verified rollouts; this command verifies cohort/partition binding and recomputes subgroup statistics, without replaying raw episodes.",
            "implementation_sha256": {Path(__file__).name: sha256_file(__file__)}, "notes": NOTES + [
                "Paired effects are other minus reference, using the same frozen case IDs. Exact McNemar p-values are unadjusted across subgroups and models.",
                "Small or empty subgroups are retained. Empty groups have null rates and intervals, not zero performance."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("data/appo_basic_supervision_v1/expert/episodes.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("configs/appo_basic_supervision_v1_cases.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/appo_basic_supervision_v1.json"))
    parser.add_argument("--novelty", type=Path, default=Path("configs/appo_basic_supervision_v1_novelty.json"))
    parser.add_argument("--summary", type=Path, help="Existing completed main summary; omit to freeze the input-only partition")
    parser.add_argument("--output", type=Path, help="Required new supplemental report path when --summary is supplied")
    args = parser.parse_args(argv)
    if args.summary is not None:
        require(args.output is not None, "--summary requires --output")
        report = apply_summary(args.novelty, args.summary)
        write_new(args.output, report)
        print(json.dumps({"complete": True, "output": str(args.output), "models": list(report["runs"])}))
    else:
        require(args.output is None, "Use --novelty for the frozen partition destination")
        require(not args.novelty.exists(), "Frozen partition already exists; use it without regenerating")
        frozen = freeze(args.episodes, args.cases, args.protocol)
        validate_partition(frozen)
        write_new(args.novelty, frozen)
        print(json.dumps({"frozen": str(args.novelty), "sha256": sha256_file(args.novelty),
                          "membership_sha256": frozen["membership_sha256"], "counts": frozen["source_initial_stats"]}))


if __name__ == "__main__":
    main()
