#!/usr/bin/env python3
"""Compare complete APPO-supervision rollouts on an explicit fresh case cohort.

Standard library only; no inference, simulator, network, or credentials. Source
files may include other splits of the same registered cases. Every supplied row
is validated before the requested cohort is selected. Source hashes always refer
to original complete JSONL files, never an invented filtered file.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from summarize_unified_games import SPLITS, canon, group_stats, is_num, load_run, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(canon(value).encode()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"{path}:{number}: expected an object")
                yield value


def mcnemar_exact(improved, regressed):
    """Two-sided exact binomial test conditional on discordant episode pairs."""
    require(type(improved) is int and type(regressed) is int and min(improved, regressed) >= 0,
            "Discordant counts must be nonnegative integers")
    n = improved + regressed
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(improved, regressed) + 1))
    return min(1.0, (2 * tail) / (2 ** n))


def paired(reference, other):
    a, b = {r["id"]: r for r in reference}, {r["id"]: r for r in other}
    require(a and a.keys() == b.keys(), "Paired comparison requires identical nonempty episode IDs")
    improved = sorted(k for k in a if b[k]["success"] and not a[k]["success"])
    regressed = sorted(k for k in a if a[k]["success"] and not b[k]["success"])
    both_success = sorted(k for k in a if a[k]["success"] and b[k]["success"])
    delta = {}
    for key in ("physical_ticks", "ammo_consumed", "native_reward"):
        require(all(key in r["metrics"] for r in [*a.values(), *b.values()]),
                f"Paired episodes lack required metric {key}")
        delta[key] = {"n": len(a), "mean_other_minus_reference": math.fsum(
            b[k]["metrics"][key] - a[k]["metrics"][key] for k in a) / len(a)}
    return {"n": len(a), "success_rate_delta": (len(improved) - len(regressed)) / len(a),
            "improved": len(improved), "regressed": len(regressed),
            "both_success": len(both_success),
            "both_failure": len(a) - len(improved) - len(regressed) - len(both_success),
            "exact_mcnemar_two_sided_p": mcnemar_exact(len(improved), len(regressed)),
            "p_adjustment": "none; descriptive predeclared comparisons",
            "improved_case_ids": improved, "regressed_case_ids": regressed,
            "paired_metrics": delta}


def controller_check(episode, policy):
    """Recompute the recorded behavior and every actual action from its seeded RNG."""
    case = episode["case"]
    rng = random.Random(int(digest([case["id"], policy["sampling_seed"]])[:16], 16))
    mode = "sample" if policy["engine"] == "random" else policy["controller"]
    epsilon = policy["epsilon"]
    ticks = 0
    rewards = []
    for i, step in enumerate(episode["steps"]):
        where = f"{case['id']} step {i}"
        obs, info = step["observation"], step["info"]
        keys = sorted(obs["candidates"])
        require(keys == ["left", "noop", "right", "shoot"], f"{where}: Basic candidate mismatch")
        require(obs["task"] == "shooting" and obs["step"] == i, f"{where}: observation identity mismatch")
        scores = step["scores"]
        require(set(scores) == set(keys) and all(is_num(v) and v >= 0 for v in scores.values()),
                f"{where}: invalid action scores")
        require(abs(math.fsum(scores.values()) - 1) <= 1e-6, f"{where}: scores are not a Choice distribution")
        require(step.get("forced") is not True, f"{where}: four-action Basic cannot be forced")
        if mode == "sample":
            if policy["engine"] == "random":
                require(all(abs(scores[k] - .25) <= 1e-12 for k in keys),
                        f"{where}: random baseline must be uniform")
            total = sum(scores.values())
            base = {k: scores[k] / total for k in keys}
        else:
            top = max(keys, key=lambda k: scores[k])
            base = {k: float(k == top) for k in keys}
        expected = {k: (1 - epsilon) * base[k] + epsilon / len(keys) for k in keys}
        actual = step["behavior_probs"]
        require(set(actual) == set(keys) and all(is_num(actual[k]) and
                abs(actual[k] - expected[k]) <= 1e-12 for k in keys), f"{where}: behavior mismatch")
        draw, cumulative, selected = rng.random(), 0.0, keys[-1]
        for k in keys:
            cumulative += expected[k]
            if draw < cumulative:
                selected = k
                break
        require(step["action"] == selected, f"{where}: actual action differs from seeded controller")
        if "sampling_draw" in step:
            require(step["sampling_draw"] == draw, f"{where}: sampling receipt differs")
        requested, physical = info["requested_ticks"], info["actual_ticks"]
        require(type(requested) is int and type(physical) is int and
                1 <= physical <= requested <= case["spec"].get("frame_skip", 4), f"{where}: invalid ticks")
        ticks += physical
        rewards.append(step["reward"])
        require(is_num(rewards[-1]), f"{where}: invalid reward")
        require(info["action_id"] == step["action"] and info["terminated"] is step["terminated"] and
                info["truncated"] is False, f"{where}: transition info differs")
        require(info["episode_metrics"]["physical_ticks"] == ticks and
                info["episode_metrics"]["decisions"] == i + 1, f"{where}: cumulative counters differ")
    final = episode["final_info"]
    require(episode["steps"] and final == episode["steps"][-1]["info"],
            f"{case['id']}: final_info differs from actual terminal transition")
    metrics = final["episode_metrics"]
    require(type(metrics.get("success")) is bool and metrics["success"] is episode["success"],
            f"{case['id']}: metric success differs")
    require(is_num(metrics.get("kills")) and (metrics["kills"] > 0) is episode["success"],
            f"{case['id']}: success must be observed kill delta > 0")
    require(all(is_num(metrics.get(k)) for k in ("physical_ticks", "ammo_consumed", "native_reward")),
            f"{case['id']}: missing physical metric")
    require(metrics["ammo_consumed"] >= 0 and abs(math.fsum(rewards) - metrics["native_reward"]) <= 1e-9,
            f"{case['id']}: reward or ammo accounting mismatch")
    return len(episode["steps"])


def load_verified(name, path, registry, case_sha, protocol):
    run = load_run(name, path)
    require(run["cases_sha256"] == case_sha, f"{name}: source was not collected from the registered fresh case file")
    p, evaluation = run["policy"], protocol["evaluation"]
    require(digest(p) == run["continuation_policy_id"], f"{name}: policy identity hash mismatch")
    require(p.get("engine") in {"checkpoint", "random", "sample_factory_appo"},
            f"{name}: unexpected engine, including legacy API runs")
    if name == "random":
        require(p["engine"] == "random", "The random run must execute the uniform random engine")
    else:
        require(p["engine"] != "random", f"{name}: named learned run is actually random")
    if name == "appo_1111":
        require(p["engine"] == "sample_factory_appo", "APPO baseline must use its actual pixel actor")
        require(p.get("checkpoint_sha256", {}).get("model") == protocol["expert"]["checkpoint_sha256"],
                "APPO baseline checkpoint differs from the frozen expert")
    elif p["engine"] == "checkpoint":
        require(isinstance(p.get("checkpoint_sha256"), dict) and p["checkpoint_sha256"].get("best.safetensors"),
                f"{name}: checkpoint identity missing")
        if name == "frozen_sft":
            require(p["checkpoint_sha256"]["best.safetensors"] == protocol["initialization"]["weights_sha256"],
                    "Frozen SFT baseline differs from registered initialization")
    for key, expected in (("controller", evaluation["primary_controller"]),
                          ("epsilon", evaluation["epsilon"]),
                          ("sampling_seed", evaluation["sampling_seed"]),
                          ("temperature", 1.0), ("tie_break", "lexicographic_first")):
        require(p.get(key) == expected, f"{name}: {key} differs from the registered evaluation")
    count = 0
    for ep in rows(path):
        case = ep["case"]
        require(case["id"] in registry and case == registry[case["id"]],
                f"{name}: unknown, old, or changed case {case['id']}")
        count += controller_check(ep, p)
    run["validation"] = {"source_episodes": len(run["episodes"]), "controller_transitions_checked": count,
                         "policy_identity_verified": True, "physical_replay_performed": False}
    return run


def build(protocol_path, case_path, run_paths, reference="frozen_sft", splits=None):
    protocol = json.loads(Path(protocol_path).read_text())
    require(protocol.get("schema_version") == "nanojev-appo-supervision-v1", "Wrong supervision protocol")
    case_path = Path(case_path or protocol["case_file"])
    case_sha = sha256_file(case_path)
    require(case_sha == protocol["case_sha256"], "Registered case hash mismatch")
    expected_names = protocol["evaluation"]["models"]
    require(set(run_paths) == set(expected_names),
            f"Supply every registered run exactly once: {expected_names}; received {sorted(run_paths)}")
    require(reference in run_paths, "Reference run is missing")
    case_list = list(rows(case_path))
    registry = {c["id"]: c for c in case_list}
    require(len(registry) == len(case_list) and registry, "Duplicate or empty case registry")
    groups = {}
    for c in case_list:
        require(c["split"] in SPLITS and c["spec"].get("task") == "shooting" and
                c["spec"].get("scenario") == "basic", "This protocol registers Basic shooting cases only")
        group = (c["spec"]["scenario"], c["seed"])
        require(group not in groups, "Repeated environment seed, including across splits")
        groups[group] = c["split"]
    require(dict(Counter(c["split"] for c in case_list)) == protocol["case_counts"], "Case counts differ from protocol")
    splits = list(protocol["evaluation"]["fresh_splits"] if splits is None else splits)
    require(splits and len(set(splits)) == len(splits) and all(s in SPLITS for s in splits), "Invalid split selection")
    selected = {cid: c for cid, c in registry.items() if c["split"] in splits}
    require(selected and all(any(c["split"] == s for c in selected.values()) for s in splits), "Empty requested split")
    loaded, summaries = {}, {}
    for name in expected_names:
        run = load_verified(name, run_paths[name], registry, case_sha, protocol)
        require(set(selected) <= set(run["cases"]), f"{name}: missing registered comparison cases")
        eps = [dict(ep, scenario=registry[ep["id"]]["spec"]["scenario"])
               for ep in run["episodes"] if ep["id"] in selected]
        loaded[name] = eps
        by_group = {f"{s}/{g}": group_stats([e for e in eps if e["split"] == s and e["scenario"] == g])
                    for s, g in sorted({(e["split"], e["scenario"]) for e in eps})}
        summaries[name] = {"source": {k: run[k] for k in (
            "episodes_path", "manifest_path", "episodes_sha256", "manifest_sha256",
            "cases_sha256", "continuation_policy_id", "policy")},
            "validation": run["validation"], "included_episodes": len(eps),
            "excluded_other_registered_splits": len(run["episodes"]) - len(eps),
            "by_split_scenario": by_group, "per_episode": eps}
    comparisons = {}
    for name in expected_names:
        if name == reference:
            continue
        comparisons[name] = {key: paired(
            [e for e in loaded[reference] if f"{e['split']}/{e['scenario']}" == key],
            [e for e in loaded[name] if f"{e['split']}/{e['scenario']}" == key])
            for key in summaries[reference]["by_split_scenario"]}
    return {"schema_version": "nanojev-appo-supervision-summary-v1", "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_path": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
            "case_file": str(case_path), "case_file_sha256": case_sha,
            "cohort_sha256": digest([selected[k] for k in sorted(selected)]),
            "cohort_case_ids": sorted(selected), "splits": splits, "reference": reference,
            "runs": summaries, "paired_vs_reference": comparisons,
            "implementation_sha256": {Path(__file__).name: sha256_file(__file__),
                "summarize_unified_games.py": sha256_file(Path(__file__).with_name("summarize_unified_games.py"))},
            "notes": [
                "Only registered new cases are included; no historical Jev cohort is pooled into this report.",
                "Source hashes identify original complete files. Explicit split selection is a view, not a replacement source.",
                "APPO consumes native-rendered recurrent pixels; NanoJev consumes structured visible text. This compares complete policies with different observations.",
                "Success means observed kill before the native deadline, not positive native reward. Physical ticks and action decisions differ.",
                "Wilson intervals use episodes as units. Exact McNemar tests use paired discordant cases; p-values are unadjusted across comparisons.",
                "Paired deltas are other minus reference. Ticks and ammo include failures and early successes and do not independently measure skill.",
                "Hard seed 17, hard seed 29 and soft seed 17 remain distinct runs. This is not an estimate over many independent training seeds.",
                "Hashes, case definitions, seeded actions and transition counters are verified here; no simulator replay or model inference is performed.",
                "Action probabilities describe action choice, not winning probability. Offline expert-action NLL is not evaluated by this rollout summary."]}


def markdown(report):
    lines = ["# APPO action-supervision evaluation", "", "Fresh matched episodes; synchronized case definitions and fixed action controllers.", "",
             "| Run | Split / scenario | Success | Wilson 95% | Physical ticks | Ammo used |", "|---|---|---:|---|---:|---:|"]
    for name, run in report["runs"].items():
        for group, stat in run["by_split_scenario"].items():
            lo, hi = stat["wilson95"]
            lines.append(f"| {name} | {group} | {stat['successes']}/{stat['n']} ({stat['success_rate']:.1%}) | "
                         f"{lo:.1%}–{hi:.1%} | {stat['metrics']['physical_ticks']['mean']:.2f} | "
                         f"{stat['metrics']['ammo_consumed']['mean']:.2f} |")
    lines += ["", f"Paired comparisons against **{report['reference']}**:", "",
              "| Run | Split / scenario | Success delta | Improved / regressed | Exact McNemar p |",
              "|---|---|---:|---:|---:|"]
    for name, groups in report["paired_vs_reference"].items():
        for group, stat in groups.items():
            lines.append(f"| {name} | {group} | {stat['success_rate_delta']:+.1%} | "
                         f"{stat['improved']} / {stat['regressed']} | {stat['exact_mcnemar_two_sided_p']:.5g} |")
    lines += ["", *["- " + note for note in report["notes"]], ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=Path("configs/appo_basic_supervision_v1.json"))
    p.add_argument("--cases", type=Path, help="Optional relocated copy; its SHA must match the protocol")
    p.add_argument("--run", action="append", required=True, metavar="NAME=EPISODES.jsonl")
    p.add_argument("--reference", default="frozen_sft")
    p.add_argument("--splits", help="Comma-separated registered splits; default is protocol fresh_splits")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output_dir.exists(), "Use a fresh output directory")
    paths = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        require(separator and name and path and name not in paths, "Use unique --run NAME=PATH values")
        paths[name] = Path(path)
    report = build(args.protocol, args.cases, paths, args.reference,
                   args.splits.split(",") if args.splits is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.md").write_text(markdown(report))
    print(json.dumps({"complete": True, "output": str(args.output_dir),
                      "runs": len(report["runs"]), "episodes_per_run": len(report["cohort_case_ids"])}))


if __name__ == "__main__":
    main()
