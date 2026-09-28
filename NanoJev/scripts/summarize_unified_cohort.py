#!/usr/bin/env python3
"""Compare verified runs on an explicit, fully covered reference cohort.

All source files and manifests are validated before filtering. The reference
file defines the cohort; no intersection or outcome-dependent selection is used.
Source SHA256 values always identify the original complete files, not a subset.
No episodes, manifests, model outputs, or training files are modified.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import summarize_unified_games as summary_tools


def cohort_view(run, reference):
    selected = set(reference["cases"])
    missing = sorted(selected - set(run["cases"]))
    if missing:
        raise ValueError(f"run {run['name']!r} does not cover the explicit cohort; missing case IDs: {missing}")
    changed = [cid for cid in sorted(selected)
               if summary_tools.canon(run["cases"][cid]) != summary_tools.canon(reference["cases"][cid])]
    if changed:
        raise ValueError(f"run {run['name']!r} has different case definitions for cohort IDs: {changed}")
    source_selection = []
    for source in run["sources"]:
        included = sorted(selected & set(source["case_ids"]))
        source_selection.append({**source, "source_case_count": len(source["case_ids"]),
                                 "included_case_count": len(included),
                                 "excluded_case_count": len(source["case_ids"]) - len(included),
                                 "included_case_ids": included,
                                 "sha256_scope": "original_complete_source_file"})
    active_sources = [source for source in source_selection if source["included_case_count"]]
    policy_ids = list(dict.fromkeys(source["continuation_policy_id"] for source in active_sources))
    common_policy = {key: value for key, value in active_sources[0]["policy"].items()
                     if all(key in source["policy"] and summary_tools.canon(source["policy"][key]) == summary_tools.canon(value)
                            for source in active_sources[1:])}
    view = {**run, "cases": {cid: run["cases"][cid] for cid in sorted(selected)},
            "case_policies": {cid: run["case_policies"][cid] for cid in sorted(selected)},
            "episodes": [episode for episode in run["episodes"] if episode["id"] in selected],
            "sources": active_sources, "policy": common_policy,
            "continuation_policy_ids": policy_ids,
            "continuation_policy_id": policy_ids[0] if len(policy_ids) == 1 else None}
    selection = {"source_case_count": len(run["cases"]), "included_case_count": len(selected),
                 "excluded_case_count": len(run["cases"]) - len(selected), "sources": source_selection}
    return view, selection


def build_summary(run_specs, cohort_from, training_specs=None):
    if not run_specs:
        raise ValueError("At least one --run NAME=path is required")
    reference = summary_tools.load_run_sources("cohort_reference", cohort_from)
    complete_runs = [summary_tools.load_run_sources(name, paths) for name, paths in run_specs.items()]
    views, selections = [], {}
    for run in complete_runs:
        view, selection = cohort_view(run, reference)
        views.append(view)
        selections[run["name"]] = selection
    by_name = {run["name"]: run for run in views}
    cohort = summary_tools.check_cohort(views)
    cohort.update({"selection_rule": "exact_case_ids_from_verified_reference; complete_coverage_required",
                   "reference_episodes_path": str(cohort_from),
                   "reference_episodes_sha256": reference["episodes_sha256"],
                   "reference_manifest_sha256": reference["manifest_sha256"],
                   "selected_case_ids": sorted(reference["cases"]),
                   "selection_uses_outcomes": False})
    summaries = {run["name"]: summary_tools.summarize_run(run) for run in views}
    training = {}
    for name, path in (training_specs or {}).items():
        if name not in by_name:
            raise ValueError(f"--training name {name!r} does not refer to a --run name")
        training[name] = summary_tools.load_training(name, path, by_name[name])
    return {
        "tool": "scripts/summarize_unified_cohort.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {run["name"]: {key: run[key] for key in
                   ("episodes_path", "episodes_sha256", "manifest_path", "manifest_sha256")}
                   | {"sources": selections[run["name"]]["sources"],
                      "sha256_scope": "original_complete_source_files"} for run in complete_runs},
        "policies": {run["name"]: {key: run[key] for key in
                     ("continuation_policy_id", "continuation_policy_ids", "policy", "sources")} for run in views},
        "cohort": cohort, "source_selection": selections,
        "policy_differences": summary_tools.policy_differences(views), "runs": summaries,
        "comparison": summary_tools.comparison_tables(list(by_name), summaries),
        "training_diagnostics": training,
        "notes": [
            "The explicit reference cohort is selected by case ID, never by success or by the intersection of available results.",
            "Every original source is hash-verified before filtering. Every compared run must cover the entire reference cohort with identical case definitions.",
            "Input episodes_sha256 and manifest_sha256 identify the original complete source files. No subset episodes file or rewritten manifest is produced.",
            "The cohort SHA256 identifies canonical selected case definitions; it is not a trajectory or source-file SHA256.",
            "Policies with no selected cases remain in input provenance but are excluded from selected-cohort policy comparisons.",
            *summary_tools.NOTES,
        ],
    }


def render_markdown(summary):
    text = summary_tools.render_markdown(summary).replace(
        "Generated by scripts/summarize_unified_games.py", "Generated by scripts/summarize_unified_cohort.py", 1)
    rows = ["", "## Explicit cohort selection", "",
            "The source hashes above refer to the complete original files. Only the explicit reference case IDs enter these comparisons; missing cases are errors.", "",
            f"Reference: `{summary['cohort']['reference_episodes_path']}`", "",
            "| Run | Source cases | Included cases | Excluded cases |",
            "|---|---:|---:|---:|"]
    for name, selection in summary["source_selection"].items():
        rows.append(f"| {name} | {selection['source_case_count']} | {selection['included_case_count']} | {selection['excluded_case_count']} |")
    return text + "\n".join(rows) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-from", required=True, help="completed reference episodes JSONL with its original manifest")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=EPISODES_JSONL",
                        help="repeat a name to combine disjoint, independently verified sources")
    parser.add_argument("--training", action="append", default=[], metavar="NAME=TRAINER_SUMMARY_JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("--output-dir must be a fresh directory")
        summary = build_summary(summary_tools.parse_named(args.run, "--run"), args.cohort_from,
                                summary_tools.parse_named(args.training, "--training"))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (args.output_dir / "summary.md").write_text(render_markdown(summary))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output_dir / 'summary.json'} and {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
