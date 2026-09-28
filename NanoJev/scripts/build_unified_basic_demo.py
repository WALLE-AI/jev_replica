#!/usr/bin/env python3
"""Replay Basic demonstrations from the current unified step-400 checkpoint.

Uses the same frozen full-cohort files as Predict Position. The selected
illustrations are accompanied by all 128 test and all 128 OOD outcomes.
Export requires a fresh directory; previous demo versions are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import tempfile

from build_predict_position_demo import (CASES_SHA, NANOJEV_SHA, PredictCapture,
    add_events, load_recordings)
from build_shooting_demo import (AtlasWriter, COLS, HEIGHT, ORDER, ROWS, WIDTH,
    match, read_rows, render_episode, require)
from replay_unified_episodes import load_json
from unified_game_pipeline import digest, file_digest

OUTPUT_NAME = "shooting_results.json"
IMPLEMENTATION = (Path(__file__).name, "build_predict_position_demo.py", "build_shooting_demo.py",
                  "unified_doom_env.py", "unified_game_pipeline.py", "replay_unified_episodes.py")
MODELS = [
    {"id": "jev", "name": "Jev", "description": "Recorded Jev API action probabilities"},
    {"id": "nanojev", "name": "NanoJev", "description": "0.6B · current unified model · step 400"},
    {"id": "base", "name": "Untuned Qwen", "description": "Original Qwen3-0.6B vocabulary head · no task fine-tuning"},
]


def aggregate_basic(sources, registry):
    summary = {k: {} for k in ORDER}
    for split in ("test", "ood"):
        ids = [cid for cid, case in registry.items() if case["split"] == split and
               case["spec"].get("scenario") == "basic"]
        require(len(ids) == 128, "Basic requires all 128 frozen test and all 128 OOD cases")
        for model_id in ORDER:
            wins = sum(sources[model_id][cid]["success"] for cid in ids)
            summary[model_id][split] = {"episodes": len(ids), "successes": wins, "success_rate": wins / len(ids)}
    return summary


def select_cases(config, sources, registry):
    rows = config["cases"]
    require(rows and len({r["id"] for r in rows}) == len(rows), "Distinct Basic illustrations are required")
    require(config["default_case_id"] in {r["id"] for r in rows}, "Default illustration is absent")
    for row in rows:
        require(row["id"] in registry, "Illustration is absent from the frozen registry")
        case = registry[row["id"]]
        require(case["split"] == "test" and case["spec"].get("scenario") == "basic", "Basic test cases only")
        outcomes = {k: sources[k][row["id"]]["success"] for k in ORDER}
        match({"jev": False, "nanojev": True, "base": False}, outcomes, row["id"] + "/required_outcomes")
    return rows, {"rule": config["selection_rule"], "selected_case_id": config["default_case_id"],
                  "selected_case_ids": [r["id"] for r in rows], "outcome_selected": True,
                  "aggregate_scope": "All 128 frozen Basic test cases; all 128 OOD cases reported separately",
                  "selection_changes_training_or_evaluation": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev", type=Path, default=Path("runs/sonic_unified_sft_v1/jev_parallel_recovery_v2/episodes.jsonl"))
    parser.add_argument("--nanojev", type=Path, default=Path("runs/sonic_unified_sft_v1/experiment/selected_test.jsonl"))
    parser.add_argument("--base", type=Path, default=Path("runs/sonic_unified_sft_v1/native_test.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("runs/sonic_unified_sft_v1/test_cases.jsonl"))
    parser.add_argument("--selection", type=Path, default=Path("configs/unified_basic_demo_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    cases_sha = file_digest(args.cases)
    require(cases_sha == CASES_SHA, "Expected the frozen 548-case unified evaluation registry")
    case_rows = read_rows(args.cases)
    registry = {r["id"]: r for r in case_rows}
    require(len(case_rows) == len(registry) == 548, "Expected 548 distinct frozen cases")
    sources, provenance = {}, {}
    for k in ORDER:
        sources[k], provenance[k] = load_recordings(getattr(args, k), k, registry, cases_sha)
    selection_sha = file_digest(args.selection)
    config = load_json(args.selection.read_text())
    selected, selection = select_cases(config, sources, registry)
    summary = aggregate_basic(sources, registry)
    if args.validate_only:
        print(json.dumps({"passed": True, "selection": selection, "summary": summary}), flush=True)
        return 0
    require(not args.output.is_symlink() and not (args.output / "media").is_symlink(), "Output cannot be a symlink")
    require(not (args.output / OUTPUT_NAME).exists() and not args.receipt.exists(), "Use fresh output and receipt paths")
    implementation = {n: file_digest(Path(__file__).with_name(n)) for n in IMPLEMENTATION}
    namespace = digest({"sources": provenance, "cases": cases_sha, "selection": selection_sha,
                        "implementation": implementation})[:12]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unified-basic-build-", dir=args.output.parent) as temporary:
        staging = Path(temporary)
        demos, audits, assets = [], [], []
        for row in selected:
            case = registry[row["id"]]
            systems, budgets = [], []
            for k in ORDER:
                writer = AtlasWriter(staging / "media", f"shooting_unified_{namespace}_{k}_{digest(case['id'])[:12]}")
                system, audit, budget = render_episode(sources[k][case["id"]], k, writer, capture_factory=PredictCapture)
                add_events(system)
                systems.append(system); audits.append(audit); budgets.append(budget); assets.extend(audit["assets"])
                print(json.dumps({"case": case["id"], "model": k, "success": system["success"],
                    "physical_ticks": system["total_ticks"], "shot_ticks": system["shot_ticks"], "replay_passed": True}), flush=True)
            require(len(set(budgets)) == 1, "Systems have different physical deadlines")
            demos.append({"id": case["id"], "split": case["split"], "seed": case["seed"],
                          "frame_skip": case["spec"]["frame_skip"], "max_ticks": budgets[0],
                          "title": row["title"], "description": row["description"], "systems": systems})
        illustration_summary = {k: {"test": {"episodes": len(demos), "successes":
            sum(next(s for s in c["systems"] if s["id"] == k)["success"] for c in demos)}} for k in ORDER}
        data = {"schema": "nanojev-shooting-demo-v1", "task": "basic", "ticks_per_second": 35,
                "default_case_id": config["default_case_id"], "models": MODELS,
                "cases": demos, "summary": summary, "illustration_summary": illustration_summary,
                "protocol": {"synchronization": "physical_tick", "ticks_per_second": 35,
                    "frame_zero": "actual reset state", "decision_index": "zero at reset; one-based executed decisions",
                    "probabilities": "Original model action probabilities before epsilon exploration; not hit probabilities",
                    "controller": "greedy", "epsilon": .1, "sampling_seed": 17,
                    "image_sampling": "Every available physical-tick RGB image; every state retained",
                    "image_fallback": "Hold the preceding real RGB image when native state is unavailable; image_tick identifies it",
                    "terminal_playback": "Freeze each system at its actual terminal transition while the shared clock continues",
                    "shot": "Observed decrease in SELECTED_WEAPON_AMMO",
                    "success": "Positive observed kill-count delta before the task deadline",
                    "selection": selection, "case_file_sha256": cases_sha,
                    "source_episodes_sha256": {k: p["episodes_sha256"] for k, p in provenance.items()},
                    "selected_checkpoint_sha256": NANOJEV_SHA, "checkpoint_step": 400,
                    "new_model_calls": 0, "new_api_calls": 0}}
        index = staging / OUTPUT_NAME
        index.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        generated = [{"path": OUTPUT_NAME, "sha256": file_digest(index), "bytes": index.stat().st_size}, *assets]
        for k, p in provenance.items():
            require(file_digest(p["path"]) == p["episodes_sha256"] and file_digest(p["manifest_path"]) == p["manifest_sha256"],
                    f"{k}: source changed during rendering")
        require(all(file_digest(Path(__file__).with_name(n)) == h for n, h in implementation.items()), "Replay code changed")
        require(file_digest(args.cases) == cases_sha and file_digest(args.selection) == selection_sha, "Frozen inputs changed")
        import PIL
        receipt = {"schema": "nanojev-unified-basic-media-receipt-v1", "passed": True,
            "created_at": datetime.now(timezone.utc).isoformat(), "output_directory": str(args.output.resolve()),
            "cases_sha256": cases_sha, "selection_sha256": selection_sha, "selection": selection,
            "sources": provenance, "implementation_sha256": implementation,
            "runtime": {"python": platform.python_version(), "pillow": PIL.__version__},
            "rendering": {"width": WIDTH, "height": HEIGHT, "atlas_columns": COLS, "atlas_rows": ROWS,
                "encoding": "lossless WebP; decoded RGB byte-for-byte verified", "state_stride_ticks": 1,
                "image_stride_ticks": 1, "extra_simulation_ticks": 0},
            "validation": {"episodes": len(audits), "all_passed": True,
                "decisions": sum(a["decisions"] for a in audits), "physical_ticks": sum(a["physical_ticks"] for a in audits),
                "float_absolute_tolerance": 1e-9, "state_text": "exact", "all_transition_info_fields": True},
            "full_cohort_summary": summary, "episodes": audits, "generated_files": generated}
        require(not (args.output / OUTPUT_NAME).exists() and not args.receipt.exists(), "Output appeared during rendering")
        for item in assets:
            require(not (args.output / item["path"]).exists(), "Generated asset already exists")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "media").mkdir(exist_ok=True)
        for item in assets:
            os.replace(staging / item["path"], args.output / item["path"])
        os.replace(index, args.output / OUTPUT_NAME)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"passed": True, "output": str(args.output), "receipt": str(args.receipt),
        "summary": summary, "generated_bytes": sum(item["bytes"] for item in generated)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
