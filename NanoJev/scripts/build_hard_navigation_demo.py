#!/usr/bin/env python3
"""Replay the original hard navigation cases with the current NanoJev model.

The unchanged Jev and original Qwen recordings use the same fixed initial states,
questions and shared controllers as two new NanoJev rollouts. CPU replay verifies
all six complete trajectories before replacing the development data. The public
site remains read-only. This exporter performs no model or API calls.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import build_arcade_demo as arcade
import build_side_by_side_demo as original
from build_predict_position_demo import NANOJEV_SHA


SNAKE_CASE = "snake:showcase:12:61005"
ORDER = ("jev", "nanojev", "base")


def rebuild_original():
    args = argparse.Namespace(
        arcade_data=Path("web/arcade_results.json"),
        arcade_manifest=Path("assets/arcade_data_manifest.json"),
        maze_cohort=Path("results/rollout_pilot_episodes.jsonl"),
        snake_cohort=Path("results/arcade_snake_cohort.jsonl"),
        native_maze_cohort=Path("results/side_by_side_maze_episode.jsonl"),
        maze_native=Path("results/model_edges_native.json"))
    data, audits, _, protected = original.build(args)
    public_path = Path("web/side_by_side_results.json")
    public = json.loads(public_path.read_text())
    arcade.require(data == public, "CPU replay differs from the original public recordings")
    manifest = json.loads(Path("assets/side_by_side_data_manifest.json").read_text())
    arcade.require(manifest["output"]["sha256"] == arcade.sha_file(public_path),
                   "Public data differs from its recorded manifest")
    return data, audits, protected


def current_model(source, expected_case, source_cases_sha):
    model = source["model"]
    arcade.require(model.get("engine") == "checkpoint" and model.get("checkpoint_sha256") == NANOJEV_SHA,
                   "A fresh rollout from the selected current checkpoint is required")
    arcade.require(source.get("selected_episode_ids") == [expected_case], "Unexpected rollout selection")
    arcade.require(source.get("source_episodes_sha256") == source_cases_sha, "New rollout case file differs")
    arcade.require(source["execution"].get("network_model_calls") == 0 and
                   source["execution"].get("model_forward_passes", 0) > 0,
                   "Expected genuine local model inference without API calls")


def build(args):
    old, old_audits, protected = rebuild_original()
    selected_bytes = args.cases.read_bytes()
    selected_sha = hashlib.sha256(selected_bytes).hexdigest()
    selected_rows = [json.loads(line) for line in selected_bytes.splitlines() if line.strip()]
    selected = {row["id"]: row for row in selected_rows}
    arcade.require(len(selected_rows) == len(selected) == 2 and set(selected) == {arcade.MAZE_CASE, SNAKE_CASE},
                   "Hard-navigation case file must contain exactly the two original cases")
    examples, audits = [], []
    for previous in old["examples"]:
        cid, game = previous["id"], previous["game"]
        previous_audit = next(row for row in old_audits if row["case_id"] == cid)
        path = args.maze if game == "maze" else args.snake
        source, episode, _ = arcade.read_source(path, cid)
        current_model(source, cid, selected_sha)
        if game == "maze":
            initial, trained, receipt, controller = arcade.maze_system(
                path, "nanojev", "NanoJev", "0.6B · current unified model · local safety + verified-edge exploration")
            horizon = 5000
        else:
            initial, trained, receipt, controller = arcade.snake_system(
                path, cid, "nanojev", "NanoJev", "0.6B · current unified model · shared safety and food planner")
            horizon = 256
        arcade.require(initial == selected[cid]["initial_state"], "New rollout initial state differs from the fixed case")
        arcade.require(arcade.digest(initial) == previous_audit["initial_state_sha256"],
                       "New and original systems start from different states")
        arcade.require(controller == previous_audit["controller"], "New and original controllers differ")
        arcade.require(episode["horizon"] == horizon, "Hard-task horizon changed")
        receipt.update(system_id="nanojev", origin="new_current_checkpoint_rollout", passed=True,
                       initial_state_sha256=arcade.digest(initial), frame_json_sha256=original.frame_hash(trained),
                       final_outcome=trained["summary"]["outcome"], source_episode_sha256=arcade.digest(episode),
                       model_forward_passes=source["execution"]["model_forward_passes"],
                       network_model_calls=source["execution"]["network_model_calls"])
        example = copy.deepcopy(previous)
        example.update(horizon=horizon, max_steps=horizon, split=selected[cid]["split"],
                       seed=initial["seed"], playback_steps_per_second=256 if game == "maze" else 16)
        example["systems"] = [trained if mid == "nanojev" else
                              copy.deepcopy(next(row for row in previous["systems"] if row["id"] == mid)) for mid in ORDER]
        system_audits = []
        for mid in ORDER:
            if mid == "nanojev":
                system_audits.append(receipt)
                continue
            retained = copy.deepcopy(next(row for row in previous_audit["sources"] if row["system_id"] == mid))
            retained.update(origin="preserved_original_recording", passed=True,
                            frame_json_sha256=original.frame_hash(next(row for row in example["systems"] if row["id"] == mid)),
                            frames_unchanged_from_original_public=True)
            system_audits.append(retained)
        scores = {row["id"]: row["summary"] for row in example["systems"]}
        if game == "maze":
            example["title"] = "50 × 50 · Find the exit"
            example["subtitle"] = "Local safety judgments and shared exploration across 2,500 cells. Every attempted move is recorded."
            example["controller"] = "Local Boolean safety + deterministic verified-edge exploration"
        else:
            example["title"] = "12 × 12 · Keep growing"
            example["subtitle"] = "A 256-move endurance game. One shared safety and food planner; model choices at the remaining forks."
            example["controller"] = "Greedy model choices + shared safety and food planner"
        examples.append(example)
        audits.append({"case_id": cid, "game": game, "size": initial["size"], "horizon": horizon,
                       "initial_state_sha256": arcade.digest(initial), "controller": controller,
                       "sources": system_audits, "results": scores, "passed": True})
        protected.add(path.resolve())
    protocol = {"selected_checkpoint_sha256": NANOJEV_SHA,
                "checkpoint_scope": "The same current unified NanoJev checkpoint used in all four development demos",
                "case_file_sha256": selected_sha, "synchronization": "physical_step",
                "selection": {"selected_case_ids": [e["id"] for e in examples],
                              "rule": "Restore the original fixed 50x50 Maze and 12x12, 256-move Snake showcases; run the current NanoJev checkpoint on those cases without searching for new outcomes.",
                              "outcome_selected": False, "scope": "Two fixed demonstrations; not a new aggregate benchmark"},
                "source_recordings_sha256": {a["game"]: {s["system_id"]: s["sha256"] for s in a["sources"]} for a in audits},
                "controllers": {a["game"]: a["controller"] for a in audits},
                "reused_baselines": "Original Jev and original Qwen runs; exact initial states, inputs, controllers and complete physical trajectories revalidated",
                "probabilities": "Maze uses four independent local-safety probabilities; Snake uses a conditional distribution over common-planner candidates",
                "terminal_playback": "Hold each system at its actual recorded terminal frame",
                "new_api_calls": 0}
    return {"schema": "nanojev-arcade-v1", "examples": examples, "protocol": protocol}, audits, protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", type=Path, default=Path("results/hard_navigation_demo_v1/maze_nanojev.json"))
    parser.add_argument("--snake", type=Path, default=Path("results/hard_navigation_demo_v1/snake_nanojev.json"))
    parser.add_argument("--cases", type=Path, default=Path("configs/hard_navigation_demo_v1_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("web/dev/side_by_side_results.json"))
    parser.add_argument("--receipt", type=Path, default=Path("results/hard_navigation_demo_v1/build_manifest.json"))
    parser.add_argument("--replace-sha256")
    parser.add_argument("--backup", type=Path, default=Path("runs/hard_navigation_demo_v1/previous_navigation_data.json"))
    args = parser.parse_args(argv)
    expected = args.replace_sha256
    if args.output.exists():
        arcade.require(expected and arcade.sha_file(args.output) == expected, "Replacement requires the exact current output SHA")
        arcade.require(not args.backup.exists(), "Backup already exists")
    else:
        arcade.require(expected is None, "Replacement target does not exist")
    arcade.require(not args.receipt.exists(), "Use a fresh receipt path")
    public = Path("web/side_by_side_results.json")
    public_sha = arcade.sha_file(public)
    data, audits, protected = build(args)
    protected.update((public.resolve(), args.cases.resolve(), Path(__file__).resolve()))
    destinations = [args.output, args.receipt, args.backup]
    arcade.require(len({p.resolve() for p in destinations}) == 3 and
                   not {p.resolve() for p in destinations} & protected,
                   "Outputs must be distinct and cannot overwrite source files")
    raw = (json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    sha = hashlib.sha256(raw).hexdigest()
    receipt = {"schema": "nanojev-hard-navigation-demo-v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "passed": True, "export_sha256": sha, "selected_checkpoint_sha256": NANOJEV_SHA,
               "output": {"path": str(args.output), "sha256": sha}, "examples": audits,
               "public_data_sha256_before_and_after": public_sha,
               "previous_development_data_sha256": expected,
               "previous_development_data_backup": str(args.backup) if expected else None,
               "source_case_file": {"path": str(args.cases), "sha256": arcade.sha_file(args.cases)},
               "exporter_sha256": arcade.sha_file(Path(__file__)),
               "summary": {"replayed_display_episodes": 6,
                           "verified_transitions": sum(s["verified_transitions"] for a in audits for s in a["sources"]),
                           "new_nanojev_rollouts": 2, "preserved_baseline_rollouts": 4,
                           "new_model_forward_passes": sum(s.get("model_forward_passes", 0) for a in audits for s in a["sources"]),
                           "new_api_calls": 0, "exporter_model_calls": 0}}
    arcade.require(arcade.sha_file(public) == public_sha, "Public data changed during export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if expected:
        arcade.require(arcade.sha_file(args.output) == expected, "Development data changed during replay")
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        with args.backup.open("xb") as handle:
            handle.write(args.output.read_bytes())
    args.output.write_bytes(raw)
    with args.receipt.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": True, "export_sha256": sha, **receipt["summary"],
                      "results": {a["game"]: a["results"] for a in audits}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
