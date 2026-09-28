#!/usr/bin/env python3
"""Export current unified Maze and Snake recordings for the development viewer.

All navigation episodes in the complete frozen source files are independently
replayed on CPU, including exact observations, actions, controller RNG draws,
physical transitions and terminal results. Only selected real cases are shown.
Public-site assets are read-only. No inference or API requests are performed.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from build_predict_position_demo import CASES_SHA, MODEL_TEXT, NANOJEV_SHA, load_recordings
from build_shooting_demo import ORDER, match, read_rows, require
from replay_unified_episodes import load_json
from unified_game_pipeline import behavior_distribution, choose, digest, factory, file_digest

COLORS = {"jev": "#578d9a", "nanojev": "#198875", "base": "#8b77b1"}
GRID_FILES = ("unified_grid_envs.py", "scaled_maze.py", "snake_game.py", "evaluate_composed_maze.py")


def validate_grid_sources(path, provenance):
    manifest = load_json(Path(provenance["manifest_path"]).read_text())
    snapshots = path.parent / manifest["source_snapshot"]
    for name in GRID_FILES:
        sha = file_digest(Path(__file__).with_name(name))
        require(provenance["policy"]["source_sha256"].get(name) == sha and
                file_digest(snapshots / name) == sha,
                f"Recorded and replayed grid implementation differ: {name}")


def validate_decision(recorded, observation, rng):
    """Keep original model probabilities distinct from the executed controller."""
    offered = observation["candidates"]
    scores = recorded["scores"]
    require(set(scores) == set(offered) and bool(scores), "Scores differ from offered candidates")
    require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in scores.values()) and
            math.isclose(math.fsum(scores.values()), 1, abs_tol=1e-5, rel_tol=0),
            "Invalid original action probabilities")
    forced = len(offered) == 1
    require(recorded.get("forced", False) == forced, "Recorded forced-decision flag differs")
    if forced:
        require(recorded["answers"] == {} and list(scores.values()) == [1.0], "Forced move contains a model answer")
    else:
        match(recorded["answers"]["action"]["probabilities"], scores, "original_action_probabilities")
    behavior = behavior_distribution(scores, "greedy", .1)
    match(recorded["behavior_probs"], behavior, "behavior_probabilities")
    if "sampling_draw" in recorded:
        probe = random.Random()
        probe.setstate(rng.getstate())
        match(recorded["sampling_draw"], probe.random(), "sampling_draw")
    require(recorded["action"] == choose(behavior, rng), "Action differs from the frozen controller RNG")
    return copy.deepcopy(scores), forced


def render_episode(episode, model_id):
    case = episode["case"]
    game = case["spec"]["task"]
    require(game in ("maze", "snake"), "Expected a navigation case")
    env = factory(case["spec"])
    try:
        observation, info = env.reset(case["seed"])
        initial = copy.deepcopy(env._world)
        rng = random.Random(int(digest([case["id"], 17])[:16], 16))
        zero = {"score": 0, "action": None, "probabilities": {}, "collision": False,
                "decision_source": "initial", "forced": False, "done": info["terminated"],
                "decision_index": 0, "physical_step": 0}
        if game == "maze":
            zero["position"] = copy.deepcopy(initial["position"])
        else:
            zero.update(body=copy.deepcopy(initial["body"]), food=copy.deepcopy(initial["food"]),
                        direction=initial["direction"], score=initial["score"])
        frames = [zero]
        terminated, truncated = info["terminated"], info["truncated"]
        for index, recorded in enumerate(episode["steps"]):
            require(not terminated and not truncated, "Recorded action occurs after termination")
            match(recorded["observation"], observation, f"steps[{index}].observation")
            scores, forced = validate_decision(recorded, observation, rng)
            observation, reward, terminated, truncated, info = env.step(recorded["action"])
            for name, value in (("reward", reward), ("terminated", terminated),
                                ("truncated", truncated), ("info", info)):
                match(recorded[name], value, f"steps[{index}].{name}")
            events = info["physical_events"]
            require(events and events[0]["actor"] == "model", "Action lacks its physical model transition")
            for event_index, event in enumerate(events):
                macro = event["actor"] == "verified_edge_reposition"
                require(event["step"] == len(frames), "Physical frame sequence is not contiguous")
                frame = {"action": event["action"], "probabilities": {} if macro or forced else scores,
                         "controller_probabilities": {} if macro else copy.deepcopy(recorded["behavior_probs"]),
                         "collision": event["collision"], "decision_index": index + 1,
                         "physical_step": event["step"], "forced": macro or forced,
                         "decision_source": "verified_edge_reposition" if macro else
                                            "forced_single_available_action" if forced else "recorded_model_action",
                         "done": bool(terminated and event_index == len(events) - 1)}
                if game == "maze":
                    frame.update(position=copy.deepcopy(event["next_position"]),
                                 decision_position=copy.deepcopy(event["position"]), score=int(event["success"]))
                    match(frames[-1]["position"], event["position"], "contiguous_maze_position")
                else:
                    require(len(events) == 1 and not macro, "Snake action must advance one physical step")
                    frame.update(body=copy.deepcopy(env._world["body"]), food=copy.deepcopy(env._world["food"]),
                                 direction=env._world["direction"], score=env._world["score"])
                    match(frames[-1]["body"][0], event["head"], "contiguous_snake_head")
                    match(frame["body"][0], event["next_head"], "next_snake_head")
                frames.append(frame)
        require(terminated and not truncated and observation["candidates"] == {}, "Episode lacks a real terminal state")
        match(episode["final_info"], info, "final_info")
        match(episode["success"], info["success"], "success")
        if "final_observation" in episode:
            match(episode["final_observation"], observation, "final_observation")
        metrics = info["episode_metrics"]
        require(len(frames) == metrics["physical_steps"] + 1, "Frames and physical steps differ")
        require(sum(int(f["collision"]) for f in frames) == metrics["collisions"], "Collision total differs")
        outcome = "target_reached" if info["outcome"] == "target_food_reached" else info["outcome"]
        summary = {"steps": metrics["physical_steps"], "score": frames[-1]["score"],
                   "collisions": metrics["collisions"], "outcome": outcome,
                   "source_outcome": info["outcome"], "success": info["success"],
                   "decision_steps": metrics["decision_steps"]}
        if game == "snake":
            summary.update(target_food=metrics["target_food"], food_collected=metrics["food_collected"])
        system = {"id": model_id, "name": MODEL_TEXT[model_id][0], "detail": MODEL_TEXT[model_id][1],
                  "color": COLORS[model_id], "probability_kind": "action_distribution",
                  "summary": summary, "frames": frames}
        audit = {"case_id": case["id"], "system_id": model_id, "passed": True,
                 "initial_state_sha256": digest(initial), "source_episode_sha256": digest(episode),
                 "frame_json_sha256": digest(frames), "verified_decisions": len(episode["steps"]),
                 "verified_physical_steps": metrics["physical_steps"], "final_metrics": metrics,
                 "model_probabilities": "Unmodified scores; code moves carry no model probabilities",
                 "controller_rng_verified": True, "exact_observations_and_transitions_verified": True}
        return initial, system, audit
    finally:
        env.close()


def aggregate(sources, registry):
    summary = {}
    for model_id in ORDER:
        summary[model_id] = {}
        for game in ("maze", "snake"):
            summary[model_id][game] = {}
            for split in ("test", "ood"):
                ids = [cid for cid, case in registry.items() if case["split"] == split and case["spec"]["task"] == game]
                require(len(ids) == (10 if game == "maze" else 8), "Unexpected complete navigation split size")
                successes = sum(sources[model_id][cid]["success"] for cid in ids)
                summary[model_id][game][split] = {"episodes": len(ids), "successes": successes,
                                                "success_rate": successes / len(ids)}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev", type=Path, default=Path("runs/sonic_unified_sft_v1/jev_parallel_recovery_v2/episodes.jsonl"))
    parser.add_argument("--nanojev", type=Path, default=Path("runs/sonic_unified_sft_v1/experiment/selected_test.jsonl"))
    parser.add_argument("--base", type=Path, default=Path("runs/sonic_unified_sft_v1/native_test.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("runs/sonic_unified_sft_v1/test_cases.jsonl"))
    parser.add_argument("--selection", type=Path, default=Path("configs/unified_navigation_demo_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("web/dev/side_by_side_results.json"))
    parser.add_argument("--receipt", type=Path, default=Path("results/unified_navigation_demo_v1/build_manifest.json"))
    args = parser.parse_args(argv)
    require(file_digest(args.cases) == CASES_SHA, "Expected the original frozen 548-case registry")
    cases = read_rows(args.cases)
    registry = {case["id"]: case for case in cases}
    require(len(cases) == len(registry) == 548, "Expected 548 unique frozen cases")
    config = load_json(args.selection.read_text())
    selected = {row["id"]: row for row in config["cases"]}
    require(len(selected) == 2 and {row["game"] for row in selected.values()} == {"maze", "snake"},
            "Select exactly one Maze and one Snake illustration")
    root = Path(__file__).resolve().parents[1]
    public = root / "web/side_by_side_results.json"
    require(args.output.resolve() != public.resolve(), "The public-site data is read-only")
    require(args.output.resolve() != args.receipt.resolve(), "Data and receipt must be distinct")
    require(not args.output.exists() and not args.receipt.exists(), "Use fresh output and receipt paths")
    public_sha = file_digest(public)
    sources, provenance = {}, {}
    for model_id in ORDER:
        path = getattr(args, model_id)
        sources[model_id], provenance[model_id] = load_recordings(path, model_id, registry, CASES_SHA)
        validate_grid_sources(path, provenance[model_id])
    examples, audits, retained = [], [], {}
    for model_id in ORDER:
        for cid, episode in sources[model_id].items():
            if episode["case"]["spec"]["task"] not in ("maze", "snake"):
                continue
            initial, system, audit = render_episode(episode, model_id)
            audits.append(audit)
            if cid in selected:
                retained[(cid, model_id)] = (initial, system)
    for cid, row in selected.items():
        case = registry[cid]
        require(case["spec"]["task"] == row["game"], "Selected case game differs")
        match(row["expected_success"], {mid: sources[mid][cid]["success"] for mid in ORDER}, "selected_outcomes")
        initial = retained[(cid, "nanojev")][0]
        for mid in ORDER:
            match(initial, retained[(cid, mid)][0], "shared_initial_state")
        game = row["game"]
        example = {"id": cid, "game": game, "title": row["title"], "subtitle": row["subtitle"],
                   "split": case["split"], "seed": case["seed"], "size": case["spec"]["size"],
                   "max_steps": case["spec"]["max_steps"], "probability_kind": "action_distribution",
                   "controller": "Greedy actions with 10% exploration; shared sampling seed 17",
                   "playback_steps_per_second": row["playback_steps_per_second"],
                   "systems": [retained[(cid, mid)][1] for mid in ORDER]}
        if game == "maze":
            example["initial"] = {key: initial[key] for key in ("walls", "position", "goal")}
        else:
            example["initial"] = {key: initial[key] for key in ("body", "food")}
            example["target_food"] = case["spec"]["target_food"]
        examples.append(example)
    selection = {"rule": config["selection_rule"], "selected_case_ids": list(selected), "outcome_selected": True,
                 "aggregate_scope": "Every frozen Maze/Snake case, with test and OOD reported separately",
                 "selection_changes_training_or_evaluation": False}
    data = {"schema": "nanojev-arcade-v1", "examples": examples, "summary": aggregate(sources, registry),
            "protocol": {"synchronization": "physical_step", "frame_zero": "actual simulator reset state",
                         "controller": "greedy", "epsilon": .1, "sampling_seed": 17,
                         "probabilities": "Original model action probabilities before epsilon exploration; empty on code moves",
                         "maze_code_moves": "Reposition only along previously traversed open edges; all physical steps retained",
                         "snake_success": "Collect the configured target_food after reset",
                         "terminal_playback": "Hold each system at its actual terminal frame",
                         "selected_checkpoint_sha256": NANOJEV_SHA, "case_file_sha256": CASES_SHA,
                         "source_episodes_sha256": {mid: provenance[mid]["episodes_sha256"] for mid in ORDER},
                         "selection": selection, "new_model_calls": 0, "new_api_calls": 0}}
    raw = (json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(file_digest(public) == public_sha, "The public-site data changed during export")
    for mid in ORDER:
        require(file_digest(getattr(args, mid)) == provenance[mid]["episodes_sha256"], "A recording changed during export")
    receipt = {"schema": "nanojev-unified-navigation-demo-v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "passed": True, "sources": provenance, "selection": selection, "episodes": audits,
               "summary": {"replayed_episodes": len(audits), "displayed_episodes": 6,
                           "verified_decisions": sum(a["verified_decisions"] for a in audits),
                           "verified_physical_steps": sum(a["verified_physical_steps"] for a in audits),
                           "model_calls": 0, "api_calls": 0},
               "source_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in
                                 (Path(__file__).name, "build_predict_position_demo.py", "unified_game_pipeline.py", *GRID_FILES)},
               "public_data_sha256_before_and_after": public_sha,
               "selection_sha256": file_digest(args.selection), "output": {"path": str(args.output)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(raw)
    receipt["output"]["sha256"] = file_digest(args.output)
    with args.receipt.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": True, **receipt["summary"], "output": str(args.output), "sha256": receipt["output"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
