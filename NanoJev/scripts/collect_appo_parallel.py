#!/usr/bin/env python3
"""Collect Basic demonstrations with independent CPU experts in spawned workers.

The frozen appo_basic_data collector supplies all physics checks, standard-only
replay and target construction. Results are consumed in registered case order.
This wrapper does not modify an existing collection or run training/network IO.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from appo_basic_data import (
    CHECKPOINT_SHA256, SPLITS, TRAIN_SOURCE_COMMIT, collect_episode, digest,
    encode, file_digest, load_policy, read_rows, replay_standard, select_cases,
    training_rows, validate_episode, write_json,
)

_ACTOR = None
_POLICY_ID = None
_SAMPLING_SEED = None
_IDENTITY_SHA256 = None


def cpu_threads_one():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def worker_init(model_dir, config, expected_identity, policy_id, seed, source_sha256):
    global _ACTOR, _POLICY_ID, _SAMPLING_SEED, _IDENTITY_SHA256
    cpu_threads_one()
    for name, expected in source_sha256.items():
        if file_digest(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"Worker source differs from the parent: {name}")
    _ACTOR, identity = load_policy(Path(model_dir), config, "cpu")
    if identity != expected_identity:
        raise RuntimeError("Worker loaded a different expert or preprocessing/runtime contract")
    import cv2
    cv2.setNumThreads(1)
    _POLICY_ID, _SAMPLING_SEED = policy_id, seed
    _IDENTITY_SHA256 = digest(identity)


def collect_one(case):
    if _ACTOR is None:
        raise RuntimeError("Worker initializer did not load an actor")
    before_forwards, before_resets = _ACTOR.forward_count, _ACTOR.reset_count
    episode = collect_episode(case, _ACTOR, _POLICY_ID, _SAMPLING_SEED)
    episode["standard_independent_replay"] = replay_standard(episode)
    forwards = _ACTOR.forward_count - before_forwards
    resets = _ACTOR.reset_count - before_resets
    if forwards != len(episode["steps"]) or resets != 1:
        raise RuntimeError("Actual worker inference counters differ from the recorded episode")
    receipt = {"pid": os.getpid(), "expert_identity_sha256": _IDENTITY_SHA256,
               "actual_expert_forward_calls": forwards, "actual_recurrent_resets": resets,
               "actor_forward_count_before": before_forwards,
               "actor_forward_count_after": _ACTOR.forward_count,
               "actor_reset_count_before": before_resets,
               "actor_reset_count_after": _ACTOR.reset_count}
    episode["worker_execution"] = receipt
    return episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--limit", type=int, default=0,
                        help="First N selected complete episodes for smoke testing; zero retains all")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("--workers must be positive and --limit nonnegative")
    if args.output.exists():
        parser.error("Use a fresh output directory; no existing records are overwritten")
    config = json.loads(args.config.read_text())
    if config.get("case_sha256") and config["case_sha256"] != file_digest(args.cases):
        raise ValueError("Case file differs from the protocol SHA256")
    if config.get("sample_factory_source_commit") != TRAIN_SOURCE_COMMIT:
        raise ValueError("Protocol must pin the original Sample Factory source")
    cases = select_cases(read_rows(args.cases), args.splits.split(","))
    selected_before_limit = len(cases)
    cases = cases[:args.limit] if args.limit else cases
    seed = config["primary"].get("sampling_seed", 17)
    if type(seed) is not int:
        raise ValueError("sampling_seed must be an integer")
    cpu_threads_one()
    # Establish one authoritative identity in the parent. Its actor never does
    # inference and is discarded before children are spawned (no shared RNN).
    parent_actor, identity = load_policy(args.model_dir, config, "cpu")
    if parent_actor.forward_count != 0 or parent_actor.reset_count != 0:
        raise RuntimeError("Parent identity loading unexpectedly performed inference")
    del parent_actor
    implementation = {name: file_digest(Path(__file__).with_name(name)) for name in (
        "collect_appo_parallel.py", "appo_basic_data.py", "evaluate_appo_doom.py",
        "unified_doom_env.py", "unified_game_pipeline.py")}
    collection_policy = {"engine": "sample_factory_appo", "render_adapter": "mirrored_native",
        "controller": "greedy", "epsilon": .1, "temperature": 1.0,
        "sampling_seed": seed, "tie_break": "lexicographic_first", "model": identity,
        "checkpoint_sha256": {"model": CHECKPOINT_SHA256, "cfg": identity["cfg_sha256"]},
        "source_sha256": implementation, "environment_contract": "finite_task_deadline_v1",
        "observation": "standard 320x240 HUD-off visible labels, player variables and existing short history",
        "expert_input": "native 160x120 HUD-on RGB, nearest resize to 128x72, recurrent state reset per episode"}
    policy_id = digest(collection_policy)
    args.output.mkdir(parents=True)
    for variant in ("hard", "soft"):
        (args.output / variant).mkdir()
        for split in SPLITS:
            (args.output / variant / f"{split}.jsonl").touch()
    episode_path = args.output / "episodes.jsonl"
    manifest_path = args.output / "episodes.manifest.json"
    manifest = {"schema_version": "nanojev-unified-episodes-v1",
        "collector_schema": "nanojev-appo-basic-mirrored-parallel-v1", "finished": False,
        "cases_sha256": file_digest(args.cases), "selected_cases_sha256": digest(cases),
        "selected_cases": [c["id"] for c in cases], "config_sha256": file_digest(args.config),
        "policy": collection_policy, "continuation_policy_id": policy_id,
        "selected_episodes": len(cases), "selected_before_limit": selected_before_limit, "limit": args.limit,
        "retention": "all executed states from every complete selected episode, including failures",
        "mirror_checks": "exact equality after every physical tick; raw player variables, clock, reward, native terminal flags",
        "standard_independent_replay": True, "api_calls": 0, "model_downloads": 0,
        "parallel_execution": {"start_method": "spawn", "workers_requested": args.workers,
            "device": "cpu", "torch_threads_per_worker": 1, "opencv_threads_per_worker": 1,
            "result_order": "executor.map in registered case order, chunksize=1",
            "rng": "independent seeded case RNG from frozen collector; RNN reset every episode",
            "parent_actor_forward_calls": 0, "identity_verified_by_every_worker": True}}
    write_json(manifest_path, manifest)
    counts, episode_counts, successes = Counter(), Counter(), Counter()
    worker_counts = defaultdict(Counter)
    total_ticks, completed = 0, 0
    start = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"), initializer=worker_init,
        initargs=(str(args.model_dir.resolve()), config, identity, policy_id, seed, implementation))
    try:
        results = pool.map(collect_one, cases, chunksize=1)
        with episode_path.open("x") as episode_file:
            for case, episode in zip(cases, results):
                if episode["case"] != case or episode["continuation_policy_id"] != policy_id:
                    raise RuntimeError("Worker result changed case order, definition or policy identity")
                validate_episode(episode, seed)
                replay = episode["standard_independent_replay"]
                if replay.get("passed") is not True or replay["decisions"] != len(episode["steps"]):
                    raise RuntimeError("Worker result lacks successful independent standard replay")
                receipt = episode["worker_execution"]
                if (receipt["expert_identity_sha256"] != digest(identity) or
                        receipt["actual_expert_forward_calls"] != len(episode["steps"]) or
                        receipt["actual_recurrent_resets"] != 1):
                    raise RuntimeError("Worker inference receipt is inconsistent")
                episode_file.write(encode(episode) + "\n")
                episode_file.flush()
                for variant in ("hard", "soft"):
                    with (args.output / variant / f"{case['split']}.jsonl").open("a") as handle:
                        for row in training_rows(episode, soft=variant == "soft"):
                            handle.write(encode(row) + "\n")
                counts[case["split"]] += len(episode["steps"])
                episode_counts[case["split"]] += 1
                successes[case["split"]] += int(episode["success"])
                total_ticks += episode["mirrored_physical_ticks"]
                completed += 1
                worker = worker_counts[str(receipt["pid"])]
                worker["episodes"] += 1
                worker["actual_expert_forward_calls"] += receipt["actual_expert_forward_calls"]
                worker["actual_recurrent_resets"] += receipt["actual_recurrent_resets"]
                print(encode({"case": case["id"], "success": episode["success"],
                    "decisions": len(episode["steps"]), "physical_ticks": episode["mirrored_physical_ticks"],
                    "completed": completed, "worker_pid": receipt["pid"], "mirrored_and_replayed": True}), flush=True)
    except BaseException as error:
        # Keep completed records and a visibly incomplete manifest. Cancel tasks
        # not yet running rather than continuing an unconsumed collection.
        manifest.update(error_type=type(error).__name__, completed_episodes=completed,
                        elapsed_seconds=time.monotonic()-start)
        write_json(manifest_path, manifest)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    total_forwards = sum(c["actual_expert_forward_calls"] for c in worker_counts.values())
    total_resets = sum(c["actual_recurrent_resets"] for c in worker_counts.values())
    if completed != len(cases) or total_forwards != sum(counts.values()) or total_resets != completed:
        raise RuntimeError("Actual worker totals do not cover all recorded episodes")
    manifest.update(episode_sha256=file_digest(episode_path), episodes=completed,
        decisions=sum(counts.values()), mirrored_physical_ticks=total_ticks,
        split_records=dict(counts), split_episodes=dict(episode_counts), split_successes=dict(successes),
        elapsed_seconds=time.monotonic()-start, actual_expert_forward_calls=total_forwards,
        actual_recurrent_resets=total_resets, worker_execution_totals=dict(worker_counts),
        summary={s: {"episodes": episode_counts[s], "successes": successes[s]} for s in SPLITS})
    for variant in ("hard", "soft"):
        write_json(args.output / variant / "manifest.json", {
            "schema_version": "nanojev-unified-training-v1", "continuation_policy_id": policy_id,
            "episode_sha256": manifest["episode_sha256"], "episode_file": "../episodes.jsonl",
            "collection_policy": collection_policy, "max_states_per_episode": 0,
            "policy_target_kind": "expert_action" if variant == "hard" else "expert_distribution",
            "training_target": "expert_argmax" if variant == "hard" else "expert_policy_distribution",
            "gold_label_kind": "reference_argmax_compatibility", "split_records": dict(counts),
            "all_successes_and_failures_retained": True,
            "split_sha256": {s: file_digest(args.output / variant / f"{s}.jsonl") for s in SPLITS}})
    manifest["finished"] = True
    write_json(manifest_path, manifest)
    print(encode({"finished": True, "output": str(args.output), "episodes": completed,
                  "records_per_view": sum(counts.values()), "workers_used": len(worker_counts),
                  "actual_worker_forwards": total_forwards}), flush=True)


if __name__ == "__main__":
    main()
