#!/usr/bin/env python3
"""Collect a frozen Basic expert with independently rendered, tick-locked games.

The expert receives its native 160x120 HUD-on RGB frame. Training records use
only the unchanged 320x240 UnifiedDoomEnv observation. Every physical tick is
checked for identical counters, clock, reward and termination in both games.
All executed states are retained, including exploratory actions and failures.
No network client, training operation, or model download is implemented here.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

from evaluate_appo_doom import (
    CHECKPOINT_SHA256, MODEL_ID, REVISION, TRAIN_SOURCE_COMMIT, SF_ACTIONS,
    SampleFactoryPolicy, map_action_distribution, preprocess_frame,
    sample_with_receipt, select_cases,
)
from unified_doom_env import UnifiedDoomEnv
from unified_game_pipeline import (
    SPLITS, behavior_distribution, digest, encode, environment_group,
    file_digest, policy_request, read_rows,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def physics_snapshot(game, vzd):
    """Use raw numeric variables, never rounded text or hidden object metadata."""
    variables = {name: float(game.get_game_variable(getattr(vzd.GameVariable, name)))
                 for name in UnifiedDoomEnv._VARIABLES}
    total_reward = float(game.get_total_reward())
    if not all(math.isfinite(x) for x in [*variables.values(), total_reward]):
        raise RuntimeError("Nonfinite physical audit value")
    return {"variables": variables, "episode_tick": int(game.get_episode_time()),
            "native_timeout_tick": int(game.get_episode_timeout()),
            "finished": bool(game.is_episode_finished()),
            "dead": bool(game.is_player_dead()), "total_reward": total_reward,
            "native_timeout": (bool(game.is_episode_timeout_reached())
                               if hasattr(game, "is_episode_timeout_reached") else None)}


def require_equal(left, right, where):
    if left != right:
        keys = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        raise RuntimeError(f"Mirrored physics mismatch at {where}: {keys}; standard={left}; expert={right}")


class TickMirror:
    """Delegate standard reads; advance the companion once per standard tick."""

    def __init__(self, standard, expert, vzd):
        self.standard, self.expert, self.vzd = standard, expert, vzd
        self.checks = []
        self.total_checked_ticks = 0

    def __getattr__(self, name):
        return getattr(self.standard, name)

    def check(self, where):
        state = physics_snapshot(self.standard, self.vzd)
        require_equal(state, physics_snapshot(self.expert, self.vzd), where)
        return state

    def make_action(self, buttons, ticks):
        if ticks != 1:
            raise ValueError("The unchanged standard adapter must advance exactly one tick")
        before = self.check("before tick")
        standard_reward = float(self.standard.make_action(buttons, 1))
        expert_reward = float(self.expert.make_action(buttons, 1))
        if not math.isfinite(standard_reward) or standard_reward != expert_reward:
            raise RuntimeError("Mirrored per-tick reward differs")
        after = self.check("after tick")
        self.total_checked_ticks += 1
        self.checks.append({"tick_index": self.total_checked_ticks,
                            "before_sha256": digest(before), "after": after,
                            "reward": standard_reward})
        return standard_reward


def native_expert_env(spec):
    """Only rendering differs; its inherited text observation is never exported."""
    env = UnifiedDoomEnv(spec)
    try:
        env._initialize()
        game = env._game
        game.close()
        game.set_screen_resolution(env._vzd.ScreenResolution.RES_160X120)
        game.set_render_hud(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.init()
        return env
    except Exception:
        env.close()
        raise


def rendering_snapshot(game, declared):
    fields = {"width": "get_screen_width", "height": "get_screen_height",
              "hud": "is_render_hud", "decals": "is_render_decals",
              "particles": "is_render_particles"}
    result = {}
    for name, method in fields.items():
        if hasattr(game, method):
            result[name] = getattr(game, method)()
            if name in declared and result[name] != declared[name]:
                raise RuntimeError(f"Rendering getter disagrees with configured {name}")
        elif name in declared:
            # ViZDoom 1.3.0 has render setters but no render-flag getters.
            result[name] = declared[name]
        else:
            raise RuntimeError(f"Cannot attest rendering resolution: missing {method}")
    result["verification"] = {name: "runtime_getter" if hasattr(game, method)
                              else "explicit_setter_receipt_no_runtime_getter"
                              for name, method in fields.items()}
    result["explicit_setter_calls"] = {"set_render_" + k: v for k, v in declared.items()}
    return result


class MirroredEnvironment:
    def __init__(self, spec):
        self.standard = UnifiedDoomEnv(copy.deepcopy(spec))
        self.expert = None
        self.mirror = None
        try:
            self.standard._initialize()
            vzd = self.standard._vzd
            scenarios = Path(getattr(vzd, "scenarios_path", Path(vzd.__file__).parent / "scenarios"))
            config = (scenarios / "basic.cfg").read_text()
            self.standard_render = {}
            for field in ("hud", "decals", "particles"):
                match = re.search(r"^\s*render_" + field + r"\s*=\s*(true|false)\b", config, re.MULTILINE | re.IGNORECASE)
                if not match:
                    raise RuntimeError(f"Bundled Basic cfg must explicitly declare render_{field}")
                self.standard_render[field] = match.group(1).lower() == "true"
            if self.standard_render["hud"]:
                raise RuntimeError("The standard Basic cfg must preserve its existing HUD-off contract")
            # Repeat the already-loaded cfg settings explicitly. No gameplay or
            # observation configuration is altered relative to the standard env.
            for field, value in self.standard_render.items():
                getattr(self.standard._game, "set_render_" + field)(value)
            self.expert = native_expert_env(copy.deepcopy(spec))
        except Exception:
            self.standard.close()
            raise

    def reset(self, seed):
        if self.mirror is not None:
            raise RuntimeError("Create a fresh mirrored environment for each episode")
        obs, info = self.standard.reset(seed)
        self.expert.reset(seed)  # Discard its 160px-label/320px-text observation.
        require_equal(self.standard._metadata, self.expert._metadata, "adapter metadata")
        self.mirror = TickMirror(self.standard._game, self.expert._game, self.standard._vzd)
        initial = self.mirror.check("reset")
        self.standard._game = self.mirror
        render = {"standard": rendering_snapshot(self.mirror.standard, self.standard_render),
                  "expert": rendering_snapshot(self.mirror.expert, {"hud": True, "decals": False, "particles": False})}
        if (render["standard"]["width"], render["standard"]["height"]) != (320, 240):
            raise RuntimeError("Unexpected standard rendering resolution")
        if (render["expert"]["width"], render["expert"]["height"]) != (160, 120):
            raise RuntimeError("Unexpected expert rendering resolution")
        if render["expert"]["hud"] is not True or render["standard"]["hud"] is not False:
            raise RuntimeError("Expected expert HUD-on and standard HUD-off")
        if render["expert"]["decals"] or render["expert"]["particles"]:
            raise RuntimeError("Expert rendering differs from the declared native settings")
        return obs, info, {"initial_physics": initial, "rendering": render,
                           "shared_environment_metadata": copy.deepcopy(self.standard._metadata)}

    def expert_frame(self):
        self.mirror.check("before inference")
        state = self.expert._game.get_state()
        if state is None or state.screen_buffer is None:
            raise RuntimeError("Missing live expert pixels")
        return state.screen_buffer.copy()

    def step(self, action):
        self.mirror.checks = []
        result = self.standard.step(action)
        checks = copy.deepcopy(self.mirror.checks)
        if len(checks) != result[-1]["actual_ticks"]:
            raise RuntimeError("Mirror checks do not cover every executed physical tick")
        return (*result, checks)

    def close(self):
        try:
            self.standard.close()
        finally:
            if self.expert is not None:
                self.expert.close()


def predict_native_frame(policy, frame):
    """Reuse the frozen loader while proving exact original input pixels."""
    import cv2
    import numpy as np
    if frame.shape != (120, 160, 3) or frame.dtype != np.uint8:
        raise ValueError("Expert pixels must be uint8 160x120 RGB24")
    expanded = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
    expected = np.ascontiguousarray(cv2.resize(frame, (128, 72),
                                              interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1))
    if not np.array_equal(preprocess_frame(expanded)[0], expected):
        raise RuntimeError("Reused loader does not equal the native training resize")
    result = policy.predict_pixels(expanded)
    receipt = result["pixel_input"]
    expected_hash = hashlib.sha256(expected.tobytes()).hexdigest()
    if receipt["resized_uint8_sha256"] != expected_hash:
        raise RuntimeError("Loaded policy did not consume the checked native image")
    receipt.update(native_frame_shape_hwc=list(frame.shape),
                   native_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                   native_direct_resize_sha256=expected_hash,
                   loader_raw_frame_role="nearest 2x expansion of native expert frame",
                   native_resize_equivalence_verified=True,
                   student_text_fed_to_expert=False)
    return result


def expert_argmax(probs):
    return min(key for key in probs if probs[key] == max(probs.values()))


def collect_episode(case, policy, policy_id, seed=17, env_factory=MirroredEnvironment,
                    predict=predict_native_frame):
    if case["spec"].get("task") != "shooting" or case["spec"].get("scenario") != "basic":
        raise ValueError("This collector supports Basic only")
    env = env_factory(case["spec"])
    rng = random.Random(int(digest([case["id"], seed])[:16], 16))
    episode = {"case": copy.deepcopy(case), "continuation_policy_id": policy_id,
               "steps": [], "complete": False, "success": None}
    try:
        obs, info, audit = env.reset(case["seed"])
        episode["mirror_initial"] = audit
        episode["initial_observation"] = copy.deepcopy(obs)
        policy.reset_episode()
        while not info["terminated"]:
            if set(obs["candidates"]) != set(SF_ACTIONS):
                raise RuntimeError("Standard Basic must offer all four original actions")
            if json.loads(obs["state"])["screen_size"] != [320, 240]:
                raise RuntimeError("Training observation must use the standard viewport")
            result = predict(policy, env.expert_frame())
            mapped = map_action_distribution(result["sf_logits"], result["sf_probabilities"])
            probs = mapped["policy_probs"]
            behavior = behavior_distribution(probs, "greedy", .1)
            action, draw = sample_with_receipt(behavior, rng)
            next_obs, reward, done, truncated, next_info, ticks = env.step(action)
            if truncated or not 1 <= len(ticks) <= next_info["requested_ticks"]:
                raise RuntimeError("Invalid terminal or physical duration contract")
            episode["steps"].append({
                "observation": copy.deepcopy(obs), "request": policy_request(obs, case["id"]),
                "observation_role": "unchanged standard 320x240 visible-state text",
                "action": action, "expert_argmax": expert_argmax(probs),
                "scores": probs, "policy_probs": probs,
                "policy_probs_raw": mapped["policy_probs_raw"],
                "raw_probability_sum": mapped["raw_probability_sum"],
                "normalization_applied": mapped["normalization_applied"],
                "sf_action_logits": mapped["logits"], "sf_action_index": SF_ACTIONS.index(action),
                "behavior_probs": behavior, "sampling_draw": draw,
                "pixel_input": result["pixel_input"], "forced": False,
                "answers": {"action": {"type": "choice", "probabilities": probs,
                    "probability_semantics": "expert categorical action policy"}},
                "decision_source": "native_rendered_appo_on_standard_mirrored_state",
                "reward": reward, "terminated": done, "truncated": truncated,
                "info": copy.deepcopy(next_info), "mirror_tick_checks": ticks,
                "next_observation_sha256": digest(next_obs)})
            obs, info = next_obs, next_info
        if type(info.get("success")) is not bool or info.get("truncated"):
            raise RuntimeError("A complete episode needs a real terminal outcome")
        episode.update(complete=True, success=info["success"], final_info=copy.deepcopy(info),
                       final_observation=copy.deepcopy(obs),
                       mirrored_physical_ticks=sum(len(s["mirror_tick_checks"]) for s in episode["steps"]))
        validate_episode(episode, seed)
        return episode
    finally:
        env.close()


def validate_episode(episode, seed=17):
    case, steps = episode["case"], episode["steps"]
    if not episode["complete"] or type(episode["success"]) is not bool or not steps:
        raise ValueError("Only complete nonempty episodes can produce training records")
    if not episode["final_info"]["terminated"] or episode["final_info"]["success"] != episode["success"]:
        raise ValueError("Terminal success is inconsistent")
    if episode["final_observation"]["candidates"]:
        raise ValueError("Terminal candidates must be empty")
    rng = random.Random(int(digest([case["id"], seed])[:16], 16))
    ticks_total = 0
    previous = episode["mirror_initial"]["initial_physics"]
    for index, step in enumerate(steps):
        obs, probs = step["observation"], step["policy_probs"]
        if index == 0 and obs != episode["initial_observation"]:
            raise ValueError("Initial standard observation changed")
        if set(obs["candidates"]) != set(SF_ACTIONS) or step["request"] != policy_request(obs, case["id"]):
            raise ValueError("Recorded request is not the standard observation")
        raw, logits = step["policy_probs_raw"], step["sf_action_logits"]
        mapped = map_action_distribution([logits[k] for k in SF_ACTIONS], [raw[k] for k in SF_ACTIONS])
        if probs != mapped["policy_probs"] or step["expert_argmax"] != expert_argmax(probs):
            raise ValueError("Expert target differs from the frozen policy probabilities")
        behavior = behavior_distribution(probs, "greedy", .1)
        action, draw = sample_with_receipt(behavior, rng)
        if (step["behavior_probs"], step["action"], step["sampling_draw"]) != (behavior, action, draw):
            raise ValueError("Actual action does not replay from the declared behavior RNG")
        checks = step["mirror_tick_checks"]
        if len(checks) != step["info"]["actual_ticks"]:
            raise ValueError("Missing physical tick checks")
        for tick in checks:
            ticks_total += 1
            if tick["tick_index"] != ticks_total or tick["before_sha256"] != digest(previous):
                raise ValueError("Broken physical audit chain")
            previous = tick["after"]
        if not math.isclose(sum(t["reward"] for t in checks), step["reward"], abs_tol=1e-10):
            raise ValueError("Per-tick and per-decision rewards differ")
        following = steps[index+1]["observation"] if index+1 < len(steps) else episode["final_observation"]
        if digest(following) != step["next_observation_sha256"]:
            raise ValueError("Broken standard observation trajectory")
        if step["truncated"] or step["terminated"] != (index == len(steps)-1):
            raise ValueError("Episode has an early terminal or incomplete ending")
    if ticks_total != episode["mirrored_physical_ticks"] or ticks_total != episode["final_info"]["episode_metrics"]["physical_ticks"]:
        raise ValueError("Episode physical tick totals disagree")


def replay_standard(episode, env_factory=UnifiedDoomEnv):
    """A fresh standard-only game verifies every recorded visible observation."""
    env = env_factory(copy.deepcopy(episode["case"]["spec"]))
    try:
        obs, info = env.reset(episode["case"]["seed"])
        for step in episode["steps"]:
            if obs != step["observation"]:
                raise RuntimeError("Standard-only replay observation differs")
            obs, reward, done, truncated, info = env.step(step["action"])
            if (reward, done, truncated, info) != (step["reward"], step["terminated"], step["truncated"], step["info"]):
                raise RuntimeError("Standard-only replay transition differs")
        if obs != episode["final_observation"] or info != episode["final_info"]:
            raise RuntimeError("Standard-only final state differs")
        return {"passed": True, "decisions": len(episode["steps"]),
                "physical_ticks": info["episode_metrics"]["physical_ticks"],
                "standard_observation_sequence_sha256": digest([s["observation"] for s in episode["steps"]] + [obs])}
    finally:
        env.close()


def training_rows(episode, soft=False):
    case = episode["case"]
    group = environment_group(case)
    for index, step in enumerate(episode["steps"]):
        obs, probs = step["observation"], step["policy_probs"]
        identity = digest([episode["continuation_policy_id"], case["id"], index, "policy"])
        row = {**policy_request(obs, identity),
            "state_id": digest([group, case["seed"], index, obs["state"]]),
            "family_id": "unified_shooting", "split": case["split"],
            "gold": {"action": step["expert_argmax"]},
            "gold_label_kind": {"action": "reference_argmax_compatibility"},
            "expert": {"model": MODEL_ID, "revision": REVISION,
                       "checkpoint_sha256": CHECKPOINT_SHA256,
                       "policy_probs_raw": copy.deepcopy(step["policy_probs_raw"]),
                       "policy_probs": copy.deepcopy(probs),
                       "raw_probability_sum": step["raw_probability_sum"],
                       "normalization_applied": step["normalization_applied"]},
            "metadata": {"task": "shooting", "record_role": "policy",
                "policy_target_kind": "expert_action", "episode_id": case["id"],
                "source_group_id": group, "decision_index": index,
                "public_observation_sha256": digest(obs["state"]),
                "executed_action": step["action"], "conditioned_action": step["action"],
                "continuation_policy_id": episode["continuation_policy_id"],
                "episode_success": episode["success"], "spec": copy.deepcopy(case["spec"]),
                "remaining_steps": obs["remaining_steps"],
                "observation_source": "standard_320x240_visible_state_text",
                "expert_argmax": step["expert_argmax"],
                "behavior_probs": copy.deepcopy(step["behavior_probs"]),
                "sampling_draw": step["sampling_draw"],
                "expert_pixel_input": copy.deepcopy(step["pixel_input"]),
                "mirror_tick_checks_sha256": digest(step["mirror_tick_checks"])}}
        if soft:
            row["gold_probs"] = {"action": copy.deepcopy(probs)}
            row["gold_probs_kind"] = {"action": "expert_policy_distribution"}
            row["metadata"]["policy_target_kind"] = "expert_distribution"
        yield row


def load_policy(model_dir, config, device):
    primary = config["primary"]
    expected = {"checkpoint_repository": MODEL_ID, "checkpoint_revision": REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256, "controller": "greedy", "epsilon": .1}
    for key, value in expected.items():
        if primary.get(key) != value:
            raise ValueError(f"Frozen Basic expert configuration differs: {key}")
    local = json.loads((model_dir / "download_manifest.json").read_text())
    if local.get("repo") != MODEL_ID or local.get("revision") != REVISION:
        raise ValueError("Local model manifest differs from the frozen public revision")
    checkpoint = model_dir / primary["checkpoint_file"]
    if not checkpoint.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Checkpoint path must stay inside the local model directory")
    if file_digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Local checkpoint bytes differ from the frozen SHA256")
    recorded = {item["file"]: item for item in local["files"]}
    for file in (primary["checkpoint_file"], "cfg.json"):
        if file not in recorded or file_digest(model_dir / file) != recorded[file]["sha256"]:
            raise ValueError(f"Local artifact differs from its download receipt: {file}")
    policy = SampleFactoryPolicy(checkpoint, model_dir / "cfg.json", device=device,
                                 runtime_source_commit=TRAIN_SOURCE_COMMIT)
    identity = copy.deepcopy(policy.identity)
    identity.update(model=MODEL_ID, revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA256,
                    cfg_sha256=file_digest(model_dir / "cfg.json"),
                    evaluation_render_size_wh=[160, 120],
                    render_contract="native 160x120 HUD-on expert; separate standard 320x240 text environment")
    return policy, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--limit", type=int, default=0, help="First N selected complete episodes for an explicit smoke test; zero keeps all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.output.exists():
        parser.error("Use a fresh output directory; completed or partial records are never overwritten")
    config = json.loads(args.config.read_text())
    if config.get("case_sha256") and config["case_sha256"] != file_digest(args.cases):
        raise ValueError("Case file differs from the protocol SHA256")
    if config.get("sample_factory_source_commit") != TRAIN_SOURCE_COMMIT:
        raise ValueError("Protocol must pin the original Sample Factory source")
    cases = select_cases(read_rows(args.cases), args.splits.split(","))
    selected_before_limit = len(cases)
    cases = cases[:args.limit] if args.limit else cases
    policy, identity = load_policy(args.model_dir, config, args.device)
    seed = config["primary"].get("sampling_seed", 17)
    if type(seed) is not int:
        raise ValueError("sampling_seed must be an integer")
    implementation = {name: file_digest(Path(__file__).with_name(name)) for name in (
        "appo_basic_data.py", "evaluate_appo_doom.py", "unified_doom_env.py", "unified_game_pipeline.py")}
    collection_policy = {"engine": "sample_factory_appo", "render_adapter": "mirrored_native",
        "controller": "greedy", "epsilon": .1, "temperature": 1.0,
        "sampling_seed": seed, "tie_break": "lexicographic_first", "model": identity,
        "checkpoint_sha256": {"model": CHECKPOINT_SHA256, "cfg": identity["cfg_sha256"]},
        "source_sha256": implementation,
        "environment_contract": "finite_task_deadline_v1",
        "observation": "standard 320x240 HUD-off visible labels, player variables and existing short history",
        "expert_input": "native 160x120 HUD-on RGB, nearest resize to 128x72, recurrent state reset per episode"}
    policy_id = digest(collection_policy)
    args.output.mkdir(parents=True)
    for variant in ("hard", "soft"):
        (args.output / variant).mkdir()
        for split in SPLITS:
            (args.output / variant / f"{split}.jsonl").touch()
    episode_path = args.output / "episodes.jsonl"
    manifest = {"schema_version": "nanojev-unified-episodes-v1",
        "collector_schema": "nanojev-appo-basic-mirrored-v1", "finished": False,
        "cases_sha256": file_digest(args.cases), "selected_cases_sha256": digest(cases),
        "selected_cases": [c["id"] for c in cases],
        "config_sha256": file_digest(args.config), "policy": collection_policy,
        "continuation_policy_id": policy_id, "selected_episodes": len(cases),
        "selected_before_limit": selected_before_limit, "limit": args.limit,
        "retention": "all executed states from every complete selected episode, including failures",
        "mirror_checks": "exact equality after every physical tick; raw player variables, clock, reward, native terminal flags",
        "standard_independent_replay": True, "api_calls": 0, "model_downloads": 0}
    write_json(args.output / "episodes.manifest.json", manifest)
    counts, summary_episodes = Counter(), []
    start = time.monotonic()
    # Append complete episodes immediately. A failing mirror/replay leaves a
    # finished=false manifest and never enters any partial episode into training.
    with episode_path.open("x") as episode_file:
        for case in cases:
            episode = collect_episode(case, policy, policy_id, seed)
            episode["standard_independent_replay"] = replay_standard(episode)
            episode_file.write(encode(episode) + "\n")
            episode_file.flush()
            for variant in ("hard", "soft"):
                rows = list(training_rows(episode, soft=variant == "soft"))
                with (args.output / variant / f"{case['split']}.jsonl").open("a") as handle:
                    for row in rows:
                        handle.write(encode(row) + "\n")
            counts[case["split"]] += len(episode["steps"])
            summary_episodes.append({k: v for k, v in episode.items() if k not in ("steps", "initial_observation", "final_observation")})
            print(encode({"case": case["id"], "success": episode["success"], "decisions": len(episode["steps"]),
                          "physical_ticks": episode["mirrored_physical_ticks"], "mirrored_and_replayed": True}), flush=True)
    manifest.update(finished=True, episode_sha256=file_digest(episode_path),
                    episodes=len(cases), decisions=sum(counts.values()),
                    mirrored_physical_ticks=sum(e["mirrored_physical_ticks"] for e in summary_episodes),
                    split_records=dict(counts),
                    split_episodes=dict(Counter(e["case"]["split"] for e in summary_episodes)),
                    split_successes=dict(Counter(e["case"]["split"] for e in summary_episodes if e["success"])),
                    elapsed_seconds=time.monotonic()-start,
                    actual_expert_forward_calls=policy.forward_count,
                    actual_recurrent_resets=policy.reset_count)
    # Existing summarize expects steps; aggregate terminal metrics independently.
    manifest["summary"] = {split: {"episodes": sum(e["case"]["split"] == split for e in summary_episodes),
        "successes": sum(e["success"] for e in summary_episodes if e["case"]["split"] == split)} for split in SPLITS}
    write_json(args.output / "episodes.manifest.json", manifest)
    for variant in ("hard", "soft"):
        dataset_manifest = {"schema_version": "nanojev-unified-training-v1",
            "continuation_policy_id": policy_id, "episode_sha256": manifest["episode_sha256"],
            "episode_file": "../episodes.jsonl", "collection_policy": collection_policy,
            "max_states_per_episode": 0,
            "policy_target_kind": "expert_action" if variant == "hard" else "expert_distribution",
            "training_target": "expert_argmax" if variant == "hard" else "expert_policy_distribution",
            "gold_label_kind": "reference_argmax_compatibility",
            "split_records": dict(counts), "all_successes_and_failures_retained": True,
            "split_sha256": {s: file_digest(args.output / variant / f"{s}.jsonl") for s in SPLITS}}
        write_json(args.output / variant / "manifest.json", dataset_manifest)
    print(encode({"finished": True, "output": str(args.output), "episodes": len(cases), "records_per_view": sum(counts.values())}), flush=True)


if __name__ == "__main__":
    main()
