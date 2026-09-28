"""Verified finite-horizon n-step targets for a frozen recorded behavior policy.

The index is keyed by (training_row_id, question_id). Terminal entries contain
terminal_target; bootstrap entries contain only a public request, recorded
behavior probabilities, and action/question mapping. Gamma is one. Native game
score rewards are not success rewards, and physical reposition moves do not add
decision timesteps. This module performs no inference, simulation or API calls.
"""

from collections import Counter
import copy
import json
import math
from pathlib import Path
import random

from unified_game_pipeline import (
    SPLITS, behavior_distribution, choose, digest, environment_group,
    file_digest, outcome_request, validate_cases,
)


IDENTITY_TOLERANCE = 1e-9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _parse(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_nonfinite)


def _probabilities(values, candidates, where):
    _require(isinstance(values, dict) and set(values) == set(candidates), f"{where}: probability candidate mismatch")
    _require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values.values()),
             f"{where}: probabilities must be finite numbers in [0,1]")
    _require(abs(math.fsum(values.values()) - 1) <= IDENTITY_TOLERANCE,
             f"{where}: behavior probabilities must sum to one")


def weighted_bootstrap(action_values, behavior_probs):
    """Return sum_a recorded_pi(a|s) Q_target(s,a); never maximize or refit pi."""
    _require(isinstance(action_values, dict) and bool(action_values), "Nonempty action values are required")
    _require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in action_values.values()),
             "Success action values must be finite probabilities")
    _probabilities(behavior_probs, action_values, "bootstrap")
    result = math.fsum(behavior_probs[action] * action_values[action] for action in action_values)
    # A unit-sum distribution can exceed one by floating-point roundoff. Keep
    # the same tolerance as the probability validation; never repair bad pi.
    _require(-IDENTITY_TOLERANCE <= result <= 1 + IDENTITY_TOLERANCE,
             "Weighted success value is outside the probability range")
    return min(1.0, max(0.0, result))


def _verify_behavior(episode, declaration):
    case, steps = episode["case"], episode["steps"]
    rng = random.Random(int(digest([case["id"], declaration["sampling_seed"]])[:16], 16))
    for index, step in enumerate(steps):
        where = f"{case['id']} decision {index}"
        obs = step["observation"]
        _require(set(obs) == {"task", "state", "candidates", "remaining_steps", "step"}, f"{where}: invalid observation schema")
        _require(obs["task"] == case["spec"]["task"] and isinstance(obs["state"], str) and obs["state"],
                 f"{where}: invalid public task/state")
        candidates = obs["candidates"]
        _require(isinstance(candidates, dict) and bool(candidates) and
                 all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in candidates.items()),
                 f"{where}: live observation needs described candidates")
        _require(type(obs["remaining_steps"]) is int and obs["remaining_steps"] > 0 and
                 type(obs["step"]) is int and obs["step"] >= 0, f"{where}: invalid remaining/physical steps")
        _require(step["action"] in candidates, f"{where}: executed action is not a candidate")
        _probabilities(step.get("behavior_probs"), candidates, where)
        scores = step.get("scores")
        _require(isinstance(scores, dict) and set(scores) == set(candidates) and
                 all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in scores.values()),
                 f"{where}: invalid recorded scores")
        if len(candidates) == 1:
            _require(step.get("forced") is True and not step.get("answers"), f"{where}: singleton must be recorded as forced")
        else:
            _require(step.get("forced") is not True, f"{where}: multiple candidates cannot be forced")
            answers = step.get("answers", {})
            if declaration["controller"] == "q_greedy":
                _require(set(answers) == {"success_" + a for a in candidates}, f"{where}: missing Q answers")
                for action in candidates:
                    answer = answers["success_" + action]
                    _require(answer.get("type") == "boolean", f"{where}: Q answer must be Boolean")
                    probs = answer.get("probabilities", {})
                    _require(set(probs) == {"false", "true"} and all(
                        type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values()),
                        f"{where}: invalid Boolean output")
                    # The persistent predictor exposes finite-precision FP32 softmax values.
                    _require(abs(math.fsum(probs.values()) - 1) <= 1e-5, f"{where}: invalid Boolean probability sum")
                    _require(abs(probs["true"] - scores[action]) <= IDENTITY_TOLERANCE, f"{where}: Q score differs from recorded answer")
        mode = "sample" if declaration["engine"] == "random" or declaration["controller"] == "sample" else "greedy"
        expected = behavior_distribution(scores, mode, declaration["epsilon"])
        _require(all(abs(expected[a] - step["behavior_probs"][a]) <= IDENTITY_TOLERANCE for a in candidates),
                 f"{where}: behavior probabilities differ from frozen controller")
        _require(choose(expected, rng) == step["action"], f"{where}: action differs from recorded policy RNG")


def _load_episodes(path, dataset_manifest):
    path = Path(path)
    sidecar = path.with_suffix(".manifest.json")
    manifest = _parse(sidecar.read_text())
    _require(manifest.get("schema_version") == "nanojev-unified-episodes-v1" and manifest.get("finished") is True,
             "TD requires a finished episode manifest")
    episode_sha = file_digest(path)
    _require(manifest.get("episode_sha256") == episode_sha == dataset_manifest.get("episode_sha256"),
             "Episode SHA256 does not match both manifests")
    declaration, policy_id = manifest["policy"], manifest["continuation_policy_id"]
    _require(policy_id == digest(declaration) == dataset_manifest.get("continuation_policy_id"),
             "Continuation policy ID does not match its frozen declaration and dataset")
    _require(declaration == dataset_manifest.get("collection_policy"), "Collection policy declarations disagree")
    _require(declaration.get("environment_contract") == "finite_task_deadline_v1" and
             declaration.get("tie_break") == "lexicographic_first", "Unsupported frozen environment/controller contract")
    _require(declaration.get("engine") in {"checkpoint", "jev", "random"} and
             declaration.get("controller") in {"greedy", "sample", "q_greedy"} and
             type(declaration.get("sampling_seed")) is int, "Invalid frozen policy declaration")
    _require(declaration.get("source_sha256") and all(isinstance(h, str) and len(h) == 64
                 for h in declaration["source_sha256"].values()), "Frozen policy must declare source hashes")
    snapshot_name = manifest.get("source_snapshot")
    _require(snapshot_name is None or isinstance(snapshot_name, str) and
             snapshot_name not in {"", ".", ".."} and Path(snapshot_name).name == snapshot_name,
             "Source snapshot must name a sibling directory")
    snapshot = path.parent / snapshot_name if snapshot_name else None
    snapshot_verified = False
    if snapshot is not None and snapshot.exists():
        _require(snapshot.is_dir(), "Source snapshot must be a directory")
        for name, expected_sha in declaration["source_sha256"].items():
            _require(Path(name).name == name and file_digest(snapshot / name) == expected_sha,
                     f"Source snapshot hash mismatch: {name}")
        snapshot_verified = True
    episodes = [_parse(line) for line in path.read_text().splitlines() if line.strip()]
    _require(bool(episodes), "No episodes supplied")
    validate_cases([episode["case"] for episode in episodes])
    by_id = {episode["case"]["id"]: episode for episode in episodes}
    selected = manifest.get("selected_cases")
    _require(isinstance(selected, list) and len(selected) == len(set(selected)) and set(selected) == set(by_id),
             "Selected case IDs do not exactly match episode rows")
    for episode in episodes:
        _require(episode.get("complete") is True and episode.get("continuation_policy_id") == policy_id,
                 "Incomplete or mixed-policy episode")
        final, steps = episode.get("final_info", {}), episode.get("steps")
        success = episode.get("success")
        _require(type(success) is bool and final.get("success") is success and
                 final.get("terminated") is True and final.get("truncated") is False,
                 "Episode success/terminal information is inconsistent")
        _require(isinstance(steps, list), "Episode steps must be a list")
        for index, step in enumerate(steps):
            _require(step.get("truncated") is False and step.get("terminated") is (index == len(steps) - 1),
                     "External truncation, early termination, or missing terminal transition")
            _require(step.get("info", {}).get("terminated") is step["terminated"] and
                     step["info"].get("truncated") is False, "Transition terminal information disagrees")
        if steps:
            _require(steps[-1]["info"] == final, "Last transition information differs from final information")
        _verify_behavior(episode, declaration)
    return by_id, manifest, {"episodes_sha256": episode_sha, "manifest_sha256": file_digest(sidecar),
        "policy_declaration_sha256_verified": True, "source_snapshot_verified": snapshot_verified,
        "source_snapshot_note": "verified against declared hashes" if snapshot_verified else
            "snapshot unavailable; declaration identity verified, executable snapshot not reattested by this loader"}


def load_td_index(path, records, dataset_manifest, n_step):
    """Return ({(row_id, qid): entry}, JSON audit) for all outcome rows/splits.

For action a_t, n transitions cover decisions t..t+n-1. If the episode ends
inside that interval, the target is its observed success. Otherwise bootstrap
at the recorded pre-action state s_(t+n), averaging the frozen target network
with that state's recorded behavior distribution. Neither realized success nor
post-bootstrap observations are placed in a nonterminal network request.
"""
    _require(type(n_step) is int and n_step > 0, "n_step must be a positive integer")
    _require(dataset_manifest.get("max_states_per_episode") == 0, "TD requires all transitions, without outcome thinning")
    episodes, manifest, provenance = _load_episodes(path, dataset_manifest)
    policy_id = manifest["continuation_policy_id"]
    group_ids = {cid: environment_group(episode["case"]) for cid, episode in episodes.items()}
    expected = {(cid, index) for cid, episode in episodes.items() for index in range(len(episode["steps"]))}
    seen, row_ids, group_splits, index, counts = set(), set(), {}, {}, Counter()
    ignored_policy = 0
    for row in records:
        _require(isinstance(row.get("id"), str) and row["id"] and row["id"] not in row_ids, "Duplicate or invalid training row ID")
        row_ids.add(row["id"])
        meta = row.get("metadata", {})
        role, split = meta.get("record_role"), row.get("split")
        _require(role in {"policy", "outcome"} and split in SPLITS, "Invalid row role or split")
        group = meta.get("source_group_id")
        _require(isinstance(group, str) and group and (group not in group_splits or group_splits[group] == split),
                 "A source environment group crosses splits")
        group_splits[group] = split
        if role == "policy":
            ignored_policy += 1
            continue
        key = meta.get("episode_id"), meta.get("decision_index")
        _require(type(key[1]) is int and key in expected and key not in seen, "Missing, duplicate or unknown outcome transition")
        seen.add(key)
        episode = episodes[key[0]]
        case, t = episode["case"], key[1]
        step, steps = episode["steps"][t], episode["steps"]
        obs, action = step["observation"], step["action"]
        _require(split == case["split"] and group == group_ids[key[0]] and meta.get("task") == case["spec"]["task"],
                 "Outcome split/task/source group disagrees with its trajectory")
        _require(meta.get("continuation_policy_id") == policy_id and meta.get("spec") == case["spec"],
                 "Outcome policy/spec disagrees with its trajectory")
        _require(meta.get("executed_action") == meta.get("conditioned_action") == action,
                 "Outcome question is not bound to the executed action")
        _require(row.get("state") == obs["state"] and meta.get("remaining_steps") == obs["remaining_steps"] and
                 meta.get("public_observation_sha256") == digest(obs["state"]) and
                 row.get("state_id") == digest([group, case["seed"], t, obs["state"]]),
                 "Outcome state/remaining steps/identity disagrees with its trajectory")
        request = outcome_request(obs, row["id"], [action])
        _require(row.get("questions") == request["questions"], "Outcome question text differs from its recorded action")
        qid = "success_" + action
        _require(row.get("gold") == {qid: episode["success"]} and type(row["gold"][qid]) is bool and
                 row.get("gold_label_kind") == {qid: "observed_outcome"} and "gold_probs" not in row and
                 meta.get("episode_success") is episode["success"], "Outcome label is not the actual terminal success")
        remaining = len(steps) - t
        entry = {"episode_id": key[0], "decision_index": t, "split": split,
                 "n_step": n_step, "actual_decision_steps": min(n_step, remaining), "gamma": 1.0,
                 "continuation_policy_id": policy_id}
        if remaining <= n_step:
            entry["terminal_target"] = float(episode["success"])
            counts[(split, "terminal")] += 1
        else:
            future = steps[t + n_step]
            future_obs = future["observation"]
            request_id = "td-bootstrap:" + digest([policy_id, key[0], t + n_step])
            entry.update(bootstrap_decision_index=t + n_step,
                         request=copy.deepcopy(outcome_request(future_obs, request_id)),
                         behavior_probs=copy.deepcopy(future["behavior_probs"]),
                         action_to_qid={action: "success_" + action for action in future_obs["candidates"]})
            counts[(split, "bootstrap")] += 1
        index[(row["id"], qid)] = entry
    _require(seen == expected, f"Outcome rows do not cover every transition: missing {len(expected-seen)}")
    audit = {**provenance, "episodes": len(episodes), "outcome_rows": len(index), "ignored_policy_rows": ignored_policy,
             "n_step": n_step, "gamma": 1.0, "continuation_policy_id": policy_id,
             "index_key": "(row_id, question_id)", "decision_timestep": "recorded model/controller action; excludes internal physical reposition moves",
             "target": "terminal success indicator, otherwise frozen-target action values averaged with recorded behavior_probs",
             "native_game_rewards_used": False, "behavior_distribution_and_rng_verified": True,
             "cross_split_environment_groups": 0, "complete_transition_coverage": True,
             "by_split": {split: {kind: counts[(split, kind)] for kind in ("terminal", "bootstrap")} for split in SPLITS}}
    return index, audit
