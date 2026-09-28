#!/usr/bin/env python3
"""Read-only APPO data diagnostics; no simulator, model, network or training.

Verify the complete frozen collection and both exported views before reporting
action-policy statistics and exact student-observation ambiguity. Reused visible
observations are measured, not automatically classified as environment leakage.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

from appo_basic_data import training_rows, validate_episode
from unified_game_pipeline import SPLITS, digest, file_digest


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON value: " + value)


def decode(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def load_json(path):
    return decode(Path(path).read_text())


def assert_hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Missing or invalid SHA256: " + label)
    return value


def read_next_row(handle):
    for line in handle:
        if line.strip():
            return decode(line)
    return None


def text_sha(value):
    if not isinstance(value, str):
        raise ValueError("The standard student state must remain its exact recorded text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def quantiles(values):
    if not values:
        return {"n": 0, "mean": None}
    ordered = sorted(values)
    output = {"n": len(values), "mean": math.fsum(values) / len(values)}
    for name, fraction in (("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1)):
        position = fraction * (len(ordered) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        output[name] = ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])
    return output


def new_bucket():
    return {"episodes": 0, "successes": 0, "failures": 0, "decisions": 0,
            "failure_decisions": 0, "action_not_argmax": 0, "tied_argmax_decisions": 0,
            "argmax": Counter(), "actual_action": Counter(), "entropy": [], "max_probability": []}


def bucket_report(bucket):
    n = bucket["decisions"]
    return {key: bucket[key] for key in ("episodes", "successes", "failures", "decisions", "failure_decisions")} | {
        "expert_argmax_counts": dict(sorted(bucket["argmax"].items())),
        "expert_argmax_frequencies": {key: value / n for key, value in sorted(bucket["argmax"].items())} if n else {},
        "actual_action_counts": dict(sorted(bucket["actual_action"].items())),
        "action_not_argmax_count": bucket["action_not_argmax"],
        "action_not_argmax_rate": bucket["action_not_argmax"] / n if n else None,
        "tied_argmax_decisions": bucket["tied_argmax_decisions"],
        "expert_entropy_nats": quantiles(bucket["entropy"]),
        "expert_max_probability": quantiles(bucket["max_probability"])}


def observe_group(groups, key, action, split, episode_id):
    group = groups.setdefault(key, {"labels": Counter(), "splits": set(), "episodes": set()})
    group["labels"][action] += 1
    group["splits"].add(split)
    group["episodes"].add(episode_id)


def ambiguity_report(groups):
    total = sum(sum(group["labels"].values()) for group in groups.values())
    repeated = sum(sum(group["labels"].values()) > 1 for group in groups.values())
    ambiguous = [(key, group) for key, group in groups.items() if len(group["labels"]) > 1]
    ambiguous_rows = sum(sum(group["labels"].values()) for _, group in ambiguous)
    minimum_errors = sum(sum(group["labels"].values()) - max(group["labels"].values()) for group in groups.values())
    examples = sorted(ambiguous, key=lambda item: (-sum(item[1]["labels"].values()), item[0]))[:5]
    return {"decisions": total, "unique_observations": len(groups), "repeated_observation_groups": repeated,
            "ambiguous_groups": len(ambiguous),
            "ambiguous_fraction_of_unique_groups": len(ambiguous) / len(groups) if groups else None,
            "ambiguous_fraction_of_repeated_groups": len(ambiguous) / repeated if repeated else None,
            "decisions_in_ambiguous_groups": ambiguous_rows,
            "fraction_of_decisions_in_ambiguous_groups": ambiguous_rows / total if total else None,
            "ambiguous_groups_with_multiple_episodes": sum(len(group["episodes"]) > 1 for _, group in ambiguous),
            "minimum_empirical_argmax_errors_for_deterministic_observation_mapping": minimum_errors,
            "minimum_empirical_error_rate": minimum_errors / total if total else None,
            "largest_ambiguous_groups": [{"observation_sha256": key, "labels": dict(sorted(group["labels"].items())),
                                           "episode_count": len(group["episodes"]), "splits": sorted(group["splits"])}
                                          for key, group in examples]}


def overlap_report(groups):
    """One hash per episode reset; report rates over episodes and unique hashes."""
    cross = {key: value for key, value in groups.items() if len(value) > 1}
    by_split = {}
    for split in SPLITS:
        total = sum(value[split] for value in groups.values())
        shared = sum(value[split] for value in cross.values())
        by_split[split] = {"episodes": total, "unique_initial_hashes": sum(value[split] > 0 for value in groups.values()),
                           "episodes_with_a_cross_split_match": shared,
                           "cross_split_match_rate": shared / total if total else None}
    return {"unique_initial_hashes": len(groups), "cross_split_hashes": len(cross), "by_split": by_split,
            "pairwise_shared_unique_hashes": {a + "/" + b: sum(value[a] > 0 and value[b] > 0 for value in groups.values())
                                               for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}}


def load_attested_inputs(episodes_path, protocol_path):
    manifest_path = episodes_path.with_suffix(".manifest.json")
    manifest, protocol = load_json(manifest_path), load_json(protocol_path)
    if manifest.get("finished") is not True or manifest.get("limit") != 0:
        raise ValueError("Require a finished full collection with limit=0")
    if manifest.get("episode_sha256") != file_digest(episodes_path):
        raise ValueError("Episode file SHA256 differs from its manifest")
    if manifest.get("config_sha256") != file_digest(protocol_path):
        raise ValueError("Collection protocol SHA256 mismatch")
    reference = Path(protocol["case_file"])
    case_path = reference if reference.is_absolute() else Path(__file__).resolve().parents[1] / reference
    if protocol["case_sha256"] != file_digest(case_path) or manifest.get("cases_sha256") != protocol["case_sha256"]:
        raise ValueError("Frozen cohort file SHA256 mismatch")
    with case_path.open() as handle:
        cases = [decode(line) for line in handle if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate cases in the frozen cohort")
    if dict(Counter(case["split"] for case in cases)) != protocol["case_counts"]:
        raise ValueError("Frozen cohort split counts differ from protocol")
    if any(case["spec"].get("task") != "shooting" or case["spec"].get("scenario") != "basic" for case in cases):
        raise ValueError("Only the complete frozen Basic cohort is supported")
    for key in ("selected_episodes", "selected_before_limit", "episodes"):
        if manifest.get(key) != len(cases):
            raise ValueError("Incomplete full cohort: " + key)
    if manifest.get("selected_cases_sha256") != digest(cases):
        raise ValueError("Selected case identities/specifications differ from full cohort")
    policy = manifest["policy"]
    if (policy.get("engine") != "sample_factory_appo" or policy.get("render_adapter") != "mirrored_native"
            or policy.get("controller") != "greedy" or policy.get("epsilon") != .1):
        raise ValueError("Unexpected expert or behavior policy contract")
    if manifest.get("continuation_policy_id") != digest(policy):
        raise ValueError("Collection policy identity mismatch")
    primary = protocol["primary"]
    model = policy["model"]
    for field, expected in (("model", primary["checkpoint_repository"]), ("revision", primary["checkpoint_revision"]),
                            ("checkpoint_sha256", primary["checkpoint_sha256"])):
        if model.get(field) != expected:
            raise ValueError("Expert model provenance mismatch: " + field)
    if policy.get("sampling_seed") != primary["sampling_seed"]:
        raise ValueError("Expert behavior sampling seed mismatch")
    return manifest, protocol, cases, {"episodes_sha256": manifest["episode_sha256"],
        "manifest_sha256": file_digest(manifest_path), "protocol_sha256": file_digest(protocol_path),
        "cases_sha256": file_digest(case_path)}


def audit(episodes_path, protocol_path):
    episodes_path, protocol_path = Path(episodes_path), Path(protocol_path)
    manifest, protocol, cases, hashes = load_attested_inputs(episodes_path, protocol_path)
    root = episodes_path.parent
    buckets = defaultdict(new_bucket)
    state_groups, request_groups = {}, {}
    split_state_groups, split_request_groups = defaultdict(dict), defaultdict(dict)
    initial_text, initial_pixels, initial_requests = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    initial_rnn_hashes, seen = set(), set()
    row_counts, failure_rows = Counter(), Counter()
    physical_ticks = 0
    dataset_hashes = {}
    with ExitStack() as stack:
        views = {}
        for arm in ("hard", "soft"):
            path = root / arm / "manifest.json"
            dm = load_json(path)
            if dm.get("episode_sha256") != hashes["episodes_sha256"] or dm.get("max_states_per_episode") != 0:
                raise ValueError("Exported view is not the complete recorded trajectory: " + arm)
            if dm.get("all_successes_and_failures_retained") is not True:
                raise ValueError("Exported view does not attest failure retention")
            dataset_hashes[arm] = {"manifest_sha256": file_digest(path), "splits": {}}
            for split in SPLITS:
                path = root / arm / (split + ".jsonl")
                actual = file_digest(path)
                if dm.get("split_sha256", {}).get(split) != actual:
                    raise ValueError("Exported view split SHA256 mismatch: " + arm + "/" + split)
                dataset_hashes[arm]["splits"][split] = actual
                views[(arm, split)] = stack.enter_context(path.open())
        source = stack.enter_context(episodes_path.open())
        for expected_case in cases:
            episode = read_next_row(source)
            if episode is None or episode.get("case") != expected_case:
                raise ValueError("Episode is absent or differs from the ordered full frozen cohort")
            if episode.get("complete") is not True or type(episode.get("success")) is not bool:
                raise ValueError("Every episode must have an explicit complete flag and Boolean outcome")
            if episode["case"]["id"] in seen or episode.get("continuation_policy_id") != manifest["continuation_policy_id"]:
                raise ValueError("Duplicate episode or mismatched collection policy")
            seen.add(episode["case"]["id"])
            validate_episode(episode, manifest["policy"]["sampling_seed"])
            receipt = episode.get("standard_independent_replay", {})
            if receipt.get("passed") is not True or receipt.get("decisions") != len(episode["steps"]):
                raise ValueError("Missing complete standard-only replay receipt")
            observations = [step["observation"] for step in episode["steps"]] + [episode["final_observation"]]
            if receipt.get("standard_observation_sequence_sha256") != digest(observations):
                raise ValueError("Standard-only observation sequence receipt mismatch")
            ticks = episode["mirrored_physical_ticks"]
            if receipt.get("physical_ticks") != ticks:
                raise ValueError("Standard replay physical ticks mismatch")
            physical_ticks += ticks
            split, eid = expected_case["split"], expected_case["id"]
            for name in ("all", split):
                buckets[name]["episodes"] += 1
                buckets[name]["successes" if episode["success"] else "failures"] += 1
            for arm in ("hard", "soft"):
                for expected_row in training_rows(episode, soft=arm == "soft"):
                    if read_next_row(views[(arm, split)]) != expected_row:
                        raise ValueError("Exported training row differs from its actual expert transition: " + arm + "/" + eid)
                    row_counts[(arm, split)] += 1
                    if not episode["success"]:
                        failure_rows[(arm, split)] += 1
            for index, step in enumerate(episode["steps"]):
                pixel = step["pixel_input"]
                pixel_hash = assert_hash(pixel.get("native_frame_sha256"), "native expert frame")
                resized_hash = assert_hash(pixel.get("resized_uint8_sha256"), "resized expert frame")
                if (pixel.get("native_frame_shape_hwc") != [120, 160, 3]
                        or pixel.get("native_direct_resize_sha256") != resized_hash
                        or pixel.get("native_resize_equivalence_verified") is not True
                        or pixel.get("student_text_fed_to_expert") is not False):
                    raise ValueError("Expert native pixel provenance mismatch")
                text = step["observation"]["state"]
                if decode(text).get("screen_size") != [320, 240]:
                    raise ValueError("Student state is not the unchanged standard viewport")
                state_hash = text_sha(text)
                request_hash = digest({key: step["request"][key] for key in ("state", "questions")})
                if index == 0:
                    initial_text[state_hash][split] += 1
                    initial_pixels[pixel_hash][split] += 1
                    initial_requests[request_hash][split] += 1
                    initial_rnn_hashes.add(assert_hash(pixel.get("rnn_before_sha256"), "initial recurrent state"))
                label, probs = step["expert_argmax"], step["policy_probs"]
                for groups, key in ((state_groups, state_hash), (request_groups, request_hash),
                                    (split_state_groups[split], state_hash), (split_request_groups[split], request_hash)):
                    observe_group(groups, key, label, split, eid)
                entropy = -math.fsum(p * math.log(p) for p in probs.values() if p > 0)
                maximum = max(probs.values())
                for name in ("all", split):
                    bucket = buckets[name]
                    bucket["decisions"] += 1
                    bucket["failure_decisions"] += not episode["success"]
                    bucket["action_not_argmax"] += step["action"] != label
                    bucket["tied_argmax_decisions"] += sum(p == maximum for p in probs.values()) > 1
                    bucket["argmax"][label] += 1
                    bucket["actual_action"][step["action"]] += 1
                    bucket["entropy"].append(entropy)
                    bucket["max_probability"].append(maximum)
        if read_next_row(source) is not None or any(read_next_row(handle) is not None for handle in views.values()):
            raise ValueError("Extra episodes or exported training rows outside the frozen collection")
    decisions = buckets["all"]["decisions"]
    if (manifest.get("decisions") != decisions or manifest.get("actual_expert_forward_calls") != decisions
            or manifest.get("actual_recurrent_resets") != len(cases)
            or manifest.get("mirrored_physical_ticks") != physical_ticks):
        raise ValueError("Collection aggregate counters disagree with complete trajectories")
    for split in SPLITS:
        for key, expected in (("split_records", buckets[split]["decisions"]),
                              ("split_episodes", buckets[split]["episodes"]),
                              ("split_successes", buckets[split]["successes"])):
            if manifest.get(key, {}).get(split, 0) != expected:
                raise ValueError("Collection aggregate split counters disagree: " + key + "/" + split)
    if len(initial_rnn_hashes) != 1:
        raise ValueError("Episode initial recurrent-state hashes differ")
    return {"schema_version": "nanojev-appo-data-audit-v1", "validated": True,
            "source_hashes": hashes, "exported_dataset_hashes": dataset_hashes,
            "audit_implementation_sha256": file_digest(__file__),
            "collector_validator_sha256": file_digest(Path(__file__).with_name("appo_basic_data.py")),
            "full_cohort": {"episodes": len(cases), "decisions": decisions, "physical_ticks": physical_ticks,
                            "complete": True, "episode_failures_retained": buckets["all"]["failures"],
                            "all_visited_states_retained_in_both_views": True,
                            "view_rows_by_split": {"/".join(key): n for key, n in sorted(row_counts.items())},
                            "view_failure_rows_by_split": {arm + "/" + split: failure_rows[(arm, split)]
                                                           for arm in ("hard", "soft") for split in SPLITS}},
            "policy_statistics": {key: bucket_report(buckets[key]) for key in ("all", *SPLITS)},
            "exact_state_text_ambiguity": {"all": ambiguity_report(state_groups),
                                          "by_split": {split: ambiguity_report(split_state_groups[split]) for split in SPLITS}},
            "full_student_request_ambiguity": {"all": ambiguity_report(request_groups),
                                              "by_split": {split: ambiguity_report(split_request_groups[split]) for split in SPLITS}},
            "initial_cross_split_reuse": {"standard_state_text": overlap_report(initial_text),
                                          "full_student_request": overlap_report(initial_requests),
                                          "expert_native_pixels": overlap_report(initial_pixels)},
            "notes": ["Entropy uses natural logarithms; statistics are decision-weighted, not independent episode estimates.",
                      "The hard target is the original expert argmax; action_not_argmax measures actual epsilon behavior divergence, not all exploration draws.",
                      "Ambiguity is an empirical conflict at exactly repeated student inputs, not an estimate of generalization error. No observed conflict does not prove sufficient information.",
                      "The expert also receives recurrent pixel history; reset pixel reuse does not imply identical latent environment identity.",
                      "Cross-split visible-state/pixel matches are reported as observation reuse, not automatically as episode or latent-state leakage.",
                      "Stored replay/resize receipts are checked offline; no simulator or image reconstruction is executed by this audit."]}


def self_check():
    import tempfile
    groups = {}
    for label, split, episode in (("left", "train", "a"), ("right", "train", "b"), ("left", "test", "c")):
        observe_group(groups, "same-visible-state", label, split, episode)
    report = ambiguity_report(groups)
    assert report["ambiguous_groups"] == 1 and report["minimum_empirical_argmax_errors_for_deterministic_observation_mapping"] == 1
    assert report["minimum_empirical_error_rate"] == 1 / 3
    overlap = overlap_report({"shared": Counter(train=2, test=1), "unique": Counter(train=1)})
    assert overlap["by_split"]["train"]["cross_split_match_rate"] == 2 / 3
    assert overlap["pairwise_shared_unique_hashes"]["train/test"] == 1
    assert quantiles([0., 1.])["median"] == .5
    for bad in ('{"a":1,"a":2}', '{"a":NaN}'):
        try:
            decode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid JSON was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        case = {"id": "integrity-fixture", "split": "train", "seed": 17,
                "spec": {"task": "shooting", "scenario": "basic"}}
        case_path, episode_path, protocol_path = [root / name for name in ("cases.jsonl", "episodes.jsonl", "protocol.json")]
        case_path.write_text(json.dumps(case) + "\n")
        episode_path.write_text("{}\n")  # Only manifest attestation is tested here; never passed to trajectory diagnostics.
        primary = {"checkpoint_repository": "fixture-model", "checkpoint_revision": "fixture-revision",
                   "checkpoint_sha256": "0" * 64, "sampling_seed": 17}
        protocol = {"case_file": str(case_path), "case_sha256": file_digest(case_path),
                    "case_counts": {"train": 1}, "primary": primary}
        protocol_path.write_text(json.dumps(protocol))
        policy = {"engine": "sample_factory_appo", "render_adapter": "mirrored_native",
                  "controller": "greedy", "epsilon": .1, "sampling_seed": 17,
                  "model": {"model": primary["checkpoint_repository"], "revision": primary["checkpoint_revision"],
                            "checkpoint_sha256": primary["checkpoint_sha256"]}}
        manifest = {"finished": True, "limit": 0, "episode_sha256": file_digest(episode_path),
                    "config_sha256": file_digest(protocol_path), "cases_sha256": file_digest(case_path),
                    "selected_episodes": 1, "selected_before_limit": 1, "episodes": 1,
                    "selected_cases_sha256": digest([case]), "policy": policy, "continuation_policy_id": digest(policy)}
        path = episode_path.with_suffix(".manifest.json")
        path.write_text(json.dumps(manifest))
        load_attested_inputs(episode_path, protocol_path)
        for change in ({"finished": False}, {"limit": 1}, {"selected_before_limit": 2},
                       {"selected_cases_sha256": "1" * 64}, {"episode_sha256": "2" * 64}):
            path.write_text(json.dumps(manifest | change))
            try:
                load_attested_inputs(episode_path, protocol_path)
            except ValueError:
                pass
            else:
                raise AssertionError("Corrupt or partial collection manifest accepted")
    return {"self_check": "passed", "checks": ["empirical ambiguity", "episode versus unique-hash reuse", "quantiles", "strict JSON", "manifest hash and full-cohort negative controls"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "configs/appo_basic_supervision_v1.json")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
        return
    if not args.episodes or not args.output:
        parser.error("--episodes and --output are required")
    if args.output.exists():
        parser.error("Use a fresh diagnostic output path")
    result = audit(args.episodes, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"validated": True, "episodes": result["full_cohort"]["episodes"],
                      "decisions": result["full_cohort"]["decisions"],
                      "failures_retained": result["full_cohort"]["episode_failures_retained"]}))


if __name__ == "__main__":
    main()
