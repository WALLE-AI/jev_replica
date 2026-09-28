"""No API calls: verify bounded journal accounting and all-or-nothing merging."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

from run_jev_parallel_cases import (
    allocate_budgets, audit_journal, behavior_distribution, choose, digest,
    encode, file_digest, merge_results, ordered_shards, policy_request,
    prepare_journals, source_hashes, summarize, write_json, execute_jobs,
)


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def journal(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(encode(row) + "\n" for row in rows))

    def test_unknown_failures_and_interrupted_starts_remain_reserved(self):
        path = self.root / "calls.jsonl"
        self.journal(path, [
            {"id": "a", "status": "started", "reservation_usd": .004},
            {"id": "a", "status": "succeeded", "cost_usd": 0},
            {"id": "b", "status": "started", "reservation_usd": .003},
            {"id": "b", "status": "failed", "unknown_cost": True},
            {"id": "c", "status": "started", "reservation_usd": .005},
        ])
        report, calls = audit_journal(path)
        self.assertEqual(Decimal(report["unknown_reserved_usd"]), Decimal(".008"))
        self.assertEqual(report["unresolved_started_ids"], ["c"])
        self.assertEqual(report["unknown_cost_ids"], ["b", "c"])
        self.assertEqual(report["zero_reported_cost_calls"], 1)
        self.assertEqual(set(calls), {"a"})
        budgets = allocate_budgets("3", [report["accounted_usd"]] + ["0"]*7)
        self.assertLessEqual(sum(map(Decimal, budgets)), Decimal(3))
        self.assertEqual(sum(map(Decimal, budgets)), Decimal(3))
        self.assertGreater(Decimal(budgets[0]), Decimal(budgets[1]))
        with self.assertRaises(ValueError):
            allocate_budgets("3", ["3"] + ["0"]*7)
        with self.assertRaises(ValueError):
            allocate_budgets("3.01", ["0"]*8)

    def test_nonfinite_duplicate_or_partial_journal_rejected(self):
        path = self.root / "calls.jsonl"
        self.journal(path, [{"id": "a", "status": "succeeded", "cost_usd": "NaN"}])
        with self.assertRaises(ValueError):
            audit_journal(path)
        row = {"id": "a", "status": "started", "reservation_usd": .002}
        self.journal(path, [row, row])
        with self.assertRaises(ValueError):
            audit_journal(path)
        path.write_text('{"id":')
        with self.assertRaises(ValueError):
            audit_journal(path)

    def test_contiguous_548_chunks_and_prior_copied_only_once(self):
        cases = [{"id": str(i)} for i in range(548)]
        shards = ordered_shards(cases, 8)
        self.assertEqual(list(map(len, shards)), [69]*7 + [65])
        self.assertEqual(sum(shards, []), cases)
        prior = self.root / "prior"
        self.journal(prior / "calls.jsonl", [{"id": "x", "status": "started", "reservation_usd": .002}])
        target = self.root / "journals"
        declaration = prepare_journals(target, prior, "casehash", shards)
        self.assertEqual((target / "shard_00/calls.jsonl").read_bytes(), (prior / "calls.jsonl").read_bytes())
        self.assertTrue(all(not (target / f"shard_{i:02d}/calls.jsonl").exists() for i in range(1,8)))
        self.assertEqual(prepare_journals(target, None, "casehash", shards), declaration)
        with self.assertRaises(ValueError):
            prepare_journals(target, None, "wrong-casehash", shards)

    def fixtures(self):
        cases = [{"id": f"test-{i}", "seed": 100+i, "split": "test", "variant": "doom_basic",
                  "spec": {"task": "shooting", "scenario": "basic"}} for i in range(2)]
        case_path = self.root / "cases.jsonl"
        case_path.write_text("".join(encode(c) + "\n" for c in cases))
        sources = source_hashes()
        policy = {"engine": "jev", "controller": "greedy", "epsilon": .1, "sampling_seed": 17,
                  "model": "typesafe-ai/jev", "temperature": 1.0, "source_sha256": sources,
                  "tie_break": "lexicographic_first", "environment_contract": "finite_task_deadline_v1"}
        jobs = []
        for i, case in enumerate(cases):
            directory = self.root / f"shard_{i:02d}"
            directory.mkdir()
            cp, ep = directory / "cases.jsonl", directory / "episodes.jsonl"
            cp.write_text(encode(case) + "\n")
            obs = {"state": f"public observation {i}", "candidates": {"left": "Turn left", "shoot": "Attack"}}
            scores = {"left": .2, "shoot": .8}
            behavior = behavior_distribution(scores, "greedy", .1)
            rng = random.Random(int(digest([case["id"], 17])[:16], 16))
            info = {"terminated": True, "truncated": False, "success": True,
                    "episode_metrics": {"kills": 1, "physical_ticks": 4}}
            # Identical call IDs in different shards must retain separate provenance.
            receipt = {"id": "nanojev-live-000001", "status": "succeeded", "cost_usd": .001,
                "input": {"model": "typesafe-ai/jev", "state": obs["state"],
                          "questions": policy_request(obs, case["id"])["questions"]},
                "input_sha256": f"input-{i}", "native_probs": {"action": scores}}
            self.journal(directory / "journal/calls.jsonl", [receipt])
            answer = {"source_api_call_id": receipt["id"], "source_input_sha256": receipt["input_sha256"],
                      "native_probabilities": scores}
            row = {"case": case, "complete": True, "success": True, "final_info": info,
                "continuation_policy_id": digest(policy), "steps": [{"observation": obs,
                    "action": choose(behavior, rng), "scores": scores, "behavior_probs": behavior,
                    "answers": {"action": answer}, "truncated": False, "terminated": True}]}
            ep.write_text(encode(row) + "\n")
            snapshot = directory / "episodes.sources"
            snapshot.mkdir()
            for name in sources:
                shutil.copy2(Path(__file__).with_name(name), snapshot / name)
            manifest = {"finished": True, "episode_sha256": file_digest(ep), "cases_sha256": file_digest(cp),
                "selected_cases": [case["id"]], "policy": policy, "continuation_policy_id": digest(policy),
                "source_snapshot": snapshot.name, "summary": summarize([row])}
            write_json(ep.with_suffix(".manifest.json"), manifest)
            jobs.append({"shard": i, "cases": str(cp), "episodes": str(ep), "journal": str(directory / "journal"),
                         "returncode": 0, "budget_usd": ".5"})
        return cases, case_path, jobs, sources

    def test_merge_keeps_case_order_raw_bytes_and_receipt_namespaces(self):
        cases, path, jobs, sources = self.fixtures()
        output = self.root / "merged"
        output.mkdir()
        report = merge_results(cases, path, list(reversed(jobs)), output, sources)
        expected = b"".join(Path(job["episodes"]).read_bytes() for job in jobs)
        self.assertEqual((output / "episodes.jsonl").read_bytes(), expected)
        self.assertEqual(report["selected_cases"], [c["id"] for c in cases])
        self.assertEqual(report["episode_journal_shards"], {cases[0]["id"]: 0, cases[1]["id"]: 1})
        self.assertEqual(Decimal(report["reported_cost_usd"]), Decimal(".002"))

    def test_merge_rejects_failed_missing_or_changed_child(self):
        cases, path, jobs, sources = self.fixtures()
        for variant in ("failed", "missing", "changed"):
            output = self.root / variant
            output.mkdir()
            modified = copy.deepcopy(jobs)
            if variant == "failed":
                modified[0]["returncode"] = 1
            elif variant == "missing":
                modified.pop()
            else:
                altered = copy.deepcopy(cases[0])
                altered["seed"] += 7
                Path(modified[0]["cases"]).write_text(encode(altered) + "\n")
            with self.assertRaises(ValueError):
                merge_results(cases, path, modified, output, sources)
            self.assertFalse((output / "episodes.manifest.json").exists())

    def scheduler_fixture(self, plans):
        class Clock:
            elapsed = 0.0
            def now(self):
                return self.elapsed
            def sleep(self, duration):
                self.elapsed += duration
        clock = Clock()
        launched, processes = [], []
        jobs = []
        for shard in sorted(plans):
            directory = self.root / f"scheduler_{shard}"
            directory.mkdir()
            cases = directory / "cases.jsonl"
            cases.write_text("{}\n")
            journal = str(directory / "journal")
            jobs.append({"shard": shard, "cases": str(cases), "journal": journal,
                "episodes": "unused", "budget_usd": ".3725", "returncode": None,
                "command": ["python", "pipeline", "--output", "unused", "--journal-dir", journal,
                            "--budget-usd", ".3725"]})
        class Process:
            def __init__(self, pid, start, duration, rc):
                self.pid, self.start, self.duration, self.rc = pid, start, duration, rc
            def poll(self):
                return self.rc if clock.elapsed >= self.start + self.duration else None
        counts = {}
        peak = [0]
        def popen(command, **kwargs):
            output = Path(command[command.index("--output")+1])
            shard = int(output.parent.parent.name.split("_")[-1])
            count = counts.get(shard, 0)
            counts[shard] = count + 1
            duration, rc = plans[shard][min(count, len(plans[shard])-1)]
            process = Process(1000+len(processes), clock.elapsed, duration, rc)
            processes.append(process)
            launched.append((shard, count+1, clock.elapsed, list(command)))
            peak[0] = max(peak[0], sum(p.poll() is None for p in processes))
            return process
        return jobs, clock, popen, launched, peak

    def test_retry_waits_30_seconds_preserves_budget_and_healthy_children(self):
        jobs, clock, popen, launches, peak = self.scheduler_fixture({
            0: [(1, 1), (2, 0)], 1: [(8, 0)], 2: [(3, 0)]})
        with patch("run_jev_parallel_cases.stop_children") as stop:
            passed = execute_jobs(jobs, 2, 4, popen=popen, clock=clock.now, sleep=clock.sleep)
            stop.assert_not_called()
        self.assertTrue(passed)
        self.assertEqual(peak[0], 2)
        self.assertEqual([len(j["attempts"]) for j in jobs], [2, 1, 1])
        self.assertGreaterEqual(jobs[0]["attempts"][1]["started_elapsed_seconds"] -
                                jobs[0]["attempts"][0]["finished_elapsed_seconds"], 30)
        self.assertLess(jobs[2]["attempts"][0]["started_elapsed_seconds"],
                        jobs[1]["attempts"][0]["finished_elapsed_seconds"])
        for job in jobs:
            for attempt in job["attempts"]:
                command = attempt["command"]
                self.assertEqual(command[command.index("--budget-usd")+1], job["budget_usd"])
                self.assertEqual(command[command.index("--journal-dir")+1], job["journal"])
        self.assertNotEqual(jobs[0]["attempts"][0]["episodes"], jobs[0]["attempts"][1]["episodes"])

    def test_exhausted_shard_does_not_kill_other_shards(self):
        jobs, clock, popen, launches, peak = self.scheduler_fixture({
            0: [(1, 1)], 1: [(100, 0)], 2: [(4, 0)]})
        with patch("run_jev_parallel_cases.stop_children") as stop:
            passed = execute_jobs(jobs, 2, 2, popen=popen, clock=clock.now, sleep=clock.sleep)
            stop.assert_not_called()
        self.assertFalse(passed)
        self.assertEqual([j["state"] for j in jobs], ["exhausted", "succeeded", "succeeded"])
        self.assertEqual([len(j["attempts"]) for j in jobs], [2, 1, 1])
        self.assertGreaterEqual(clock.elapsed, 100)


if __name__ == "__main__":
    unittest.main()
