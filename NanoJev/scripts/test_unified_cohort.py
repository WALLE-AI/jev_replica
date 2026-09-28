"""Offline cohort filtering regressions; all fixtures are synthetic."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import summarize_unified_cohort as cohort
from test_unified_summary import episode, write_run


class ExplicitCohortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reference_rows = [episode("test-a", False, seed=1), episode("ood-b", True, split="ood", seed=2)]
        self.reference = write_run(self.root, "reference", self.reference_rows)

    def test_complete_sources_filtered_by_id_with_original_hashes(self):
        rows = [episode("excluded-train", True, split="train", seed=99), *copy.deepcopy(self.reference_rows)]
        rows[1]["success"] = rows[1]["final_info"]["success"] = True
        source = write_run(self.root, "full", rows)
        before = Path(source).read_bytes()
        report = cohort.build_summary({"Greedy": self.reference, "Full": source}, self.reference)
        self.assertEqual(report["cohort"]["case_count"], 2)
        self.assertEqual(report["cohort"]["selected_case_ids"], ["ood-b", "test-a"])
        self.assertNotIn("train/maze", report["runs"]["Full"]["by_split_task"])
        self.assertEqual(report["runs"]["Full"]["by_split_task"]["test/maze"]["successes"], 1)
        self.assertEqual(report["source_selection"]["Full"]["source_case_count"], 3)
        self.assertEqual(report["source_selection"]["Full"]["included_case_count"], 2)
        self.assertEqual(report["source_selection"]["Full"]["excluded_case_count"], 1)
        self.assertEqual(report["inputs"]["Full"]["episodes_sha256"], cohort.summary_tools.sha256_file(source))
        self.assertEqual(Path(source).read_bytes(), before)
        self.assertIn("complete original files", cohort.render_markdown(report))

    def test_missing_case_rejected_without_intersection_fallback(self):
        source = write_run(self.root, "incomplete-cohort", self.reference_rows[:1])
        with self.assertRaisesRegex(ValueError, "missing case IDs"):
            cohort.build_summary({"Partial": source}, self.reference)

    def test_same_id_different_seed_or_spec_or_split_is_rejected(self):
        for field in ("seed", "spec", "split"):
            rows = copy.deepcopy(self.reference_rows)
            rows[0]["case"][field] = {"seed": 777, "spec": {"task": "maze", "size": 9}, "split": "dev"}[field]
            source = write_run(self.root, "changed-" + field, rows)
            with self.assertRaisesRegex(ValueError, "different case definitions"):
                cohort.build_summary({"Changed": source}, self.reference)

    def test_full_source_is_verified_before_filtering(self):
        source = write_run(self.root, "corrupted", self.reference_rows)
        Path(source).write_text(Path(source).read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "episode_sha256"):
            cohort.build_summary({"Corrupt": source}, self.reference)

    def test_disjoint_sources_and_excluded_policy_are_auditable(self):
        first = write_run(self.root, "first", self.reference_rows[:1])
        second_rows = copy.deepcopy(self.reference_rows[1:])
        second_rows[0]["continuation_policy_id"] = "second-policy"
        second = write_run(self.root, "second", second_rows, policy={"epsilon": .15})
        excluded = write_run(self.root, "unused", [episode("train-unused", False, split="train", seed=99, policy="unused-policy")],
                             policy={"controller": "sample"})
        report = cohort.build_summary({"Combined": [first, second, excluded]}, self.reference)
        self.assertEqual(report["policies"]["Combined"]["continuation_policy_ids"], ["polA", "second-policy"])
        self.assertEqual(len(report["inputs"]["Combined"]["sources"]), 3)
        self.assertEqual(len(report["policies"]["Combined"]["sources"]), 2)
        self.assertEqual(report["source_selection"]["Combined"]["sources"][-1]["included_case_count"], 0)

    def test_cli_writes_new_report_and_rejects_overwrite(self):
        out = self.root / "report"
        args = ["--cohort-from", self.reference, "--run", "Reference=" + self.reference, "--output-dir", str(out)]
        self.assertEqual(cohort.main(args), 0)
        value = json.loads((out / "summary.json").read_text())
        self.assertEqual(value["cohort"]["case_count"], 2)
        self.assertEqual(cohort.main(args), 2)


if __name__ == "__main__":
    unittest.main()
