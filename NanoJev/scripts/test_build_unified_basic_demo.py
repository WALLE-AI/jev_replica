"""Full-cohort and current-recording selection contracts for the Basic viewer."""
import unittest
from build_unified_basic_demo import aggregate_basic, select_cases


class UnifiedBasic(unittest.TestCase):
    def test_aggregate_counts_all128_and_separates_ood(self):
        registry = {f"{split}-{i}": {"split": split, "spec": {"scenario": "basic"}}
                    for split in ("test", "ood") for i in range(128)}
        sources = {k: {cid: {"success": int(cid.split("-")[1]) <
                   (128 if k == "nanojev" else 56 if cid.startswith("test") else 59)} for cid in registry}
                   for k in ("jev", "nanojev", "base")}
        result = aggregate_basic(sources, registry)
        self.assertEqual(result["nanojev"]["test"], {"episodes": 128, "successes": 128, "success_rate": 1.0})
        self.assertEqual(result["jev"]["test"]["successes"], 56)
        self.assertEqual(result["base"]["ood"]["successes"], 59)
        del registry["test-127"]
        with self.assertRaises(ValueError): aggregate_basic(sources, registry)

    def test_selected_illustration_must_match_actual_outcomes(self):
        registry = {"a": {"split": "test", "spec": {"scenario": "basic"}}}
        sources = {k: {"a": {"success": k == "nanojev"}} for k in ("jev", "nanojev", "base")}
        config = {"cases": [{"id": "a"}], "default_case_id": "a", "selection_rule": "Outcome selected"}
        _, selection = select_cases(config, sources, registry)
        self.assertTrue(selection["outcome_selected"])
        sources["jev"]["a"]["success"] = True
        with self.assertRaises(ValueError): select_cases(config, sources, registry)

    def test_wrong_scenario_and_duplicate_selection_fail(self):
        registry = {"a": {"split": "test", "spec": {"scenario": "predict_position"}}}
        sources = {k: {"a": {"success": k == "nanojev"}} for k in ("jev", "nanojev", "base")}
        config = {"cases": [{"id": "a"}], "default_case_id": "a", "selection_rule": "Outcome selected"}
        with self.assertRaises(ValueError): select_cases(config, sources, registry)
        registry["a"]["spec"]["scenario"] = "basic"
        config["cases"].append({"id": "a"})
        with self.assertRaises(ValueError): select_cases(config, sources, registry)


if __name__ == "__main__":
    unittest.main()
