"""Contract tests for APPO action mapping, selection, and the frozen cohort."""
import math
import random
import unittest
from pathlib import Path

from evaluate_appo_doom import map_action_distribution, sample_with_receipt, select_cases
from unified_game_pipeline import behavior_distribution, choose, read_rows


class AppoContractTests(unittest.TestCase):
    def test_original_test_and_ood_intervals_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        cases = select_cases(read_rows(root / "configs/unified_games_v1_cases.jsonl"),
                             ["train", "dev", "calibration", "test", "ood"])
        self.assertEqual(len(cases), 48)
        self.assertEqual({c["spec"]["frame_skip"] for c in cases if c["split"] == "ood"}, {8})
        self.assertEqual({c["spec"]["frame_skip"] for c in cases if c["split"] == "test"}, {4})
        self.assertTrue(all(c["spec"]["scenario"] == "basic" for c in cases))

    def test_native_action_indices_are_not_alphabetical(self):
        logits = [0., 1., 2., 3.]
        z = sum(math.exp(x) for x in logits)
        mapped = map_action_distribution(logits, [math.exp(x) / z for x in logits])
        self.assertEqual(mapped["logits"], {"noop": 0., "left": 1., "right": 2., "shoot": 3.})
        self.assertEqual(max(mapped["policy_probs"], key=mapped["policy_probs"].get), "shoot")
        with self.assertRaises(ValueError):
            map_action_distribution(logits, [.25] * 4)

    def test_sampling_matches_existing_controller(self):
        scores = {"noop": .1, "left": .2, "right": .3, "shoot": .4}
        for controller, epsilon in [("greedy", 0.), ("greedy", .1), ("sample", 0.)]:
            a, b = random.Random(17), random.Random(17)
            behavior = behavior_distribution(scores, controller, epsilon)
            for _ in range(100):
                actual, _ = sample_with_receipt(behavior, a)
                self.assertEqual(actual, choose(behavior, b))
        self.assertAlmostEqual(behavior_distribution(scores, "greedy", .1)["shoot"], .925)


if __name__ == "__main__":
    unittest.main()
