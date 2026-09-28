"""Reject stale-model relabeling and mismatched rollout identities."""
import copy
import unittest

from build_hard_navigation_demo import NANOJEV_SHA, current_model


class CurrentRolloutIdentity(unittest.TestCase):
    def source(self):
        return {"model": {"engine": "checkpoint", "checkpoint_sha256": NANOJEV_SHA},
                "selected_episode_ids": ["fixed-case"], "source_episodes_sha256": "fixed-file-sha",
                "execution": {"network_model_calls": 0, "model_forward_passes": 72}}

    def test_original_model_cannot_be_relabelled_current(self):
        for engine, sha in (("checkpoint", "5a395c56fad2a9bd4fb664ef78e1e06c19102886d97dd56dfe2676e5896c7e3a"),
                            ("native", NANOJEV_SHA), ("jev", NANOJEV_SHA)):
            source = self.source()
            source["model"] = {"engine": engine, "checkpoint_sha256": sha}
            with self.subTest(engine=engine), self.assertRaises(ValueError):
                current_model(source, "fixed-case", "fixed-file-sha")

    def test_case_selection_file_and_real_local_inference_are_required(self):
        source = self.source()
        current_model(source, "fixed-case", "fixed-file-sha")
        for field in ("case", "file", "forward", "api"):
            changed = copy.deepcopy(source)
            if field == "case":
                changed["selected_episode_ids"].append("searched-case")
            elif field == "file":
                changed["source_episodes_sha256"] = "different-case-file"
            elif field == "forward":
                changed["execution"]["model_forward_passes"] = 0
            else:
                changed["execution"]["network_model_calls"] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                current_model(changed, "fixed-case", "fixed-file-sha")


if __name__ == "__main__":
    unittest.main()
