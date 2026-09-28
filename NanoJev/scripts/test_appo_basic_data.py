"""Dependency-free checks of mirrored physics, target export and replay."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from appo_basic_data import (
    MirroredEnvironment, TickMirror, collect_episode, digest, physics_snapshot,
    replay_standard, rendering_snapshot, training_rows, validate_episode,
)
from test_unified_doom_env import FakeGame
from unified_doom_env import UnifiedDoomEnv
from train_pipeline_decisions import validate_training_row


class RenderGame(FakeGame):
    def get_screen_width(self):
        return 160 if self.configured.get("set_screen_resolution") == "160" else 320

    def get_screen_height(self):
        return self.get_screen_width() * 3 // 4

    def is_render_hud(self):
        return self.configured.get("set_render_hud", False)

    def is_render_decals(self):
        return self.configured.get("set_render_decals", False)

    def is_render_particles(self):
        return self.configured.get("set_render_particles", False)

    def get_state(self):
        state = super().get_state()
        if state is not None:
            state.screen_buffer = SimpleNamespace(copy=lambda: "native-pixel-fixture")
        return state


class FakePolicy:
    def __init__(self):
        self.resets = 0

    def reset_episode(self):
        self.resets += 1


def predict_fixture(policy, frame):
    if frame != "native-pixel-fixture":
        raise AssertionError("Expert received an unexpected input")
    return {"sf_logits": [math.log(.05), math.log(.8), math.log(.05), math.log(.1)],
            "sf_probabilities": [.05, .8, .05, .1],
            "pixel_input": {"native_frame_sha256": "fixture-only",
                            "rnn_before_sha256": "fixture-zero", "rnn_after_sha256": "fixture-one"}}


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        for suffix in ("cfg", "wad"):
            (Path(self.directory.name) / ("basic." + suffix)).write_text(
                "render_hud = false\nrender_decals = false\nrender_particles = false\n")
        self.games = []
        self.options = {"kill": True, "kill_after_calls": 5, "reward": -1.}

        def factory():
            game = RenderGame(**self.options)
            self.games.append(game)
            return game

        self.vzd = SimpleNamespace(
            __file__=str(Path(self.directory.name) / "__init__.py"),
            __version__="mock", scenarios_path=self.directory.name, DoomGame=factory,
            Mode=SimpleNamespace(PLAYER="PLAYER"),
            ScreenResolution=SimpleNamespace(RES_320X240="320", RES_160X120="160"),
            ScreenFormat=SimpleNamespace(RGB24="RGB24"),
            Button=SimpleNamespace(MOVE_LEFT="MOVE_LEFT", MOVE_RIGHT="MOVE_RIGHT", ATTACK="ATTACK"),
            GameVariable=SimpleNamespace(**{n: n for n in UnifiedDoomEnv._VARIABLES}))
        self.patch = patch("unified_doom_env._load_vizdoom", return_value=self.vzd)
        self.patch.start()
        self.case = {"id": "mirror-fixture", "split": "train", "seed": 73,
                     "variant": "doom_basic", "spec": {"task": "shooting", "scenario": "basic",
                                                          "max_steps": 75, "frame_skip": 4}}

    def tearDown(self):
        self.patch.stop()
        self.directory.cleanup()

    def episode(self):
        return collect_episode(self.case, FakePolicy(), "fixture-policy", predict=predict_fixture)

    def test_tick_lock_and_standard_only_replay(self):
        episode = self.episode()
        self.assertTrue(episode["success"])
        self.assertEqual(episode["mirrored_physical_ticks"], 5)
        self.assertEqual([len(s["mirror_tick_checks"]) for s in episode["steps"]], [4, 1])
        self.assertEqual(episode["final_info"]["episode_metrics"]["native_reward"], -5.)
        self.assertEqual(episode["final_observation"]["candidates"], {})
        self.assertTrue(replay_standard(episode)["passed"])
        self.assertTrue(all(g.closed for g in self.games))

    def test_standard_viewport_and_history_are_the_training_input(self):
        episode = self.episode()
        render = episode["mirror_initial"]["rendering"]
        self.assertEqual((render["standard"]["width"], render["standard"]["hud"]), (320, False))
        self.assertEqual((render["expert"]["width"], render["expert"]["hud"]), (160, True))
        for row in training_rows(episode):
            self.assertEqual(json.loads(row["state"])["screen_size"], [320, 240])
            self.assertNotIn("INVISIBLE_SENTINEL", row["state"])
            self.assertNotIn("KILLCOUNT", row["state"])

    def test_hard_and_soft_target_contract(self):
        episode = self.episode()
        hard = list(training_rows(episode))
        soft = list(training_rows(episode, soft=True))
        self.assertEqual(len(hard), len(episode["steps"]))
        for h, s in zip(hard, soft):
            self.assertNotIn("gold_probs", h)
            self.assertEqual(h["gold"], {"action": "left"})
            self.assertEqual(h["metadata"]["policy_target_kind"], "expert_action")
            self.assertEqual(s["metadata"]["policy_target_kind"], "expert_distribution")
            self.assertEqual(s["gold_probs_kind"], {"action": "expert_policy_distribution"})
            self.assertEqual(s["gold_probs"]["action"], h["expert"]["policy_probs"])
            self.assertEqual(validate_training_row(h)["action"]["gold_distribution_probs"], [1., 0., 0., 0.])
            self.assertIsNotNone(validate_training_row(s)["action"]["gold_probs"])
            equivalent = copy.deepcopy({k: v for k, v in s.items() if k not in ("gold_probs", "gold_probs_kind")})
            equivalent["metadata"]["policy_target_kind"] = "expert_action"
            self.assertEqual(h, equivalent)

    def test_exploration_action_is_not_the_hard_target(self):
        for i in range(1000):
            identity = f"exploratory-{i}"
            rng = random.Random(int(digest([identity, 17])[:16], 16))
            if rng.random() > .925:
                self.case["id"] = identity
                break
        episode = self.episode()
        self.assertNotEqual(episode["steps"][0]["action"], "left")
        self.assertEqual(next(training_rows(episode))["gold"]["action"], "left")

    def test_failure_and_partial_deadline_are_retained(self):
        self.options = {"timeout": 20, "reward": 100.}
        episode = self.episode()
        self.assertFalse(episode["success"])
        self.assertEqual(episode["mirrored_physical_ticks"], 6)
        self.assertEqual([len(s["mirror_tick_checks"]) for s in episode["steps"]], [4, 2])
        self.assertEqual(len(list(training_rows(episode))), 2)
        self.assertTrue(replay_standard(episode)["passed"])

    def test_wrong_physical_variable_or_clock_fails(self):
        env = MirroredEnvironment(self.case["spec"])
        try:
            env.reset(73)
            env.expert._game.values["POSITION_X"] += .00001
            with self.assertRaisesRegex(RuntimeError, "physics mismatch"):
                env.step("noop")
            env.expert._game.values["POSITION_X"] -= .00001
            env.expert._game.time += 1
            with self.assertRaisesRegex(RuntimeError, "physics mismatch"):
                env.step("noop")
        finally:
            env.close()

    def test_reward_mismatch_and_multitick_proxy_rejected(self):
        left, right = RenderGame(reward=-1.), RenderGame(reward=-2.)
        left.new_episode()
        right.new_episode()
        mirror = TickMirror(left, right, self.vzd)
        with self.assertRaises(ValueError):
            mirror.make_action([False, False, False], 4)
        with self.assertRaisesRegex(RuntimeError, "reward differs"):
            mirror.make_action([False, False, False], 1)

    def test_tampered_action_and_missing_tick_fail_validation(self):
        episode = self.episode()
        for mutate in (lambda e: e["steps"][0].update(action="shoot"),
                       lambda e: e["steps"][0]["mirror_tick_checks"].pop(),
                       lambda e: e.update(complete=False),
                       lambda e: e["steps"][0].update(terminated=True)):
            broken = copy.deepcopy(episode)
            mutate(broken)
            with self.assertRaises(ValueError):
                validate_episode(broken)

    def test_nonfinite_snapshot_rejected(self):
        game = RenderGame()
        game.new_episode()
        game.values["HEALTH"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            physics_snapshot(game, self.vzd)

    def test_vizdoom_130_setter_only_render_api_is_recorded_honestly(self):
        game = FakeGame()
        game.get_screen_width = lambda: 160
        game.get_screen_height = lambda: 120
        settings = {"hud": True, "decals": False, "particles": False}
        for name, value in settings.items():
            getattr(game, "set_render_" + name)(value)
        receipt = rendering_snapshot(game, settings)
        self.assertEqual(receipt["verification"]["width"], "runtime_getter")
        self.assertEqual(receipt["verification"]["hud"], "explicit_setter_receipt_no_runtime_getter")
        self.assertEqual(receipt["explicit_setter_calls"]["set_render_hud"], True)

    def test_predict_position_is_rejected(self):
        self.case["spec"]["scenario"] = "predict_position"
        with self.assertRaises(ValueError):
            self.episode()


if __name__ == "__main__":
    unittest.main()
