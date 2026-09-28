"""Adversarial contracts for physical frames and frozen action controllers."""
import copy
import random
import unittest

from build_unified_navigation_demo import (behavior_distribution, choose, digest,
                                            factory, render_episode)
from test_unified_grid_envs import corridor_env, snake_state


def recording(spec):
    case = {"id": "test-navigation-export", "seed": 17, "split": "test", "spec": spec}
    env = factory(spec)
    observation, info = env.reset(case["seed"])
    rng = random.Random(int(digest([case["id"], 17])[:16], 16))
    steps = []
    while not info["terminated"]:
        offered = observation["candidates"]
        weights = {key: {"east": 4, "north": 3, "south": 2, "west": 1}[key] for key in offered}
        scores = {key: value / sum(weights.values()) for key, value in weights.items()}
        forced = len(offered) == 1
        behavior = behavior_distribution(scores, "greedy", .1)
        action = choose(behavior, rng)
        following, reward, terminated, truncated, info = env.step(action)
        steps.append({"observation": observation, "action": action, "scores": scores,
                      "answers": {} if forced else {"action": {"probabilities": scores}},
                      "forced": forced, "behavior_probs": behavior, "reward": reward,
                      "terminated": terminated, "truncated": truncated, "info": info})
        observation = following
    env.close()
    return {"case": case, "steps": steps, "complete": True, "success": info["success"], "final_info": info}


class NavigationExport(unittest.TestCase):
    def maze(self):
        env = corridor_env(20)
        return recording(env.spec)

    def test_macro_moves_keep_every_physical_frame_without_stale_probabilities(self):
        source = self.maze()
        _, system, audit = render_episode(source, "nanojev")
        frames = system["frames"]
        macros = [frame for frame in frames if frame["decision_source"] == "verified_edge_reposition"]
        self.assertTrue(macros)
        self.assertTrue(all(frame["forced"] and frame["probabilities"] == {} and
                            frame["controller_probabilities"] == {} for frame in macros))
        self.assertEqual([frame["physical_step"] for frame in frames], list(range(len(frames))))
        self.assertEqual(len(frames) - 1, audit["verified_physical_steps"])
        self.assertGreater(audit["verified_physical_steps"], audit["verified_decisions"])
        self.assertTrue(frames[-1]["done"])
        self.assertFalse(any(frame["done"] for frame in frames[:-1]))

    def test_snake_target_success_does_not_claim_the_board_was_cleared(self):
        source = recording({"task": "snake", "size": 4, "max_steps": 10, "target_food": 1,
                            "initial_state": snake_state(food=(1, 2))})
        _, system, audit = render_episode(source, "nanojev")
        self.assertTrue(system["summary"]["success"])
        self.assertEqual(system["summary"]["outcome"], "target_reached")
        self.assertEqual(system["summary"]["source_outcome"], "target_food_reached")
        self.assertEqual(system["summary"]["food_collected"], 1)
        self.assertLess(len(system["frames"][-1]["body"]), 4 * 4)
        self.assertTrue(audit["passed"])

    def test_changed_observation_environment_feedback_or_final_outcome_is_rejected(self):
        original = self.maze()
        for field in ("observation", "transition", "success"):
            changed = copy.deepcopy(original)
            if field == "observation":
                changed["steps"][0]["observation"]["state"] += " altered"
            elif field == "transition":
                changed["steps"][0]["info"]["physical_events"][0]["next_position"] = [0, 0]
            else:
                changed["success"] = not changed["success"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_episode(changed, "nanojev")

    def test_changed_controller_probabilities_or_action_is_rejected(self):
        original = self.maze()
        for field in ("scores", "behavior", "action", "forced"):
            changed = copy.deepcopy(original)
            step = changed["steps"][0]
            if field == "scores":
                step["scores"]["north"] = float("nan")
            elif field == "behavior":
                step["behavior_probs"]["east"] = .5
            elif field == "action":
                step["action"] = next(key for key in step["scores"] if key != step["action"])
            else:
                step["forced"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_episode(changed, "nanojev")


if __name__ == "__main__":
    unittest.main()
