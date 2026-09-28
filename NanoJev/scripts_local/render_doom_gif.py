#!/usr/bin/env python3
"""Replay a recorded ViZDoom episode's actions through a fresh env instance,
capturing the native RGB screen buffer each tick, and write an animated GIF.

Determinism: same scenario + same seed + same action sequence as the recorded
rollout reproduces the same visual trajectory (UnifiedDoomEnv.reset(seed) calls
game.set_seed(seed) then new_episode()).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "checkpoints/NanoJev-unified/source/scripts"))

from PIL import Image
from unified_doom_env import UnifiedDoomEnv


def find_episode(path, episode_id):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("case", {}).get("id") == episode_id:
                return d
    raise SystemExit(f"episode {episode_id!r} not found in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    episode = find_episode(args.episodes, args.episode_id)
    case = episode["case"]
    actions = [s["action"] for s in episode["steps"]]

    env = UnifiedDoomEnv(case["spec"])
    env.reset(seed=case["seed"])
    game = env._game  # noqa: SLF001 -- intentional: this repo's env has no public frame hook

    frames = []

    def capture(caption):
        state = game.get_state()
        if state is None or state.screen_buffer is None:
            return
        arr = state.screen_buffer  # (H, W, 3) RGB24
        img = Image.fromarray(arr, mode="RGB")
        frames.append((img, caption))

    capture(f"reset seed={case['seed']}")
    for i, action in enumerate(actions):
        env.step(action)
        capture(f"step {i+1}/{len(actions)} action={action}")

    final_outcome = episode.get("final_info", {}).get("outcome", "?")
    success = episode.get("success")
    env.close()

    if not frames:
        raise SystemExit("no frames captured")

    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    rendered = []
    for img, caption in frames:
        canvas = Image.new("RGB", (img.width, img.height + 24), "black")
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, img.height + 4), caption, fill="white", font=font)
        rendered.append(canvas)

    tail = Image.new("RGB", rendered[0].size, "black")
    ImageDraw.Draw(tail).text((4, 4), f"FINAL success={success} outcome={final_outcome}", fill="red", font=font)
    rendered.extend([tail, tail])

    duration_ms = int(1000 / args.fps)
    rendered[0].save(args.output, save_all=True, append_images=rendered[1:], duration=duration_ms, loop=0)
    print(f"wrote {args.output} ({len(rendered)} frames)")


if __name__ == "__main__":
    main()
