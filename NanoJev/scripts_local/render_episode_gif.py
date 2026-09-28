#!/usr/bin/env python3
"""Render a recorded Maze/Snake episode from a rollout JSONL file into an animated GIF.

Parses the per-step observation text (no extra simulation), draws a simple grid,
and writes an animated GIF. Only depends on Pillow.
"""
import argparse
import ast
import json
import re
import sys

from PIL import Image, ImageDraw, ImageFont

CELL = 48
PAD = 24


def find_episode(path, episode_id):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("case", {}).get("id") == episode_id:
                return d
    raise SystemExit(f"episode {episode_id!r} not found in {path}")


def parse_snake_state(text):
    size = int(re.search(r"on a (\d+)x\d+ board", text).group(1))
    body = ast.literal_eval(re.search(r"Body head first: (\[\[.*?\]\])\.", text).group(1))
    food = ast.literal_eval(re.search(r"Current food: (\[\d+, ?\d+\])\.", text).group(1))
    return size, body, food


def render_snake(episode, out_path, scale=CELL):
    steps = episode["steps"]
    frames = []
    font = ImageFont.load_default()
    for i, step in enumerate(steps):
        size, body, food = parse_snake_state(step["observation"]["state"])
        img = Image.new("RGB", (size * scale + 2 * PAD, size * scale + 2 * PAD + 40), "white")
        draw = ImageDraw.Draw(img)
        for r in range(size + 1):
            draw.line([(PAD, PAD + r * scale), (PAD + size * scale, PAD + r * scale)], fill="#dddddd")
        for c in range(size + 1):
            draw.line([(PAD + c * scale, PAD), (PAD + c * scale, PAD + size * scale)], fill="#dddddd")

        fr, fc = food
        draw.ellipse(
            [PAD + fc * scale + 6, PAD + fr * scale + 6, PAD + fc * scale + scale - 6, PAD + fr * scale + scale - 6],
            fill="#e74c3c",
        )
        for j, (r, c) in enumerate(body):
            color = "#1b5e20" if j == 0 else "#4caf50"
            draw.rectangle(
                [PAD + c * scale + 3, PAD + r * scale + 3, PAD + c * scale + scale - 3, PAD + r * scale + scale - 3],
                fill=color,
            )
        info = step["observation"].get("state", "")
        action = step.get("action", "")
        outcome = step["info"].get("outcome", "in_progress")
        caption = f"step {i+1}/{len(steps)}  action={action}  outcome={outcome}"
        draw.text((PAD, size * scale + PAD + 8), caption, fill="black", font=font)
        frames.append(img)

    final = episode.get("final_info", {}).get("outcome", "?")
    success = episode.get("success")
    tail = Image.new("RGB", frames[0].size, "white")
    d = ImageDraw.Draw(tail)
    d.text((PAD, PAD), f"FINAL: success={success} outcome={final}", fill="red", font=font)
    frames.append(tail)
    frames.append(tail)

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=600, loop=0
    )
    print(f"wrote {out_path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    episode = find_episode(args.episodes, args.episode_id)
    variant = episode["case"]["variant"]
    if variant.startswith("snake"):
        render_snake(episode, args.output)
    else:
        raise SystemExit(f"variant {variant!r} not supported by this renderer yet")


if __name__ == "__main__":
    main()
