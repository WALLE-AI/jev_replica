#!/usr/bin/env python3
"""Render a recorded Maze episode into an animated GIF.

The model only ever sees a local 5x5 window centered on the agent (a POMDP),
so this renderer accumulates each step's revealed window into a persistent
"discovered map" canvas and re-draws it every frame, growing as the agent
explores. No extra simulation is run; everything comes from the recorded
observation text.
"""
import argparse
import json
import re

from PIL import Image, ImageDraw, ImageFont

CELL = 28
PAD = 24

COLORS = {
    "#": "#333333",
    ".": "#f0f0f0",
    "X": "#ffffff",
    "?": "#bbbbbb",
}


def find_episode(path, episode_id):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("case", {}).get("id") == episode_id:
                return d
    raise SystemExit(f"episode {episode_id!r} not found in {path}")


def parse_maze_state(text):
    goal = tuple(int(x) for x in re.search(r"reach goal \[(\d+),\s*(\d+)\]", text).groups())
    agent = tuple(int(x) for x in re.search(r"Agent coordinate: \((\d+),\s*(\d+)\)", text).groups())
    lines_block = re.search(r"Local map:\n((?:.+\n?){5})", text).group(1)
    grid = [row for row in lines_block.splitlines() if row]
    return agent, goal, grid


def render(episode, out_path):
    steps = episode["steps"]
    discovered = {}
    positions = []
    for step in steps:
        agent, goal, grid = parse_maze_state(step["observation"]["state"])
        positions.append(agent)
        for lr, row in enumerate(grid):
            for lc, ch in enumerate(row):
                gr, gc = agent[0] - 2 + lr, agent[1] - 2 + lc
                if ch != "X":
                    discovered[(gr, gc)] = ch

    all_rc = list(discovered.keys()) + positions + [goal]
    min_r = min(r for r, c in all_rc) - 1
    max_r = max(r for r, c in all_rc) + 1
    min_c = min(c for r, c in all_rc) - 1
    max_c = max(c for r, c in all_rc) + 1
    width = (max_c - min_c + 1) * CELL + 2 * PAD
    height = (max_r - min_r + 1) * CELL + 2 * PAD + 30
    font = ImageFont.load_default()

    frames = []
    revealed = {}
    for i, step in enumerate(steps):
        agent, goal, grid = parse_maze_state(step["observation"]["state"])
        for lr, row in enumerate(grid):
            for lc, ch in enumerate(row):
                gr, gc = agent[0] - 2 + lr, agent[1] - 2 + lc
                if ch != "X":
                    revealed[(gr, gc)] = ch

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                ch = revealed.get((r, c), "?")
                x0 = PAD + (c - min_c) * CELL
                y0 = PAD + (r - min_r) * CELL
                draw.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=COLORS.get(ch, "#bbbbbb"))
        gx = PAD + (goal[1] - min_c) * CELL
        gy = PAD + (goal[0] - min_r) * CELL
        draw.rectangle([gx + 3, gy + 3, gx + CELL - 3, gy + CELL - 3], fill="#f1c40f")
        ax = PAD + (agent[1] - min_c) * CELL
        ay = PAD + (agent[0] - min_r) * CELL
        draw.ellipse([ax + 3, ay + 3, ax + CELL - 3, ay + CELL - 3], fill="#2980b9")

        outcome = step["info"].get("outcome", "in_progress")
        caption = f"step {i+1}/{len(steps)}  action={step['action']}  outcome={outcome}"
        draw.text((PAD, height - 22), caption, fill="black", font=font)
        frames.append(img)

    final = episode.get("final_info", {}).get("outcome", "?")
    success = episode.get("success")
    tail = Image.new("RGB", frames[0].size, "white")
    ImageDraw.Draw(tail).text((PAD, PAD), f"FINAL: success={success} outcome={final}", fill="red", font=font)
    frames.extend([tail, tail])

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=250, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    episode = find_episode(args.episodes, args.episode_id)
    render(episode, args.output)


if __name__ == "__main__":
    main()
