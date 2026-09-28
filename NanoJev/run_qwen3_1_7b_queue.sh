#!/usr/bin/env bash
# Task queue: wait for the running 4-arm Qwen3-1.7B SFT comparison to finish,
# pick the winning arm, run the 548-case closed-loop rollout evaluation on it,
# and write a final 0.6B-vs-1.7B comparison report.
#
# This script is meant to be launched once and left running in the background;
# it polls for the upstream training script's completion marker rather than
# assuming a particular process tree.

set -euo pipefail

RUNS_DIR="${RUNS_DIR:-/tmp/nanojev_runs/qwen3_1_7b}"
MAIN_LOG="${MAIN_LOG:-/home/dataset1/gaojing/jev_replica/NanoJev/full_sft_reproduction_qwen3_1_7b.log}"
DATA_DIR="${DATA_DIR:-data/NanoJev-unified}"
MODEL_DIR="${MODEL_DIR:-checkpoints/NanoJev-unified}"
GPU="${GPU:-2}"
PY=".venv/bin/python"
ARMS=(hard_lr2e5 hard_lr1e5 soft_lr2e5 soft_lr1e5)

echo "[queue] waiting for the 4-arm training script to finish (polling $MAIN_LOG)..."
while ! grep -q "^Done\. Per-arm checkpoints" "$MAIN_LOG" 2>/dev/null; do
  sleep 60
done
echo "[queue] training script finished."

echo "[queue] selecting the winning arm by dev CE..."
WINNER=$("$PY" - "$RUNS_DIR" "${ARMS[@]}" <<'PY'
import json, sys, pathlib
runs_dir = pathlib.Path(sys.argv[1])
arms = sys.argv[2:]
best_name, best_ce = None, None
init_summary = runs_dir / "init_eval" / "summary.json"
if init_summary.exists():
    d = json.loads(init_summary.read_text())
    ce = d.get("metrics_by_split", d).get("dev", d).get("selection_ce")
    if ce is not None:
        best_name, best_ce = "init", ce
for name in arms:
    p = runs_dir / name / "summary.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    ce = d.get("best_dev_selection_ce")
    if ce is not None and (best_ce is None or ce < best_ce):
        best_name, best_ce = name, ce
if best_name is None or best_name == "init":
    raise SystemExit("no trained arm beat initialization; refusing to pick a winner")
print(best_name)
PY
)
echo "[queue] winning arm: $WINNER"

EVAL_DIR="$RUNS_DIR/eval"
mkdir -p "$EVAL_DIR"
EPISODES="$EVAL_DIR/${WINNER}_test_ood_episodes.jsonl"

if [[ ! -s "$EPISODES" ]]; then
  echo "[queue] running 548-case closed-loop rollout for $WINNER on GPU $GPU..."
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$MODEL_DIR/source/scripts/unified_game_pipeline.py" rollout \
    --cases "$DATA_DIR/evaluation/test_cases.jsonl" \
    --engine checkpoint --checkpoint "$RUNS_DIR/$WINNER" \
    --controller q_greedy --epsilon 0.1 --seed 17 \
    --env-batch 16 --batch-questions 16 --max-length 8192 \
    --output "$EPISODES" \
    > "$EVAL_DIR/rollout.log" 2>&1
  echo "[queue] rollout done."
else
  echo "[queue] rollout episodes already exist, skipping."
fi

echo "[queue] writing final 0.6B vs 1.7B comparison report..."
"$PY" - "$WINNER" "$RUNS_DIR" "$EPISODES" <<'PY' > /home/dataset1/gaojing/jev_replica/NanoJev/qwen3_1_7b_vs_0_6b_comparison.md
import json, sys, pathlib

winner, runs_dir, episodes_path = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])

def offline_ce(summary_path):
    d = json.loads(summary_path.read_text())
    out = {}
    for split, m in d.get("metrics_by_split", {}).items():
        out[split] = m.get("selection_ce")
    return out

def agg_rollout(path):
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    table = {}
    for split in ("test", "ood"):
        for task, keys in [
            ("Maze", ["maze8", "maze16", "maze50"]),
            ("Snake", ["snake8", "snake12"]),
            ("Basic", ["doom_basic"]),
            ("Predict Position", ["doom_predict_position"]),
        ]:
            eps = succ = 0
            for r in rows:
                c = r["case"]
                if c["split"] == split and c["variant"] in keys:
                    eps += 1
                    succ += int(bool(r["success"]))
            table[(split, task)] = (succ, eps)
    return table

ce_1_7b = offline_ce(runs_dir / winner / "summary.json")
ce_0_6b_path = pathlib.Path("/tmp/nanojev_runs/hard_lr1e5/summary.json")
ce_0_6b = offline_ce(ce_0_6b_path) if ce_0_6b_path.exists() else {}

rollout_1_7b = agg_rollout(episodes_path)
rollout_0_6b_path = pathlib.Path("/tmp/nanojev_runs/eval/hard_lr1e5_test_ood_episodes.jsonl")
rollout_0_6b = agg_rollout(rollout_0_6b_path)

print(f"# Qwen3-0.6B vs Qwen3-1.7B — NanoJev unified SFT comparison\n")
print(f"Winning 1.7B arm: **{winner}**\n")

print("## Offline dev/calibration/test/ood cross-entropy\n")
print("| Split | 0.6B (hard_lr1e5) | 1.7B ({}) |".format(winner))
print("|---|---:|---:|")
for split in ("dev", "calibration", "test", "ood"):
    a = ce_0_6b.get(split)
    b = ce_1_7b.get(split)
    a_s = f"{a:.4f}" if a is not None else "n/a"
    b_s = f"{b:.4f}" if b is not None else "n/a"
    print(f"| {split} | {a_s} | {b_s} |")

if rollout_1_7b and rollout_0_6b:
    print("\n## Closed-loop 548-case rollout (test split)\n")
    print("| Task | 0.6B test | 1.7B test |")
    print("|---|---:|---:|")
    for task in ("Maze", "Snake", "Basic", "Predict Position"):
        s0, e0 = rollout_0_6b[("test", task)]
        s1, e1 = rollout_1_7b[("test", task)]
        print(f"| {task} | {s0}/{e0} | {s1}/{e1} |")
    print("\n## Closed-loop 548-case rollout (ood split)\n")
    print("| Task | 0.6B ood | 1.7B ood |")
    print("|---|---:|---:|")
    for task in ("Maze", "Snake", "Basic", "Predict Position"):
        s0, e0 = rollout_0_6b[("ood", task)]
        s1, e1 = rollout_1_7b[("ood", task)]
        print(f"| {task} | {s0}/{e0} | {s1}/{e1} |")
else:
    print("\n(0.6B rollout comparison unavailable — missing episodes file.)")
PY

echo "[queue] wrote /home/dataset1/gaojing/jev_replica/NanoJev/qwen3_1_7b_vs_0_6b_comparison.md"
echo "[queue] all done."
