#!/usr/bin/env bash
# Reproduce the four-arm unified SFT comparison (hard/soft targets x backbone lr 1e-5/2e-5)
# and select the best arm by lowest weighted development CE, per docs/SONIC_PREDICT_POSITION.md
# and TRAINING_RECIPE.md. Uses the pre-published dataset/checkpoint snapshots, so it needs
# neither the paid Jev API nor the private Sonic Doom expert weights.
#
# Usage:
#   ./run_full_sft_reproduction.sh
#
# Env overrides:
#   DATA_DIR, MODEL_DIR   local snapshot paths (defaults below)
#   RUNS_DIR              where per-arm output dirs are written (default: /tmp/nanojev_runs)
#   GPU_A, GPU_B          two physical GPU indices to run arms on in parallel (default: 1 6)
#   INIT_CHECKPOINT       DecisionModel bundle to warm-start from (default: $MODEL_DIR/training_initialization)

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/NanoJev-unified}"
MODEL_DIR="${MODEL_DIR:-checkpoints/NanoJev-unified}"
RUNS_DIR="${RUNS_DIR:-/tmp/nanojev_runs}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-6}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$MODEL_DIR/training_initialization}"
MICROBATCH_QUESTIONS="${MICROBATCH_QUESTIONS:-8}"
MAX_MICROBATCH_TOKENS="${MAX_MICROBATCH_TOKENS:-32768}"
PY=".venv/bin/python"
TRAIN="$MODEL_DIR/source/scripts/train_unified_games.py"
EVAL="$MODEL_DIR/source/scripts/evaluate_unified_checkpoint.py"
POOL_WEIGHTS="$DATA_DIR/configs/sonic_policy_pool_weights.json"

mkdir -p "$RUNS_DIR"

COMMON_ARGS=(
  --init-checkpoint "$INIT_CHECKPOINT"
  --stage sft --loss ce --balance task
  --policy-pool-weights "$POOL_WEIGHTS"
  --steps 600 --head-steps 0 --seed 17
  --batch-questions 24 --microbatch-questions "$MICROBATCH_QUESTIONS" --max-microbatch-tokens "$MAX_MICROBATCH_TOKENS"
  --max-length 8192 --eval-every 100
  --weight-decay 0.01 --precision bf16 --gradient-checkpointing --disable-native-triton
)

# name  target(hard/soft)  backbone-lr  head-lr
ARMS=(
  "hard_lr2e5 hard 2e-5 2e-4"
  "hard_lr1e5 hard 1e-5 1e-4"
  "soft_lr2e5 soft 2e-5 2e-4"
  "soft_lr1e5 soft 1e-5 1e-4"
)

echo "== Data-only validation (hard + soft) =="
"$PY" "$TRAIN" --input "$DATA_DIR/unified/hard" --stage sft --loss ce \
  --policy-pool-weights "$POOL_WEIGHTS" --validate-only >/dev/null
"$PY" "$TRAIN" --input "$DATA_DIR/unified/soft" --stage sft --loss ce \
  --policy-pool-weights "$POOL_WEIGHTS" --validate-only >/dev/null
echo "validation ok"

run_arm() {
  local name="$1" target="$2" backbone_lr="$3" head_lr="$4" gpu="$5"
  local out="$RUNS_DIR/$name"
  if [[ -f "$out/summary.json" ]]; then
    echo "[$name] already complete, skipping"
    return 0
  fi
  echo "[$name] starting on GPU $gpu (target=$target backbone_lr=$backbone_lr head_lr=$head_lr)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$TRAIN" \
    --input "$DATA_DIR/unified/$target" \
    --output-dir "$out" \
    --backbone-lr "$backbone_lr" --head-lr "$head_lr" \
    "${COMMON_ARGS[@]}" \
    > "$RUNS_DIR/${name}.log" 2>&1
  echo "[$name] done"
}

echo "== Training four arms, two at a time on GPU $GPU_A / GPU $GPU_B =="
run_arm ${ARMS[0]} "$GPU_A" &
pid_a=$!
run_arm ${ARMS[1]} "$GPU_B" &
pid_b=$!
wait "$pid_a" "$pid_b"

run_arm ${ARMS[2]} "$GPU_A" &
pid_a=$!
run_arm ${ARMS[3]} "$GPU_B" &
pid_b=$!
wait "$pid_a" "$pid_b"

echo "== Evaluating the untrained initialization checkpoint as a selection candidate =="
INIT_EVAL_DIR="$RUNS_DIR/init_eval"
if [[ ! -f "$INIT_EVAL_DIR/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_A" "$PY" "$EVAL" \
    --input "$DATA_DIR/unified/hard" \
    --checkpoint "$INIT_CHECKPOINT" \
    --output-dir "$INIT_EVAL_DIR" --stage sft \
    --microbatch-questions "$MICROBATCH_QUESTIONS" --max-microbatch-tokens "$MAX_MICROBATCH_TOKENS" --max-length 8192 \
    --precision bf16 --disable-native-triton \
    > "$RUNS_DIR/init_eval.log" 2>&1
fi

echo "== Selecting the best arm by weighted dev CE =="
"$PY" - "$RUNS_DIR" "${ARMS[@]}" <<'PY'
import json, sys, pathlib

runs_dir = pathlib.Path(sys.argv[1])
arms = [a.split() for a in sys.argv[2:]]

candidates = []
init_summary = runs_dir / "init_eval" / "summary.json"
if init_summary.exists():
    d = json.loads(init_summary.read_text())
    ce = d.get("metrics_by_split", d).get("dev", d).get("selection_ce")
    candidates.append(("initialization (untrained)", None, ce))

for name, target, backbone_lr, head_lr in arms:
    p = runs_dir / name / "summary.json"
    if not p.exists():
        print(f"[warn] missing {p}")
        continue
    d = json.loads(p.read_text())
    candidates.append((name, d.get("best_step"), d.get("best_dev_selection_ce")))

candidates = [c for c in candidates if c[2] is not None]
candidates.sort(key=lambda c: c[2])

print(f"{'candidate':28s} {'step':>6s} {'dev_ce':>10s}")
for name, step, ce in candidates:
    print(f"{name:28s} {str(step):>6s} {ce:>10.4f}")

if candidates:
    winner = candidates[0]
    print(f"\nSelected: {winner[0]} (step {winner[1]}, dev CE {winner[2]:.4f})")
PY

echo "Done. Per-arm checkpoints and logs are under $RUNS_DIR."
