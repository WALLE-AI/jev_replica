#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== mmBERT-base ==="
hf download jhu-clsp/mmBERT-base --local-dir models/mmBERT-base

log "=== typed-decisions dataset ==="
hf download LocalLLaMA/typed-decisions --repo-type dataset --local-dir data/typed-decisions

log "=== MASSIVE intent dataset ==="
hf download mteb/amazon_massive_intent --repo-type dataset --local-dir data/massive_intent

log "=== XNLI dataset ==="
hf download facebook/xnli --repo-type dataset --local-dir data/xnli

log "=== ALL DOWNLOADS DONE ==="
