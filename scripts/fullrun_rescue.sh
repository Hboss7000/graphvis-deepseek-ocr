#!/usr/bin/env bash
# Run only on Henrique's manually created single-GPU pod. No infrastructure API.
# Optional later session only. Never called by the strict launcher.
# Launch: tmux new-session -d -s rescue 'cd /workspace/bachelorArbeit && MAX_HOURS=1 bash scripts/fullrun_rescue.sh llava'
# Monitor: tail -F /workspace/logs/fullrun_2026-10-04_B_results/*/rescue_*.log
# GPU: watch -n 30 nvidia-smi
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace}" # The single path root for inputs, outputs, logs and caches.
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=13
export TMPDIR="$WORKSPACE/.cache/fullrun/tmp" XDG_CACHE_HOME="$WORKSPACE/.cache"
mkdir -p "$TMPDIR"
cd "$WORKSPACE/bachelorArbeit"
exec "$WORKSPACE/venvs/venv_qwen/bin/python" scripts/fullrun_rescue.py "${1:?model: llava, qwen, gemma, deepseek}" \
  --root "$WORKSPACE/bachelorArbeit" --workspace "$WORKSPACE" --max-hours "${MAX_HOURS:-1}"
