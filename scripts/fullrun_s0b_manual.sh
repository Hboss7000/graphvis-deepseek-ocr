#!/usr/bin/env bash
# Runs only on the existing approved pod; no infrastructure API.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
export HF_HOME="$WORKSPACE/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=13
export TMPDIR="$WORKSPACE/.cache/fullrun/tmp" XDG_CACHE_HOME="$WORKSPACE/.cache"
mkdir -p "$TMPDIR"
cd "$WORKSPACE/bachelorArbeit"
exec "$WORKSPACE/venvs/venv_qwen/bin/python" -u scripts/fullrun_s0b.py \
  --root "$WORKSPACE/bachelorArbeit" --workspace "$WORKSPACE" --max-hours "${MAX_HOURS:-1.5}" "$@"
