#!/usr/bin/env bash
# Launch inside tmux on the already-approved single-GPU Pod; no infrastructure API.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
export HF_HOME="$WORKSPACE/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=13
export TMPDIR="$WORKSPACE/.cache/fullrun/tmp" XDG_CACHE_HOME="$WORKSPACE/.cache"
mkdir -p "$TMPDIR"
cd "$WORKSPACE/bachelorArbeit"
exec "$WORKSPACE/venvs/venv_qwen/bin/python" scripts/llava_stage1_jobs.py run \
  --root "$WORKSPACE/bachelorArbeit" --workspace "$WORKSPACE" \
  --phase "${1:?phase: pilot-dry, pilot, full-dry, full}" --arm "${2:?arm: P1, P2, P3; all only for pilot}" "${@:3}"
