#!/usr/bin/env bash
# Read-only selector: prefer usable python3.12, otherwise Qwen's base interpreter.
set -euo pipefail
WORKSPACE=${1:-/workspace}
if command -v python3.12 >/dev/null 2>&1; then
  CANDIDATE=$(command -v python3.12)
  if "$CANDIDATE" -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
    "$CANDIDATE" -c 'import os,sys; print(os.path.realpath(sys.executable))'
    exit 0
  fi
fi
QWEN_PY="$WORKSPACE/venvs/venv_qwen/bin/python"
if [[ -x "$QWEN_PY" ]]; then
  CANDIDATE=$(readlink -f "$QWEN_PY")
  if [[ -x "$CANDIDATE" ]] && "$CANDIDATE" -c 'import sys' >/dev/null 2>&1; then
    printf '%s\n' "$CANDIDATE"
    exit 0
  fi
fi
echo 'Neither usable python3.12 nor the base interpreter of venv_qwen exists; STOP. No Python installation.' >&2
exit 1
