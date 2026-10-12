#!/usr/bin/env bash
# Read-only CPU checks: full strict checkpoint, three hashes, versions, cached weights.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/synth_mcq_name_grounded_llava_env.sh"
"$PY" -u scripts/run_synth_mcq.py "${LLAVA_ARGS[@]}" --preflight
