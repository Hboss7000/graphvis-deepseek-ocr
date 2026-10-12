#!/usr/bin/env bash
# Glue only: delegates the planned Stage 1 + QA eval to its unchanged pod job.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/synth_mcq_name_grounded_llava_env.sh"
[[ -n "${TMUX:-}" ]] || { echo 'Use the documented tmux launch command.' >&2; exit 2; }
SMOKE_OUT=${SMOKE_OUT:-$WORKSPACE/outputs/synth_mcq_name_grounded_llava_smoke}
SMOKE_REPORT=${SMOKE_REPORT:-$SMOKE_OUT/smoke_report_regated.json}
[[ -f "$SMOKE_REPORT" ]] || SMOKE_REPORT="$SMOKE_OUT/smoke_report.json"
"$PY" -c 'import json,sys; sys.path.insert(0,"scripts"); from synth_mcq_llava_common import PREFIX_GATE_PROTOCOL; r=json.load(open(sys.argv[1])); g=r.get("prefix_agreement", {}); assert r["passed"] and g.get("protocol") == PREFIX_GATE_PROTOCOL and g.get("pooled", {}).get("passed"), "STOP: failed or obsolete synthetic smoke"' "$SMOKE_REPORT"
"$PY" -u scripts/run_synth_mcq.py "${LLAVA_ARGS[@]}" --preflight
export BACKBONE=llava RUN_TAG=llava_stage1 EVAL_TAG=trained EVAL_SCOPE=all
export ADAPTER_PATH="$ADAPTER"
PLAN=${PLAN:-$WORKSPACE/outputs/llava_stage1/execution_plan.json}
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["eval"]["guard_hours"]*3600))' "$PLAN")
echo "Planned trained eval guard: $MAX_SECONDS seconds; no eval settings changed"
exec bash "$REPO/scripts/llava_stage1_pod_job.sh" eval
