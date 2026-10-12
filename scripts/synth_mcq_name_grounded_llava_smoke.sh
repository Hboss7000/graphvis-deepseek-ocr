#!/usr/bin/env bash
# Launch manually in tmux after authorization; never starts the next session job.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/synth_mcq_name_grounded_llava_env.sh"
[[ -n "${TMUX:-}" ]] || { echo 'Use the documented tmux launch command.' >&2; exit 2; }
: "${MAX_SECONDS:?Set the approved smoke job ceiling; the pod billing guard is separate}"
OUT=${OUT:-$WORKSPACE/outputs/synth_mcq_name_grounded_llava_smoke}
mkdir -p "$OUT"
ln -sfn "$OUT/run.log" "$WORKSPACE/logs/synth_mcq_name_grounded_llava_smoke.log"
exec > >(tee -a "$OUT/run.log") 2>&1
exec 9>>"$WORKSPACE/.fullrun_gpu.lock"
flock -n 9 || { echo 'Another GPU session holds the workspace lock.'; exit 2; }
trap 'rc=$?; echo "Smoke ended ($rc): review smoke_report.json before launching anything else; STOP the pod if finished or failed."' EXIT
"$PY" -u scripts/run_synth_mcq.py "${LLAVA_ARGS[@]}" --preflight
timeout --signal=INT --kill-after=120s "$MAX_SECONDS" "$PY" -u scripts/run_synth_mcq.py \
  "${LLAVA_ARGS[@]}" --out "$OUT" --limit 2 --n-perms 1 --smoke-report
wc -l "$OUT/predictions.jsonl"
