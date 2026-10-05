#!/usr/bin/env bash
# Strict-only grouped sessions. No rescue and no infrastructure API calls.
set -euo pipefail
case "${1:?session: S1, S2, S3}" in
  S1) MODELS=(llava); BUDGET_SECONDS=8100 ;;
  S2) MODELS=(qwen deepseek); BUDGET_SECONDS=12096 ;;
  S3) MODELS=(gemma); BUDGET_SECONDS=10584 ;;
  *) echo 'Session must be S1, S2 or S3' >&2; exit 2 ;;
esac
trap 'echo "Strict session ended. STOP the Pod; billing continues."' EXIT
SECONDS=0
for MODEL in "${MODELS[@]}"; do
  REMAINING=$((BUDGET_SECONDS - SECONDS))
  if (( REMAINING <= 0 )); then echo 'Session budget exhausted; rerun to resume.'; exit 1; fi
  MAX_HOURS="$(awk -v seconds="$REMAINING" 'BEGIN {printf "%.9f", seconds/3600}')" \
    bash scripts/fullrun_manual.sh "$MODEL"
done
