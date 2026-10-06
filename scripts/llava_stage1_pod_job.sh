#!/usr/bin/env bash
# Run inside an already authorized pod; no infrastructure/lifecycle operations.
set -euo pipefail
JOB=${1:?Use probe, smoke-first, smoke-resume, train, eval-dry or eval}
export WORKSPACE=${WORKSPACE:-/workspace}
PROJECT="$WORKSPACE/bachelorArbeit"
PY="$WORKSPACE/venvs/venv_train/bin/python"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORKSPACE/.cache" TORCH_HOME="$WORKSPACE/.cache/torch"
export CUDA_CACHE_PATH="$WORKSPACE/.cache/cuda" TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WORKSPACE/.cache/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLFLOW_TRACKING_URI="file:$WORKSPACE/mlruns"
export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=true
export TMPDIR="$WORKSPACE/.cache/tmp" MPLCONFIGDIR="$WORKSPACE/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
RUN_TAG=${RUN_TAG:-llava_stage1}
LORA_LR=${LORA_LR:-2e-5}
PROJECTOR_LR=${PROJECTOR_LR:-2e-5}
BASE="$WORKSPACE/outputs/$RUN_TAG"
DATA="$PROJECT/outputs/llava_stage1_training_2026-10-06"
EVAL_DATA="$PROJECT/outputs/fullrun_2026-10-04_B"
EVAL_INPUTS="$PROJECT/outputs/llava_stage1_eval_inputs"
LOGS="$WORKSPACE/logs/$RUN_TAG"
mkdir -p "$LOGS" "$TMPDIR" "$MPLCONFIGDIR"
cd "$PROJECT"
test -n "${TMUX:-}" || { echo 'Use the documented tmux launch command.' >&2; exit 1; }
exec 9>"$WORKSPACE/.fullrun_gpu.lock"
flock -n 9 || { echo 'Another GPU session holds the workspace lock.' >&2; exit 1; }
trap 'rc=$?; df -h "$WORKSPACE"; if (( rc )); then echo "FAILED/PAUSED ($rc): STOP the Pod. Inspect logs before retrying."; else echo "JOB COMPLETED/PAUSED: STOP the Pod if finished."; fi' EXIT
df -h "$WORKSPACE"
COMMON=(--data-dir "$DATA" --lora-lr "$LORA_LR" --projector-lr "$PROJECTOR_LR")
if [[ "$JOB" != probe ]]; then
  PROBE="$BASE/probe/batch_probe_report.json"
  BATCH=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch"])' "$PROBE")
fi
MAX_SECONDS=${MAX_SECONDS:-3600} # Initial probe ceiling; later jobs use measured plans.
case "$JOB" in
  probe)
    COMMAND=("$PY" scripts/probe_llava_stage1_batch.py "${COMMON[@]}" --output-dir "$BASE/probe" --log-dir "$LOGS/probe") ;;
  smoke-first|smoke-resume)
    COMMAND=("$PY" scripts/train_llava_stage1.py "${COMMON[@]}" --mode smoke --per-device-batch-size "$BATCH" --output-dir "$BASE/smoke")
    if [[ "$JOB" == smoke-first ]]; then COMMAND+=(--stop-after-step 10)
    else COMMAND+=(--resume "$BASE/smoke/checkpoint-000010"); fi ;;
  train)
    COMMAND=("$PY" scripts/train_llava_stage1.py "${COMMON[@]}" --mode full --per-device-batch-size "$BATCH" --output-dir "$BASE/train" --smoke-report "$BASE/smoke/smoke_report.json")
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then COMMAND+=(--resume "$RESUME_CHECKPOINT"); fi ;;
  eval-dry|eval)
    ADAPTER_PATH=${ADAPTER_PATH:?Set the checkpoint root containing adapter/ and projector.pt}
    EVAL_TAG=${EVAL_TAG:-trained}
    PHASE=dry; OUTPUT="$BASE/eval_${EVAL_TAG}_dry"
    if [[ "$JOB" == eval ]]; then PHASE=full; OUTPUT="$BASE/eval_${EVAL_TAG}"; fi
    COMMAND=("$PY" scripts/evaluate_llava_stage1_adapter.py --phase "$PHASE" --adapter "$ADAPTER_PATH" --data-dir "$EVAL_DATA" --dry-inputs "$EVAL_INPUTS" --output-dir "$OUTPUT" --log-dir "$LOGS/eval")
    if [[ "$PHASE" == full ]]; then COMMAND+=(--dry-report "$BASE/eval_${EVAL_TAG}_dry/evaluation_report.json"); fi ;;
  *) echo "Unknown job: $JOB" >&2; exit 2 ;;
esac
unset FULLRUN_CONTRACT STAGE1_MATRIX_CELL_JSON STAGE1_FROZEN_PROMPT_POLICY_JSON
printf 'LAUNCH:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
# This bounds the job, not pod billing. The approved pod termination guard is separate.
timeout --signal=INT --kill-after=120s "$MAX_SECONDS" "${COMMAND[@]}"
