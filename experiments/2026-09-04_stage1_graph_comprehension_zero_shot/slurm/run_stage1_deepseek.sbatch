#!/usr/bin/env bash
#SBATCH --job-name=stage1-ds2-zs
#SBATCH --output=/storage/home/hleonel/outputs/slurm-%x-%j.out
#SBATCH --error=/storage/home/hleonel/outputs/slurm-%x-%j.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${HOME}/graphvis"
CONTAINER_IMAGE="${HOME}/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif"
DEEPSEEK_PYTHON="/storage/home/hleonel/venv_ocr2/bin/python"
STAGE1_ROOT="${STAGE1_ROOT:-${HOME}/outputs/graphvis_obqa}"
INPUT_JSONL="${INPUT_JSONL:-${STAGE1_ROOT}/test/stage1_graph_comprehension_0_500.jsonl}"
GRAPH_METADATA="${STAGE1_ROOT}/test/graph_metadata_0_500.jsonl"
IMAGE_ROOT="${STAGE1_ROOT}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/outputs/experiments/2026-09-04_stage1_graph_comprehension_zero_shot/deepseek}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-OCR-2}"
MODEL_REVISION="${MODEL_REVISION:-aaa02f3811945a91062062994c5c4a3f4c0af2b0}"
APPROVE_PROMPTS="${APPROVE_PROMPTS:-0}"
TASK_SET="${TASK_SET:-paper}"
ANSWER_FORMAT="${ANSWER_FORMAT:-none}"
EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
# The extended test input currently has 4496 rows; set EXPECTED_COUNT accordingly.
EXPECTED_COUNT="${EXPECTED_COUNT:-3000}"

HOME_HF_CACHE="${HOME}/.cache/huggingface"
NOBACKUP_HF_CACHE="${HOME}/nobackup/huggingface"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
if [[ -z "${HF_CACHE_DIR:-}" ]]; then
  if [[ -d "${HOME_HF_CACHE}/hub/${MODEL_CACHE_NAME}/snapshots/${MODEL_REVISION}" ]]; then
    HF_CACHE_DIR="$HOME_HF_CACHE"
  elif [[ -d "${NOBACKUP_HF_CACHE}/hub/${MODEL_CACHE_NAME}/snapshots/${MODEL_REVISION}" ]]; then
    HF_CACHE_DIR="$NOBACKUP_HF_CACHE"
  else
    HF_CACHE_DIR="$NOBACKUP_HF_CACHE"
    echo "Exact revision is not cached; model loading may download it into $HF_CACHE_DIR" >&2
  fi
fi

mkdir -p "$RESULTS_DIR" "$HF_CACHE_DIR"
cd "$PROJECT_DIR"
export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

RUNNER_ARGS=(
  "$DEEPSEEK_PYTHON"
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py
  --model-id "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --input-jsonl "$INPUT_JSONL"
  --graph-metadata "$GRAPH_METADATA"
  --image-root "$IMAGE_ROOT"
  --output-dir "$RESULTS_DIR"
  --expected-count "$EXPECTED_COUNT" \
  --expected-split "$EXPECTED_SPLIT" \
  --answer-format "$ANSWER_FORMAT" \
  --task-set "$TASK_SET"
)
APPTAINER_ARGS=(
  exec
  --nv
  --bind /storage/nobackup
  "$CONTAINER_IMAGE"
)

# Safe default: print one exact raw image prompt for every task and exit before
# importing torch, checking CUDA, or loading the model.
if [[ "$APPROVE_PROMPTS" != "1" ]]; then
  apptainer "${APPTAINER_ARGS[@]}" "${RUNNER_ARGS[@]}" --preview-only
  exit 0
fi

apptainer "${APPTAINER_ARGS[@]}" \
  "${RUNNER_ARGS[@]}" \
  --approve-prompts \
  --resume
