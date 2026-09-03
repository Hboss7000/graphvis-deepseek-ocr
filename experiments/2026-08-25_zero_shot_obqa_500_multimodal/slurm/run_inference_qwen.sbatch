#!/usr/bin/env bash
#SBATCH --job-name=obqa-qwen3vl-zs-500
#SBATCH --output=/storage/home/hleonel/outputs/slurm-%x-%j.out
#SBATCH --error=/storage/home/hleonel/outputs/slurm-%x-%j.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${HOME}/graphvis"
CONTAINER_IMAGE="${HOME}/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif"
QWEN_PYTHON="/storage/home/hleonel/venv_qwen/bin/python"
STAGE2_ROOT="${STAGE2_ROOT:-${HOME}/outputs/graphvis_obqa}"
INPUT_JSONL="${STAGE2_ROOT}/test/stage2_obqa_0_500.jsonl"
GRAPH_METADATA="${STAGE2_ROOT}/test/graph_metadata_0_500.jsonl"
IMAGE_ROOT="$STAGE2_ROOT"
RESULTS_DIR="${HOME}/outputs/2026-08-25_zero_shot_obqa_500_qwen3vl"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/nobackup/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-}"
MIN_PIXELS="${MIN_PIXELS:-262144}"
MAX_PIXELS="${MAX_PIXELS:-1310720}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
APPROVE_PROMPT_DIFF="${APPROVE_PROMPT_DIFF:-0}"

mkdir -p "$RESULTS_DIR/image" "$RESULTS_DIR/text_noref" "$HF_CACHE_DIR"
cd "$PROJECT_DIR"

export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

COMMON_ARGS=(
  "$QWEN_PYTHON"
  experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot_qwen.py
  --model-id "$MODEL_ID"
  --input-jsonl "$INPUT_JSONL"
  --graph-metadata "$GRAPH_METADATA"
  --image-root "$IMAGE_ROOT"
  --expected-count 500
  --min-pixels "$MIN_PIXELS"
  --max-pixels "$MAX_PIXELS"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
if [[ -n "$MODEL_REVISION" ]]; then
  COMMON_ARGS+=(--revision "$MODEL_REVISION")
fi

APPTAINER_ARGS=(
  exec
  --nv
  --bind /storage/nobackup
  "$CONTAINER_IMAGE"
)

if [[ "$APPROVE_PROMPT_DIFF" != "1" ]]; then
  apptainer "${APPTAINER_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    --condition image \
    --output-jsonl "$RESULTS_DIR/image/predictions.jsonl" \
    --preview-only
  exit 0
fi

if [[ -z "$MODEL_REVISION" ]]; then
  echo "Set MODEL_REVISION to the immutable commit printed by download_qwen.sbatch" >&2
  exit 1
fi

apptainer "${APPTAINER_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --condition image \
  --output-jsonl "$RESULTS_DIR/image/predictions.jsonl" \
  --approve-prompt-diff \
  --resume

apptainer "${APPTAINER_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --condition text_noref \
  --output-jsonl "$RESULTS_DIR/text_noref/predictions.jsonl" \
  --approve-prompt-diff \
  --resume
