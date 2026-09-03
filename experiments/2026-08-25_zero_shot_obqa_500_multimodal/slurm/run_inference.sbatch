#!/usr/bin/env bash
#SBATCH --job-name=obqa-ds2-zs-500
#SBATCH --output=/storage/home/hleonel/outputs/slurm-%x-%j.out
#SBATCH --error=/storage/home/hleonel/outputs/slurm-%x-%j.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

# COMA paths. Override STAGE2_ROOT if this generated dataset has another name.
PROJECT_DIR="${HOME}/graphvis"
CONTAINER_IMAGE="${HOME}/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif"
# Raw QA-GNN data is under ~/nobackup/dpr; this evaluator consumes the Stage 2
# JSONL and rendered PNGs produced from it, which are under ~/outputs.
STAGE2_ROOT="${STAGE2_ROOT:-${HOME}/outputs/graphvis_obqa}"
INPUT_JSONL="${STAGE2_ROOT}/test/stage2_obqa_0_500.jsonl"
GRAPH_METADATA="${STAGE2_ROOT}/test/graph_metadata_0_500.jsonl"
IMAGE_ROOT="${STAGE2_ROOT}"
RESULTS_DIR="${HOME}/outputs/2026-08-25_zero_shot_obqa_500"
KG_TEXT_RESULTS_DIR="${HOME}/outputs/experiments/2026-08-25_zero_shot_obqa_500_multimodal/kg_text"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-OCR-2}"
MODEL_REVISION="${MODEL_REVISION:-aaa02f3811945a91062062994c5c4a3f4c0af2b0}"
APPROVE_PROMPT_DIFF="${APPROVE_PROMPT_DIFF:-0}"

# Prefer an existing exact-revision cache instead of silently downloading the
# checkpoint again. An explicitly supplied HF_CACHE_DIR always wins.
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
    echo "Exact model revision was not found in either known cache; downloads will use $HF_CACHE_DIR" >&2
  fi
fi

mkdir -p "$RESULTS_DIR" "$KG_TEXT_RESULTS_DIR" "$HF_CACHE_DIR"
cd "$PROJECT_DIR"

echo "Using Hugging Face cache: $HF_CACHE_DIR"

export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

COMMON_ARGS=(
  /storage/home/hleonel/venv_ocr2/bin/python experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot.py
  --model-id "$MODEL_ID"
  --input-jsonl "$INPUT_JSONL"
  --graph-metadata "$GRAPH_METADATA"
  --image-root "$IMAGE_ROOT"
  --expected-count 500
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

# Safe default: this job loads the exact model and prints the fully rendered
# image/text/text_noref/kg_text prompts, but generates no predictions. Inspect
# the Slurm log.
if [[ "$APPROVE_PROMPT_DIFF" != "1" ]]; then
  apptainer "${APPTAINER_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    --condition image \
    --output-jsonl "$RESULTS_DIR/predictions_image.jsonl" \
    --infer-output-dir "$RESULTS_DIR/infer_image" \
    --preview-only
  exit 0
fi

# After approving the prompt diff, all four conditions run in this one restart-safe
# job. The same model ID/revision, input JSONL, parser, and decoding settings are
# used. text_noref removes only IMAGE_REFERENCE_SENTENCE; kg_text prepends the
# metadata-derived triple block to text_noref.
apptainer "${APPTAINER_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --condition image \
  --output-jsonl "$RESULTS_DIR/predictions_image.jsonl" \
  --infer-output-dir "$RESULTS_DIR/infer_image" \
  --approve-prompt-diff \
  --resume

apptainer "${APPTAINER_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --condition text \
  --output-jsonl "$RESULTS_DIR/predictions_text.jsonl" \
  --infer-output-dir "$RESULTS_DIR/infer_text" \
  --approve-prompt-diff \
  --resume

apptainer "${APPTAINER_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --condition text_noref \
  --output-jsonl "$RESULTS_DIR/predictions_text_noref.jsonl" \
  --infer-output-dir "$RESULTS_DIR/infer_text_noref" \
  --approve-prompt-diff \
  --resume

apptainer "${APPTAINER_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --condition kg_text \
  --output-jsonl "$KG_TEXT_RESULTS_DIR/predictions.jsonl" \
  --infer-output-dir "$KG_TEXT_RESULTS_DIR/infer" \
  --approve-prompt-diff \
  --resume
