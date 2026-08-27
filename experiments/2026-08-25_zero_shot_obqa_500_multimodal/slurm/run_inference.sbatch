#!/usr/bin/env bash
#SBATCH --job-name=obqa-ds2-zs-500
#SBATCH --output=/home/hleonel/outputs/slurm-%x-%j.out
#SBATCH --error=/home/hleonel/outputs/slurm-%x-%j.err
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
IMAGE_ROOT="${STAGE2_ROOT}"
RESULTS_DIR="${HOME}/outputs/2026-08-25_zero_shot_obqa_500"
HF_CACHE_DIR="${HOME}/nobackup/huggingface"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-OCR-2}"
MODEL_REVISION="${MODEL_REVISION:-}"
APPROVE_PROMPT_DIFF="${APPROVE_PROMPT_DIFF:-0}"

mkdir -p "$RESULTS_DIR" "$HF_CACHE_DIR"
cd "$PROJECT_DIR"

export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

COMMON_ARGS=(
  python experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot.py
  --model-id "$MODEL_ID"
  --input-jsonl "$INPUT_JSONL"
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
# image/text prompt pair, but generates no predictions. Inspect the Slurm log.
if [[ "$APPROVE_PROMPT_DIFF" != "1" ]]; then
  apptainer "${APPTAINER_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    --condition image \
    --output-jsonl "$RESULTS_DIR/predictions_image.jsonl" \
    --infer-output-dir "$RESULTS_DIR/infer_image" \
    --preview-only
  exit 0
fi

# After approving the prompt diff, both conditions run in this one restart-safe
# job. The same model ID/revision, input JSONL, prompts, parser, and decoding
# settings are used. Only the graph image and its <image> marker differ.
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
