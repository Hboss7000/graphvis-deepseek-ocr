#!/usr/bin/env bash
#SBATCH --job-name=stage1-qwen-zs
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
# If a manual Hub download is needed, the executable in this venv is `hf`, not
# the deprecated `huggingface-cli` shim.
STAGE1_ROOT="${STAGE1_ROOT:-${HOME}/outputs/graphvis_obqa}"
INPUT_JSONL="${INPUT_JSONL:-${STAGE1_ROOT}/test/stage1_graph_comprehension_0_500.jsonl}"
GRAPH_METADATA="${STAGE1_ROOT}/test/graph_metadata_0_500.jsonl"
IMAGE_ROOT="${STAGE1_ROOT}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/outputs/experiments/2026-09-04_stage1_graph_comprehension_zero_shot/qwen}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/nobackup/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
MIN_PIXELS="${MIN_PIXELS:-262144}"
MAX_PIXELS="${MAX_PIXELS:-1310720}"
APPROVE_PROMPTS="${APPROVE_PROMPTS:-0}"
TASK_SET="${TASK_SET:-paper}"
ANSWER_FORMAT="${ANSWER_FORMAT:-none}"
if [[ -z "${MAX_NEW_TOKENS:-}" ]]; then
  [[ "${ANSWER_FORMAT}" == "constrained" ]] && MAX_NEW_TOKENS=2048 || MAX_NEW_TOKENS=1024
fi
EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
# The extended test input currently has 4496 rows; set EXPECTED_COUNT accordingly.
EXPECTED_COUNT="${EXPECTED_COUNT:-3000}"

mkdir -p "$RESULTS_DIR" "$HF_CACHE_DIR"
cd "$PROJECT_DIR"
export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

RUNNER_ARGS=(
  "$QWEN_PYTHON"
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_qwen.py
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
  --min-pixels "$MIN_PIXELS"
  --max-pixels "$MAX_PIXELS"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
APPTAINER_ARGS=(
  exec
  --nv
  --bind /storage/nobackup
  "$CONTAINER_IMAGE"
)

# Safe default: load only the processor, print the six exact chat-template
# prompts and first-image budget, then exit before loading the model.
if [[ "$APPROVE_PROMPTS" != "1" ]]; then
  apptainer "${APPTAINER_ARGS[@]}" "${RUNNER_ARGS[@]}" --preview-only
  exit 0
fi

apptainer "${APPTAINER_ARGS[@]}" \
  "${RUNNER_ARGS[@]}" \
  --approve-prompts \
  --resume
