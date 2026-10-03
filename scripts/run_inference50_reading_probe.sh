#!/usr/bin/env bash
set -euo pipefail

# Run this script inside tmux. All mutable artifacts belong on /workspace.
readonly REPO_ROOT="${REPO_ROOT:-/workspace/bachelorArbeit}"
readonly OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_ROOT}/outputs}"
readonly DATA_ROOT="${DATA_ROOT:-${OUTPUTS_ROOT}/inference50_2026-10-03_corekeep_budget18_e30_tb30}"
readonly DEFAULT_RENDER_ROOT="${DEFAULT_RENDER_ROOT:-${OUTPUTS_ROOT}/inference50_2026-10-03_corekeep_budget18_e30_default_render}"
readonly PROBE_ROOT="${PROBE_ROOT:-${OUTPUTS_ROOT}/phase1_reading_probe}"
readonly RUN_ROOT="${RUN_ROOT:-${OUTPUTS_ROOT}/phase1_reading_probe/runs}"
readonly LOG_DIR="${LOG_DIR:-/workspace/logs}"
readonly LLAVA_PYTHON="${LLAVA_PYTHON:?Set LLAVA_PYTHON to the verified existing venv Python}"
readonly QWEN_PYTHON="${QWEN_PYTHON:-/workspace/venvs/venv_qwen/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

readonly STAGE1_RUNNER="${REPO_ROOT}/experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
readonly INPUT_JSONL="${PROBE_ROOT}/stage1_node_description_first20.jsonl"
readonly METADATA="${PROBE_ROOT}/graph_metadata_first20.jsonl"
readonly LLAVA_REV="2424fdd47412fccc66d91719126b420e9fbd7065"
readonly QWEN_REV="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${HF_HOME}"
cd "${REPO_ROOT}"

common_args=(
  --input-jsonl "${INPUT_JSONL}"
  --graph-metadata "${METADATA}"
  --expected-count 20
  --expected-split test
  --task-set extended
  --answer-format none
  --max-new-tokens 1024
  --behavior-only
  --approve-prompts
  --resume
)

run_llava() {
  local label="$1" image_root="$2" output="$3"
  "${LLAVA_PYTHON}" "${STAGE1_RUNNER}/run_stage1_llava.py" \
    "${common_args[@]}" --revision "${LLAVA_REV}" --image-root "${image_root}" \
    --output-dir "${output}" 2>&1 | tee -a "${LOG_DIR}/${label}.log"
}

run_llava llava_default_render "${DEFAULT_RENDER_ROOT}" "${RUN_ROOT}/llava_default_render"
run_llava llava_tb30 "${DATA_ROOT}" "${RUN_ROOT}/llava_tb30"

"${QWEN_PYTHON}" "${STAGE1_RUNNER}/run_stage1_qwen.py" \
  "${common_args[@]}" --revision "${QWEN_REV}" --image-root "${DATA_ROOT}" \
  --output-dir "${RUN_ROOT}/qwen_tb30" \
  2>&1 | tee -a "${LOG_DIR}/qwen_tb30.log"

"${QWEN_PYTHON}" "${REPO_ROOT}/scripts/report_inference50_reading_probe.py" \
  --graph-metadata "${METADATA}" \
  --arm "llava_default_render=${RUN_ROOT}/llava_default_render/predictions_llava_node_description.jsonl" \
  --arm "llava_tb30=${RUN_ROOT}/llava_tb30/predictions_llava_node_description.jsonl" \
  --arm "qwen_tb30=${RUN_ROOT}/qwen_tb30/predictions_qwen_node_description.jsonl" \
  --expected-count 20 --output "${RUN_ROOT}/reading_probe_report.json"

echo "Reading probe complete: ${RUN_ROOT}/reading_probe_report.json"
