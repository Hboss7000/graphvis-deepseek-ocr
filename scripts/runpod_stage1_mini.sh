#!/usr/bin/env bash
# Run the three real paper-task Stage 1 entry points on the same five graphs.
# Expected Runpod mount: /workspace. This script never creates Runpod resources.
set -Eeuo pipefail

readonly WORKSPACE="/workspace"
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SYSTEM_PYTHON="/opt/conda/bin/python"
readonly OCR2_PYTHON="${WORKSPACE}/venvs/venv_ocr2/bin/python"
readonly QWEN_PYTHON="${WORKSPACE}/venvs/venv_qwen/bin/python"
readonly RENDER_ENV="${WORKSPACE}/venvs/venv_gen"
readonly RENDER_PYTHON="${RENDER_ENV}/bin/python"
readonly HF_HOME="${WORKSPACE}/.cache/huggingface"
readonly RAW_DATA_ROOT="${WORKSPACE}/data_preprocessed_release"
readonly MINI_ROOT="${WORKSPACE}/outputs/mini"
readonly DATASET_ROOT="${MINI_ROOT}/data"
readonly TEST_ROOT="${DATASET_ROOT}/test"
readonly INPUT_JSONL="${TEST_ROOT}/stage1_graph_comprehension_0_5.jsonl"
readonly GRAPH_METADATA="${TEST_ROOT}/graph_metadata_0_5.jsonl"
readonly LOG_DIR="${WORKSPACE}/logs"
readonly DATA_LOG="${LOG_DIR}/stage1_mini_data.log"
readonly RENDER_HASHES="${LOG_DIR}/mini_render_hashes.txt"
readonly EXPECTED_COUNT=30

readonly DEEPSEEK_ID="deepseek-ai/DeepSeek-OCR-2"
readonly DEEPSEEK_REV="aaa02f3811945a91062062994c5c4a3f4c0af2b0"
readonly QWEN_ID="Qwen/Qwen3-VL-8B-Instruct"
readonly QWEN_REV="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
readonly GEMMA_ID="google/gemma-3-12b-it"
readonly GEMMA_REV="96b6f1eccf38110c56df3a15bffe176da04bfd80"
readonly GEMMA_PREFLIGHT_REPORT="${GEMMA_PREFLIGHT_REPORT:-${MINI_ROOT}/gemma_preflight/preflight_report.json}"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=13

mkdir -p "${MINI_ROOT}" "${LOG_DIR}"
cd "${REPO_ROOT}"

# Keep the pod bootstrap self-contained even when the template predates these
# required download/render utilities.
missing_host_tools=()
for host_tool in git wget curl unzip rsync tmux dot; do
  command -v "${host_tool}" >/dev/null 2>&1 || missing_host_tools+=("${host_tool}")
done
if ((${#missing_host_tools[@]})); then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: missing host tools and cannot apt-install as non-root: ${missing_host_tools[*]}" >&2
    exit 1
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates curl git graphviz rsync tmux unzip wget
fi

for executable in "${OCR2_PYTHON}" "${QWEN_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "ERROR: missing Runpod venv interpreter: ${executable}" >&2
    exit 1
  fi
done
if [[ ! -x "${SYSTEM_PYTHON}" ]]; then
  echo "ERROR: expected image Python at ${SYSTEM_PYTHON}." >&2
  exit 1
fi
if [[ "$("${SYSTEM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.11" ]]; then
  echo "ERROR: requirements_render_py311.txt requires /opt/conda Python 3.11." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi is unavailable; cannot record the mini-run GPU." >&2
  exit 1
}
command -v dot >/dev/null 2>&1 || {
  echo "ERROR: Graphviz 'dot' is not installed." >&2
  exit 1
}

{
  echo "mini_run_scope=pipeline_check_only"
  echo "comparison_eligible=false"
  echo "gpu_inventory:"
  nvidia-smi -L
  nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader
  "${QWEN_PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable while recording mini-run GPU metadata")
props = torch.cuda.get_device_properties(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"torch_gpu_name={props.name}")
print(f"torch_gpu_total_memory_bytes={props.total_memory}")
print(f"torch_gpu_architecture=sm_{major}{minor}")
PY
} 2>&1 | tee -a "${DATA_LOG}"

# Keep rendering isolated from the recorded inference environments. This also
# creates/repairs venv_gen when the setup script ran before it was introduced.
if [[ ! -x "${RENDER_PYTHON}" ]]; then
  "${SYSTEM_PYTHON}" -m venv "${RENDER_ENV}"
fi
if ! grep -q '^include-system-site-packages = false$' "${RENDER_ENV}/pyvenv.cfg"; then
  echo "ERROR: existing ${RENDER_ENV} is not an isolated render venv." >&2
  exit 1
fi
"${RENDER_PYTHON}" -m pip install --upgrade pip 2>&1 | tee -a "${DATA_LOG}"
"${RENDER_PYTHON}" -m pip install --upgrade \
  --requirement "${REPO_ROOT}/requirements_render_py311.txt" \
  2>&1 | tee -a "${DATA_LOG}"
"${RENDER_PYTHON}" - <<'PY' 2>&1 | tee -a "${DATA_LOG}"
import graphviz
import numpy
import scipy
print(f"render env: graphviz={graphviz.__version__} numpy={numpy.__version__} scipy={scipy.__version__}")
PY
"${RENDER_PYTHON}" -m pip freeze > "${LOG_DIR}/freeze_venv_gen.txt"
echo "Wrote ${LOG_DIR}/freeze_venv_gen.txt" \
  2>&1 | tee -a "${DATA_LOG}"
dot -V 2>&1 | tee -a "${DATA_LOG}"
if generator_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null)"; then
  echo "repository_commit=${generator_commit}" | tee -a "${DATA_LOG}"
else
  echo "repository_commit=unavailable" | tee -a "${DATA_LOG}"
fi
sha256sum "${REPO_ROOT}/scripts/generate_graphvis_datasets.py" | tee -a "${DATA_LOG}"

# Download directly on the pod. setup_data.sh is idempotent once cpnet/ and
# obqa/ exist, and copies only the two QA-GNN trees this project needs.
mkdir -p "${WORKSPACE}/.tmp"
echo "Filesystem before QA-GNN setup:" | tee -a "${DATA_LOG}"
df -h "${WORKSPACE}" | tee -a "${DATA_LOG}"
SETUP_DATA_WORK_ROOT="${WORKSPACE}/.tmp" \
SETUP_DATA_DF_PATH="${WORKSPACE}" \
  bash "${REPO_ROOT}/scripts/setup_data.sh" "${RAW_DATA_ROOT}" \
  2>&1 | tee -a "${DATA_LOG}"
echo "Filesystem after QA-GNN setup and temporary-archive cleanup:" | tee -a "${DATA_LOG}"
df -h "${WORKSPACE}" | tee -a "${DATA_LOG}"

# Reuse a complete fixed rendering; the generator itself refuses partial or
# colliding output, which prevents silently mixing render configurations.
image_count=0
if [[ -d "${TEST_ROOT}/images" ]]; then
  image_count="$(find "${TEST_ROOT}/images" -maxdepth 1 -type f -name 'q*_clean.png' | wc -l)"
fi
if [[ ! -f "${INPUT_JSONL}" || ! -f "${GRAPH_METADATA}" \
      || "${image_count}" -ne 5 ]]; then
  "${RENDER_PYTHON}" scripts/generate_graphvis_datasets.py \
    --split test \
    --start 0 \
    --limit 5 \
    --tasks-per-graph 6 \
    --stage1-task-set paper \
    --data-root "${RAW_DATA_ROOT}" \
    --out-dir "${DATASET_ROOT}" \
    --seed 13 \
    --max-nodes 18 \
    --max-edges 30 \
    --max-degree 5 \
    --bridge-rule qa-bridge \
    --engine dot \
    --node-fontsize 18 \
    --edge-fontsize 14 \
    --nodesep 0.5 \
    --ranksep 0.7 \
    --rankdir LR \
    --dpi 200 \
    --disconnected-rows 3 \
    2>&1 | tee -a "${DATA_LOG}"
fi

(cd "${DATASET_ROOT}" && sha256sum test/images/q*_clean.png) > "${RENDER_HASHES}"
cat "${RENDER_HASHES}" | tee -a "${DATA_LOG}"

# Run the real legibility gate as part of the mini pipeline. It evaluates all
# five graphs with pan-and-scan both enabled and disabled. The runner exits
# unsuccessfully when the selected pan-and-scan setting misses the threshold.
if ! "${QWEN_PYTHON}" scripts/eval_gemma3_stage1.py \
  --model-id "${GEMMA_ID}" \
  --revision "${GEMMA_REV}" \
  --input-jsonl "${INPUT_JSONL}" \
  --graph-metadata "${GRAPH_METADATA}" \
  --image-root "${DATASET_ROOT}" \
  --output-dir "${MINI_ROOT}/gemma_preflight" \
  --expected-count "${EXPECTED_COUNT}" \
  --task-set paper \
  --preflight 5 \
  --preflight-min-recall 0.95 \
  --attn-impl eager \
  --pan-and-scan \
  --pan-and-scan-min-crop-size 256 \
  --pan-and-scan-max-num-crops 4 \
  --pan-and-scan-min-ratio-to-activate 1.2 \
  --approve-prompts \
  --resume \
  2>&1 | tee -a "${LOG_DIR}/stage1_mini_gemma_preflight.log"; then
  echo "ERROR: Gemma Stage 1 legibility preflight failed; stopping before all mini-runs." >&2
  exit 1
fi

"${OCR2_PYTHON}" \
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py \
  --model-id "${DEEPSEEK_ID}" \
  --revision "${DEEPSEEK_REV}" \
  --input-jsonl "${INPUT_JSONL}" \
  --graph-metadata "${GRAPH_METADATA}" \
  --image-root "${DATASET_ROOT}" \
  --output-dir "${MINI_ROOT}/deepseek" \
  --expected-count "${EXPECTED_COUNT}" \
  --task-set paper \
  --approve-prompts \
  --resume \
  2>&1 | tee -a "${LOG_DIR}/stage1_mini_deepseek.log"

"${QWEN_PYTHON}" \
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_qwen.py \
  --model-id "${QWEN_ID}" \
  --revision "${QWEN_REV}" \
  --input-jsonl "${INPUT_JSONL}" \
  --graph-metadata "${GRAPH_METADATA}" \
  --image-root "${DATASET_ROOT}" \
  --output-dir "${MINI_ROOT}/qwen" \
  --expected-count "${EXPECTED_COUNT}" \
  --task-set paper \
  --min-pixels 262144 \
  --max-pixels 1310720 \
  --max-new-tokens 1024 \
  --approve-prompts \
  --resume \
  2>&1 | tee -a "${LOG_DIR}/stage1_mini_qwen.log"

"${QWEN_PYTHON}" scripts/eval_gemma3_stage1.py \
  --model-id "${GEMMA_ID}" \
  --revision "${GEMMA_REV}" \
  --input-jsonl "${INPUT_JSONL}" \
  --graph-metadata "${GRAPH_METADATA}" \
  --image-root "${DATASET_ROOT}" \
  --output-dir "${MINI_ROOT}/gemma" \
  --expected-count "${EXPECTED_COUNT}" \
  --task-set paper \
  --attn-impl eager \
  --pan-and-scan \
  --pan-and-scan-min-crop-size 256 \
  --pan-and-scan-max-num-crops 4 \
  --pan-and-scan-min-ratio-to-activate 1.2 \
  --preflight-report "${GEMMA_PREFLIGHT_REPORT}" \
  --approve-prompts \
  --resume \
  2>&1 | tee -a "${LOG_DIR}/stage1_mini_gemma.log"

echo "Stage 1 mini-runs completed under ${MINI_ROOT}."
