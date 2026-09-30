#!/usr/bin/env bash
# Ordered Runpod Stage 1 workflow. Invoke one phase at a time; every phase exits.
# After the pilot and prompt freeze, run one resumable main block at a time:
#   CONFIRM_STAGE1_MAIN=yes ./scripts/runpod_stage1_session.sh main --cap 18 --model qwen
# Rehearse the complete pilot locally, with rendered graphs and stub inference:
#   DRY_RUN=1 ./scripts/runpod_stage1_session.sh pilot
# This script never creates, stops, or terminates a Runpod resource.
set -Eeuo pipefail
trap 'status=$?; echo "ERROR line ${LINENO}: ${BASH_COMMAND} (exit ${status})" >&2; exit "${status}"' ERR

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DRY_RUN="${DRY_RUN:-0}"
[[ "${DRY_RUN}" =~ ^(0|1)$ ]] || { echo "ERROR: DRY_RUN must be 0 or 1." >&2; exit 2; }
PHASE="${1:-}"
if [[ "${DRY_RUN}" == "1" && -z "${PHASE}" ]]; then
  PHASE="pilot"
fi
readonly PHASE
if [[ "${DRY_RUN}" == "1" ]]; then
  readonly DRY_RUN_BASE="${DRY_RUN_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/stage1-session-dry-run.XXXXXXXX")}"
  readonly WORKSPACE="${DRY_RUN_BASE}/workspace"
  if [[ -e "${WORKSPACE}/outputs" ]]; then
    echo "ERROR: dry-run output tree must be absent: ${WORKSPACE}/outputs" >&2
    exit 1
  fi
  SYSTEM_PYTHON="${DRY_RUN_PYTHON:-}"
  if [[ -z "${SYSTEM_PYTHON}" ]]; then
    if [[ -x "${REPO_ROOT}/myvenv/bin/python" ]]; then
      SYSTEM_PYTHON="${REPO_ROOT}/myvenv/bin/python"
    else
      SYSTEM_PYTHON="$(command -v python3)"
    fi
  fi
  readonly SYSTEM_PYTHON
  readonly OCR2_PYTHON="${SYSTEM_PYTHON}"
  readonly QWEN_PYTHON="${SYSTEM_PYTHON}"
  readonly RENDER_ENV=""
  readonly RENDER_PYTHON="${SYSTEM_PYTHON}"
  readonly RAW_DATA_ROOT="${DRY_RUN_DATA_ROOT:-${REPO_ROOT}/data_preprocessed_release}"
else
  readonly DRY_RUN_BASE=""
  readonly WORKSPACE="/workspace"
  readonly SYSTEM_PYTHON="/opt/conda/bin/python"
  readonly OCR2_PYTHON="${WORKSPACE}/venvs/venv_ocr2/bin/python"
  readonly QWEN_PYTHON="${WORKSPACE}/venvs/venv_qwen/bin/python"
  readonly RENDER_ENV="${WORKSPACE}/venvs/venv_gen"
  readonly RENDER_PYTHON="${RENDER_ENV}/bin/python"
  readonly RAW_DATA_ROOT="${WORKSPACE}/data_preprocessed_release"
fi
readonly LOG_DIR="${WORKSPACE}/logs"
readonly PILOT_DATA_ROOT="${WORKSPACE}/outputs/stage1_prompt_pilot"
readonly PILOT_ROOT="${WORKSPACE}/outputs/stage1_prompt_repilot_v2"
readonly MAIN_ROOT="${WORKSPACE}/outputs/stage1_subset100"
readonly STATE_DIR="${WORKSPACE}/outputs/stage1_session_state"
readonly PILOT_DONE="${STATE_DIR}/pilot_v2.complete"
readonly FREEZE_FILE="${STATE_DIR}/prompts_frozen.json"

readonly DEEPSEEK_ID="deepseek-ai/DeepSeek-OCR-2"
readonly DEEPSEEK_REV="aaa02f3811945a91062062994c5c4a3f4c0af2b0"
readonly QWEN_ID="Qwen/Qwen3-VL-8B-Instruct"
readonly QWEN_REV="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
readonly GEMMA_ID="google/gemma-3-12b-it"
readonly GEMMA_REV="96b6f1eccf38110c56df3a15bffe176da04bfd80"

export HF_HOME="${WORKSPACE}/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=13
mkdir -p "${LOG_DIR}" "${STATE_DIR}"
cd "${REPO_ROOT}"

usage() {
  echo "Usage: $0 pilot|freeze|main --cap 18|40 --model deepseek|qwen|gemma" >&2
  echo "       DRY_RUN=1 $0 [pilot]  # local CPU/stub rehearsal; pilot only" >&2
  echo "Run pilot first, inspect its report, then run CONFIRM_PROMPT_FREEZE=yes $0 freeze." >&2
  exit 2
}
[[ "${PHASE}" =~ ^(pilot|freeze|main)$ ]] || usage
if [[ "${DRY_RUN}" == "1" && "${PHASE}" != "pilot" ]]; then
  echo "ERROR: DRY_RUN=1 rehearses the complete pilot and stops; freeze/main are intentionally unavailable." >&2
  exit 2
fi
shift || true

CAP=""
MODEL=""
while (($#)); do
  case "$1" in
    --cap) [[ $# -ge 2 ]] || usage; CAP="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || usage; MODEL="$2"; shift 2 ;;
    *) usage ;;
  esac
done
if [[ "${PHASE}" == "main" ]]; then
  [[ "${CAP}" =~ ^(18|40)$ ]] || usage
  [[ "${MODEL}" =~ ^(deepseek|qwen|gemma)$ ]] || usage
elif [[ -n "${CAP}" || -n "${MODEL}" ]]; then
  usage
fi

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: required file is absent: $1" >&2; exit 1; }
}

verify_prompt_freeze() {
  "${SYSTEM_PYTHON}" - "${FREEZE_FILE}" <<'PY'
import hashlib, json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
path = pathlib.Path.cwd() / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/stage1_common.py'
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != record['stage1_common_sha256']:
    raise SystemExit(f"Frozen prompt code changed: expected {record['stage1_common_sha256']}, found {actual}")
PY
}

write_render_configs() {
  local path="$1"
  "${SYSTEM_PYTHON}" - "${path}" <<'PY'
import json, pathlib, sys
common = {'max_edges': 30, 'max_degree': 5, 'bridge_rule': 'qa-bridge', 'lifelines': True,
          'engine': 'dot', 'node_fontsize': 18, 'edge_fontsize': 14,
          'nodesep': 0.5, 'ranksep': 0.7, 'rankdir': 'LR', 'dpi': 200,
          'disconnected_rows': 3, 'hide_relatedto_labels': False,
          'reveal_correct_answer': False}
payload = {
  'nodes18': {**common, 'max_nodes': 18},
  'nodes40': {**common, 'max_nodes': 40},
  'provenance': (
    'Shared values are the COMA run-manifest expected renderer/pruning settings plus the '
    'generator defaults used by its recorded shell command. The copied nodes40 manifest '
    'incorrectly says max_nodes=18; max_nodes=40 comes from the named COMA condition and '
    'the explicit experimental instruction.'),
}
target = pathlib.Path(sys.argv[1]); target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY
}

prepare_host() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "PHASE prepare_host: local CPU dry run"
    echo "DRY-RUN ROOT: ${DRY_RUN_BASE}"
    echo "DRY-RUN WORKSPACE: ${WORKSPACE}"
    [[ -x "${SYSTEM_PYTHON}" ]] || { echo "ERROR: missing local Python ${SYSTEM_PYTHON}" >&2; exit 1; }
    command -v dot >/dev/null 2>&1 || { echo "ERROR: local Graphviz 'dot' is required." >&2; exit 1; }
    require_file "${RAW_DATA_ROOT}/cpnet/concept.txt"
    require_file "${RAW_DATA_ROOT}/obqa/graph/dev.graph.adj.pk"
    require_file "${RAW_DATA_ROOT}/obqa/statement/dev.statement.jsonl"
    "${RENDER_PYTHON}" - <<'PY'
import graphviz, numpy, scipy
print(f"Local render imports: graphviz={graphviz.__version__} numpy={numpy.__version__} scipy={scipy.__version__}")
PY
    dot -V 2>&1 | tee -a "${LOG_DIR}/stage1_session_data.log"
    echo "Dry run uses existing local data at ${RAW_DATA_ROOT}; no downloads or package installs."
    return
  fi
  for executable in "${SYSTEM_PYTHON}" "${OCR2_PYTHON}" "${QWEN_PYTHON}"; do
    [[ -x "${executable}" ]] || { echo "ERROR: missing interpreter ${executable}" >&2; exit 1; }
  done
  local missing=()
  local command
  for command in git wget curl unzip rsync tmux dot; do
    command -v "${command}" >/dev/null 2>&1 || missing+=("${command}")
  done
  if ((${#missing[@]})); then
    [[ "$(id -u)" -eq 0 ]] || { echo "ERROR: missing host commands: ${missing[*]}" >&2; exit 1; }
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      ca-certificates curl git graphviz rsync tmux unzip wget
  fi
  dot -V 2>&1 | tee -a "${LOG_DIR}/stage1_session_data.log"
  if [[ ! -x "${RENDER_PYTHON}" ]]; then
    "${SYSTEM_PYTHON}" -m venv "${RENDER_ENV}"
  fi
  grep -q '^include-system-site-packages = false$' "${RENDER_ENV}/pyvenv.cfg" || {
    echo "ERROR: ${RENDER_ENV} must be an isolated rendering environment." >&2
    exit 1
  }
  "${RENDER_PYTHON}" -m pip install --upgrade pip
  "${RENDER_PYTHON}" -m pip install --upgrade --requirement requirements_render_py311.txt
  "${RENDER_PYTHON}" -m pip freeze > "${LOG_DIR}/freeze_venv_gen.txt"
  mkdir -p "${WORKSPACE}/.tmp"
  SETUP_DATA_WORK_ROOT="${WORKSPACE}/.tmp" SETUP_DATA_DF_PATH="${WORKSPACE}" \
    bash scripts/setup_data.sh "${RAW_DATA_ROOT}" \
    2>&1 | tee -a "${LOG_DIR}/stage1_session_data.log"
}

gpu_is_comparison_eligible() {
  "${QWEN_PYTHON}" - <<'PY'
import subprocess, torch
p = torch.cuda.get_device_properties(0)
inventory = subprocess.run(['nvidia-smi', '-L'], check=True, text=True, capture_output=True).stdout
ok = ('RTX PRO 6000' in p.name and 'MIG' not in p.name.upper()
      and p.total_memory >= 90 * 1024**3 and 'MIG' not in inventory.upper())
raise SystemExit(0 if ok else 1)
PY
}

record_gpu_session() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      echo "phase=${PHASE}"
      date -u +started_utc=%FT%TZ
      echo "device=CPU stub (DRY_RUN=1; no CUDA/model inference)"
    } | tee -a "${LOG_DIR}/stage1_session_gpu.log"
    return
  fi
  {
    echo "phase=${PHASE}"
    date -u +started_utc=%FT%TZ
    nvidia-smi -L
    nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader
    "${QWEN_PYTHON}" - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
major, minor = torch.cuda.get_device_capability(0)
print(f'torch_device_name={p.name}')
print(f'torch_total_memory_bytes={p.total_memory}')
print(f'torch_architecture=sm_{major}{minor}')
PY
  } 2>&1 | tee -a "${LOG_DIR}/stage1_session_gpu.log"
}

run_stub_model() {
  local model="$1" input="$2" metadata="$3" image_root="$4" output="$5"
  local expected_count="$6" split="$7" answer_format="$8" prompt_variant="$9"
  local model_id revision
  shift 9
  case "${model}" in
    deepseek) model_id="${DEEPSEEK_ID}"; revision="${DEEPSEEK_REV}" ;;
    qwen) model_id="${QWEN_ID}"; revision="${QWEN_REV}" ;;
    gemma) model_id="${GEMMA_ID}"; revision="${GEMMA_REV}" ;;
    *) echo "ERROR: unsupported stub model: ${model}" >&2; return 2 ;;
  esac
  "${SYSTEM_PYTHON}" scripts/stub_stage1_model.py \
    --model "${model}" --model-id "${model_id}" --revision "${revision}" \
    --input-jsonl "${input}" --graph-metadata "${metadata}" --image-root "${image_root}" \
    --output-dir "${output}" --expected-count "${expected_count}" --expected-split "${split}" \
    --task-set paper --answer-format "${answer_format}" --prompt-variant "${prompt_variant}" "$@"
}

generate_dataset() {
  local split="$1" limit="$2" cap="$3" destination="$4"
  local input="${destination}/${split}/stage1_graph_comprehension_0_${limit}.jsonl"
  local metadata="${destination}/${split}/graph_metadata_0_${limit}.jsonl"
  local images="${destination}/${split}/images"
  local count=0
  if [[ -d "${images}" ]]; then
    count="$(find "${images}" -maxdepth 1 -type f -name 'q*_clean.png' | wc -l)"
  fi
  if [[ -f "${input}" && -f "${metadata}" && "${count}" -eq "${limit}" ]]; then
    return
  fi
  if [[ -e "${destination}/${split}" ]]; then
    echo "ERROR: incomplete/colliding rendered split at ${destination}/${split}; preserve it and choose a fresh root." >&2
    exit 1
  fi
  "${RENDER_PYTHON}" scripts/generate_graphvis_datasets.py \
    --split "${split}" --start 0 --limit "${limit}" --tasks-per-graph 6 \
    --stage1-task-set paper --data-root "${RAW_DATA_ROOT}" --out-dir "${destination}" \
    --seed 13 --max-nodes "${cap}" --max-edges 30 --max-degree 5 \
    --bridge-rule qa-bridge --engine dot --node-fontsize 18 --edge-fontsize 14 \
    --nodesep 0.5 --ranksep 0.7 --rankdir LR --dpi 200 --disconnected-rows 3
}

run_model_set() {
  local split="$1" limit="$2" cap="$3" dataset="$4" root="$5" answer_format="$6" prefix="$7"
  local selected_model="${8:-all}"
  local count=$((limit * 6))
  local input="${dataset}/${split}/stage1_graph_comprehension_0_${limit}.jsonl"
  local metadata="${dataset}/${split}/graph_metadata_0_${limit}.jsonl"
  local preflight="${root}/gemma_preflight/preflight_report.json"
  local common=(--input-jsonl "${input}" --graph-metadata "${metadata}" --image-root "${dataset}"
    --expected-count "${count}" --expected-split "${split}" --task-set paper
    --answer-format "${answer_format}" --approve-prompts --resume)

  if [[ "${DRY_RUN}" == "1" ]]; then
    local stub_model
    for stub_model in deepseek qwen gemma; do
      if [[ "${selected_model}" == "all" || "${selected_model}" == "${stub_model}" ]]; then
        echo "PHASE re-pilot: cap=${cap}, model=${stub_model}, CPU stub"
        if [[ "${stub_model}" == "gemma" ]]; then
          run_stub_model gemma "${input}" "${metadata}" "${dataset}" \
            "${root}/gemma_preflight" "${count}" "${split}" "${answer_format}" standard --preflight \
            2>&1 | tee -a "${LOG_DIR}/${prefix}_gemma_preflight.log"
        fi
        run_stub_model "${stub_model}" "${input}" "${metadata}" "${dataset}" \
          "${root}/${stub_model}" "${count}" "${split}" "${answer_format}" standard \
          2>&1 | tee -a "${LOG_DIR}/${prefix}_${stub_model}.log"
      fi
    done
    return
  fi

  if [[ "${selected_model}" == "all" || "${selected_model}" == "gemma" ]]; then
    if ! "${QWEN_PYTHON}" scripts/eval_gemma3_stage1.py \
      --model-id "${GEMMA_ID}" --revision "${GEMMA_REV}" "${common[@]}" \
      --output-dir "${root}/gemma_preflight" --preflight "${limit}" --preflight-min-recall 0.95 \
      --attn-impl eager --pan-and-scan --pan-and-scan-min-crop-size 256 \
      --pan-and-scan-max-num-crops 4 --pan-and-scan-min-ratio-to-activate 1.2 \
      2>&1 | tee -a "${LOG_DIR}/${prefix}_gemma_preflight.log"; then
      echo "ERROR: Gemma legibility preflight failed for node cap ${cap}; Gemma was not started." >&2
      exit 1
    fi
  fi

  if [[ "${selected_model}" == "all" || "${selected_model}" == "deepseek" ]]; then
    "${OCR2_PYTHON}" experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py \
      --model-id "${DEEPSEEK_ID}" --revision "${DEEPSEEK_REV}" "${common[@]}" \
      --output-dir "${root}/deepseek" 2>&1 | tee -a "${LOG_DIR}/${prefix}_deepseek.log"
    "${QWEN_PYTHON}" scripts/record_runpod_gpu.py "${root}/deepseek/run_config.json"
  fi

  if [[ "${selected_model}" == "all" || "${selected_model}" == "qwen" ]]; then
    "${QWEN_PYTHON}" experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_qwen.py \
      --model-id "${QWEN_ID}" --revision "${QWEN_REV}" "${common[@]}" \
      --output-dir "${root}/qwen" --min-pixels 262144 --max-pixels 1310720 \
      --max-new-tokens 2048 2>&1 | tee -a "${LOG_DIR}/${prefix}_qwen.log"
    "${QWEN_PYTHON}" scripts/record_runpod_gpu.py "${root}/qwen/run_config.json"
  fi

  if [[ "${selected_model}" == "all" || "${selected_model}" == "gemma" ]]; then
    "${QWEN_PYTHON}" scripts/eval_gemma3_stage1.py \
      --model-id "${GEMMA_ID}" --revision "${GEMMA_REV}" "${common[@]}" \
      --output-dir "${root}/gemma" --attn-impl eager --pan-and-scan \
      --pan-and-scan-min-crop-size 256 --pan-and-scan-max-num-crops 4 \
      --pan-and-scan-min-ratio-to-activate 1.2 --preflight-report "${preflight}" \
      --max-new-tokens 2048 \
      2>&1 | tee -a "${LOG_DIR}/${prefix}_gemma.log"
    "${QWEN_PYTHON}" scripts/record_runpod_gpu.py "${root}/gemma/run_config.json"
  fi
}

make_ablation_subset() {
  local dataset="$1" destination="$2"
  local input="${dataset}/dev/stage1_graph_comprehension_0_10.jsonl"
  local metadata="${dataset}/dev/graph_metadata_0_10.jsonl"
  mkdir -p "${destination}"
  "${SYSTEM_PYTHON}" - "${input}" "${metadata}" "${destination}" <<'PY'
import json, pathlib, sys
source, metadata, destination = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
indices = sorted({int(row['statement_idx']) for row in rows})[:5]
selected = [row for row in rows if int(row['statement_idx']) in indices]
meta = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
meta = [row for row in meta if int(row['statement_idx']) in indices]
if len(selected) != 30 or len(meta) != 5:
    raise SystemExit(f'Expected 30 task rows and 5 metadata rows; found {len(selected)} and {len(meta)}')
(destination / 'stage1_graph_comprehension_0_5.jsonl').write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
(destination / 'stage1_describe_0_5.jsonl').write_text(
    ''.join(json.dumps(next(row for row in selected
                            if int(row['statement_idx']) == idx
                            and row['task_type'] == 'node_description'), ensure_ascii=False) + '\n'
            for idx in indices))
(destination / 'graph_metadata_0_5.jsonl').write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in meta))
PY
}

run_deepseek_ablation() {
  local dataset="$1" root="$2"
  local subset="${root}/data/dev"
  make_ablation_subset "${dataset}" "${subset}"
  local input="${subset}/stage1_graph_comprehension_0_5.jsonl"
  local metadata="${subset}/graph_metadata_0_5.jsonl"
  local name answer_format prompt_variant
  for spec in \
    "original:none:standard" \
    "old_placeholder:constrained:old-placeholder" \
    "natural:constrained:standard" \
    "completion:none:completion"; do
    IFS=: read -r name answer_format prompt_variant <<<"${spec}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "PHASE ablation: variant=${name}, CPU stub"
      run_stub_model deepseek "${input}" "${metadata}" "${dataset}" \
        "${root}/${name}" 30 dev "${answer_format}" "${prompt_variant}" \
        2>&1 | tee -a "${LOG_DIR}/deepseek_ablation_${name}.log"
      continue
    fi
    "${OCR2_PYTHON}" experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py \
      --model-id "${DEEPSEEK_ID}" --revision "${DEEPSEEK_REV}" \
      --input-jsonl "${input}" --graph-metadata "${metadata}" --image-root "${dataset}" \
      --output-dir "${root}/${name}" --expected-count 30 --expected-split dev \
      --task-set paper --answer-format "${answer_format}" --prompt-variant "${prompt_variant}" \
      --approve-prompts --resume 2>&1 | tee -a "${LOG_DIR}/deepseek_ablation_${name}.log"
    "${QWEN_PYTHON}" scripts/record_runpod_gpu.py "${root}/${name}/run_config.json"
  done
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "PHASE ablation: variant=describe, CPU stub"
    run_stub_model deepseek "${subset}/stage1_describe_0_5.jsonl" "${metadata}" "${dataset}" \
      "${root}/describe" 5 dev none describe --behavior-only \
      2>&1 | tee -a "${LOG_DIR}/deepseek_ablation_describe.log"
  else
  "${OCR2_PYTHON}" experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py \
    --model-id "${DEEPSEEK_ID}" --revision "${DEEPSEEK_REV}" \
    --input-jsonl "${subset}/stage1_describe_0_5.jsonl" --graph-metadata "${metadata}" \
    --image-root "${dataset}" --output-dir "${root}/describe" --expected-count 5 \
    --expected-split dev --task-set paper --answer-format none --prompt-variant describe \
    --behavior-only --approve-prompts --resume \
    2>&1 | tee -a "${LOG_DIR}/deepseek_ablation_describe.log"
  "${QWEN_PYTHON}" scripts/record_runpod_gpu.py "${root}/describe/run_config.json"
  fi
  "${QWEN_PYTHON}" scripts/report_deepseek_prompt_ablation.py \
    --run "original=${root}/original" --run "old_placeholder=${root}/old_placeholder" \
    --run "natural=${root}/natural" --run "completion=${root}/completion" \
    --run "describe=${root}/describe" --output-json "${root}/report.json" \
    --output-markdown "${root}/report.md" | tee -a "${LOG_DIR}/deepseek_ablation_report.log"
}

pilot_phase() {
  echo "PHASE pilot: begin"
  prepare_host
  record_gpu_session
  if [[ "${DRY_RUN}" != "1" ]] && ! gpu_is_comparison_eligible; then
    echo "ERROR: the pilot requires a full non-MIG RTX PRO 6000." >&2
    exit 1
  fi
  echo "PHASE render: nodes18 dev graphs"
  write_render_configs "${PILOT_ROOT}/render_configs.json"
  local cap dataset root
  dataset="${PILOT_DATA_ROOT}/nodes18/data"
  generate_dataset dev 10 18 "${dataset}"
  run_deepseek_ablation "${dataset}" "${PILOT_ROOT}/deepseek_ablation"
  for cap in 18 40; do
    dataset="${PILOT_DATA_ROOT}/nodes${cap}/data"
    root="${PILOT_ROOT}/nodes${cap}"
    echo "PHASE render: nodes${cap} dev graphs"
    generate_dataset dev 10 "${cap}" "${dataset}"
    run_model_set dev 10 "${cap}" "${dataset}" "${root}" constrained "stage1_pilot_nodes${cap}"
  done
  echo "PHASE report: combined re-pilot report"
  "${QWEN_PYTHON}" scripts/report_stage1_answer_format.py \
    --answer-format constrained \
    --run "nodes18/deepseek=${PILOT_ROOT}/nodes18/deepseek" \
    --run "nodes18/qwen=${PILOT_ROOT}/nodes18/qwen" \
    --run "nodes18/gemma=${PILOT_ROOT}/nodes18/gemma" \
    --run "nodes40/deepseek=${PILOT_ROOT}/nodes40/deepseek" \
    --run "nodes40/qwen=${PILOT_ROOT}/nodes40/qwen" \
    --run "nodes40/gemma=${PILOT_ROOT}/nodes40/gemma" \
    --output-json "${PILOT_ROOT}/pilot_report.json" \
    --output-markdown "${PILOT_ROOT}/pilot_report.md" \
    | tee -a "${LOG_DIR}/stage1_pilot_report.log"
  date -u +%FT%TZ > "${PILOT_DONE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN COMPLETE. Preserved temporary workspace at ${WORKSPACE}."
  fi
  echo "PILOT COMPLETE. Stop here. Review pilot_report.md; do not freeze prompts or run main until you approve the dev result."
}

freeze_phase() {
  require_file "${PILOT_DONE}"
  [[ "${CONFIRM_PROMPT_FREEZE:-}" == "yes" ]] || {
    echo "ERROR: after reviewing the dev pilot, run CONFIRM_PROMPT_FREEZE=yes $0 freeze" >&2
    exit 1
  }
  "${SYSTEM_PYTHON}" - "${FREEZE_FILE}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path.cwd()
common = root / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/stage1_common.py'
payload = {'frozen_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
           'stage1_common_sha256': hashlib.sha256(common.read_bytes()).hexdigest(),
           'decision': 'dev pilot accepted; prompts frozen'}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY
  echo "Prompt freeze recorded in ${FREEZE_FILE}. Stop here."
}

main_phase() {
  require_file "${PILOT_DONE}"; require_file "${FREEZE_FILE}"
  verify_prompt_freeze
  [[ "${CONFIRM_STAGE1_MAIN:-}" == "yes" ]] || {
    echo "ERROR: main is gated; after explicit approval run CONFIRM_STAGE1_MAIN=yes $0 main" >&2
    exit 1
  }
  prepare_host
  record_gpu_session
  write_render_configs "${MAIN_ROOT}/render_configs.json"
  gpu_is_comparison_eligible || {
    echo "ERROR: main comparison run requires a full non-MIG RTX PRO 6000." >&2
    exit 1
  }
  local dataset="${MAIN_ROOT}/nodes${CAP}/data"
  local root="${MAIN_ROOT}/nodes${CAP}"
  generate_dataset test 100 "${CAP}" "${dataset}"
  run_model_set test 100 "${CAP}" "${dataset}" "${root}" constrained \
    "stage1_subset100_nodes${CAP}_${MODEL}" "${MODEL}"
  echo "MAIN BLOCK COMPLETE: cap=${CAP}, model=${MODEL}, output=${root}/${MODEL}."
}

case "${PHASE}" in
  pilot) pilot_phase ;;
  freeze) freeze_phase ;;
  main) main_phase ;;
esac
