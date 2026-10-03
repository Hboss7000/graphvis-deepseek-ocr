#!/usr/bin/env bash
# LLaVA QA/probe verification on an already running pod. This script never
# creates, stops, or terminates a pod. Use a clean clone of inference50-core-keep.
#
# From the repository root on the pod:
#   tmux new-session -d -s phase1b 'cd /workspace/bachelorArbeit && bash scripts/phase1b_manual.sh'
# Monitor current job:
#   tail -F /workspace/logs/phase1b/<job>.log
#   watch -n 2 nvidia-smi
# At completion:
#   cd /workspace/bachelorArbeit && bash scripts/phase1b_summary.sh

set -u -o pipefail

readonly WORKSPACE=/workspace
readonly REPO_ROOT=/workspace/bachelorArbeit
readonly OUTPUTS_ROOT="${REPO_ROOT}/outputs"
readonly DRY_ROOT="${OUTPUTS_ROOT}/phase1_dryrun"
readonly PHASE1B_ROOT="${OUTPUTS_ROOT}/phase1b"
readonly PROBE_ROOT="${PHASE1B_ROOT}/reading_probe"
readonly PROBE_DATA="${OUTPUTS_ROOT}/phase1_reading_probe/data"
readonly A_ROOT="${OUTPUTS_ROOT}/inference50_2026-10-03_corekeep_budget18_e30_tb30"
readonly B_ROOT="${OUTPUTS_ROOT}/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_tb30"
readonly C_ROOT="${OUTPUTS_ROOT}/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_wrap14_tb30"
readonly D_ROOT="${OUTPUTS_ROOT}/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_parens_tb30"
readonly QA_INPUT="${DRY_ROOT}/data/stage2_qa_8graphs.jsonl"
readonly QA_METADATA="${DRY_ROOT}/data/graph_metadata_8graphs.jsonl"
readonly LOG_ROOT="${WORKSPACE}/logs/phase1b"
readonly STATUS_ROOT="${PHASE1B_ROOT}/status"
readonly QWEN_PYTHON="${WORKSPACE}/venvs/venv_qwen/bin/python"
readonly OCR2_PYTHON="${WORKSPACE}/venvs/venv_ocr2/bin/python"
readonly LLAVA_PYTHON="${QWEN_PYTHON}"
readonly LLAVA_ID=llava-hf/llava-v1.6-mistral-7b-hf
readonly LLAVA_REV=2424fdd47412fccc66d91719126b420e9fbd7065
readonly DEEPSEEK_ID=deepseek-ai/DeepSeek-OCR-2
readonly DEEPSEEK_REV=aaa02f3811945a91062062994c5c4a3f4c0af2b0

export HF_HOME="${WORKSPACE}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=13

mkdir -p "${PHASE1B_ROOT}" "${PROBE_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}" "${HF_HOME}"
cd "${REPO_ROOT}"
for required in "${QWEN_PYTHON}" "${OCR2_PYTHON}" "${QA_INPUT}" "${QA_METADATA}" \
  "${PROBE_DATA}/stage1_node_description_first20.jsonl" \
  "${PROBE_DATA}/graph_metadata_first20.jsonl"; do
  [[ -e "${required}" ]] || { echo "ERROR: missing required phase-1 artifact: ${required}" >&2; exit 2; }
done
for root in "${A_ROOT}" "${B_ROOT}" "${C_ROOT}" "${D_ROOT}"; do
  [[ -f "${root}/test/graph_metadata_subset50_seed13.jsonl" && -d "${root}/test/images" ]] || {
    echo "ERROR: config images/metadata not synced: ${root}" >&2; exit 2;
  }
done

cat <<EOF
Required B/C/D sync from the laptop, run from the repository root:
rsync -rltvz --no-owner --no-group -e 'ssh -i ~/.ssh/id_ed25519 -p 15268' \\
  outputs/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_tb30 \\
  outputs/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_wrap14_tb30 \\
  outputs/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_parens_tb30 \\
  root@216.243.220.174:/workspace/bachelorArbeit/outputs/
EOF

monitor_gpu() {
  local pid="$1" csv="$2"
  printf 'timestamp,memory_used_mib,utilization_gpu_percent\n' >"${csv}"
  while kill -0 "${pid}" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null >>"${csv}" || true
    sleep 0.5
  done
}

run_job() {
  local label="$1"
  shift
  local log="${LOG_ROOT}/${label}.log"
  local csv="${LOG_ROOT}/${label}.gpu.csv"
  local status="${STATUS_ROOT}/${label}.json"
  echo
  echo "=== ${label} ==="
  echo "Launch: tmux new-session -d -s phase1b 'cd ${REPO_ROOT} && bash scripts/phase1b_manual.sh'"
  echo "Monitor: tail -F ${log}"
  echo "GPU monitor: watch -n 2 nvidia-smi"
  {
    printf 'started_utc=%s\ncommand=' "$(date -u +%FT%TZ)"
    printf '%q ' "$@"
    printf '\n'
  } >"${log}"
  local started ended code monitor_pid
  started="$(date +%s.%N)"
  "$@" >>"${log}" 2>&1 &
  local pid=$!
  monitor_gpu "${pid}" "${csv}" & monitor_pid=$!
  if wait "${pid}"; then code=0; else code=$?; fi
  wait "${monitor_pid}" || true
  ended="$(date +%s.%N)"
  "${QWEN_PYTHON}" - "${status}" "${label}" "${started}" "${ended}" "${code}" <<'PY'
import json, sys
from pathlib import Path
path, label, started, ended, code = sys.argv[1:]
Path(path).write_text(json.dumps({'label': label, 'started_epoch': float(started),
    'ended_epoch': float(ended), 'wall_seconds': float(ended)-float(started),
    'exit_code': int(code)}, indent=2) + '\n')
PY
  if (( code == 0 )); then
    echo "COMPLETED; see ${log}"
  else
    echo "FAILED (${code}); continuing to next job; see ${log}" >&2
  fi
  return 0
}

readonly STAGE1_RUNNER="${REPO_ROOT}/experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py"
readonly QA_RUNNER="${REPO_ROOT}/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot_llava.py"
readonly PROBE_INPUT="${PROBE_DATA}/stage1_node_description_first20.jsonl"
readonly PROBE_METADATA="${PROBE_DATA}/graph_metadata_first20.jsonl"

echo "Effective phase-1b token limits: LLaVA QA=64 (all four conditions); DeepSeek QA=64."
for condition in image text text_noref kg_text; do
  run_job "llava_qa_${condition}" "${LLAVA_PYTHON}" "${QA_RUNNER}" \
    --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" \
    --input-jsonl "${QA_INPUT}" --graph-metadata "${QA_METADATA}" --image-root "${A_ROOT}" \
    --output-jsonl "${PHASE1B_ROOT}/qa/llava_${condition}/predictions.jsonl" \
    --condition "${condition}" --expected-count 8 --max-new-tokens 64 --seed 13 \
    --approve-prompt-diff --resume
done

for config in A B C; do
  case "${config}" in
    A) image_root="${A_ROOT}" ;;
    B) image_root="${B_ROOT}" ;;
    C) image_root="${C_ROOT}" ;;
  esac
  run_job "llava_probe_${config}" "${LLAVA_PYTHON}" "${STAGE1_RUNNER}" \
    --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" \
    --input-jsonl "${PROBE_INPUT}" --graph-metadata "${PROBE_METADATA}" --image-root "${image_root}" \
    --output-dir "${PROBE_ROOT}/llava_${config}" --expected-count 20 --expected-split test \
    --task-set extended --answer-format none --max-new-tokens 1024 --seed 13 \
    --behavior-only --approve-prompts --resume
done

run_job "llava_probe_D" "${LLAVA_PYTHON}" "${STAGE1_RUNNER}" \
  --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" \
  --input-jsonl "${PROBE_INPUT}" --graph-metadata "${PROBE_METADATA}" --image-root "${D_ROOT}" \
  --output-dir "${PROBE_ROOT}/llava_D" --expected-count 20 --expected-split test \
  --task-set extended --answer-format none --max-new-tokens 1024 --seed 13 \
  --behavior-only --approve-prompts --resume

run_job deepseek_qa_image "${OCR2_PYTHON}" \
  "${REPO_ROOT}/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot.py" \
  --model-id "${DEEPSEEK_ID}" --revision "${DEEPSEEK_REV}" \
  --input-jsonl "${QA_INPUT}" --graph-metadata "${QA_METADATA}" --image-root "${A_ROOT}" \
  --output-jsonl "${PHASE1B_ROOT}/qa/deepseek_image/predictions.jsonl" \
  --infer-output-dir "${PHASE1B_ROOT}/qa/deepseek_image/infer" \
  --condition image --expected-count 8 --max-new-tokens 64 --approve-prompt-diff --resume

run_job phase1b_probe_report "${QWEN_PYTHON}" scripts/report_inference50_reading_probe.py \
  --graph-metadata "${PROBE_METADATA}" \
  --arm "llava_A=${PROBE_ROOT}/llava_A/predictions_llava_node_description.jsonl" \
  --arm "llava_B=${PROBE_ROOT}/llava_B/predictions_llava_node_description.jsonl" \
  --arm "llava_C=${PROBE_ROOT}/llava_C/predictions_llava_node_description.jsonl" \
  --arm "llava_D=${PROBE_ROOT}/llava_D/predictions_llava_node_description.jsonl" \
  --expected-count 20 --output "${PROBE_ROOT}/reading_probe_report.json"

echo
echo "Phase 1b sequence ended; individual failures are recorded and later jobs were attempted."
echo "Summary: cd ${REPO_ROOT} && bash scripts/phase1b_summary.sh"
