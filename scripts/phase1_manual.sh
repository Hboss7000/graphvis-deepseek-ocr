#!/usr/bin/env bash
# Manual, single-GPU phase-1 runner.  Run this *on the existing pod* only.
# It creates no RunPod resources and never stops or terminates a pod.
#
# Launch (from the pod):
#   tmux new-session -d -s phase1 'cd /workspace/bachelorArbeit && bash scripts/phase1_manual.sh'
# Monitor the current job:
#   tail -F /workspace/logs/phase1/<job>.log
#   watch -n 2 nvidia-smi
# The script also prints the exact tail command before each sequential job.

set -u -o pipefail

readonly WORKSPACE=/workspace
readonly REPO_ROOT=/workspace/bachelorArbeit
readonly SPLIT_ROOT="${WORKSPACE}/outputs/inference50_2026-10-03_corekeep_budget18_e30_tb30"
readonly DEFAULT_RENDER_ROOT="${WORKSPACE}/outputs/inference50_2026-10-03_corekeep_budget18_e30_default_render"
readonly SPLIT=test
readonly TEST_ROOT="${SPLIT_ROOT}/${SPLIT}"
readonly DEFAULT_TEST_ROOT="${DEFAULT_RENDER_ROOT}/${SPLIT}"
readonly DRY_ROOT="${WORKSPACE}/outputs/phase1_dryrun"
readonly PROBE_ROOT="${WORKSPACE}/outputs/phase1_reading_probe"
readonly LOG_ROOT="${WORKSPACE}/logs/phase1"
readonly STATUS_ROOT="${DRY_ROOT}/status"
readonly OCR2_PYTHON="${WORKSPACE}/venvs/venv_ocr2/bin/python"
# LLaVA, Qwen, and Gemma use the validated Transformers 5.x environment.
readonly QWEN_PYTHON="${WORKSPACE}/venvs/venv_qwen/bin/python"
readonly LLAVA_PYTHON="${QWEN_PYTHON}"

readonly DEEPSEEK_ID=deepseek-ai/DeepSeek-OCR-2
readonly DEEPSEEK_REV=aaa02f3811945a91062062994c5c4a3f4c0af2b0
readonly QWEN_ID=Qwen/Qwen3-VL-8B-Instruct
readonly QWEN_REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
readonly GEMMA_ID=google/gemma-3-12b-it
readonly GEMMA_REV=96b6f1eccf38110c56df3a15bffe176da04bfd80
readonly LLAVA_ID=llava-hf/llava-v1.6-mistral-7b-hf
readonly LLAVA_REV=2424fdd47412fccc66d91719126b420e9fbd7065

export HF_HOME="${WORKSPACE}/.cache/huggingface"
# Phase 1 must use the volume cache; a missing pinned snapshot is a failure,
# not permission to silently re-download model weights to an unknown location.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=13

mkdir -p "${DRY_ROOT}" "${PROBE_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}" "${HF_HOME}"
cd "${REPO_ROOT}"

for required in "${OCR2_PYTHON}" "${QWEN_PYTHON}"; do
  [[ -x "${required}" ]] || { echo "ERROR: missing interpreter: ${required}" >&2; exit 2; }
done
for required in \
  "${TEST_ROOT}/stage1_graph_comprehension_subset50_seed13.jsonl" \
  "${TEST_ROOT}/stage2_obqa_subset50_seed13.jsonl" \
  "${TEST_ROOT}/graph_metadata_subset50_seed13.jsonl" \
  "${DEFAULT_TEST_ROOT}/graph_metadata_subset50_seed13.jsonl"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing synced split artifact: ${required}" >&2; exit 2; }
done

# Build deterministic, exact-size dry inputs.  Eight examples means eight
# question graphs: Stage 1 contains all 9 extended tasks for each (72 rows),
# and QA contains 8 rows.  Images remain in the synced split and are not copied.
if [[ ! -f "${DRY_ROOT}/data/manifest.json" ]]; then
  "${QWEN_PYTHON}" - "${TEST_ROOT}" "${DRY_ROOT}/data" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)

def rows(name):
    return [json.loads(line) for line in (source / name).read_text().splitlines() if line]

stage1 = rows('stage1_graph_comprehension_subset50_seed13.jsonl')
stage2 = rows('stage2_obqa_subset50_seed13.jsonl')
metadata = rows('graph_metadata_subset50_seed13.jsonl')
ids = sorted({int(row['statement_idx']) for row in stage2})[:8]
selected_stage1 = [row for row in stage1 if int(row['statement_idx']) in ids]
selected_stage2 = [row for row in stage2 if int(row['statement_idx']) in ids]
selected_metadata = [row for row in metadata if int(row['statement_idx']) in ids]
if len(selected_stage1) != 72 or len(selected_stage2) != 8 or len(selected_metadata) != 8:
    raise SystemExit(f'unexpected dry selection sizes: stage1={len(selected_stage1)}, '
                     f'stage2={len(selected_stage2)}, metadata={len(selected_metadata)}')
if sorted({row['task_type'] for row in selected_stage1}) != [
        'edge_number', 'highest_node_degree', 'neighbor_listing', 'node_degree',
        'node_description', 'node_number', 'relation_identification',
        'shortest_path_listing', 'triple_listing']:
    raise SystemExit('dry Stage 1 selection does not contain the complete extended task set')
for name, value in {
    'stage1_extended_8graphs.jsonl': selected_stage1,
    'stage2_qa_8graphs.jsonl': selected_stage2,
    'graph_metadata_8graphs.jsonl': selected_metadata,
}.items():
    (destination / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in value))
(destination / 'manifest.json').write_text(json.dumps({
    'statement_indices': ids, 'stage1_records': len(selected_stage1),
    'stage2_records': len(selected_stage2), 'task_set': 'extended',
}, indent=2) + '\n')
PY
fi

# Build the fixed first-20 node_description probe once.  It contains no
# rendering-dependent bytes: only --image-root changes between the two LLaVA arms.
if [[ ! -f "${PROBE_ROOT}/data/manifest.json" ]]; then
  "${QWEN_PYTHON}" scripts/build_inference50_reading_probe.py \
    --stage1-jsonl "${TEST_ROOT}/stage1_graph_comprehension_subset50_seed13.jsonl" \
    --graph-metadata "${TEST_ROOT}/graph_metadata_subset50_seed13.jsonl" \
    --output-dir "${PROBE_ROOT}/data" --n 20
fi

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
  local gpu_csv="${LOG_ROOT}/${label}.gpu.csv"
  local status="${STATUS_ROOT}/${label}.json"
  local started ended exit_code monitor_pid
  echo
  echo "=== ${label} ==="
  echo "Launch: bash scripts/phase1_manual.sh   # runs this sequential job in the phase1 tmux session"
  echo "Monitor: tail -F ${log}"
  echo "GPU monitor: watch -n 2 nvidia-smi"
  printf 'started_utc=%s\ncommand=' "$(date -u +%FT%TZ)" >"${log}"
  printf '%q ' "$@" >>"${log}"
  printf '\n' >>"${log}"
  started="$(date +%s.%N)"
  "$@" >>"${log}" 2>&1 &
  local job_pid=$!
  monitor_gpu "${job_pid}" "${gpu_csv}" &
  monitor_pid=$!
  if wait "${job_pid}"; then exit_code=0; else exit_code=$?; fi
  wait "${monitor_pid}" || true
  ended="$(date +%s.%N)"
  "${QWEN_PYTHON}" - "${status}" "${label}" "${started}" "${ended}" "${exit_code}" <<'PY'
import json, sys
from pathlib import Path
path, label, started, ended, exit_code = sys.argv[1:]
payload = {'label': label, 'started_epoch': float(started), 'ended_epoch': float(ended),
           'wall_seconds': float(ended) - float(started), 'exit_code': int(exit_code)}
Path(path).write_text(json.dumps(payload, indent=2) + '\n')
PY
  if (( exit_code != 0 )); then
    echo "FAILED (${exit_code}); continuing to the next sequential job. See ${log}" >&2
  else
    echo "completed; see ${log}"
  fi
  return 0
}

readonly DRY_STAGE1="${DRY_ROOT}/data/stage1_extended_8graphs.jsonl"
readonly DRY_STAGE2="${DRY_ROOT}/data/stage2_qa_8graphs.jsonl"
readonly DRY_METADATA="${DRY_ROOT}/data/graph_metadata_8graphs.jsonl"
readonly PROBE_STAGE1="${PROBE_ROOT}/data/stage1_node_description_first20.jsonl"
readonly PROBE_METADATA="${PROBE_ROOT}/data/graph_metadata_first20.jsonl"
readonly STAGE1_RUNNERS="${REPO_ROOT}/experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
readonly QA_RUNNERS="${REPO_ROOT}/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts"

# Stage 1: eight question graphs / 72 records, normal extended-task inference.
run_job deepseek_stage1_8g "${OCR2_PYTHON}" "${STAGE1_RUNNERS}/run_stage1_deepseek.py" \
  --model-id "${DEEPSEEK_ID}" --revision "${DEEPSEEK_REV}" \
  --input-jsonl "${DRY_STAGE1}" --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${DRY_ROOT}/runs/deepseek_stage1" --expected-count 72 --expected-split test \
  --task-set extended --answer-format none --prompt-variant standard --seed 13 --max-new-tokens 8192 \
  --approve-prompts --resume

run_job qwen_stage1_8g "${QWEN_PYTHON}" "${STAGE1_RUNNERS}/run_stage1_qwen.py" \
  --model-id "${QWEN_ID}" --revision "${QWEN_REV}" \
  --input-jsonl "${DRY_STAGE1}" --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${DRY_ROOT}/runs/qwen_stage1" --expected-count 72 --expected-split test \
  --task-set extended --answer-format none --min-pixels 262144 --max-pixels 1310720 \
  --max-new-tokens 1024 --seed 13 --approve-prompts --resume

run_job llava_stage1_8g "${LLAVA_PYTHON}" "${STAGE1_RUNNERS}/run_stage1_llava.py" \
  --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" \
  --input-jsonl "${DRY_STAGE1}" --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${DRY_ROOT}/runs/llava_stage1" --expected-count 72 --expected-split test \
  --task-set extended --answer-format none --max-new-tokens 1024 --seed 13 --approve-prompts --resume

# Gemma's own runner requires this double-crop legibility gate before image inference.
run_job gemma_stage1_preflight_8g "${QWEN_PYTHON}" scripts/eval_gemma3_stage1.py \
  --model-id "${GEMMA_ID}" --revision "${GEMMA_REV}" \
  --input-jsonl "${DRY_STAGE1}" --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${DRY_ROOT}/runs/gemma_stage1_preflight" --expected-count 72 --expected-split test \
  --task-set extended --answer-format none --max-new-tokens 1024 --attn-impl eager --pan-and-scan \
  --pan-and-scan-min-crop-size 256 --pan-and-scan-max-num-crops 4 \
  --pan-and-scan-min-ratio-to-activate 1.2 --cache-implementation dynamic --batch-size 1 --seed 13 \
  --preflight 8 --preflight-min-recall 0.95 --approve-prompts --resume

run_job gemma_stage1_8g "${QWEN_PYTHON}" scripts/eval_gemma3_stage1.py \
  --model-id "${GEMMA_ID}" --revision "${GEMMA_REV}" \
  --input-jsonl "${DRY_STAGE1}" --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${DRY_ROOT}/runs/gemma_stage1" --expected-count 72 --expected-split test \
  --task-set extended --answer-format none --max-new-tokens 1024 --attn-impl eager --pan-and-scan \
  --pan-and-scan-min-crop-size 256 --pan-and-scan-max-num-crops 4 \
  --pan-and-scan-min-ratio-to-activate 1.2 --cache-implementation dynamic --batch-size 1 --seed 13 \
  --preflight-report "${DRY_ROOT}/runs/gemma_stage1_preflight/preflight_report.json" --approve-prompts --resume

# QA: eight of the same question graphs, image condition, unchanged prompt contract.
run_job deepseek_qa_8g "${OCR2_PYTHON}" "${QA_RUNNERS}/run_zero_shot.py" \
  --model-id "${DEEPSEEK_ID}" --revision "${DEEPSEEK_REV}" --input-jsonl "${DRY_STAGE2}" \
  --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-jsonl "${DRY_ROOT}/runs/deepseek_qa/predictions.jsonl" \
  --infer-output-dir "${DRY_ROOT}/runs/deepseek_qa/infer" --condition image --expected-count 8 \
  --max-new-tokens 8192 --approve-prompt-diff --resume

run_job qwen_qa_8g "${QWEN_PYTHON}" "${QA_RUNNERS}/run_zero_shot_qwen.py" \
  --model-id "${QWEN_ID}" --revision "${QWEN_REV}" --input-jsonl "${DRY_STAGE2}" \
  --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-jsonl "${DRY_ROOT}/runs/qwen_qa/predictions.jsonl" --condition image --expected-count 8 \
  --min-pixels 262144 --max-pixels 1310720 --max-new-tokens 64 --approve-prompt-diff --resume

run_job llava_qa_8g "${LLAVA_PYTHON}" "${QA_RUNNERS}/run_zero_shot_llava.py" \
  --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" --input-jsonl "${DRY_STAGE2}" \
  --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-jsonl "${DRY_ROOT}/runs/llava_qa/predictions.jsonl" --condition image --expected-count 8 \
  --max-new-tokens 64 --seed 13 --approve-prompt-diff --resume

run_job gemma_qa_8g "${QWEN_PYTHON}" scripts/eval_gemma3_stage2.py \
  --model-id "${GEMMA_ID}" --revision "${GEMMA_REV}" --input-jsonl "${DRY_STAGE2}" \
  --graph-metadata "${DRY_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-jsonl "${DRY_ROOT}/runs/gemma_qa/predictions.jsonl" --condition image --expected-count 8 \
  --max-new-tokens 64 --attn-impl eager --pan-and-scan --pan-and-scan-min-crop-size 256 \
  --pan-and-scan-max-num-crops 4 --pan-and-scan-min-ratio-to-activate 1.2 \
  --cache-implementation dynamic --batch-size 1 --seed 13 \
  --preflight-report "${DRY_ROOT}/runs/gemma_stage1_preflight/preflight_report.json" \
  --approve-prompt-diff --resume

# Reading probe: first 20 node_description graphs.  Only the default-vs-TB
# image root differs between the two LLaVA arms; inputs/metadata are identical.
run_job probe_llava_default_render "${LLAVA_PYTHON}" "${STAGE1_RUNNERS}/run_stage1_llava.py" \
  --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" --input-jsonl "${PROBE_STAGE1}" \
  --graph-metadata "${PROBE_METADATA}" --image-root "${DEFAULT_RENDER_ROOT}" \
  --output-dir "${PROBE_ROOT}/runs/llava_default_render" --expected-count 20 --expected-split test \
  --task-set extended --answer-format none --max-new-tokens 1024 --seed 13 --behavior-only --approve-prompts --resume

run_job probe_llava_tb30 "${LLAVA_PYTHON}" "${STAGE1_RUNNERS}/run_stage1_llava.py" \
  --model-id "${LLAVA_ID}" --revision "${LLAVA_REV}" --input-jsonl "${PROBE_STAGE1}" \
  --graph-metadata "${PROBE_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${PROBE_ROOT}/runs/llava_tb30" --expected-count 20 --expected-split test \
  --task-set extended --answer-format none --max-new-tokens 1024 --seed 13 --behavior-only --approve-prompts --resume

run_job probe_qwen_tb30 "${QWEN_PYTHON}" "${STAGE1_RUNNERS}/run_stage1_qwen.py" \
  --model-id "${QWEN_ID}" --revision "${QWEN_REV}" --input-jsonl "${PROBE_STAGE1}" \
  --graph-metadata "${PROBE_METADATA}" --image-root "${SPLIT_ROOT}" \
  --output-dir "${PROBE_ROOT}/runs/qwen_tb30" --expected-count 20 --expected-split test \
  --task-set extended --answer-format none --min-pixels 262144 --max-pixels 1310720 \
  --max-new-tokens 1024 --seed 13 --behavior-only --approve-prompts --resume

run_job probe_report "${QWEN_PYTHON}" scripts/report_inference50_reading_probe.py \
  --arm "llava_default_render=${PROBE_ROOT}/runs/llava_default_render/predictions_llava_node_description.jsonl" \
  --arm "llava_tb30=${PROBE_ROOT}/runs/llava_tb30/predictions_llava_node_description.jsonl" \
  --arm "qwen_tb30=${PROBE_ROOT}/runs/qwen_tb30/predictions_qwen_node_description.jsonl" \
  --expected-count 20 --output "${PROBE_ROOT}/runs/reading_probe_report.json"

echo
echo "Phase 1 command sequence finished (individual failures were recorded and did not block later jobs)."
echo "Summary: cd ${REPO_ROOT} && bash scripts/phase1_summary.sh"
