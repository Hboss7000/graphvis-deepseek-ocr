#!/usr/bin/env bash
#SBATCH --job-name=gemma3-zs
#SBATCH --output=/storage/home/hleonel/outputs/slurm-%x-%j.out
#SBATCH --error=/storage/home/hleonel/outputs/slurm-%x-%j.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/graphvis}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif}"
QWEN_PYTHON="/storage/home/hleonel/venv_qwen/bin/python"
DATA_ROOT="${DATA_ROOT:-${HOME}/outputs/graphvis_obqa}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/outputs/experiments/gemma3_zero_shot}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
MODEL_ID="${MODEL_ID:-google/gemma-3-12b-it}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the prefetched immutable Hub commit}"
PHASE="${PHASE:-preflight}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
PAN_AND_SCAN="${PAN_AND_SCAN:-1}"
# The pinned checkpoint metadata leaves these controls null. The 256/4/1.2
# fallbacks come from Gemma3ProcessorKwargs._defaults in Transformers 4.57.1
# (processing_gemma3.py:43-48) and 5.16.1 (processing_gemma3.py:29-34); keep
# them explicit so the experiment does not depend on implicit library defaults.
PAN_AND_SCAN_MIN_CROP_SIZE="${PAN_AND_SCAN_MIN_CROP_SIZE:-256}"
PAN_AND_SCAN_MAX_NUM_CROPS="${PAN_AND_SCAN_MAX_NUM_CROPS:-4}"
PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE="${PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE:-1.2}"
TASK_SET="${TASK_SET:-paper}"
ANSWER_FORMAT="${ANSWER_FORMAT:-none}"
EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
APPROVE_PROMPTS="${APPROVE_PROMPTS:-0}"
STAGE1_JSONL="${STAGE1_JSONL:-${DATA_ROOT}/test/stage1_graph_comprehension_0_500.jsonl}"
STAGE2_JSONL="${STAGE2_JSONL:-${DATA_ROOT}/test/stage2_obqa_0_500.jsonl}"
GRAPH_METADATA="${GRAPH_METADATA:-${DATA_ROOT}/test/graph_metadata_0_500.jsonl}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${RESULTS_DIR}/preflight/preflight_report.json}"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_PYTHONHASHSEED=13
APPTAINER_ARGS=(exec --nv --bind /storage/nobackup "$CONTAINER_IMAGE")
COMMON_ARGS=(--model-id "$MODEL_ID" --revision "$MODEL_REVISION"
  --graph-metadata "$GRAPH_METADATA" --image-root "$DATA_ROOT" --seed 13)
if [[ "$PAN_AND_SCAN" == 1 ]]; then
  COMMON_ARGS+=(--pan-and-scan)
else
  COMMON_ARGS+=(--no-pan-and-scan)
fi
if [[ -n "${PAN_AND_SCAN_MIN_CROP_SIZE:-}" ]]; then
  COMMON_ARGS+=(--pan-and-scan-min-crop-size "$PAN_AND_SCAN_MIN_CROP_SIZE")
fi
if [[ -n "${PAN_AND_SCAN_MAX_NUM_CROPS:-}" ]]; then
  COMMON_ARGS+=(--pan-and-scan-max-num-crops "$PAN_AND_SCAN_MAX_NUM_CROPS")
fi
if [[ -n "${PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE:-}" ]]; then
  COMMON_ARGS+=(--pan-and-scan-min-ratio-to-activate "$PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE")
fi
run_python() { apptainer "${APPTAINER_ARGS[@]}" "$QWEN_PYTHON" "$@"; }

# Also run this cache check on the login node before sbatch (documented in README).
run_python - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import json, pathlib, re, sys
from huggingface_hub import snapshot_download
if not re.fullmatch(r'[0-9a-fA-F]{40}', sys.argv[2]):
    raise SystemExit('MODEL_REVISION must be a 40-character commit hash')
snapshot = pathlib.Path(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
index = snapshot / 'model.safetensors.index.json'
weights = (set(json.loads(index.read_text())['weight_map'].values()) if index.exists()
           else {'model.safetensors'})
missing = [name for name in weights if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0]
if missing:
    raise SystemExit(f'Incomplete offline weight cache: {missing}')
print(snapshot)
PY

# The initial attention/repeat checks must have passed before any full run.
if [[ "$APPROVE_PROMPTS" == 1 && ( "$PHASE" == stage1 || "$PHASE" == stage2 ) ]]; then
  run_python - "$RESULTS_DIR/sanity" "$MODEL_ID" "$MODEL_REVISION" <<'PYCODE'
import json, pathlib, sys
# compare_sdpa.json is informational: eager vs SDPA differ in kernel numerics,
# which flips near-tie tokens. Runs use eager, so only repeat-determinism gates.
for name in ('compare_eager_repeat.json',):
    report = json.loads((pathlib.Path(sys.argv[1]) / name).read_text())
    if not (report['identical'] and report['item_count'] == 50
            and report['model_id'] == sys.argv[2] and report['revision'] == sys.argv[3]):
        raise SystemExit('Run PHASE=sanity and resolve its differences before full inference')
PYCODE
fi

case "$PHASE" in
  preflight|stage1)
    ARGS=(scripts/eval_gemma3_stage1.py "${COMMON_ARGS[@]}"
      --input-jsonl "$STAGE1_JSONL" --expected-count "${STAGE1_EXPECTED_COUNT:-3000}"
      --expected-split "$EXPECTED_SPLIT" --answer-format "$ANSWER_FORMAT"
      --task-set "$TASK_SET" --attn-impl "$ATTN_IMPL")
    if [[ "$PHASE" == preflight ]]; then
      ARGS+=(--preflight 50 --output-dir "$RESULTS_DIR/preflight"
        --preflight-min-recall "${PREFLIGHT_MIN_RECALL:-0.95}")
    else
      ARGS+=(--output-dir "$RESULTS_DIR/stage1" --preflight-report "$PREFLIGHT_REPORT" --resume)
    fi
    if [[ "$APPROVE_PROMPTS" == 1 ]]; then
      ARGS+=(--approve-prompts)
    else
      ARGS+=(--preview-only)
    fi
    run_python "${ARGS[@]}"
    ;;
  sanity)
    # Same first 50 items for eager, sdpa and the eager repeat. No rerendering.
    SLICE="$RESULTS_DIR/stage2_first50.jsonl"
    run_python - "$STAGE2_JSONL" "$SLICE" <<'PY'
import json, pathlib, sys
rows = sorted((json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line),
              key=lambda row: int(row['statement_idx']))[:50]
if len(rows) != 50:
    raise SystemExit('Need at least 50 Stage 2 items')
pathlib.Path(sys.argv[2]).write_text(''.join(json.dumps(row) + '\n' for row in rows))
PY
    for TRIAL in eager eager_repeat sdpa; do
      BACKEND="${TRIAL%_repeat}"
      ARGS=(scripts/eval_gemma3_stage2.py "${COMMON_ARGS[@]}"
        --input-jsonl "$SLICE" --expected-count 50 --condition text --attn-impl "$BACKEND"
        --output-jsonl "$RESULTS_DIR/sanity/$TRIAL/predictions.jsonl")
      if [[ "$APPROVE_PROMPTS" != 1 ]]; then
        run_python "${ARGS[@]}" --preview-only
        exit 0
      fi
      run_python "${ARGS[@]}" --approve-prompt-diff --resume
      run_python experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/score_predictions.py \
        --predictions "$RESULTS_DIR/sanity/$TRIAL/predictions.jsonl" --expected-count 50 \
        --metrics-out "$RESULTS_DIR/sanity/$TRIAL/metrics.json"
    done
    # Both diffs are reported even if one comparison fails.
    STATUS=0
    for TRIAL in eager_repeat sdpa; do
      run_python scripts/compare_gemma3_runs.py \
        --left "$RESULTS_DIR/sanity/eager/predictions.jsonl" \
        --right "$RESULTS_DIR/sanity/$TRIAL/predictions.jsonl" \
        --report "$RESULTS_DIR/sanity/compare_$TRIAL.json" || STATUS=1
    done
    exit "$STATUS"
    ;;
  stage2)
    CONDITION="${CONDITION:-image}"
    ARGS=(scripts/eval_gemma3_stage2.py "${COMMON_ARGS[@]}"
      --input-jsonl "$STAGE2_JSONL" --expected-count "${STAGE2_EXPECTED_COUNT:-500}"
      --condition "$CONDITION" --attn-impl "$ATTN_IMPL"
      --output-jsonl "$RESULTS_DIR/stage2/$CONDITION/predictions.jsonl" --resume)
    if [[ "$CONDITION" == image ]]; then
      ARGS+=(--preflight-report "$PREFLIGHT_REPORT")
    fi
    if [[ "$APPROVE_PROMPTS" == 1 ]]; then
      ARGS+=(--approve-prompt-diff)
    else
      ARGS+=(--preview-only)
    fi
    run_python "${ARGS[@]}"
    ;;
  *) echo 'PHASE must be preflight, sanity, stage1 or stage2' >&2; exit 1 ;;
esac
