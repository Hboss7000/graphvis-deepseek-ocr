#!/usr/bin/env bash
# Prepared Pod commands only; no model execution happened while writing this file.
set -euo pipefail
REPO=${REPO:-/workspace/bachelorArbeit}
PY=${PY:-/workspace/venvs/venv_qwen/bin/python}
DATASET_PATH=${DATASET_PATH:-/workspace/knowledge_graphs_50_name_grounded_100_questions.json}
ASSETS=${ASSETS:-/workspace/outputs/synth_mcq_name_grounded_assets}
AUDIT_GATE=${AUDIT_GATE:-$ASSETS/audit/legibility_gate.json}
PROMPT_STYLES=${PROMPT_STYLES:-obqa}
: "${ADAPTER:?Set ADAPTER to the full checkpoint-000450 directory}"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=${OUT:-/workspace/outputs/synth_mcq_name_grounded_smoke}
LOG_NAME=synth_mcq_name_grounded_smoke.log
cd "$REPO"
"$PY" -c 'import json,sys; g=json.load(open(sys.argv[1])); assert g["passed"] and not g["nodes_below_10px"] and not g["edges_below_10px"], "STOP: failed 10 px image gate; user decision required"' "$AUDIT_GATE"
mkdir -p "$OUT" /workspace/logs
if [[ -e "$OUT/graphs" && ! -L "$OUT/graphs" ]]; then
  echo 'OUT/graphs must be a symlink to the audited assets; choose a new OUT' >&2
  exit 2
fi
ln -sfn "$ASSETS/graphs" "$OUT/graphs"
cp "$ASSETS/render_manifest.json" "$OUT/render_manifest.json"
ln -sfn "$OUT/run.log" "/workspace/logs/$LOG_NAME"
"$PY" -u scripts/run_synth_mcq.py \
  --dataset-path "$DATASET_PATH" --adapter "$ADAPTER" --out "$OUT" \
  --image-audit "$AUDIT_GATE" --models base,tuned --render-style training \
  --conditions image,text,none,wrong_graph --wrong-graph-modality both \
  --prompt-styles "$PROMPT_STYLES" --seed 0 --resume --limit 2 --n-perms 1 \
  2>&1 | tee "$OUT/run.log"
wc -l "$OUT/predictions.jsonl"
