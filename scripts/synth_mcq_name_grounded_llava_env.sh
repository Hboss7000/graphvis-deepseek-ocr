#!/usr/bin/env bash
# Shared settings for an already authorized pod; source this file.
set -euo pipefail
export WORKSPACE=${WORKSPACE:-/workspace}
export REPO=${REPO:-$WORKSPACE/bachelorArbeit}
export PY=${PY:-$WORKSPACE/venvs/venv_train/bin/python}
export DATASET_PATH=${DATASET_PATH:-$WORKSPACE/knowledge_graphs_50_name_grounded_100_questions.json}
export ASSETS=${ASSETS:-$WORKSPACE/outputs/synth_mcq_name_grounded_llava_assets}
export ADAPTER=${ADAPTER:-$WORKSPACE/outputs/llava_stage1/train/checkpoint-000450}
export AUDIT_GATE=${AUDIT_GATE:-$ASSETS/audit/legibility_gate.json}
export PAIRING=${PAIRING:-$ASSETS/wrong_graph_pairing.json}
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORKSPACE/.cache" TORCH_HOME="$WORKSPACE/.cache/torch"
export CUDA_CACHE_PATH="$WORKSPACE/.cache/cuda" TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WORKSPACE/.cache/torchinductor"
export TMPDIR="$WORKSPACE/.cache/tmp" MPLCONFIGDIR="$WORKSPACE/.cache/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset FULLRUN_CONTRACT STAGE1_MATRIX_CELL_JSON STAGE1_FROZEN_PROMPT_POLICY_JSON
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$WORKSPACE/logs"
cd "$REPO"
LLAVA_ARGS=(--backend llava --dataset-path "$DATASET_PATH" --assets "$ASSETS"
  --adapter "$ADAPTER" --image-audit "$AUDIT_GATE" --pairing "$PAIRING"
  --expected-render-checksum 2317d3b134e9390f987e4a233438617da8e119a0a288e242c21168e8689903e1
  --models base,tuned --conditions image,text,none,wrong_graph --wrong-graph-modality both
  --prompt-styles obqa --seed 0 --max-new-tokens 64 --resume)
