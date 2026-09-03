#!/usr/bin/env bash
#SBATCH --job-name=download-qwen3vl-8b
#SBATCH --output=/storage/home/hleonel/outputs/slurm-%x-%j.out
#SBATCH --error=/storage/home/hleonel/outputs/slurm-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

CONTAINER_IMAGE="${HOME}/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif"
QWEN_HF_CLI="/storage/home/hleonel/venv_qwen/bin/huggingface-cli"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/nobackup/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
REQUESTED_REVISION="${MODEL_REVISION:-main}"

mkdir -p "$HF_CACHE_DIR"
export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"

SNAPSHOT_PATH="$(
  apptainer exec \
    --bind /storage/nobackup \
    "$CONTAINER_IMAGE" \
    "$QWEN_HF_CLI" download "$MODEL_ID" --revision "$REQUESTED_REVISION"
)"
RESOLVED_COMMIT="$(basename "$(readlink -f "$SNAPSHOT_PATH")")"

echo "Snapshot path: $SNAPSHOT_PATH"
echo "Resolved commit: $RESOLVED_COMMIT"
