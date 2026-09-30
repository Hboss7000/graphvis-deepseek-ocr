#!/usr/bin/env bash
#SBATCH --job-name=prunmat
#SBATCH --output=/storage/home/hleonel/outputs/slurm-prunmat-%A_%a.out
#SBATCH --error=/storage/home/hleonel/outputs/slurm-prunmat-%A_%a.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The launcher supplies --array=1-<manifest lines>%<concurrency> to sbatch.
# SBATCH directives cannot expand shell variables; no fixed matrix size here.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit via build_pruning_matrix.py --submit}"
MANIFEST="${1:?Manifest path is required}"
PROJECT_DIR="${2:?Project path is required}"
GEN_PYTHON="${3:?Absolute venv_gen python path is required}"
[[ "$GEN_PYTHON" = /* ]] || { echo "GEN_PYTHON must be absolute" >&2; exit 1; }
cd "$PROJECT_DIR"
export PYTHONHASHSEED=13
exec "$GEN_PYTHON" scripts/build_pruning_matrix.py \
  --manifest "$MANIFEST" --run-cell "$SLURM_ARRAY_TASK_ID"
