#!/usr/bin/env bash
# Approved environment setup; run only on a separately authorized pod.
set -euo pipefail
export WORKSPACE=${WORKSPACE:-/workspace}
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORKSPACE/.cache" TORCH_HOME="$WORKSPACE/.cache/torch"
export CUDA_CACHE_PATH="$WORKSPACE/.cache/cuda" TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
export PIP_NO_CACHE_DIR=1
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= PIP_FIND_LINKS=
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=true
PROJECT="$WORKSPACE/bachelorArbeit"
TRAIN_ENV="$WORKSPACE/venvs/venv_train"
test -n "${TMUX:-}" || { echo 'Setup must run inside tmux.' >&2; exit 1; }
trap 'df -h "$WORKSPACE"; du -sh "$WORKSPACE"; echo "SETUP FINISHED/FAILED: STOP the Pod after reviewing the log."' EXIT
df -h "$WORKSPACE"
du -sh "$WORKSPACE"
echo "Configured volume cap: $VOLUME_CAP_GB GB"
exec 9>>"$WORKSPACE/.fullrun_gpu.lock"
flock -n 9 || { echo 'Another GPU session holds the workspace lock.' >&2; exit 1; }
test ! -e "$TRAIN_ENV"
SYSTEM_PYTHON=$(bash "$PROJECT/scripts/stage1_train_interpreter.sh" "$WORKSPACE")
"$SYSTEM_PYTHON" -c 'import sys; print("Selected interpreter:",sys.executable); print("Python:",sys.version)'
if "$SYSTEM_PYTHON" "$PROJECT/scripts/stage1_weight_cache.py" --cache "$HF_HOME" --backbone llava; then
  echo 'Pinned LLaVA cache is complete: skipping all weight downloads; cache remains untouched.'
else
  echo 'Pinned cache is incomplete: setup will not download or overwrite weights; offline jobs must wait for separately authorized caching.'
fi
echo 'Approved sources: official PyTorch cu128 index for torch/torchvision only; all dependencies and other packages from PyPI. Pins unchanged.'
test ! -e "$WORKSPACE/venvs/freeze_venv_train.txt"
mkdir -p "$WORKSPACE/venvs"
"$SYSTEM_PYTHON" -m venv "$TRAIN_ENV"
export TMPDIR="$TRAIN_ENV/.install_tmp"
mkdir "$TMPDIR"
"$TRAIN_ENV/bin/python" -m pip install --no-deps --no-cache-dir --constraint "$PROJECT/env/constraints_llava_train.txt" \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$TRAIN_ENV/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  --constraint "$PROJECT/env/constraints_llava_train.txt" \
  --requirement "$PROJECT/env/requirements_llava_train.txt"
"$TRAIN_ENV/bin/python" -m pip check
{ "$TRAIN_ENV/bin/python" -c 'import sys; print("# interpreter="+sys.executable); print("# base_interpreter="+sys._base_executable); print("# python="+sys.version.replace("\n"," "))';
  "$TRAIN_ENV/bin/python" -m pip freeze; } > "$WORKSPACE/venvs/freeze_venv_train.txt"
"$TRAIN_ENV/bin/python" - <<'PY'
for name in ('torch', 'transformers', 'peft', 'accelerate', 'mlflow'):
    print(name, __import__(name).__version__, flush=True)
PY
