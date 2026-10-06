#!/usr/bin/env bash
# Reviewable future setup; run only after separate pod/environment approval.
set -euo pipefail
export WORKSPACE=${WORKSPACE:-/workspace}
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORKSPACE/.cache" TORCH_HOME="$WORKSPACE/.cache/torch"
export CUDA_CACHE_PATH="$WORKSPACE/.cache/cuda" TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export TMPDIR="$WORKSPACE/.cache/tmp"
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=true
PROJECT="$WORKSPACE/bachelorArbeit"
TRAIN_ENV="$WORKSPACE/venvs/venv_train"
test -n "${TMUX:-}" || { echo 'Setup must run inside tmux.' >&2; exit 1; }
trap 'echo "SETUP FINISHED/FAILED: STOP the Pod after reviewing the log."' EXIT
exec 9>"$WORKSPACE/.fullrun_gpu.lock"
flock -n 9 || { echo 'Another GPU session holds the workspace lock.' >&2; exit 1; }
test ! -e "$TRAIN_ENV"
command -v python3.12 >/dev/null || { echo 'Python 3.12 unavailable; stop without substituting.' >&2; exit 1; }
mkdir -p "$WORKSPACE/venvs" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"
python3.12 -m venv "$TRAIN_ENV"
"$TRAIN_ENV/bin/python" -m pip install --constraint "$PROJECT/env/constraints_llava_train.txt" \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
"$TRAIN_ENV/bin/python" -m pip install --constraint "$PROJECT/env/constraints_llava_train.txt" \
  --requirement "$PROJECT/env/requirements_llava_train.txt"
"$TRAIN_ENV/bin/python" -m pip check
"$TRAIN_ENV/bin/python" -m pip freeze > "$WORKSPACE/venvs/freeze_venv_train.txt"
"$TRAIN_ENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for name in ('torch', 'transformers', 'peft', 'accelerate', 'mlflow'):
    print(name, __import__(name).__version__, flush=True)
snapshot_download('llava-hf/llava-v1.6-mistral-7b-hf',
                  revision='2424fdd47412fccc66d91719126b420e9fbd7065',
                  allow_patterns=['*.safetensors', '*.json', '*.model'])
PY
