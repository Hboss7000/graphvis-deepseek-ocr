#!/usr/bin/env bash
# Idempotent Runpod setup and one-image GPU smoke test.
# Expected image: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
# Run from the rsynced repository with a real rendered graph supplied as
# SMOKE_IMAGE=/workspace/<repo>/outputs/.../graph.png.
set -Eeuo pipefail

# The custom template snapshots Docker/Runpod variables here before sshd starts.
# Source it explicitly as well, so this script also works through `ssh host cmd`
# (a non-login shell does not read /etc/profile.d automatically).
if [[ -r /etc/profile.d/container_env.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/container_env.sh
fi

readonly WORKSPACE="/workspace"
readonly LOG_DIR="${WORKSPACE}/logs"
readonly LOG_FILE="${LOG_DIR}/setup.log"
readonly VENV_ROOT="${WORKSPACE}/venvs"
readonly HF_HOME="${WORKSPACE}/.cache/huggingface"
export HF_HOME

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OCR2_ENV="${VENV_ROOT}/venv_ocr2"
readonly QWEN_ENV="${VENV_ROOT}/venv_qwen"
readonly GEN_ENV="${VENV_ROOT}/venv_gen"
readonly SYSTEM_PYTHON="/opt/conda/bin/python"

readonly DEEPSEEK_ID="deepseek-ai/DeepSeek-OCR-2"
readonly DEEPSEEK_REV="aaa02f3811945a91062062994c5c4a3f4c0af2b0"
readonly QWEN_ID="Qwen/Qwen3-VL-8B-Instruct"
readonly QWEN_REV="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
readonly GEMMA_ID="google/gemma-3-12b-it"
readonly GEMMA_REV="96b6f1eccf38110c56df3a15bffe176da04bfd80"

mkdir -p "${LOG_DIR}" "${VENV_ROOT}" "${HF_HOME}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "[$(date -u +%FT%TZ)] ERROR line ${LINENO}: ${BASH_COMMAND} (exit ${status})"; exit "${status}"' ERR

if [[ ! -x "${SYSTEM_PYTHON}" ]]; then
  echo "ERROR: expected image Python at ${SYSTEM_PYTHON}." >&2
  exit 1
fi
readonly IMAGE_TORCH_VERSION="$("${SYSTEM_PYTHON}" -c 'import torch; print(torch.__version__)')"

echo
echo "[$(date -u +%FT%TZ)] Runpod setup starting"
echo "repository=${REPO_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "SYSTEM_PYTHON=${SYSTEM_PYTHON}"
echo "HF_TOKEN=${HF_TOKEN:+set}"
"${SYSTEM_PYTHON}" --version
"${SYSTEM_PYTHON}" - <<'PY'
import torch
print(f"torch.__version__={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
major, minor = torch.cuda.get_device_capability()
running_arch = f"sm_{major}{minor}"
compiled_arches = torch.cuda.get_arch_list()
print(f"torch.cuda.device={torch.cuda.get_device_name()}")
print(f"torch.cuda.running_arch={running_arch}")
print(f"torch.cuda.get_arch_list()={compiled_arches}")
if running_arch not in compiled_arches:
    raise SystemExit(
        f"Running GPU architecture {running_arch} is absent from this Torch build: "
        f"{compiled_arches}"
    )
PY

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: run as root so apt can install the setup system packages." >&2
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git graphviz rsync tmux unzip wget
rsync --version | sed -n '1p'
tmux -V
dot -V

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be present in this shell; do not store it in the repo." >&2
  exit 1
fi
if [[ -z "${SMOKE_IMAGE:-}" || ! -f "${SMOKE_IMAGE}" ]]; then
  echo "ERROR: set SMOKE_IMAGE to one real, rsynced rendered graph image." >&2
  exit 1
fi
case "${SMOKE_IMAGE}" in
  "${WORKSPACE}"/*) ;;
  *) echo "ERROR: SMOKE_IMAGE must be under ${WORKSPACE}: ${SMOKE_IMAGE}" >&2; exit 1 ;;
esac
echo "smoke_image=${SMOKE_IMAGE}"
stat --format='smoke_image_bytes=%s' "${SMOKE_IMAGE}"

ensure_venv() {
  local env_path="$1"
  if [[ ! -x "${env_path}/bin/python" ]]; then
    "${SYSTEM_PYTHON}" -m venv --system-site-packages "${env_path}"
  fi
  if ! grep -q '^include-system-site-packages = true$' "${env_path}/pyvenv.cfg"; then
    echo "ERROR: existing ${env_path} was not built with --system-site-packages." >&2
    exit 1
  fi
  "${env_path}/bin/python" -m pip install --upgrade pip
}
ensure_venv "${OCR2_ENV}"
ensure_venv "${QWEN_ENV}"

ensure_gen_venv() {
  if [[ "$("${SYSTEM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.11" ]]; then
    echo "ERROR: requirements_render_py311.txt requires /opt/conda Python 3.11." >&2
    exit 1
  fi
  if [[ ! -x "${GEN_ENV}/bin/python" ]]; then
    "${SYSTEM_PYTHON}" -m venv "${GEN_ENV}"
  fi
  if ! grep -q '^include-system-site-packages = false$' "${GEN_ENV}/pyvenv.cfg"; then
    echo "ERROR: existing ${GEN_ENV} is not an isolated render venv." >&2
    exit 1
  fi
  "${GEN_ENV}/bin/python" -m pip install --upgrade pip
  "${GEN_ENV}/bin/python" -m pip install --upgrade \
    --requirement "${REPO_ROOT}/requirements_render_py311.txt"
  "${GEN_ENV}/bin/python" - <<'PY'
import graphviz
import numpy
import scipy
print(f"render env: graphviz={graphviz.__version__} numpy={numpy.__version__} scipy={scipy.__version__}")
PY
  "${GEN_ENV}/bin/python" -m pip freeze > "${LOG_DIR}/freeze_venv_gen.txt"
  echo "Wrote ${LOG_DIR}/freeze_venv_gen.txt"
}
ensure_gen_venv

# Transformers does not install Torch. The remaining names cover direct imports
# in the repo and DeepSeek-OCR-2's pinned remote code; Torch/torchvision stay
# inherited from the image through --system-site-packages.
"${OCR2_ENV}/bin/python" -m pip install --upgrade \
  "transformers==4.46.3" pillow packaging numpy requests tqdm \
  einops addict accelerate sentencepiece safetensors
"${QWEN_ENV}/bin/python" -m pip install --upgrade \
  "transformers==5.16.1" pillow packaging accelerate \
  sentencepiece safetensors

verify_env() {
  local env_path="$1"
  local expected_transformers="$2"
  "${env_path}/bin/python" - "${expected_transformers}" "${IMAGE_TORCH_VERSION}" <<'PY'
import sys
import torch
import transformers
expected_transformers, expected_torch = sys.argv[1:]
print(f"env={sys.prefix}")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}; cuda={torch.version.cuda}; transformers={transformers.__version__}")
if transformers.__version__ != expected_transformers:
    raise SystemExit(f"Expected transformers {expected_transformers}, found {transformers.__version__}")
if torch.__version__ != expected_torch:
    raise SystemExit(f"Torch changed: image={expected_torch}, venv={torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the venv")
PY
}
verify_env "${OCR2_ENV}" "4.46.3"
verify_env "${QWEN_ENV}" "5.16.1"

# snapshot_download is idempotent: completed pinned snapshots are reused.
"${QWEN_ENV}/bin/python" - \
  "${DEEPSEEK_ID}" "${DEEPSEEK_REV}" \
  "${QWEN_ID}" "${QWEN_REV}" \
  "${GEMMA_ID}" "${GEMMA_REV}" <<'PY'
import os
import re
import sys
from huggingface_hub import snapshot_download

for model_id, revision in zip(sys.argv[1::2], sys.argv[2::2]):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"Refusing non-commit revision for {model_id}: {revision!r}")
    path = snapshot_download(model_id, revision=revision, token=os.environ["HF_TOKEN"])
    print(f"cached {model_id}@{revision} at {path}")
PY

echo "[$(date -u +%FT%TZ)] DeepSeek-OCR-2 smoke inference"
"${OCR2_ENV}/bin/python" - "${SMOKE_IMAGE}" "${LOG_DIR}/smoke_deepseek" \
  "${DEEPSEEK_ID}" "${DEEPSEEK_REV}" <<'PY'
import pathlib
import sys
import torch
from transformers import AutoModel, AutoTokenizer

image_path, output_dir, model_id, revision = sys.argv[1:]
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=revision, trust_remote_code=True, local_files_only=True
)
model = AutoModel.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
    use_safetensors=True,
).eval().cuda().to(torch.bfloat16)
model.generation_config.do_sample = False
model.generation_config.num_beams = 1
with torch.inference_mode():
    result = model.infer(
        tokenizer,
        prompt="<image>\nList the visible graph nodes briefly.",
        image_file=image_path,
        output_path=output_dir,
        save_results=False,
        eval_mode=True,
        base_size=1024,
        image_size=768,
        crop_mode=True,
    )
print(f"DeepSeek-OCR-2 result: {result!r}")
PY

echo "[$(date -u +%FT%TZ)] Qwen3-VL smoke inference"
"${QWEN_ENV}/bin/python" - "${SMOKE_IMAGE}" "${REPO_ROOT}" \
  "${QWEN_ID}" "${QWEN_REV}" <<'PY'
import gc
import sys
from argparse import Namespace
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

image_path, repo_root, model_id, revision = sys.argv[1:]
sys.path.insert(0, str(Path(repo_root) / "experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts"))
import run_zero_shot_qwen as qwen

args = Namespace(
    model_id=model_id, revision=revision,
    min_pixels=262144, max_pixels=1310720, max_new_tokens=16,
)
processor, _ = qwen.load_processor(AutoProcessor, args)
model, _, _ = qwen.load_model(Qwen3VLForConditionalGeneration, torch, args)
image = Image.open(image_path).convert("RGB")
body = "<image>\nList the visible graph nodes briefly."
prompt = qwen.format_qwen_prompt(processor, body, "image")
result = qwen.infer_one(model, processor, prompt, image, args, torch)
print(f"Qwen3-VL result: {result!r}")
del model, processor
gc.collect()
torch.cuda.empty_cache()
PY

echo "[$(date -u +%FT%TZ)] Gemma 3 smoke inference"
"${QWEN_ENV}/bin/python" - "${SMOKE_IMAGE}" "${REPO_ROOT}" \
  "${GEMMA_ID}" "${GEMMA_REV}" <<'PY'
import sys
from argparse import Namespace
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

image_path, repo_root, model_id, revision = sys.argv[1:]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import gemma3_common as gemma

args = Namespace(
    model_id=model_id,
    revision=revision,
    seed=13,
    attn_impl="eager",
    pan_and_scan=True,
    pan_and_scan_min_crop_size=256,
    pan_and_scan_max_num_crops=4,
    pan_and_scan_min_ratio_to_activate=1.2,
    cache_implementation="dynamic",
    max_new_tokens=16,
)
processor, details = gemma.load_processor(AutoProcessor, args)
model, _, _ = gemma.load_model(Gemma3ForConditionalGeneration, torch, args)
image = Image.open(image_path).convert("RGB")
body = "<image>\nList the visible graph nodes briefly."
result, tokens, hit_limit, image_views, image_soft_tokens = gemma.infer_one(
    model, processor, body, "image", image, args, torch, details
)
print(f"Gemma 3 result: {result!r}; generated_tokens={tokens}; hit_limit={hit_limit}; "
      f"image_views={image_views}; image_soft_tokens={image_soft_tokens}")
PY

"${OCR2_ENV}/bin/python" -m pip freeze > "${LOG_DIR}/freeze_venv_ocr2.txt"
"${QWEN_ENV}/bin/python" -m pip freeze > "${LOG_DIR}/freeze_venv_qwen.txt"
echo "Wrote ${LOG_DIR}/freeze_venv_ocr2.txt"
echo "Wrote ${LOG_DIR}/freeze_venv_qwen.txt"

echo "[$(date -u +%FT%TZ)] Runpod setup and all three smoke inferences completed"
