# LLaVA Stage 1 — CPU preparation, stop point 1

The requested scope is Stage 1 only. Do not launch training or evaluation at this
stop point. Henrique pushes the commits. Existing uncommitted work is excluded.

## Environment proposal (approval required before installation)

Verified in `env/freeze_venv_qwen.txt`: torch 2.8.0+cu128, torchvision
0.23.0+cu128, transformers 5.16.1, accelerate 1.15.0. PEFT and MLflow are absent.
Do not modify `venv_qwen`. Proposed separate Python 3.12 environment:
PEFT 0.21.2, accelerate 1.15.0, MLflow 2.22.2. These releases exist on PyPI;
compatibility with this exact model/library combination must be tested, not assumed.
MLflow 2.22.2 retains the requested file store. No server is used in training.

After explicit installation approval, on the pod:

```bash
export WORKSPACE=/workspace
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
test ! -e "$WORKSPACE/venvs/venv_train"
python3.12 -m venv "$WORKSPACE/venvs/venv_train"
"$WORKSPACE/venvs/venv_train/bin/python" -m pip install \
  --constraint "$WORKSPACE/bachelorArbeit/env/constraints_llava_train.txt" \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
"$WORKSPACE/venvs/venv_train/bin/python" -m pip install \
  --constraint "$WORKSPACE/bachelorArbeit/env/constraints_llava_train.txt" \
  --requirement "$WORKSPACE/bachelorArbeit/env/requirements_llava_train.txt"
"$WORKSPACE/venvs/venv_train/bin/python" -m pip check
"$WORKSPACE/venvs/venv_train/bin/python" -m pip freeze \
  > "$WORKSPACE/venvs/freeze_venv_train.txt"
```

If Python 3.12 or a required pinned release is unavailable, stop; do not substitute
another version. CPU tests need a separate **laptop** environment because the
existing laptop environments lack transformers/PEFT and use Python 3.14.
After explicit installation approval, on the laptop:

```bash
export CPU_ENV=/tmp/llava-stage1-cpu
test ! -e "$CPU_ENV"
/home/henrique-leonel/.pyenv/versions/3.12.7/bin/python -m venv "$CPU_ENV"
"$CPU_ENV/bin/python" -m pip install --constraint env/constraints_llava_train.txt \
  torch==2.8.0+cpu torchvision==0.23.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
"$CPU_ENV/bin/python" -m pip install --constraint env/constraints_llava_train.txt \
  transformers==5.16.1 peft==0.21.2 \
  accelerate==1.15.0 pytest==8.4.2 pillow==12.3.0
"$CPU_ENV/bin/python" -m pip check
```

CPU torch is explicitly a test-only build; it does not validate CUDA or bf16.
No dependency installation or pod operation is implied by these commands.

## Reproduce the CPU preparation

Rendering uses the existing laptop renderer and preserves every selected graph.
The sample is drawn from sorted **question IDs**, then rendered in statement-index
order: 1,200 training IDs, followed by 100 from the sorted remainder using the same
`random.Random(13)` instance. Train and validation IDs are checked against **all
500 test question IDs**, covering both the Stage 1 and QA test subsets. The
question-level graph is the evaluation pipeline's union of four answer-choice
graphs, an implementation interpretation already documented by that pipeline.

```bash
myvenv/bin/python scripts/prepare_llava_stage1_training.py --render-only \
  --out-dir outputs/llava_stage1_training_2026-10-06
```

After approval and installation of the isolated CPU environment:

```bash
export CPU_ENV=/tmp/llava-stage1-cpu
"$CPU_ENV/bin/python" scripts/prepare_llava_stage1_training.py --audit-only \
  --out-dir outputs/llava_stage1_training_2026-10-06
"$CPU_ENV/bin/python" -m pytest -q tests/llava_stage1_training_cpu_checks.py \
  tests/test_llava_stage1_training_data.py
```

The rendering command requires a fresh output directory. The audit is read-only
for data and writes diagnostics; it checks every data/image hash. Reviewable
manifests and summary distributions are copied to
`experiments/2026-10-06_llava_stage1_training/`. Images and per-example diagnostic
records remain under `outputs/`. Expanded lengths include anyres image tokens,
the exact `llava_v1` prompt, the gold answer and the final EOS. Zero truncations
is a hard requirement: the audit writes overlength diagnostics and fails if any
example exceeds 4096; it does not drop, shorten or resample records.

CPU tests use the pinned real tokenizer and an actual LlavaNextProcessor with
a tiny synthetic vision/image config and random model weights. They verify pad
masking even when pad equals EOS, image-token expansion against processor features,
answer-only loss, final EOS, overlength rejection, finite projector/LoRA gradients,
frozen vision weights, adapter/projector/optimizer/scheduler checkpoint restoration,
exact continuation with restored RNG, and rejection of resume drift/corruption.
Synthetic image sizes and float32 are test fixtures, not proposed run settings.
The heavyweight checks are invoked explicitly by the command above. Missing
training dependencies cause a collection failure, never a skipped successful run.

## Fixed experiment and source discrepancy

Model: `llava-hf/llava-v1.6-mistral-7b-hf`, revision
`2424fdd47412fccc66d91719126b420e9fbd7065`. Processor is inference's NeXT anyres
processor. Training uses the existing `llava_v1` system/user/assistant text,
one space before the gold answer, and final `</s>`. Only answer and final EOS
tokens contribute to loss. Maximum length is 4096; overlength examples fail.

[GraphVis's released script](https://github.com/yihedeng9/GraphVis/blob/main/scripts/finetune_lora.sh)
sets LoRA r=128, alpha=256, projector and language-model lr=2e-5,
one epoch, warmup .03, cosine schedule, weight decay 0, bf16 and checkpointing.
It uses per-device batch 16 on two GPUs, so its effective batch is 32. The
requested single-GPU global batch is **16**, an explicit replication decision.
The released script sets length 2048; its image handling is the original LLaVA
path. [The published paper, Table 5 (p.16)](https://proceedings.neurips.cc/paper_files/paper/2024/file/7cb04f510593c9ba30da398f5e0a7e7b-Paper-Conference.pdf)
independently confirms lr=1e-7, global batch 4 and image_aspect_ratio=pad,
differing from the released recipe. This experiment explicitly
uses NeXT anyres, length 4096, dropout .05 and all language-model linear layers
(excluding the output `lm_head`, matching GraphVis's target finder), with the
projector trained separately. The approved target-module scope comes from the
current request. The future trainer must record actual resolved module names.

The paper's Table 4 (p.10) also contains an inconsistency: edge number is printed
as **9.7 → 16.2 (+9.7)**. The request's **9.7 → 19.4** matches the printed gain,
but not the printed final score. Preserve both the literal table value and this
gain-derived value with a footnote in the future comparison; do not label 19.4
as the literal published score. These CSQA exact-match results are directional
references for this OBQA experiment, not directly comparable headline metrics.

## Pending beyond stop point 1

The full trainer, MLflow/CSV logger and plotter, GPU smoke, adapter evaluation
wiring, comparison report and measured launch/time/cost commands follow after
CPU preparation. CPU checks cannot prove real-weight bf16 behavior, CUDA memory,
GPU speed, learning quality or a readable MLflow training run. No paid run is
authorized. Full-run estimates require the real-model smoke measurements.
