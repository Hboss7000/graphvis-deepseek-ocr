# LLaVA Stage 1 — verified CPU preparation and pod rehearsal

The requested scope is Stage 1 only. Stop point 1 is complete. The trainer, smoke
mode, evaluation wiring and commands are prepared; no real-model GPU job has
been run. Henrique pushes the commits. Existing uncommitted work is excluded.

## Environment proposal (approval required before installation)

Verified in `env/freeze_venv_qwen.txt`: torch 2.8.0+cu128, torchvision
0.23.0+cu128, transformers 5.16.1, accelerate 1.15.0. PEFT and MLflow are absent.
Do not modify `venv_qwen`. Proposed separate Python 3.12 environment:
PEFT 0.21.2, accelerate 1.15.0, MLflow 3.16.1. The initially proposed MLflow 2.22.2
was rejected by the resolver because it requires packaging<25 while COMA pins
packaging==26.3. No COMA dependency was downgraded. MLflow 3.16.1 supports that
pin and the requested file store with `MLFLOW_ALLOW_FILE_STORE=true`.
[Pinned MLflow file-store source](https://github.com/mlflow/mlflow/blob/v3.16.1/mlflow/store/tracking/file_store.py).
CPU tests verified PEFT compatibility, file-store writes/reads, and no network
connections during logging. Training uses no MLflow server or model artifacts.

The setup selector prefers usable `python3.12`; otherwise it uses
`readlink -f /workspace/venvs/venv_qwen/bin/python`. The existing setup log verifies
`/opt/conda/bin/python` at 3.11.13 as the original pod interpreter. No apt,
deadsnakes or Python installation is performed. Stop only if neither interpreter
exists or a pinned dependency cannot install. The selected executable and Python
version are recorded in the setup log and `freeze_venv_train.txt`.

Read-only pre-check, on an existing authorized pod:

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export SYSTEM_PYTHON=$(bash "$PROJECT/scripts/stage1_train_interpreter.sh" "$WORKSPACE")
"$SYSTEM_PYTHON" -c 'import sys; print("Selected interpreter:",sys.executable); print("Python:",sys.version)'
df -h "$WORKSPACE"
```

The exact installation commands are in `llava_stage1_pod_setup.sh`, launched in
tmux below. Environment installation does not download or overwrite weights.
The last installation approval said **PyPI only**. PyPI does not publish the exact
`torch==2.8.0+cu128` / `torchvision==0.23.0+cu128` requirement versions, so setup
refuses before creating the venv unless a source exception for the official CUDA
index is separately granted (`ALLOW_PYTORCH_CUDA_INDEX=yes`). Other packages use
PyPI, pinned constraints remain unchanged, and no other installs occur. No pip
cache is written; installation temporary files stay inside the new venv.

The approved laptop environment was installed separately in `.venv_llava_cpu`:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
test ! -e "$CPU_ENV"
/home/henrique-leonel/.pyenv/versions/3.12.7/bin/python -m venv "$CPU_ENV"
"$CPU_ENV/bin/python" -m pip install --constraint env/constraints_llava_train.txt \
  torch==2.8.0+cpu torchvision==0.23.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
"$CPU_ENV/bin/python" -m pip install --constraint env/constraints_llava_train.txt \
  --requirement env/requirements_llava_cpu.txt
"$CPU_ENV/bin/python" -m pip check
```

CPU torch is explicitly a test-only build; it does not validate CUDA or bf16.
Installed versions (printed and import-checked; `pip check` passed):

| torch | transformers | peft | accelerate | mlflow |
|---|---|---|---|---|
| 2.8.0+cpu | 5.16.1 | 0.21.2 | 1.15.0 | 3.16.1 |

`torch.version.cuda is None`. Full freeze: `env/freeze_venv_llava_cpu.txt`.
No system packages, sudo, other venv changes or model-weight downloads occurred.
The pinned tokenizer/processor/config were already cached and loaded offline.

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

Repeat the verified CPU checks:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CPU_ENV/bin/python" scripts/prepare_llava_stage1_training.py --audit-only \
  --out-dir outputs/llava_stage1_training_2026-10-06
"$CPU_ENV/bin/python" -m pytest -q tests/llava_stage1_training_cpu_checks.py \
  tests/test_llava_stage1_training_data.py tests/test_llava_stage1_diagnostics.py \
  tests/test_llava_runners.py tests/test_llava_stage1_execution.py
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

Token audit: 7,200 training records, 600 validation records; zero truncations and
zero examples above 4096. The complete per-task distributions and example prompts
are in `experiments/2026-10-06_llava_stage1_training/`.

| Split | Total tokens min / median / p95 / max | Answer+EOS min / median / p95 / max | Above 2048 |
|---|---|---|---:|
| Train | 1804 / 2244 / 2892 / 3234 | 10 / 17.5 / 256 / 306 | 5536 / 7200 |
| Validation | 1804 / 2247.5 / 2740.7 / 3166 | 10 / 17 / 250.15 / 293 | 489 / 600 |

55 CPU/regression checks passed, including the production loop,
exact optimizer/scheduler/RNG continuation, adapter+projector inference reload
with identical logits, readable file store with sockets forbidden, and CSV
continuation after simulated MLflow failure. These checks use tiny random weights,
synthetic small vision settings and float32. They do not prove real-weight bf16,
CUDA capacity, GPU speed, learning quality or tuned accuracy.

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
current request. The trainer records every resolved module name.

The paper's Table 4 (p.10) also contains an inconsistency: edge number is printed
as **9.7 → 16.2 (+9.7)**. The request's **9.7 → 19.4** matches the printed gain,
but not the printed final score. Preserve both the literal table value and this
gain-derived value with a footnote in the future comparison; do not label 19.4
as the literal published score. These CSQA exact-match results are directional
references for this OBQA experiment, not directly comparable headline metrics.

## Implemented training and evaluation contract

`scripts/train_llava_stage1.py` implements HF+PEFT training with separate LoRA and
projector learning-rate flags, global batch 16, a mean microbatch-loss accumulation,
AdamW (.9/.999, epsilon 1e-8), gradient norm clipping at 1 (the released Trainer
default), .03 warmup and cosine scheduling. bf16/TF32, frozen vision, checkpointing
and the fixed 4096 limit are recorded. No quantization is used. LoRA applies to
the language decoder's linear layers, excluding lm_head as in GraphVis; the
projector trains at full rank. Actual resolved names are in run_config.json.

Full training is one shuffled epoch, 450 optimizer steps. Dry mode performs two
steps on the eight longest records; smoke performs 20 steps on 64 seed-13 records,
cycling them to preserve global batch 16. These modes use the same image loader,
anyres processor, vision encoder, collator, optimizer and loop as full training.
The smoke's validation slice is 64 held-out records; full validation uses all 600.
Validation runs every 100 steps and at the final step, including per-task loss.

Atomic checkpoints contain adapter/, projector.pt, optimizer/scheduler/RNG/cursor,
run_config and verified file hashes. They are saved every 100 steps, at stops and
after at most 20 minutes between completed steps. Save before validation; a
validation crash retains the update and replays missing validation on resume.
`--resume CHECKPOINT_ROOT`
restores the exact continuation and rejects recipe/code/data drift. Failed writes
never publish an incomplete checkpoint. Keep the latest **two intact checkpoints
plus the final one**. Prune this run's older saves only after the newly published
checkpoint verifies; preserve the previous resume point until then. Before every
save, stop cleanly with exit 75 if free space is below **2 × the conservative
checkpoint size estimate**, keeping the last intact save. Each save logs tensor
sizes/dtypes and disk budget; the pod worker logs `df -h /workspace` at start/end.
Checkpoints/caches/logs stay in /workspace.

The pinned LLaVA architecture gives 335,544,320 LoRA parameters (PEFT float32)
and 20,979,712 projector parameters (bf16). AdamW has two moment tensors at each
parameter's dtype. The tensor payload of one complete checkpoint is approximately
**4.15 GB / 3.87 GiB**, excluding small serialization/config/RNG overhead. The
save guard adds 5% plus overhead: roughly **8.12 GiB free** is required before a
save. Retention temporarily holds three saves while the new one verifies:
approximately **12.5 GB raw**, or **13.1 GB budgeted**, then two remain. These are
architecture/dtype estimates; the real run logs actual tensor counts and file sizes.

With the reported 100 GB volume / 70 GB used, cached weights already included,
reserve about 0.65 GB for the new rendered training data and 0.25 GB for evaluation,
metrics and logs. Expected peak is approximately **84 GB + the new training venv
size** (about 90 GB if the venv adds 6 GB). This is a planning example, not a
measured free-space claim. Installation scratch and other active jobs can add
usage; the pre-check and per-save guard use actual disk free space. Evaluation
reuses the retained adapter and base snapshot without creating base-weight copies.

Mandatory metrics.csv logs loss, both rates, gradient norm, actual tokens/s,
examples/s, step time and `torch.cuda.max_memory_allocated`. MLflow calls are
wrapped; CSV is always flushed/fsynced. A resumed run keeps its MLflow run ID.
Only checkpoint path/SHA tags are logged, never weights. Long resolved module
lists are logged by SHA with their complete value in run_config.json. A failed
MLflow store does not interrupt training, but fails the smoke verification.

Both inference runners accept `--adapter CHECKPOINT_ROOT` and record the adapter
path and verified adapter/projector SHA. The default baseline path/config remains
unchanged. Stage 1 uses llava_v1 without prefix as primary and gold-template prefix
as secondary, all 100 graphs × nine tasks. QA uses llava_v1 for all four conditions
at strict-64 with the existing shared parser. Existing frozen zero-shot fullrun
contracts reject adapters; tuned outputs have separate directories.

`compare_llava_stage1_training.py` re-scores main/P2/P3/tuned arms using the existing
span_extended scorer and preserves every metric in JSON/CSV, including the three
transfer tasks. It also compares 500 × four QA conditions and writes the paper
reference table. A missing arm remains explicitly missing. On the laptop, main
and full P3 are available; full P2 and tuned outputs are not available yet.

## Pod prerequisites and costs

These are **pod-local commands for a separately authorized pod and training
venv**. Today's approval covers the laptop environment only. No pod was created,
started, changed or stopped, and no pod installation or real GPU smoke was run.
Before provisioning, list running pods and flag idle ones; obtain an explicit
current-message “yes” with RTX PRO 6000 96GB, actual data center, Secure/Community
tier, live $/hour, expected runtime and total cost. No H100, multi-GPU, Serverless
or Flash. The requested $2.09/hour is the extrapolation rate, not a verified live
quote. No data center/tier is assumed here.

Setup requires the actual **4-hour automatic pod termination guard**. Training,
smoke and evaluation require their measured expected runtime +50% termination
guard in the approval proposal. The shell `timeout` below stops the job; **it does
not stop pod billing or implement Runpod's termination guard**. Every completion,
pause or failure prints a STOP reminder. Pod termination remains confirmation-only.

Put the pushed checkout at `/workspace/bachelorArbeit`. Upload the laptop-generated
`outputs/llava_stage1_training_2026-10-06/`, `outputs/llava_stage1_eval_inputs/`,
and frozen `outputs/fullrun_2026-10-04_B/` to the same relative paths in that
checkout. These output directories are ignored by git. Use the authorized pod's
actual SSH/volume connection; no host has been supplied or guessed. Training and
evaluation verify the manifests before starting. Graphviz never runs on the pod.

The isolated installation commands in the first section are also packaged in
`scripts/llava_stage1_pod_setup.sh`; it checks the pinned cache read-only.
A complete snapshot is reused and all weight downloads are skipped. An incomplete
snapshot is reported without downloading or overwriting the cache; offline jobs
must wait for a separately authorized cache fill. No model cache is altered. Invoke it **only after separate GPU-pod and environment approval**:

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export LOG="$WORKSPACE/logs/llava_stage1_setup_$(date -u +%Y%m%dT%H%M%SZ).log"
export ALLOW_PYTORCH_CUDA_INDEX=${ALLOW_PYTORCH_CUDA_INDEX:-no}
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s llava-setup "env WORKSPACE='$WORKSPACE' ALLOW_PYTORCH_CUDA_INDEX='$ALLOW_PYTORCH_CUDA_INDEX' timeout --signal=INT --kill-after=120s 14400 bash '$PROJECT/scripts/llava_stage1_pod_setup.sh' >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

If the pinned dependency resolver fails, stop without changing pinned versions.
All following jobs are offline, single GPU, protected by the existing workspace
GPU lock, and run in `/workspace/venvs/venv_train`. Cache and temporary directories
are on the volume, including Hugging Face, pip, torch, CUDA and Triton caches.
The setup script does not touch venv_qwen or venv_ocr2.

## Today's probe and smoke commands

Laptop → pod upload: sizes first, then pull Henrique's pushed commits and upload
the two new input directories. This block prompts for the real IP/port; it does
not create/start a pod. `--ignore-existing` preserves existing data files; any
incorrect old file produces a manifest failure instead of an overwrite.

```bash
export LOCAL_ROOT="$PWD"
export REMOTE_ROOT=/workspace/bachelorArbeit
read -rp 'Existing pod IP: ' POD_IP
read -rp 'SSH port: ' POD_PORT
read -rp 'SSH private key path: ' POD_KEY
export RSYNC_RSH="ssh -p $POD_PORT -i \"$POD_KEY\""
du -sh "$LOCAL_ROOT/outputs/llava_stage1_training_2026-10-06" "$LOCAL_ROOT/outputs/llava_stage1_eval_inputs"
ssh -p "$POD_PORT" -i "$POD_KEY" "root@$POD_IP" 'cd /workspace/bachelorArbeit && git pull --ff-only && df -h /workspace'
rsync -a --checksum --ignore-existing -e "$RSYNC_RSH" "$LOCAL_ROOT/outputs/llava_stage1_training_2026-10-06/" "root@$POD_IP:$REMOTE_ROOT/outputs/llava_stage1_training_2026-10-06/"
rsync -a --checksum --ignore-existing -e "$RSYNC_RSH" "$LOCAL_ROOT/outputs/llava_stage1_eval_inputs/" "root@$POD_IP:$REMOTE_ROOT/outputs/llava_stage1_eval_inputs/"
```

After each laptop plan calculation, copy the measured plan back with this separate,
self-contained block. Preserve any preceding plan as a timestamped backup:

```bash
export LOCAL_PLAN="$PWD/outputs/llava_stage1_fetched/execution_plan.json"
export REMOTE_PLAN=/workspace/outputs/llava_stage1/execution_plan.json
read -rp 'Existing pod IP: ' POD_IP
read -rp 'SSH port: ' POD_PORT
read -rp 'SSH private key path: ' POD_KEY
export RSYNC_RSH="ssh -p $POD_PORT -i \"$POD_KEY\""
test -f "$LOCAL_PLAN"
rsync -a --backup --suffix=".previous.$(date -u +%Y%m%dT%H%M%SZ)" -e "$RSYNC_RSH" "$LOCAL_PLAN" "root@$POD_IP:$REMOTE_PLAN"
```

Fetch reports, predictions and MLflow after a job; optimizer checkpoint files stay
on the volume. This block works for probe/smoke/training/evaluation and preserves
the expected laptop paths used by the plan/plot/comparison commands:

```bash
export FETCHED="$PWD/outputs/llava_stage1_fetched"
read -rp 'Existing pod IP: ' POD_IP
read -rp 'SSH port: ' POD_PORT
read -rp 'SSH private key path: ' POD_KEY
export RSYNC_RSH="ssh -p $POD_PORT -i \"$POD_KEY\""
mkdir -p "$FETCHED/mlruns"
rsync -a --exclude='checkpoint-*/' -e "$RSYNC_RSH" "root@$POD_IP:/workspace/outputs/llava_stage1/" "$FETCHED/"
rsync -a -e "$RSYNC_RSH" "root@$POD_IP:/workspace/mlruns/" "$FETCHED/mlruns/"
```

First measure capacity on the **eight longest audited training records**, two
optimizer steps, with the complete real model/image/vision/collator path. Probe
16, 8, 4, 2, then 1 and select the largest passing microbatch; only CUDA OOM
permits a smaller retry. A code/dependency/data error stops immediately. Global
batch stays 16. No full run launches after a failure or fix.

No real-model speed exists yet. The initial probe has a **one-hour job ceiling**
($2.09 at the requested rate), not a predicted runtime. Failed attempts and model
loading are recorded. The setup pod's actual 4-hour guard remains separate.

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export RUN_TAG=llava_stage1
export MAX_SECONDS=3600
export LOG="$WORKSPACE/logs/${RUN_TAG}_probe.log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s llava-probe "env WORKSPACE='$WORKSPACE' RUN_TAG='$RUN_TAG' MAX_SECONDS='$MAX_SECONDS' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' probe >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Fetch `/workspace/outputs/llava_stage1/probe/` to laptop
`outputs/llava_stage1_fetched/probe/`. Calculate the smoke runtime/cost/50% margin
**on the laptop** from the passing real-GPU report:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export FETCHED="$PWD/outputs/llava_stage1_fetched"
"$CPU_ENV/bin/python" scripts/llava_stage1_execution_plan.py \
  --probe-report "$FETCHED/probe/batch_probe_report.json" \
  --output "$FETCHED/execution_plan.json"
```

Review that report before the paid smoke and copy it back to
`/workspace/outputs/llava_stage1/execution_plan.json`. CPU timings cannot satisfy
this gate. The report states expected hours, total cost and expected +50% guard.
Once the separately approved pod/guard is ready, this **single tmux launch** runs
steps 1–10, saves a checkpoint, exits, starts a fresh process and resumes to 20.
The outer timeout bounds both processes together. It uses 64 real examples,
checks adapter+projector reload through the evaluation loader, requires identical
greedy probe IDs, and verifies the file store is readable with one resumed run ID.

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export RUN_TAG=llava_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["smoke"]["guard_hours"]*3600))' "$PLAN")
export LOG="$WORKSPACE/logs/${RUN_TAG}_smoke.log"
mkdir -p "$WORKSPACE/logs"
cat "$PLAN"
tmux new-session -d -s llava-smoke "env WORKSPACE='$WORKSPACE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' timeout --signal=INT --kill-after=120s $MAX_SECONDS bash -c 'bash \"$PROJECT/scripts/llava_stage1_pod_job.sh\" smoke-first && bash \"$PROJECT/scripts/llava_stage1_pod_job.sh\" smoke-resume' >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Stop point 2: inspect `/workspace/outputs/llava_stage1/smoke/smoke_report.json`
and the log, report measured examples/s, peak VRAM, save/resume, artifact SHA and
MLflow status, and **STOP the pod** after review. Do not launch training automatically.
A failed smoke must be fixed and followed by a new dry run and approval.

For continuous monitoring, use separate terminals; each command is last in its block:

```bash
export LOG=/workspace/logs/llava_stage1_smoke.log
tail -f "$LOG"
```

```bash
watch -n 30 nvidia-smi
```

## Training and adapter evaluation after the smoke

Fetch `smoke/`, `probe/` and `/workspace/mlruns/` to the corresponding laptop
`outputs/llava_stage1_fetched/` subdirectories. Recalculate the training estimate:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export FETCHED="$PWD/outputs/llava_stage1_fetched"
"$CPU_ENV/bin/python" scripts/llava_stage1_execution_plan.py \
  --probe-report "$FETCHED/probe/batch_probe_report.json" \
  --smoke-report "$FETCHED/smoke/smoke_report.json" \
  --output "$FETCHED/execution_plan.json"
```

The smoke report separates train-step compute from validation/checkpoint/loading
cost. The plan adds five full validations, checkpoint cost and measured load
allowance. This remains an extrapolation; time-triggered saves and variable
lengths may add overhead. Upload the plan and obtain the required separate full-run
approval with its expected +50% pod termination guard, then launch:

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export RUN_TAG=llava_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["train"]["guard_hours"]*3600))' "$PLAN")
export LOG="$WORKSPACE/logs/${RUN_TAG}_train.log"
unset RESUME_CHECKPOINT
mkdir -p "$WORKSPACE/logs"
cat "$PLAN"
tmux new-session -d -s llava-train "env WORKSPACE='$WORKSPACE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' RESUME_CHECKPOINT='${RESUME_CHECKPOINT:-}' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' train >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

To resume after a separately authorized restart, use the same block with
`export RESUME_CHECKPOINT=/workspace/outputs/llava_stage1/train/checkpoint-000100`
instead of `unset RESUME_CHECKPOINT`, using the actual latest intact checkpoint.
Do not change code, data or flags between save/resume. The alternative 2e-4 recipe
requires a fresh run tag and corresponding new dry/smoke evidence.

Before full evaluation, rehearse **the same adapter** through both Stage 1 arms
(one frozen graph × nine tasks) and QA's four conditions (eight examples each).
For today's smoke artifact use `EVAL_TAG=smoke` and checkpoint-000020. After full
training use `EVAL_TAG=trained` and checkpoint-000450; its dry run must be repeated
for that new adapter. This initial evaluation rehearsal uses the one-hour ceiling;
its measured report supplies the full evaluation estimate.

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export RUN_TAG=llava_stage1
export EVAL_TAG=smoke
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/smoke/checkpoint-000020"
export MAX_SECONDS=3600
export LOG="$WORKSPACE/logs/${RUN_TAG}_${EVAL_TAG}_eval_dry.log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s llava-eval-dry "env WORKSPACE='$WORKSPACE' RUN_TAG='$RUN_TAG' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' MAX_SECONDS='$MAX_SECONDS' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval-dry >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Fetch `eval_trained_dry/` after repeating that block for the trained checkpoint,
then compute the full evaluation estimate on the laptop:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export FETCHED="$PWD/outputs/llava_stage1_fetched"
"$CPU_ENV/bin/python" scripts/llava_stage1_execution_plan.py \
  --probe-report "$FETCHED/probe/batch_probe_report.json" \
  --smoke-report "$FETCHED/smoke/smoke_report.json" \
  --evaluation-dry-report "$FETCHED/eval_trained_dry/evaluation_report.json" \
  --output "$FETCHED/execution_plan.json"
```

Upload the plan, review the estimate and obtain the separate evaluation approval
and expected +50% actual pod guard before launching full 100×9×2 and 500×4:

```bash
export WORKSPACE=/workspace
export PROJECT="$WORKSPACE/bachelorArbeit"
export RUN_TAG=llava_stage1
export EVAL_TAG=trained
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/train/checkpoint-000450"
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["eval"]["guard_hours"]*3600))' "$PLAN")
export LOG="$WORKSPACE/logs/${RUN_TAG}_${EVAL_TAG}_eval.log"
mkdir -p "$WORKSPACE/logs"
cat "$PLAN"
tmux new-session -d -s llava-eval "env WORKSPACE='$WORKSPACE' RUN_TAG='$RUN_TAG' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' MAX_SECONDS='$MAX_SECONDS' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Stop point 3: report training and evaluation, then **STOP the pod**. Fetch
`eval_trained/`, `train/` and `mlruns/`; compare and plot on the laptop:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export FETCHED="$PWD/outputs/llava_stage1_fetched"
"$CPU_ENV/bin/python" scripts/compare_llava_stage1_training.py \
  --tuned "$FETCHED/eval_trained" --output-dir "$FETCHED/comparison"
"$CPU_ENV/bin/python" scripts/plot_training.py \
  --csv "$FETCHED/train/metrics.csv" --output-dir "$FETCHED/plots"
```

CSV plotting survives an unavailable MLflow store. To plot the file store instead:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export FETCHED="$PWD/outputs/llava_stage1_fetched"
export RUN_ID=$("$CPU_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$FETCHED/train/mlflow_run.json")
"$CPU_ENV/bin/python" scripts/plot_training.py --mlflow-store "$FETCHED/mlruns" \
  --run-id "$RUN_ID" --output-dir "$FETCHED/plots_mlflow"
```

Open the UI on the laptop at `http://127.0.0.1:5000` after fetching the file store.
No training server is used; the UI command runs in its own terminal:

```bash
export CPU_ENV="$PWD/.venv_llava_cpu"
export FETCHED="$PWD/outputs/llava_stage1_fetched"
export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=true
"$CPU_ENV/bin/mlflow" ui --backend-store-uri "file:$FETCHED/mlruns" --host 127.0.0.1 --port 5000
```
