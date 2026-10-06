# Qwen3-VL Stage 1 transfer test — CPU verified, stop before pod work

The four LLaVA fixes were committed first (`15974ef`): interpreter selection,
verified latest-two checkpoint retention, read-only weight-cache reuse, and the
missing upload/plan-return blocks. Qwen reuses those primitives and the real
training loop. No pod was contacted, created, changed or started during this work.
Henrique pushes the commits before the documented pull. Existing user edits were
excluded from all commits.

Model: `Qwen/Qwen3-VL-8B-Instruct`, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Native pinned chat template, user-only
messages, no default system prompt. Image budget exactly matches inference:
`min_pixels=262144`, `max_pixels=1310720`, sdpa attention.
[Immutable model configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json),
[immutable native template](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/chat_template.json).

The original 7,200/600 manifests, question IDs, six paper tasks and 1,300 rendered
images are unchanged. No Graphviz runs on the pod. Data manifest SHA256:
`643dfe4d0eb248bd7e8050b9d77e06c6fdd155d996c279df3c6e0eb40e932574`.
The model fields in that manifest record its LLaVA preparation origin;
Qwen's run config and independent token audit identify the Qwen backbone.

Loss supervises only the answer and final `<|im_end|>`, excluding the native
post-end newline. User/end/image tokens and padding are masked. The main vision
encoder is frozen. **All four merger bridges train** at 2e-5, as explicitly
approved: `model.visual.merger` and `model.visual.deepstack_merger_list.{0,1,2}`.
Language LoRA uses r=128, alpha=256, dropout=0.05 on every linear layer inside
`model.language_model`: q/k/v/o and gate/up/down in 36 layers (252 resolved names).
The separate `lm_head` stays frozen, matching the LLaVA scope. Full names and
counts are recorded in `experiments/2026-10-06_qwen_stage1_training/trainable_scope.json`
and again from the loaded GPU model in each run config. The meta-device scope
check allocates no pretrained weights.

The shared AdamW/cosine loop uses LR 2e-5 for both groups, betas .9/.999,
epsilon 1e-8, weight decay 0, warmup .03, clipping 1, global batch 16, seed 13,
gradient checkpointing and bf16/TF32 on the GPU. One epoch is exactly 450 optimizer
steps; validation runs at 100/200/300/400/450. No tuning or alternative recipe is
introduced. CSV, file-only MLflow (experiment `qwen-stage1`), resume, checkpoint
verification/retention and plotting are shared with LLaVA.

## CPU evidence and reproducibility

The combined CPU suite passed 75 checks; 43 documented Bash blocks and the shared
pod worker passed shell syntax validation. A CPU stand-in also executed every
Qwen worker dispatch to validate interpreter, cache flags, native runners and
shared input paths without using a GPU or contacting a pod.

Only nine pinned tokenizer/config/processor files were fetched into a NEW laptop
output directory. No pretrained model weights were downloaded, and no existing
Hugging Face cache or other environment was changed. Exact asset checksums are
recorded in `processor_assets_manifest.json`; the fetcher refuses differing
existing files and uses an explicit metadata whitelist. The existing approved
`.venv_llava_cpu` passed `pip check`, with torch **2.8.0+cpu**, transformers
**5.16.1**, peft **0.21.2**, accelerate **1.15.0**, mlflow **3.16.1**.
No additional installs occurred for Qwen.

The full pinned processor runs once per image, with exact image-grid expansion
checked against all native tokenized task answers. No truncation, dropping or
experimental adjustment. Maximum length stays 4,096.

| Split | N | Min | Median | p95 | Max | Over 4096 |
|---|---:|---:|---:|---:|---:|---:|
| Training | 7200 | 471 | 1290 | 1497 | 1567 | 0 |
| Validation | 600 | 1240 | 1292 | 1493.05 | 1547 | 0 |

Answer+end lengths: train median 16.5, p95 244, max 298; validation median 17,
p95 239.2, max 271. Per-task and image-token distributions are in the committed
`token_summary.json`. Full audit SHA256:
`b851363c56a4da3121918aced031d7d5133169aa4261a16ca4cf81e32e3601e3`.
The pod trainer checks this audit against the original data hashes, native
revision/template, length and pixel settings before reading batches.

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export ASSETS="$PWD/outputs/qwen_stage1_assets/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
"$PY" scripts/fetch_qwen_stage1_processor.py --output-dir "$ASSETS"
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" scripts/prepare_qwen_stage1_training.py \
  --data-dir outputs/llava_stage1_training_2026-10-06 --processor-assets "$ASSETS" \
  --output outputs/qwen_stage1_training_2026-10-06/token_diagnostics.json
"$PY" -m pip check
"$PY" -c 'import torch,transformers,peft,accelerate,mlflow; print({m.__name__:m.__version__ for m in (torch,transformers,peft,accelerate,mlflow)})'
```

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=true \
  .venv_llava_cpu/bin/python -m pytest -q \
  tests/llava_stage1_training_cpu_checks.py tests/test_llava_stage1_training_data.py \
  tests/test_llava_stage1_diagnostics.py tests/test_llava_runners.py \
  tests/test_llava_stage1_execution.py tests/test_stage1_storage.py \
  tests/qwen_stage1_training_cpu_checks.py tests/test_qwen_stage1_execution.py \
  --basetemp outputs/shared_stage1_cpu_checks
```

Tiny CPU tests exercise actual Qwen vision/image loading, all three DeepStack
injections plus main merger, answer masking/padding, finite gradients through all
four bridges and language LoRA, frozen encoder, exact optimizer/scheduler/RNG
resume through the production loop, verified adapter+all-bridge reload, identical
image and text-only greedy outputs, CSV and readable resumed MLflow run. A fourth
tiny decoder layer follows the third DeepStack injection so all four bridges can
influence the answer. CPU evidence does not measure 8B/bf16 CUDA speed or VRAM.
The regression comparison also re-scores the existing frozen Qwen zero-shot
outputs (all nine tasks and QA 500 × four) with unchanged scorers.

## Storage and environment before any paid job

The shared environment is `/workspace/venvs/venv_train`, never `venv_qwen` or
`venv_ocr2`. Interpreter selection and environment creation are documented in
[LLAVA_STAGE1_TRAINING.md](LLAVA_STAGE1_TRAINING.md). A read-only pre-check:

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export SYSTEM_PYTHON=$(bash "$PROJECT/scripts/stage1_train_interpreter.sh" "$WORKSPACE")
"$SYSTEM_PYTHON" -c 'import sys; print("Selected interpreter:",sys.executable); print("Python:",sys.version)'
df -h "$WORKSPACE"
du -sh "$WORKSPACE"
printf 'Configured volume cap: %s GB\n' "$VOLUME_CAP_GB"
```

The 2026-10-06 source exception is **approved**: only `torch==2.8.0+cu128` and
`torchvision==0.23.0+cu128` use the official PyTorch index
(`https://download.pytorch.org/whl/cu128`). Setup installs these two wheels with
`--no-deps` and no extra index, then installs all dependencies and other
requirements from **PyPI only**. All pins and constraints stay unchanged. The
previous `ALLOW_PYTORCH_CUDA_INDEX` gate is removed; no further source approval
is needed for these two wheels. No install or pod step was executed in this
update. Once `venv_train` exists, both backbones use it.

The pinned Qwen weight snapshot must already be complete in
`/workspace/.cache/huggingface`. The preflight and every pod job check it read-only,
including safetensors header/size integrity; jobs then run offline. Missing or
truncated snapshots STOP the job, with no download, overwrite or cache repair.

```bash
export PY=/workspace/venvs/venv_train/bin/python
export PROJECT=/workspace/bachelorArbeit
"$PY" "$PROJECT/scripts/stage1_weight_cache.py" --backbone qwen --cache /workspace/.cache/huggingface
df -h /workspace
du -sh /workspace
printf 'Configured volume cap: %s GB\n' "${VOLUME_CAP_GB:-200}"
```

RunPod's `df` reports the datacenter pool, so it is diagnostic only. Before
**each** checkpoint save, the guard runs `du -sb -- /workspace` again and computes
`remaining_bytes = VOLUME_CAP_GB * 1,000,000,000 - workspace_used_bytes`.
`VOLUME_CAP_GB` defaults to **200 decimal GB**; set it to the actual volume quota.
`WORKSPACE` selects the measured root when overridden. The log records cap bytes,
used bytes, remaining bytes and the required `2 × checkpoint estimate`, including
on an insufficient-space pause. Invalid caps or failed usage measurements also
pause cleanly instead of falling back to pool space. The last intact checkpoint
is preserved, and no new incomplete checkpoint directory is created. The configured
cap is passed explicitly into tmux so a custom value reaches every save.

Planning estimate from the pinned architecture: 349,175,808 LoRA parameters
in float32; main bridge 40,119,040 and each DeepStack bridge 40,125,952 in bf16.
Two AdamW moments at each parameter's dtype give **5,153,091,072 bytes per
checkpoint (~5.15 GB, 4.80 GiB)**, plus scalar/serialization overhead. The saver
records actual tensor bytes and dtypes, adds 5% + 1 MiB overhead, and requires
**2 × that budget remaining (~10.08 GiB)** before EVERY save. It exits cleanly with code
75 if insufficient, preserving the last intact save. It verifies the new save
before pruning and retains the latest two complete checkpoints. The final save
is retained. Both `df -h /workspace` and `du -sh /workspace` are logged at job
start and end.

At most three checkpoints coexist inside one active run while publishing a new
save: about 15.46 GB raw (16.24 GB with budget overhead). If measured workspace usage of
70 GB includes cached weights, shared rendered inputs ~0.65 GB and
an explicit 0.25 GB evaluation/log allowance, training plus evaluation peaks at
**~87.1 GB + new-venv size** for a run with no other new checkpoints on the volume.
At a hypothetical 6 GB venv this is ~93.1 GB; these are estimates, not measured
`du` values. The quota guard still needs ~10.82 GB remaining before saving.
Keeping the probe checkpoint and two smoke checkpoints adds another ~15.46 GB:
**~102.6 GB + new-venv size** for the documented complete sequence. Retaining
LLaVA artifacts adds its separately documented disk use too. All these artifacts
count toward the configured cap (default 200 GB). No checkpoints, cache files or volume are deleted/resized here.
Inspect actual `du` usage and the configured cap before full training;
the guard will pause safely if capacity is insufficient. Evaluation writes only
predictions/metrics, no extra base-model copy or MLflow weight artifacts.

## Transfer to an existing authorized pod

Every block below is independent; use only after the relevant paid-run approval.
No block creates, starts, changes or terminates a pod. Before any such operation,
AGENTS.md requires a current explicit yes after listing running/idle pods and
quoting GPU, data center, Secure/Community tier, live hourly rate, expected
runtime and total. Use one RTX PRO 6000 96 GB. A real pod termination guard is
required: setup/probe pod 4 h; training/evaluation expected +50%. Shell timeout
bounds the process and **does not stop pod billing**. Report success/failure
immediately and STOP the pod after review. No full run follows a fix without a
fresh dry run and approval. Reference calculations use the user's $2.09/h;
confirm the live quote before a paid approval.

This laptop block shows sizes, pulls pushed commits and uploads the reused data
and dry evaluation inputs plus the Qwen audit. Existing files are preserved;
manifest checks reject any differing old inputs. Qwen processor assets are needed
only for laptop audit/tests; pod processing uses the verified cached snapshot.

```bash
export LOCAL_ROOT="$PWD"
export REMOTE_ROOT=/workspace/bachelorArbeit
read -rp 'Existing pod IP: ' POD_IP
read -rp 'SSH port: ' POD_PORT
read -rp 'SSH private key path: ' POD_KEY
export RSYNC_RSH="ssh -p $POD_PORT -i \"$POD_KEY\""
du -sh "$LOCAL_ROOT/outputs/llava_stage1_training_2026-10-06" "$LOCAL_ROOT/outputs/llava_stage1_eval_inputs" "$LOCAL_ROOT/outputs/qwen_stage1_training_2026-10-06"
ssh -p "$POD_PORT" -i "$POD_KEY" "root@$POD_IP" 'cd /workspace/bachelorArbeit && git pull --ff-only && df -h /workspace && du -sh /workspace'
rsync -a --checksum --ignore-existing -e "$RSYNC_RSH" "$LOCAL_ROOT/outputs/llava_stage1_training_2026-10-06/" "root@$POD_IP:$REMOTE_ROOT/outputs/llava_stage1_training_2026-10-06/"
rsync -a --checksum --ignore-existing -e "$RSYNC_RSH" "$LOCAL_ROOT/outputs/llava_stage1_eval_inputs/" "root@$POD_IP:$REMOTE_ROOT/outputs/llava_stage1_eval_inputs/"
rsync -a --checksum --ignore-existing -e "$RSYNC_RSH" "$LOCAL_ROOT/outputs/qwen_stage1_training_2026-10-06/" "root@$POD_IP:$REMOTE_ROOT/outputs/qwen_stage1_training_2026-10-06/"
```

Fetch after each job (optimizer checkpoints stay on the volume):

```bash
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
read -rp 'Existing pod IP: ' POD_IP
read -rp 'SSH port: ' POD_PORT
read -rp 'SSH private key path: ' POD_KEY
export RSYNC_RSH="ssh -p $POD_PORT -i \"$POD_KEY\""
mkdir -p "$FETCHED/mlruns"
rsync -a --exclude='checkpoint-*/' -e "$RSYNC_RSH" "root@$POD_IP:/workspace/outputs/qwen_stage1/" "$FETCHED/"
rsync -a -e "$RSYNC_RSH" "root@$POD_IP:/workspace/mlruns/" "$FETCHED/mlruns/"
```

Return the laptop-calculated plan after each recalculation, preserving any old plan:

```bash
export LOCAL_PLAN="$PWD/outputs/qwen_stage1_fetched/execution_plan.json"
export REMOTE_PLAN=/workspace/outputs/qwen_stage1/execution_plan.json
read -rp 'Existing pod IP: ' POD_IP
read -rp 'SSH port: ' POD_PORT
read -rp 'SSH private key path: ' POD_KEY
export RSYNC_RSH="ssh -p $POD_PORT -i \"$POD_KEY\""
test -f "$LOCAL_PLAN"
rsync -a --backup --suffix=".previous.$(date -u +%Y%m%dT%H%M%SZ)" -e "$RSYNC_RSH" "$LOCAL_PLAN" "root@$POD_IP:$REMOTE_PLAN"
```

## Probe, measured smoke and separate training

Initial probe uses eight longest audited examples and two optimizer steps with
the full real Qwen model, images, frozen vision encoder and collator. It tries
microbatch 16/8/4/2/1; only CUDA OOM permits a smaller candidate, and global batch
stays 16. Each attempt uses a fresh process. There is no Qwen GPU timing yet;
**3600 s is an initial job ceiling, not a runtime estimate** ($2.09 maximum job
compute at the reference rate). It logs wall time, peak allocated VRAM and step
speed. All launches use tmux, a GPU lock, volume-only caches and a log file.

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=3600
export LOG="$WORKSPACE/logs/qwen-probe_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-probe "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' probe >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-probe_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

Fetch with the independent block above, then calculate the smoke estimate ON THE
LAPTOP from Qwen's own measured bf16 probe (CPU/LLaVA speeds are rejected).
The smoke plan includes 20 steps, two load/save/reload overhead allowances and a
50% guard. Return the plan with the independent upload block and review the
measured cost before paid smoke approval.

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
"$PY" scripts/llava_stage1_execution_plan.py --probe-report "$FETCHED/probe/batch_probe_report.json" --output "$FETCHED/execution_plan.json"
```

Smoke uses 64 examples, stops at step 10, saves, starts a fresh process and resumes
to 20. One outer timeout bounds both processes. The report must pass exact adapter
reload, identical greedy IDs, resumed run ID and readable MLflow before full
training can proceed. No training auto-launch follows smoke.

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["smoke"]["guard_hours"]*3600))' "$PLAN")
cat "$PLAN"
export LOG="$WORKSPACE/logs/qwen-smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-smoke "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' timeout --signal=INT --kill-after=120s $MAX_SECONDS bash -c 'bash \"$PROJECT/scripts/llava_stage1_pod_job.sh\" smoke-first && bash \"$PROJECT/scripts/llava_stage1_pod_job.sh\" smoke-resume' >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-smoke_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

Fetch again. Recalculate ON THE LAPTOP from the successful smoke. The shared plan
adds all 450 optimizer steps, five validations scaled to 600 examples, checkpoint
and loading overhead, then 50% guard. It remains an extrapolation; time-triggered
saves and variable lengths may add overhead. Return the plan, review disk space
and obtain the separate full-training approval/real pod guard before launching.

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
"$PY" scripts/llava_stage1_execution_plan.py --probe-report "$FETCHED/probe/batch_probe_report.json" \
  --smoke-report "$FETCHED/smoke/smoke_report.json" --output "$FETCHED/execution_plan.json"
```

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["train"]["guard_hours"]*3600))' "$PLAN")
cat "$PLAN"
unset RESUME_CHECKPOINT
export LOG="$WORKSPACE/logs/qwen-train_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-train "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' train >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-train_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

For resume after an approved restart, replace `unset RESUME_CHECKPOINT` above
with the actual latest verified path and add `RESUME_CHECKPOINT='$RESUME_CHECKPOINT'`
to the tmux command's environment. Keep code, data, recipe and output directory
identical. Never guess a checkpoint path. The wrapper saves every 100 steps and
at least every 20 minutes; it verifies and retains the latest two.

## Stage 1 evaluation first; QA optional second

Both Qwen runners now accept `--adapter CHECKPOINT_ROOT`; the default remains
zero-shot. Tuned outputs are separate from the frozen baseline. Stage 1 uses
`span_extended`, nine tasks including relation/neighbor/path held-out transfer,
greedy 1024 tokens. QA uses the unchanged strict parser, greedy 64 tokens and
four conditions (`image`, `text_noref`, `text`, `kg_text`), 500 questions each.
Native template, no system and pinned image settings remain identical in both
arms. Each adapter's checkpoint, run config and SHA256 are verified.

For TODAY'S smoke checkpoint, run the independent Stage 1 dry block. It exercises
one frozen graph × nine tasks. Its report provides the measured Stage 1 estimate.
The initial dry evaluation has a one-hour job ceiling, not a measured estimate.

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=3600
export EVAL_SCOPE=stage1
export EVAL_TAG=smoke
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/smoke/checkpoint-000020"
export LOG="$WORKSPACE/logs/qwen-smoke-stage1-dry_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-smoke-stage1-dry "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' EVAL_SCOPE='$EVAL_SCOPE' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval-dry >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-smoke-stage1-dry_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

After full training, repeat the Stage 1 rehearsal for the NEW final adapter;
smoke-adapter timing/verification cannot approve another adapter's evaluation.

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=3600
export EVAL_SCOPE=stage1
export EVAL_TAG=trained
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/train/checkpoint-000450"
export LOG="$WORKSPACE/logs/qwen-trained-stage1-dry_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-trained-stage1-dry "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' EVAL_SCOPE='$EVAL_SCOPE' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval-dry >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-trained-stage1-dry_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

Fetch and calculate the measured Stage 1-only estimate on the laptop. Return the
plan and obtain evaluation approval/expected +50% actual pod guard. The full
driver rejects adapter/code/input/scope drift from its rehearsal.

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
"$PY" scripts/llava_stage1_execution_plan.py --probe-report "$FETCHED/probe/batch_probe_report.json" \
  --smoke-report "$FETCHED/smoke/smoke_report.json" \
  --evaluation-dry-report "$FETCHED/eval_trained_stage1_dry/evaluation_report.json" --output "$FETCHED/execution_plan.json"
```

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["eval_stage1"]["guard_hours"]*3600))' "$PLAN")
cat "$PLAN"
export EVAL_SCOPE=stage1
export EVAL_TAG=trained
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/train/checkpoint-000450"
export LOG="$WORKSPACE/logs/qwen-trained-stage1_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-trained-stage1 "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' EVAL_SCOPE='$EVAL_SCOPE' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-trained-stage1_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

STOP here if QA is cut. To continue, rehearse eight questions in each of four QA
conditions AFTER Stage 1, using the same final adapter. QA has its own measured
estimate and independent approval so it can be omitted without discarding Stage 1.

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export MAX_SECONDS=3600
export EVAL_SCOPE=qa
export EVAL_TAG=trained
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/train/checkpoint-000450"
export LOG="$WORKSPACE/logs/qwen-trained-qa-dry_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-trained-qa-dry "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' EVAL_SCOPE='$EVAL_SCOPE' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval-dry >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-trained-qa-dry_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

Fetch, calculate QA's own estimate (preserving Stage 1's estimate), return the
plan and review/approve its separate cost and actual guard:

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
"$PY" scripts/llava_stage1_execution_plan.py --probe-report "$FETCHED/probe/batch_probe_report.json" \
  --smoke-report "$FETCHED/smoke/smoke_report.json" \
  --evaluation-dry-report "$FETCHED/eval_trained_stage1_dry/evaluation_report.json" \
  --qa-evaluation-dry-report "$FETCHED/eval_trained_qa_dry/evaluation_report.json" --output "$FETCHED/execution_plan.json"
```

```bash
export WORKSPACE=/workspace
export VOLUME_CAP_GB=${VOLUME_CAP_GB:-200}
export PROJECT="$WORKSPACE/bachelorArbeit"
export BACKBONE=qwen RUN_TAG=qwen_stage1
export LORA_LR=2e-5 PROJECTOR_LR=2e-5
export PY="$WORKSPACE/venvs/venv_train/bin/python"
export PLAN="$WORKSPACE/outputs/$RUN_TAG/execution_plan.json"
export MAX_SECONDS=$("$PY" -c 'import json,math,sys; print(math.ceil(json.load(open(sys.argv[1]))["jobs"]["eval_qa"]["guard_hours"]*3600))' "$PLAN")
cat "$PLAN"
export EVAL_SCOPE=qa
export EVAL_TAG=trained
export ADAPTER_PATH="$WORKSPACE/outputs/$RUN_TAG/train/checkpoint-000450"
export LOG="$WORKSPACE/logs/qwen-trained-qa_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$WORKSPACE/logs"
tmux new-session -d -s qwen-trained-qa "env WORKSPACE='$WORKSPACE' VOLUME_CAP_GB='$VOLUME_CAP_GB' BACKBONE='$BACKBONE' RUN_TAG='$RUN_TAG' LORA_LR='$LORA_LR' PROJECTOR_LR='$PROJECTOR_LR' MAX_SECONDS='$MAX_SECONDS' EVAL_SCOPE='$EVAL_SCOPE' EVAL_TAG='$EVAL_TAG' ADAPTER_PATH='$ADAPTER_PATH' bash '$PROJECT/scripts/llava_stage1_pod_job.sh' eval >'$LOG' 2>&1"
tail -n 80 "$LOG"
nvidia-smi
```

Continuous monitoring (separate terminals): `tail -f /workspace/logs/qwen-trained-qa_*.log` and `watch -n 30 nvidia-smi`. STOP the pod after completion or failure.

## Fetch, compare and plot on the laptop

Use the independent fetch block. Stage-only comparison when QA was cut:

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
"$PY" scripts/compare_llava_stage1_training.py --backbone qwen --stage1-only \
  --tuned "$FETCHED/eval_trained_stage1" --output-dir outputs/qwen_stage1_comparison
```

After completing QA, collect both tuned families in a NEW local evaluation root
before comparing all nine Stage 1 tasks and QA 500 × four. No baseline predictions
or scorers are changed. Missing arms remain explicit missing results, never zeros.
This reference comparison uses paper gains directionally; it does not claim the
paper's CSQA accuracy is comparable to OBQA or held-out transfer metrics.

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
export COMBINED="$FETCHED/eval_trained_combined"
test ! -e "$COMBINED"
mkdir -p "$COMBINED"
cp -a "$FETCHED/eval_trained_stage1/stage1" "$COMBINED/"
for CONDITION in image text_noref text kg_text; do cp -a "$FETCHED/eval_trained_qa/qa_$CONDITION" "$COMBINED/"; done
"$PY" scripts/compare_llava_stage1_training.py --backbone qwen --tuned "$COMBINED" --output-dir outputs/qwen_stage1_comparison
```

The shared plots accept CSV or the fetched MLflow file store. Produce exportable
1920×1080, 300 dpi loss/LR plots with integer step ticks and validation markers:

```bash
export PY="$PWD/.venv_llava_cpu/bin/python"
export FETCHED="$PWD/outputs/qwen_stage1_fetched"
"$PY" scripts/plot_training.py --csv "$FETCHED/train/metrics.csv" --output-dir outputs/qwen_stage1_plots
```

Verified: CPU processor audit, random-model gradient/resume/reload, file logging,
frozen-baseline comparisons, and shell syntax. Unmeasured: real 8B bf16 GPU
VRAM/speed, checkpoint bytes on pod, validation/evaluation wall time and live
pod quote. Those values are produced by the documented real probe/smoke/dry
commands before the corresponding paid approvals. No pod step executed.
