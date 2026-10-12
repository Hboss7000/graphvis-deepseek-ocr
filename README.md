# Boosting LLM Question Answering with Visual Knowledge Graph

Code for reproducing Henrique Val Pereira Sucaria Leonel's bachelor's thesis at
TUM. The thesis asks whether a knowledge graph helps a vision-language model
answer questions when the graph is shown as an image instead of text.

The experiments use OpenBookQA (OBQA), ConceptNet graphs, and four models:
Qwen3-VL, LLaVA-v1.6, Gemma 3 and DeepSeek-OCR-2. Qwen and LLaVA are also
fine-tuned on graph-reading tasks.

## Start with your own code agent

Copy this prompt into your code agent. Fill in the result you want to reproduce
and the resources you have. You can ask for just one experiment.

```text
Help me reproduce a result from this thesis repository.

Result: <thesis section, table or figure; models and conditions>
Task: <check saved predictions, rebuild data, run inference, or fine-tune>
Resources: <CPU/GPU, available data/model caches/checkpoints, spending limit>

Read README.md, AGENTS.md and reproduction/manifest.json first.
Find the scripts and exact inputs for my chosen result. Explain what I need,
what is available, and what is missing before running anything.

Preserve the model revisions, input graphs, prompts, image settings, training
recipe and scoring method for that result. Use the saved 100-graph records
for final graph-comprehension evaluation and all 500 test questions for OBQA.
Use seed 13 for OBQA and seed 0 for the synthetic test. Do not change settings
to make the results match. Ask about unresolved experimental choices.

Start with CPU checks in a fresh output directory. Render images locally.
Follow AGENTS.md for paid compute: obtain my approval before resource changes,
quote costs and the termination guard, then run a tiny real-model dry run.
Report peak VRAM, measured speed and estimated full-run cost before asking
for full-run approval. After a fix, repeat the dry run before continuing.
Run long GPU jobs in tmux, give launch and monitoring commands together, and
keep caches, logs and checkpoints on /workspace. Report completion or failure
and remind me to STOP the Pod. Get confirmation before terminating resources.

Return the commands, input hashes, environment versions, completed example
counts and results. Compare with the reference scores using the correct parser
and paired metrics. Distinguish an exact replay from a fresh replication, and
state clearly which results could not be reproduced.
```

## Choose what to reproduce

You do not need to run the entire thesis. Start with the part you need:

| Goal | What you need | Where to start |
| --- | --- | --- |
| Rebuild the test graph images | CPU, Graphviz | [CPU setup](#1-set-up-the-cpu-environment) |
| Check scores from saved model answers | CPU, prediction files | [Scoring](#6-score-and-compare-results) |
| Evaluate the four base models | GPU, pinned model weights | [Zero-shot evaluation](#3-evaluate-the-base-models) |
| Fine-tune Qwen or LLaVA | GPU, training data, pinned weights | [Stage 1 training](#4-fine-tune-qwen-or-llava) |
| Run the blank-image or synthetic tests | GPU, base model and trained checkpoint | [Control experiments](#5-run-the-control-experiments) |
| Reproduce the earlier retrieval/graph-size studies | CPU; GPU for model answers | [Earlier experiments](#earlier-experiments) |

**Zero-shot** means evaluating a model before fine-tuning. **Stage 1** trains it
to read graphs: count nodes and edges, list triples, and identify node degrees.
The thesis does not perform Stage 2 question-answering fine-tuning. Gemma and
DeepSeek are evaluated without fine-tuning.

The main finding is that better graph reading does not improve OBQA accuracy.
After Stage 1, image-based QA accuracy changes from 74.2% to 58.6% for Qwen
and from 56.4% to 49.0% for LLaVA. The control experiments investigate why.

## Repository contents

| Directory | Contents |
| --- | --- |
| `scripts/` | Data preparation, training, evaluation and analysis scripts |
| `experiments/` | Shared model runners, versioned scorers and training manifests |
| `reproduction/inputs/` | Saved evaluation records and the synthetic dataset |
| `reproduction/reference/` | Reference scores for comparison |
| `env/` | Dependency pins and recorded environments |
| `tests/` | CPU regression checks |

[reproduction/files.txt](reproduction/files.txt) lists the included files.
[reproduction/manifest.json](reproduction/manifest.json) records artifact hashes
and their original locations. The dated experiment folders contain scripts used
by the current workflow.

Model weights, trained checkpoints, full prediction archives, rendered images
and raw QA-GNN downloads are not included. See [Reproduction limits](#reproduction-limits)
for experiments that require additional artifacts.

## 1. Set up the CPU environment

Run the following commands from the repository root. Render graphs on the CPU
machine, then copy the images to the GPU machine.

Install the Graphviz system program with the `dot` engine and fonts available
as `Helvetica` and `Helvetica-Bold`. The recorded renderer used Python 3.14.4,
Graphviz 14.1.2 and the package versions below. Keep those versions when checking
for identical images.

```bash
python3.14 -m venv .venv_render
.venv_render/bin/python -m pip install -r requirements_full.txt \
  pillow==12.3.0 matplotlib==3.11.1 tqdm==4.70.1
export RENDER_PY="$PWD/.venv_render/bin/python"
dot -V
```

First, try one graph:

```bash
"$RENDER_PY" reproduction/prepare.py --out outputs/replay_smoke --limit 1
```

The script compares the rendered PNG with the original image hash. If it differs,
check your Graphviz and fonts. The report is `outputs/replay_smoke/render_replay.json`.

Then prepare all 500 test images and the small evaluation subsets:

```bash
"$RENDER_PY" reproduction/prepare.py --out outputs/fullrun_2026-10-04_B
"$RENDER_PY" scripts/prepare_llava_stage1_eval.py
```

Use fresh output directories. The one-graph check does not prepare the other
499 images. The full command reuses the original graph structures and prompts;
it does not draw a new sample. This matters because sampling 100 graphs again
with seed 13 does not recover the original task prompts.

The final renderer uses dot, 200 DPI, 30/24 pt node/edge fonts, spacing
`nodesep=.5` and `ranksep=.4`, and uniform light-blue nodes (`#ADD8E6`).
Each graph keeps its recorded LR or TB orientation.

To check the CPU code:

```bash
"$RENDER_PY" -m pip install pytest==9.1.1
"$RENDER_PY" -m pytest -q tests/test_pruning_legibility.py \
  tests/test_stage1_template_v1_scoring.py tests/test_qa_explicit_v2.py \
  tests/test_synth_mcq_name_grounded.py tests/test_synth_mcq_llava_audit.py
```

### Download raw data when needed

The saved test inputs are enough to replay test images. Training, retrieval
comparisons and rebuilding graphs from scratch also need the upstream
[QA-GNN](https://github.com/michiyasunaga/qagnn) data:

```bash
bash scripts/setup_data.sh
```

The expected files are:

```text
data_preprocessed_release/
  cpnet/concept.txt
  obqa/graph/{train,dev,test}.graph.adj.pk
  obqa/statement/{train,dev,test}.statement.jsonl
```

The script runs QA-GNN's downloader and copies the required folders. If the
external download fails, follow the upstream instructions and use this layout.
The full `tests/test_*.py` suite needs this data for its 20-question regression.

For a new graph build, preserve the final settings: seed 13, `--core-policy keep`,
`--max-nodes 18`, `--max-edges 30`, `--max-degree 0`, `--bridge-rule qa-bridge`,
and protected lifelines. Core nodes and protected edges can exceed the budgets.
Use the original renderer settings above, `--auto-orient llava`, uniform nodes,
and no answer-revealing labels. See `scripts/generate_graphvis_datasets.py --help`
for the input and output arguments.

## 2. Prepare the GPU environment

Use one RTX PRO 6000 with 96GB VRAM. Read [AGENTS.md](AGENTS.md) before an agent
starts paid work. Run a small real-model test first, measure peak VRAM and speed,
and approve the estimated cost before the full run. A CPU test with a tiny
random model does not measure the real model's GPU requirements.

Keep the environments separate:

| Environment | Use | Transformers version |
| --- | --- | --- |
| `/workspace/venvs/venv_ocr2` | DeepSeek inference | 4.46.3 |
| `/workspace/venvs/venv_qwen` | Qwen, Gemma and HF LLaVA-NeXT inference | 5.16.1 |
| `/workspace/venvs/venv_train` | Qwen/LLaVA training and adapter evaluation | 5.16.1 |

Training also uses torch 2.8.0+cu128, torchvision 0.23.0+cu128, PEFT 0.21.2,
accelerate 1.15.0 and MLflow 3.16.1. The exact pins are in [env/](env/);
the training setup script is [llava_stage1_pod_setup.sh](scripts/llava_stage1_pod_setup.sh).
Stop if a pinned dependency cannot install instead of substituting a version.

Download each model once at its fixed revision, then use the cached snapshot:

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| `llava-hf/llava-v1.6-mistral-7b-hf` | `2424fdd47412fccc66d91719126b420e9fbd7065` |
| `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| `deepseek-ai/DeepSeek-OCR-2` | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` |

Keep outputs, checkpoints and caches under `/workspace`, including
`HF_HOME=/workspace/.cache/huggingface`. Run long jobs in `tmux`, write logs to
`/workspace/logs/`, and monitor with `tail -f` and `watch -n 30 nvidia-smi`.
A process timeout does not stop Pod billing. Arrange the separate termination
guard required by AGENTS.md and confirm resource termination with the owner.

## 3. Evaluate the base models

This reproduces the main zero-shot experiment (§5.7):

- Graph comprehension: 100 saved test graphs, nine tasks, **900 answers** per model.
- Question answering: all **500 OBQA test questions**, four conditions per model.

Copy `outputs/fullrun_2026-10-04_B/` and `outputs/llava_stage1_eval_inputs/`
to the same relative locations in the GPU checkout. On the GPU machine, set:

```bash
export PROJECT=/workspace/bachelorArbeit
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EVAL_DATA="$PROJECT/outputs/fullrun_2026-10-04_B"
export S1="$PROJECT/experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
export QA="$PROJECT/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts"
export QWEN_REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
mkdir -p /workspace/logs
```

### Graph comprehension: Qwen example

For the dry run, use `outputs/llava_stage1_eval_inputs/stage1_dry.jsonl` and
`--expected-count 9`. Keep the same model and image settings. Save dry-run
results in a separate directory and review the time/VRAM report first.

After full-run approval, launch:

```bash
export RUN_SECONDS=APPROVED_JOB_TIMEOUT_SECONDS # Replace with the approved integer.
tmux new-session -d -s qwen-stage1 "timeout --signal=INT --kill-after=120s $RUN_SECONDS \
 /workspace/venvs/venv_qwen/bin/python -u '$S1/run_stage1_qwen.py' \
 --revision '$QWEN_REV' --input-jsonl '$EVAL_DATA/test/stage1_subset100.jsonl' \
 --graph-metadata '$EVAL_DATA/test/graph_metadata_0_500.jsonl' --image-root '$EVAL_DATA' \
 --output-dir /workspace/outputs/repro_qwen_base/stage1 --expected-count 900 \
 --task-set extended --answer-format none --extractor span_extended \
 --min-pixels 262144 --max-pixels 1310720 --max-new-tokens 1024 \
 --seed 13 --approve-prompts --resume >/workspace/logs/qwen_stage1.log 2>&1"
tail -f /workspace/logs/qwen_stage1.log
# In another terminal:
watch -n 30 nvidia-smi
```

The result should contain nine prediction files, each with 100 unique question
indices. Rescore the raw answers using the method for your target figure;
the runner's initial score is not necessarily the thesis's final score.

### Question answering: Qwen example

Use this command inside the same tmux/log/timeout setup. For a dry run, replace
the input with `qa_dry.jsonl` and set `--expected-count 8`.

```bash
/workspace/venvs/venv_qwen/bin/python -u "$QA/run_zero_shot_qwen.py" \
  --revision "$QWEN_REV" --input-jsonl "$EVAL_DATA/test/stage2_obqa_0_500.jsonl" \
  --graph-metadata "$EVAL_DATA/test/graph_metadata_0_500.jsonl" --image-root "$EVAL_DATA" \
  --condition image --output-jsonl /workspace/outputs/repro_qwen_base/qa_image/predictions.jsonl \
  --expected-count 500 --min-pixels 262144 --max-pixels 1310720 \
  --max-new-tokens 64 --approve-prompt-diff --resume
```

Run each condition separately, changing both `--condition` and the output path:

| Condition | Input |
| --- | --- |
| `image` | Question, choices and graph image |
| `kg_text` | Question, choices and the same graph triples as text |
| `text_noref` | Question and choices only |
| `text` | Question and choices, with the image-reference sentence still present |

Isolated answer nodes appear in the image but not in the triple list.

### Use the other models

| Model | Graph-comprehension runner | QA runner |
| --- | --- | --- |
| LLaVA | `$S1/run_stage1_llava.py` | `$QA/run_zero_shot_llava.py` |
| Gemma | `scripts/eval_gemma3_stage1.py` | `scripts/eval_gemma3_stage2.py` |
| DeepSeek | `$S1/run_stage1_deepseek.py` | `$QA/run_zero_shot.py` |

Use the corresponding revision and environment from step 2. Preserve these
model-specific settings:

- **LLaVA:** native NeXT anyres processing. The original zero-shot arm uses
  `--prompt-template hf-chat --assistant-prefix-mode none`. The fine-tuning
  comparison needs a separate base arm with `--prompt-template llava_v1
  --assistant-prefix-mode none`.
- **Gemma:** eager attention, dynamic cache, batch 1, pan-and-scan enabled,
  min crop size 256, max crops 4, activation ratio 1.2. Keep the patch in
  `gemma3_common.py`. Run `--preflight 5` separately and supply its
  `--preflight-report`. The historical run used a failed-gate override;
  obtain approval before using `--allow-failed-preflight`.
- **DeepSeek:** eager attention, remote code, base size 1024, image size 768,
  crop mode enabled. Its `infer()` code has an effective 8192-token ceiling,
  even when the CLI requests fewer. For QA, set `--infer-output-dir` under
  `/workspace`.

Gemma's `eval_gemma3_stage2.py` runs QA evaluation, not Stage 2 training.

## 4. Fine-tune Qwen or LLaVA

Stage 1 (§§5.8–5.9) uses 1,200 training graphs and 100 validation graphs from
the OBQA training split, selected from sorted question IDs with `Random(13)`.
Each graph supplies six original GraphVis tasks: **7,200 training examples and
600 validation examples**. The 500 test questions are excluded. Relation,
neighbor and shortest-path tasks are used only for evaluation.

After downloading QA-GNN data, prepare the shared training images on CPU:

```bash
"$RENDER_PY" scripts/prepare_llava_stage1_training.py --render-only \
  --out-dir outputs/llava_stage1_training_2026-10-06
```

Use a separate CPU processor environment matching
[env/requirements_llava_cpu.txt](env/requirements_llava_cpu.txt).
Cache the pinned LLaVA processor/config, then run the same preparation script
with `--audit-only`. For Qwen's processor and token-length checks:

```bash
export CPU_PY="$PWD/.venv_llava_cpu/bin/python"
export QWEN_ASSETS="$PWD/outputs/qwen_stage1_assets/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
"$CPU_PY" scripts/fetch_qwen_stage1_processor.py --output-dir "$QWEN_ASSETS"
"$CPU_PY" scripts/prepare_qwen_stage1_training.py \
  --data-dir outputs/llava_stage1_training_2026-10-06 --processor-assets "$QWEN_ASSETS" \
  --output outputs/qwen_stage1_training_2026-10-06/token_diagnostics.json
```

Copy the images, records and audits to the GPU checkout. The fixed training
settings are:

| Setting | Value |
| --- | --- |
| Training length | One epoch, 450 optimizer steps; evaluate `checkpoint-000450` |
| Global batch size | 16 |
| Optimizer | AdamW, learning rates 2e-5, weight decay 0, clipping 1 |
| Schedule | 3% warmup, cosine decay |
| Precision and length | BF16, gradient checkpointing, maximum 4096 tokens |
| Loss | Answer and end-of-sequence tokens only |
| Validation | 600 examples at steps 100/200/300/400/450 |
| LoRA | Rank 128, alpha 256, dropout .05 |

LoRA targets the language model's q/k/v/o and gate/up/down linear layers,
excluding `lm_head`. The vision encoder stays frozen. LLaVA's projector trains
directly; Qwen's main merger and all three DeepStack mergers train directly.

Follow the model-specific commands in
[QWEN_STAGE1_TRAINING.md](scripts/QWEN_STAGE1_TRAINING.md) or
[LLAVA_STAGE1_TRAINING.md](scripts/LLAVA_STAGE1_TRAINING.md):

1. Run `llava_stage1_pod_job.sh probe` with `BACKBONE=qwen` or `llava`.
   It tests eight real examples and two optimizer steps per batch trial.
2. Review `batch_probe_report.json`, measured VRAM, speed and cost.
3. After approval, run the 20-step smoke/resume check.
4. Review its report and approve the full `train` job separately.
5. Evaluate with `evaluate_llava_stage1_adapter.py --phase dry` first,
   then `--phase full` with its reviewed `--dry-report`.

Those guides contain the tmux launch and monitoring commands. Their preparation
status notes are historical. The cost planner's $2.09/hour is also historical;
use the current quoted rate. Keep the full checkpoint directory, including
adapters, projector/mergers, optimizer and RNG state. The saver checks each save
and retains the latest two complete checkpoints, saving within 30 minutes.

## 5. Run the control experiments

### Blank-image control (§5.8.3)

Run the Qwen QA command from step 3 with `--condition image_blank`. Run it once
for the base model and once with `--adapter` pointing to the full step-450
checkpoint. The runner replaces each graph with a white image of the same size.
Keep the prompt and image-processing settings identical.

### Synthetic graph-reading test (§§5.8.4, 5.9.3)

The included [synthetic dataset](reproduction/inputs/synthetic_graphs.json)
has 50 graphs and 100 questions. Each question is tested under four cyclic
answer orders. Use **seed 0**, rather than the OBQA seed 13.

The five conditions are `image`, `text`, `none`, `wrong_image` and `wrong_text`.
The wrong-graph conditions use a paired graph that does not contain the correct
person's name. Two model arms × 100 questions × four orders × five conditions
produce **4,000 predictions**.

For Qwen, render and check the images on CPU. This uses `CPU_PY` and
`QWEN_ASSETS` from step 4; the processor setup is needed even without training.

```bash
"$RENDER_PY" scripts/run_synth_mcq.py \
  --dataset-path reproduction/inputs/synthetic_graphs.json \
  --out outputs/synth_mcq_name_grounded_2026-10-10 --render-only --render-style training --seed 0
"$RENDER_PY" scripts/audit_synth_mcq_images.py --measure-fonts-only \
  --out outputs/synth_mcq_name_grounded_2026-10-10/audit/font_references
"$CPU_PY" scripts/audit_synth_mcq_images.py \
  --dataset-path reproduction/inputs/synthetic_graphs.json \
  --renders outputs/synth_mcq_name_grounded_2026-10-10 --synthetic-only \
  --qwen-processor-assets "$QWEN_ASSETS" \
  --font-references outputs/synth_mcq_name_grounded_2026-10-10/audit/font_references/font_references.json \
  --out outputs/synth_mcq_name_grounded_2026-10-10/audit
```

If the image-readability gate fails, stop and review it. Copy passing assets
and the audit to the GPU machine. Set `DATASET_PATH`, `PY`, `ASSETS`,
`AUDIT_GATE` and `ADAPTER` to their actual paths. Run
`synth_mcq_name_grounded_smoke.sh` in tmux first, then approve
`synth_mcq_name_grounded_full.sh` separately. Generation is greedy, with
64 new tokens. Fetch the results and analyze them on CPU:

```bash
"$RENDER_PY" scripts/analyze_synth_mcq.py \
  --pred outputs/synth_qwen_fetched/predictions.jsonl \
  --out outputs/synth_qwen_analysis --n-perms 4 --bootstrap 10000 --seed 0
```

LLaVA needs its own renders: fixed LR, 42/34 pt fonts, `nodesep=.2`,
`ranksep=.15`, 200 DPI. Its reported score uses the A–D logits after the prefix
`The correct answer is`; Qwen uses parsed answer letters. Analyze the two model
families separately. Follow [SYNTH_MCQ.md](scripts/SYNTH_MCQ.md) for the LLaVA
workflow and its original-artifact requirements.

## 6. Score and compare results

Score the saved raw model responses on CPU. Choose the parser for the result
you are reproducing:

| Result | Scorer |
| --- | --- |
| Final zero-shot graph-comprehension heatmap | `audit_v4_proposed`; highest degree uses joint name-and-degree accuracy |
| Updated Qwen graph-comprehension comparison | `audit_v4_proposed` for both base and fine-tuned answers |
| Original thesis fine-tuning task comparisons | `template_v1`; distinguish degree-only from joint accuracy |
| Earlier graph-comprehension audit | `audit_v3` |
| Final zero-shot QA figure | `explicit_v2`, including Gemma's post-hoc first-line fallback |
| Historical QA and Qwen synthetic test | `explicit_v1` |
| LLaVA synthetic test | Prefix-logit accuracy |

List tasks use macro F1; other graph tasks use accuracy. Example for fetched
Qwen predictions:

```bash
"$RENDER_PY" reproduction/score.py stage1 \
  --input-jsonl reproduction/inputs/stage1_subset100.jsonl \
  --graph-metadata reproduction/inputs/graph_metadata_0_500.jsonl \
  --predictions-dir outputs/repro_qwen_base/stage1 --model-name qwen \
  --extractor audit_v4_proposed --output outputs/qwen_stage1_v4.json

"$RENDER_PY" reproduction/score.py qa \
  --input-jsonl reproduction/inputs/stage2_obqa_0_500.jsonl \
  --predictions outputs/repro_qwen_base/qa_image/predictions.jsonl \
  --parser-version explicit_v2 --output outputs/qwen_qa_image_v2.json
```

For `--model-name`, use `qwen`, `llava`, `gemma3` or `deepseek`, matching the
prediction filenames. The scorer rejects missing/duplicate question keys and
existing output files. Unparsed QA answers count as wrong.

Compare against `reproduction/reference/`. Each CSV identifies its source;
the LLaVA reference table contains supplied thesis values rather than scores
independently recovered from raw answers. Summary CSVs alone cannot reproduce
per-question statistical tests.

Compare models on the same question IDs. Use paired McNemar tests for binary
correctness and paired bootstrap intervals for score changes. For synthetic
data, bootstrap whole **graphs**, keeping both questions and all four answer
orders together. Include incomplete outputs, truncations and failed parses in
the completion checks.

## Earlier experiments

For the retrieval comparison (§5.2), follow
[mhgrn_code/README_retrieval_compare.md](mhgrn_code/README_retrieval_compare.md).
It describes the separate CPU environment, TransE embeddings and the exact
11-question order: `60,0,1,2,3,4,5,6,7,8,9`. The vendor checkouts are excluded;
fetch these recorded revisions:

```bash
git clone https://github.com/INK-USC/MHGRN mhgrn_code/MHGRN
git -C mhgrn_code/MHGRN checkout e199dcb0fd820c47c73dc992f97e793fd28c4a7c
git clone https://github.com/yihedeng9/GraphVis mhgrn_code/GraphVis
git -C mhgrn_code/GraphVis checkout 8d52ec9697e84d5c765f7beddbb020b1cda8e42b
```

These revisions were recorded locally; their match to the earlier COMA runs is
unconfirmed. GraphVis path retrieval and QA-GNN adjacency retrieval use different
inputs and should be compared as separate methods.

For graph-size statistics (§5.6), use:

```bash
"$RENDER_PY" scripts/node_cap_sweep.py --part-a \
  --data-root data_preprocessed_release --out-dir outputs/node_cap_replication
```

`--part-b` adds CPU rendering and readability checks. Model evaluation requires
separate GPU jobs. The historical 18-versus-40 comparison also changed edge and
degree limits (18/30/5 versus 40/60/0), prompts and parsers; it was not a
node-limit-only comparison. The earlier rendering study (§5.3) uses
`legibility_sweep.py`, `build_inference50_reading_probe.py` and
`report_inference50_reading_probe.py` with its original 20-graph subsets.

## Reproduction limits

- Some early COMA prediction files were lost. Their summary scores can be
  compared with a new run, but cannot be exactly rescored from this repository.
- Original trained checkpoints and full prediction archives must be obtained
  separately for exact replay.
- The LLaVA synthetic runner checks the author's original checkpoint, renders
  and training-reference audit hashes. A newly trained checkpoint will not pass
  those checks. Obtain the original artifacts, or explicitly review and approve
  a new provenance baseline while preserving the experiment settings.
- The thesis describes white nodes and fixed LR layout in §4.4. The final saved
  model inputs use light-blue nodes and per-graph LR/TB orientation. Use the
  saved inputs when reproducing the final results.
- Greedy decoding and fixed seeds do not guarantee identical results across
  different GPU hardware or library versions.
