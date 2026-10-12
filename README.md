# Boosting LLM Question Answering with Visual Knowledge Graph

Reproduction code for Henrique Val Pereira Sucaria Leonel's bachelor's thesis at
TUM. The study adapts GraphVis to OpenBookQA (OBQA) and compares
DeepSeek-OCR-2, Qwen3-VL-8B-Instruct, Gemma 3 12B-it and LLaVA-v1.6-Mistral-7B.
It tests graph images against the same triples as text and against no graph,
then fine-tunes **Qwen and LLaVA on Stage 1 graph comprehension only**.
Stage 2 QA fine-tuning, Gemma fine-tuning and DeepSeek fine-tuning are outside
the reported experiments.

The main finding is that better graph reading does not improve OBQA accuracy:
image QA changes from 74.2% to 58.6% for Qwen and from 56.4% to 49.0% for
LLaVA after Stage 1. The blank-image and synthetic controls distinguish graph
content from image processing. On graph-dependent synthetic questions, text
triples outperform images. The thesis is the reference for interpretation;
this README specifies the actual code paths and remaining reproduction limits.

## What is included

`.gitignore` is an explicit publication allowlist. It keeps the experiment entry
points, their local helper dependencies, relevant regression tests, environment
pins, subset manifests and the small text artifacts needed for reproduction.
[reproduction/files.txt](reproduction/files.txt) inventories that selection.
The dated experiment directories contain active shared runners and scorers;
their names do not mean that they can be deleted.

Raw QA-GNN/ConceptNet downloads, external vendor checkouts, virtual environments,
weights, checkpoints, predictions, rendered images, logs, thesis PDF, figure
exports and session notes stay local. New files are ignored until added to the
allowlist. The prepared repository does not distribute trained adapters or a
complete archive of the original predictions.

[reproduction/manifest.json](reproduction/manifest.json) records SHA-256 hashes
and original locations for the frozen evaluation records, synthetic dataset,
post-hoc v4 scorer and reference CSVs. The evaluation records are necessary:
the 100-graph Stage 1 cohort combines two historical 50-graph task sets. Drawing
100 graphs again with seed 13 does not recover the exact sampled task prompts.
The reference CSVs allow comparison with reported scores; they cannot recover
per-question predictions or statistical tests.

| Thesis experiment | Entry points / artifacts |
| --- | --- |
| §5.2 retrieval comparison | `mhgrn_code/README_retrieval_compare.md`, `compare_retrieval.py` |
| §5.3 rendering and reading probes | `legibility_sweep.py`, `build_inference50_reading_probe.py`, `report_inference50_reading_probe.py` |
| §5.6 graph size / connectivity | `node_cap_sweep.py`, `pruning_stats.py`, `image_audit.py` |
| §5.7 four-model zero-shot | `experiments/*/scripts/run_stage1_*.py`, `run_zero_shot*.py`, `scripts/eval_gemma3_stage{1,2}.py` |
| §§5.8–5.9 Stage 1 training | `prepare_llava_stage1_training.py`, `prepare_qwen_stage1_training.py`, `train_{qwen,llava}_stage1.py` |
| Tuned evaluation | `evaluate_llava_stage1_adapter.py`, `prepare_llava_stage1_eval.py` |
| §5.8.3 blank image | Qwen QA runner, `--condition image_blank` |
| §§5.8.4, 5.9.3 synthetic controls | `run_synth_mcq.py`, `prepare_synth_mcq_llava.py`, `analyze_synth_mcq.py` |
| Offline parsing and scoring | `reproduction/score.py`, versioned scorers in the Stage 1 experiment directory |

Names in the table are under `scripts/` unless another directory is given.

## Start on the CPU machine

Run commands from the checkout root. Rendering runs on the laptop/CPU machine,
then images are transferred to the GPU machine. Install Graphviz with the `dot`
engine and the fonts resolved by `Helvetica` and `Helvetica-Bold`.
The local verification environment used **Python 3.14.4, Graphviz 14.1.2,
graphviz 0.21, NumPy 2.5.1, SciPy 1.18.0 and Pillow 12.3.0**. These are measured
local versions, not a promise of compatibility with another platform.

```bash
python3.14 -m venv .venv_render
.venv_render/bin/python -m pip install -r requirements_full.txt \
  pillow==12.3.0 matplotlib==3.11.1 tqdm==4.70.1
export RENDER_PY="$PWD/.venv_render/bin/python"
dot -V

# Cheap CPU smoke: one actual graph, with original PNG hash comparison.
"$RENDER_PY" reproduction/prepare.py --out outputs/replay_smoke --limit 1

# Separate CPU preparation for all 500 test images; use a fresh directory.
"$RENDER_PY" reproduction/prepare.py --out outputs/fullrun_2026-10-04_B
"$RENDER_PY" scripts/prepare_llava_stage1_eval.py
```

`prepare.py` reuses the frozen graph topology, node roles, orientation and input
records, with the production renderer: dot, 200 DPI, 30/24 pt node/edge fonts,
`nodesep=.5`, `ranksep=.4`, uniform `#ADD8E6` node fill. It does not re-prune or
resample tasks. It writes `render_replay.json` and stops if generated PNG hashes
differ from the evaluated images. Different Graphviz/font installations can
change the pixels; inspect that difference before claiming exact replay.
`--limit` is for CPU smoke only and leaves the other images absent.

For training, retrieval comparisons or rebuilding graphs from the raw source,
fetch the upstream [QA-GNN](https://github.com/michiyasunaga/qagnn) data:

```bash
bash scripts/setup_data.sh
```

Required layout:

```text
data_preprocessed_release/
  cpnet/concept.txt
  obqa/graph/{train,dev,test}.graph.adj.pk
  obqa/statement/{train,dev,test}.statement.jsonl
```

The downloader uses QA-GNN's own download script and downloads more than OBQA
before copying the required folders. Its external download was not rerun in
this documentation update. If it fails, follow the upstream download instructions
and populate this layout. Use trusted upstream pickle files.

The final graph generation settings are **core-policy keep, 18 nodes, 30 edges,
no degree cap, qa-bridge fillers, protected lifelines**, seed 13. The node and
edge budgets can be exceeded by protected core concepts/edges. Rendering chooses
LR or TB per graph with `--auto-orient llava`. For a fresh raw-data smoke:

```bash
"$RENDER_PY" scripts/generate_graphvis_datasets.py \
  --split test --limit 8 --data-root data_preprocessed_release \
  --out-dir outputs/raw_data_smoke --seed 13 \
  --core-policy keep --max-nodes 18 --max-edges 30 --max-degree 0 \
  --bridge-rule qa-bridge --stage1-task-set extended --stage1-balance per-task \
  --rankdir TB --auto-orient llava --node-fontsize 30 --edge-fontsize 24 \
  --nodesep 0.5 --ranksep 0.4 --wrap-labels 0 --edge-label-style plain \
  --node-style uniform --node-fill '#ADD8E6'
```

Use the frozen input replay for the published cohort. The raw-data smoke validates
the retrieval-to-rendering pipeline, not the full thesis result.

The portable CPU checks need pytest (the renderer verification used 9.1.1):

```bash
"$RENDER_PY" -m pip install pytest==9.1.1
"$RENDER_PY" -m pytest -q tests/test_pruning_legibility.py \
  tests/test_stage1_template_v1_scoring.py tests/test_qa_explicit_v2.py \
  tests/test_synth_mcq_name_grounded.py tests/test_synth_mcq_llava_audit.py
```

The full `tests/test_*.py` suite also includes a 20-question QA-GNN regression,
so download the upstream data first. Native tiny-model training checks in
`*_cpu_checks.py` additionally need the separate processor/test environment
and pinned cached processor assets; they allocate no pretrained GPU weights.

## GPU environments and fixed model settings

Read [AGENTS.md](AGENTS.md) before any agent launches paid work. No GPU operation
is authorized merely by copying a README command. Each paid stage needs a tiny
real-model dry run, reported peak allocated VRAM and time per step/item, a cost
estimate using a current quote, and approval before the full job. CPU tests with
tiny random models do not replace that dry run.

Use one RTX PRO 6000 96GB. Keep Hugging Face caches and every output/checkpoint
under `/workspace`, with `HF_HOME=/workspace/.cache/huggingface`. Long jobs run
in `tmux` and log under `/workspace/logs/`. A process timeout does **not** stop
Pod billing: arrange the approved Pod termination guard separately (4 hours for
setup, expected runtime +50% for training/evaluation). Report completion/failure
immediately and stop the Pod when finished, subject to the owner's confirmation.

| Model | Immutable Hub revision | Environment and image settings |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-OCR-2` | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | `venv_ocr2`, transformers 4.46.3; remote code, eager; base 1024, crop 768, crop mode on |
| `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `venv_qwen`, transformers 5.16.1; SDPA, min pixels 262144, max pixels 1310720; native chat, no system prompt |
| `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | `venv_qwen`, transformers 5.16.1; eager, dynamic cache; pan-and-scan: min crop 256, max crops 4, activation ratio 1.2 |
| `llava-hf/llava-v1.6-mistral-7b-hf` | `2424fdd47412fccc66d91719126b420e9fbd7065` | HF LLaVA-NeXT runners, transformers 5.16.1 in this implementation; native anyres processor |

The later training implementation uses a third, separate `venv_train` with
transformers 5.16.1, torch 2.8.0+cu128, torchvision 0.23.0+cu128, PEFT 0.21.2,
accelerate 1.15.0 and MLflow 3.16.1. The thesis's description of an older LLaVA
environment refers to the original implementation; these runners use HF NeXT.
Keep the inference environments intact. The exact package constraints/freezes
are in [env/](env/); training setup is in
[scripts/llava_stage1_pod_setup.sh](scripts/llava_stage1_pod_setup.sh).
The freezes are records, including platform-specific packages. Stop if a pinned
version cannot install; do not silently change it. Download each pinned model
snapshot once, obtain any required model access yourself, and then use offline
cache reads. `stage1_weight_cache.py --backbone qwen|llava --cache ...` checks
training snapshots without repairing or downloading them.

## Reproduce zero-shot inference (§5.7)

Transfer `outputs/fullrun_2026-10-04_B/` and
`outputs/llava_stage1_eval_inputs/` to the same relative locations in the GPU
checkout. Set these shell variables on an already approved machine:

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

A Qwen Stage 1 launch, **after a separate dry run and full-run approval**:

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
# Monitor in another terminal:
watch -n 30 nvidia-smi
```

For the dry run, use `outputs/llava_stage1_eval_inputs/stage1_dry.jsonl` and
`--expected-count 9`, keeping the same model, image root, settings and runner.
Keep dry and full results in separate directories. Stage 1 requires all nine
per-task prediction files with exactly 100 unique question indices each.

For the other models, use the corresponding Stage 1 runner, pinned revision and
interpreter from the model table. DeepSeek uses `run_stage1_deepseek.py` and its
remote `infer()` path: the effective generation ceiling is **8192**, regardless
of a smaller CLI limit. LLaVA's original zero-shot arm uses `--prompt-template
hf-chat --assistant-prefix-mode none`; the fine-tuning comparison also needs a
base arm with `--prompt-template llava_v1 --assistant-prefix-mode none`.
Those are distinct prompt conditions. Gemma uses `scripts/eval_gemma3_stage1.py`
and the explicit flags:

```text
--attn-impl eager --pan-and-scan --pan-and-scan-min-crop-size 256
--pan-and-scan-max-num-crops 4 --pan-and-scan-min-ratio-to-activate 1.2
--cache-implementation dynamic --batch-size 1
```

Keep the compatibility patch in `gemma3_common.py`. Gemma needs a saved
`--preflight-report`; run its `--preflight 5` trial separately first. The final
historical launcher used a failed-preflight override. Report the measured gate
and obtain approval before using `--allow-failed-preflight`; do not represent it
as a passing legibility test.

QA uses all 500 records in `test/stage2_obqa_0_500.jsonl` and the same metadata
and image root. For Qwen, replace the Stage 1 command inside the same tmux/log/
monitor pattern with:

```bash
/workspace/venvs/venv_qwen/bin/python -u "$QA/run_zero_shot_qwen.py" \
  --revision "$QWEN_REV" --input-jsonl "$EVAL_DATA/test/stage2_obqa_0_500.jsonl" \
  --graph-metadata "$EVAL_DATA/test/graph_metadata_0_500.jsonl" --image-root "$EVAL_DATA" \
  --condition image --output-jsonl /workspace/outputs/repro_qwen_base/qa_image/predictions.jsonl \
  --expected-count 500 --min-pixels 262144 --max-pixels 1310720 \
  --max-new-tokens 64 --approve-prompt-diff --resume
```

Run the four conditions separately: `image`, `kg_text`, `text_noref`, `text`.
Change both `--condition` and the output directory. `text` has no graph but
retains the image-reference sentence; `text_noref` removes it. The image/text
triple graph is identical, except isolated answer nodes appear only in images.
For DeepSeek use `run_zero_shot.py` and an `/workspace` `--infer-output-dir`;
LLaVA uses `run_zero_shot_llava.py`; Gemma uses `eval_gemma3_stage2.py` (QA
**evaluation**, despite the filename). Apply each model's fixed settings above.
For a QA dry run, use the eight-record `qa_dry.jsonl` and `--expected-count 8`.

## Reproduce Stage 1 fine-tuning (§§5.8–5.9)

Generate the shared training inputs on the CPU machine:

```bash
"$RENDER_PY" scripts/prepare_llava_stage1_training.py --render-only \
  --out-dir outputs/llava_stage1_training_2026-10-06
```

Selection is `Random(13)` over sorted **question IDs**: 1,200 training graphs,
then 100 validation graphs from the remaining OBQA training questions. They
produce **7,200 training / 600 validation** examples, six original GraphVis
tasks each. All 500 test question IDs are excluded. The three new task types
(relation, neighbors, shortest path) are evaluation-only. The saved selection
manifests in `experiments/2026-10-06_llava_stage1_training/` are the reference.

In a separate CPU processor/test environment matching
`env/requirements_llava_cpu.txt`, cache the pinned LLaVA processor/config,
then run `prepare_llava_stage1_training.py --audit-only` on that directory.
For Qwen's native image/token audit:

```bash
export CPU_PY="$PWD/.venv_llava_cpu/bin/python"
export QWEN_ASSETS="$PWD/outputs/qwen_stage1_assets/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
"$CPU_PY" scripts/fetch_qwen_stage1_processor.py --output-dir "$QWEN_ASSETS"
"$CPU_PY" scripts/prepare_qwen_stage1_training.py \
  --data-dir outputs/llava_stage1_training_2026-10-06 --processor-assets "$QWEN_ASSETS" \
  --output outputs/qwen_stage1_training_2026-10-06/token_diagnostics.json
```

Transfer the training inputs and audits to the GPU checkout. Use the separate
`venv_train`. The recorded recipe is global batch 16, one epoch/450 optimizer
steps, BF16, gradient checkpointing, answer+EOS-only loss, sequence limit 4096,
AdamW, both learning rates 2e-5, weight decay 0, 3% warmup, cosine decay,
gradient clipping 1, seed 13. Validate at steps 100/200/300/400/450 and evaluate
**checkpoint-000450**, rather than selecting a checkpoint on test accuracy.

The recorded LoRA scope is r=128, alpha=256, dropout .05, q/k/v/o and
gate/up/down linear layers in the language model, excluding `lm_head`.
The vision encoder is frozen. LLaVA's projector trains directly; Qwen's main
merger and all three DeepStack mergers train directly. These are experimental
settings to reproduce, not defaults an agent may change.

The complete probe → smoke/resume → training → evaluation commands are in
[QWEN_STAGE1_TRAINING.md](scripts/QWEN_STAGE1_TRAINING.md) and
[LLAVA_STAGE1_TRAINING.md](scripts/LLAVA_STAGE1_TRAINING.md). Their status notes
are historical; their job commands remain useful. Use
`llava_stage1_pod_job.sh probe` with `BACKBONE=qwen` or `llava`, launched with
its documented tmux/log/monitor pair. It runs eight real examples and two
optimizer steps per batch trial. Review `batch_probe_report.json`, approve the
20-step resume smoke, then review its report before approving `train`.
`llava_stage1_execution_plan.py` extrapolates measured times, but its $2.09/hour
is a historical assumption: recalculate using the current quoted price.
Checkpoints include adapters, projector/mergers, optimizer/scheduler and RNG
state; keep the full directory. The saver checkpoints at most 30 minutes apart
and verifies each save before retaining the latest two complete checkpoints.

Run `evaluate_llava_stage1_adapter.py --backbone qwen|llava --phase dry` first,
then `--phase full` with its reviewed `--dry-report`. Use separate base/tuned
folders and identical task inputs. For §5.8.3, run Qwen QA twice with
`--condition image_blank`, once without `--adapter` and once with the full
step-450 checkpoint. The runner creates a white image of each original image's
size. Keep the image prompt and processor settings unchanged.

## Reproduce the synthetic controls (§§5.8.4, 5.9.3)

The exact dataset is included at
[reproduction/inputs/synthetic_graphs.json](reproduction/inputs/synthetic_graphs.json).
It has 50 graphs, 18 nodes per graph, 100 name-grounded questions, two clues
per question, and four cyclic option orders. Use **seed 0** for this protocol,
not the OBQA seed 13. Preserve its five conditions: `image`, `text`, `none`,
`wrong_image`, `wrong_text`. Wrong graphs are paired over the full cohort;
the correct person name must be absent from the paired graph.
The synthetic runner now defaults to this bundled file; the explicit path below
also works independently of the directory name used for the checkout.

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

Stop if the native-processor legibility gate fails. Qwen uses its training-style
30/24 pt renders and fixed pixel bounds. With assets/audit copied to the GPU
machine, use `synth_mcq_name_grounded_smoke.sh`, then, after review and approval,
`synth_mcq_name_grounded_full.sh` inside tmux. Set `DATASET_PATH` explicitly to
the included JSON's location; set `PY`, `ASSETS`, `AUDIT_GATE` and `ADAPTER` to
real paths. Both model arms, five conditions, 100 questions and four permutations
give **4,000 predictions**. Greedy QA generation uses 64 new tokens. Fetch the
results and analyze on CPU:

```bash
"$RENDER_PY" scripts/analyze_synth_mcq.py \
  --pred outputs/synth_qwen_fetched/predictions.jsonl \
  --out outputs/synth_qwen_analysis --n-perms 4 --bootstrap 10000 --seed 0
```

LLaVA uses a separately rendered, fixed LR variant: node/edge fonts 42/34 pt,
`nodesep=.2`, `ranksep=.15`, 200 DPI. Its thesis metric is the A–D **prefix
logit** score after `The correct answer is`, rather than Qwen's parsed-letter
score. Analyze families separately; inspect the prefix-logit rows for LLaVA.
See [SYNTH_MCQ.md](scripts/SYNTH_MCQ.md) for the render/audit/smoke/full sequence.
There is a material portability limit: that LLaVA branch pins the author's
checkpoint checksum, original render checksum and training-reference audit.
A newly trained checkpoint cannot pass its exact historical-identity check.
For exact replay, obtain those original artifacts from the author; for a fresh
replication, review and explicitly approve a new provenance baseline while
preserving the model, rendering and prefix metric. Do not bypass its guards or
silently substitute the Qwen renders.

## Score predictions on CPU

Inference and extraction are separate. Always specify the parser; the runners'
default scores are not interchangeable with the final thesis analyses.

| Reported analysis | Scorer / metric |
| --- | --- |
| Final four-model zero-shot Stage 1 figure | preserved `audit_v4_proposed`; list macro F1, accuracy otherwise, joint name+degree for highest degree |
| Earlier frozen Stage 1 audit | `audit_v3` (retain for the version comparison) |
| Qwen/LLaVA fine-tuning task comparisons | `template_v1`; highest-degree degree accuracy and joint accuracy must be distinguished |
| Final zero-shot QA figure, including Gemma correction | `explicit_v2`, a post-hoc first-line fallback |
| Historical QA and Qwen synthetic parsed-letter analysis | `explicit_v1` |
| LLaVA synthetic thesis table | prefix logits; also retain parsed-letter diagnostics |

Example after fetching Qwen predictions:

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

Use prediction prefix `gemma3` for Gemma; other prefixes are `qwen`, `llava`,
`deepseek`. `score.py` rejects missing/duplicate question keys and existing
output files. Failed extraction counts as wrong for QA. It recomputes from raw
responses, rather than trusting stored `is_correct` values. For tuned task
comparisons run `--extractor template_v1` on both prompt-matched arms. Reference
CSVs under `reproduction/reference/` state their scorer/source; the LLaVA CSV
contains the supplied thesis table values, not independently recovered raw rows.

Accuracy comparisons must use the same question keys. Use exact paired McNemar
tests for binary correctness; use paired bootstrap intervals for score changes,
and resample **graphs** for synthetic data so both questions and all four option
orders remain together. Preserve denominators, truncations and failed parses.

## Earlier experiments and reproduction limits

For §5.2, the wrappers need external checkouts and QA-GNN data plus TransE
arrays. The local vendor revisions inspected for this update were:

```bash
git clone https://github.com/INK-USC/MHGRN mhgrn_code/MHGRN
git -C mhgrn_code/MHGRN checkout e199dcb0fd820c47c73dc992f97e793fd28c4a7c
git clone https://github.com/yihedeng9/GraphVis mhgrn_code/GraphVis
git -C mhgrn_code/GraphVis checkout 8d52ec9697e84d5c765f7beddbb020b1cda8e42b
```

Follow [mhgrn_code/README_retrieval_compare.md](mhgrn_code/README_retrieval_compare.md)
for its separate CPU environment, required embedding files and the exact
11-question order `60,0,1,2,3,4,5,6,7,8,9`. These are verified local checkout
revisions; their correspondence to the earlier COMA runs is not established.
The released GraphVis path retrieval and QA-GNN adjacency retrieval are different
inputs. Keep that distinction when interpreting the comparison.

For §5.6 CPU statistics, use a fresh directory:

```bash
"$RENDER_PY" scripts/node_cap_sweep.py --part-a \
  --data-root data_preprocessed_release --out-dir outputs/node_cap_replication
```

`--part-b` additionally renders and audits the CPU legibility grid; it does not
run model inference. Reproducing the model scores requires separate evaluator jobs.
The historical 18-versus-40 comparison also changed edge/degree caps
(18/30/5 versus 40/60/0) and used older prompts/parsers. It is not a controlled
node-cap-only comparison. Rendering probes likewise belong to their original
20-graph cohorts; use the probe builder/reporter, with explicitly named arms,
without replacing them with the final 100-graph cohort.

Some early COMA outputs were lost, as disclosed in the thesis. Those summary
numbers can be compared with a new replication but cannot be exactly rescored
from this checkout. Original GPU checkpoints, prediction archives and some
historical LLaVA audit assets must be obtained separately for exact replay.
There are also thesis prose/figure inconsistencies: §4.4 describes fixed LR and
white nodes, whereas the final recorded inputs use per-graph LR/TB and uniform
blue; the final zero-shot figure uses v4 while fine-tuning sections use older
extraction. Preserve these distinctions instead of changing conditions until
numbers agree. Greedy decoding and seeds do not guarantee bit-identical training
or inference across hardware/library versions.

Verified during this repository update: the thesis scope, CLI flags, frozen
artifact hashes, local dependency/Graphviz versions, three-image CPU replay and
offline rescoring against saved predictions. No dependency installation,
external data download, new model inference, training or paid GPU run was
performed. The commands for those stages are instructions, not a claim that a
fresh-machine end-to-end replication has passed.
An isolated copy of the publication files passed 299 CPU regression checks
(2 skipped); its QA-GNN regression used the existing local upstream download.
The portable five-file suite passed 117 checks without that download. Three
replayed images matched their original PNG hashes; offline checks recovered
Gemma image QA at 59.8% and all nine LLaVA v4 zero-shot scores.

## Prompt for your own code agent

Copy this prompt and fill in the requested target and available resources:

```text
Reproduce a selected result from this thesis repository.
Target: <section/table/figure, models, base/tuned arms and conditions>.
Available resources: <CPU OS/Python, local datasets/caches, GPU or cluster,
original checkpoints/predictions if any, spending limit>.
Desired outcome: <offline rescore, CPU dataset rebuild, fresh inference,
Stage 1 training, or synthetic/blank-image control>.

Read README.md, AGENTS.md, reproduction/manifest.json and the relevant entry
points. Inspect what is actually available before running anything. First map
my target to its exact inputs, cohort, seed, model revisions, renderer,
prompt/template, parser version, metric, denominator and output files. Explain
what is verified and what is missing. Distinguish exact replay from a fresh
replication and historical summary-only results. Ask about unresolved
experimental choices; never adjust conditions to make the numbers match.

Use the preserved 100-graph records for the final Stage 1 cohort and all 500
OBQA test questions for QA. Keep OBQA seed 13 and synthetic seed 0 separate.
Preserve the six-task 7200/600 training split, held-out tasks, LoRA/projector
scope, image settings and checkpoint-000450 evaluation. Select parsers explicitly
(v4 zero-shot Stage 1, template_v1 fine-tuning, explicit_v2 final zero-shot QA,
explicit_v1 Qwen synthetic, prefix logits LLaVA synthetic). Keep distinct LLaVA
prompt arms and graph-bootstrap units. Check for the documented artifact and
historical checksum requirements before promising a runnable full experiment.

Start with read-only inspection and CPU checks in a fresh output directory.
Render locally, then transfer images. Keep caches, outputs and checkpoints on
persistent /workspace storage. Do not change existing environments, overwrite
results, replace dependency pins, or bypass hash/legibility gates.

Before paid compute, follow AGENTS.md: get my explicit current approval for
resource changes after listing running/idle Pods and quoting GPU, datacenter,
tier, hourly rate, runtime, total cost and termination guard. Use one GPU Pod,
never Serverless/Flash/multi-GPU. Run the same real-model/image/collator path on
a tiny subset; report max allocated VRAM, measured speed and extrapolated cost.
Fix -> dry run -> approval -> full run. Launch long jobs inside tmux, provide
both the launch and monitoring commands, checkpoint within 30 minutes, and
report completion/failure immediately with a reminder to STOP the Pod. Obtain
confirmation before termination or deleting/shrinking persistent storage.

Return the actual commands, input/source hashes, environment versions,
completion/key checks, metrics and paired comparisons against the preserved
reference. Disclose differences and unreproduced parts. Do not claim success
from a CPU stand-in, missing predictions or a passing syntax check alone.
```

## Prepare the publication commit

`.gitignore` does not remove already tracked files or erase Git history. The
cleanup removes excluded files from the index while leaving working copies on
disk. Review the staged deletions and the allowlisted additions before committing:

```bash
git ls-files -ci --exclude-standard       # Should be empty after cleanup.
git status --short
git add -A
git diff --cached --stat
git diff --cached --check
# Then commit and push to your own configured remote.
```

Do not use `git add -f` for data, environments, checkpoints or logs. Existing
committed artifacts remain in history; this update does not rewrite history.
When adding a genuinely required script, add its exact path to `.gitignore`
and update `reproduction/files.txt` together.
