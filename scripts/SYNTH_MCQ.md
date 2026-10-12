# Synthetic MCQ: name-grounded 100-question evaluation

This protocol replaces the earlier 50-question dataset for the primary experiment.
Use knowledge_graphs_50_name_grounded_100_questions.json next to the old JSON.
The old file remains usable only when explicitly selected. No GPU jobs, Pod
operations, training, checkpoint edits or dependency installs are part of this
CPU implementation. Existing Stage 1 renderer and Qwen/adapter loaders, explicit_v1
parser, cyclic permutations and graph-level bootstrap remain unchanged.

## Dataset selection and validation

--dataset-path and --graphs are equivalent input selectors. Both resolve to an
absolute real path; if both are supplied they must resolve to the same path or
argparse fails. Without either, the default is the NEW name-grounded JSON, never
the old file. config.json and its identity record dataset_path and dataset_sha256.
There never was an existing --dataset flag in this repo; the 48-test baseline
(43 + 5 native CPU checks) replaces the handout's stale 32-test reference, as
confirmed by the user.

The loader requires 50 graphs / 100 unique questions, exactly two questions per
graph, and equality of compatibility question and questions[0]. Only questions
is iterated. Each graph has 18 nodes (4 person_name, 14 concept); every node occurs
in a triple. All choices must be the four person_name nodes; validate A-D keys,
distinct choices, gold mapping and supporting-triple membership. No reasoning_hops
or query_style is required. Observed data: 2169 triples, 40–49 per graph.
Independent full-graph solving checks both clues against each person's created
artifacts (both artifact clues must hold on the same artifact). All **100/100**
questions have **2 clue-1 matches, 2 clue-2 matches, and exactly the gold conjunction**.
Per-question matches are in solver_check.json; run.log prints the distribution.

## Fixed evidence controls

Conditions: image, text, none, wrong_graph. --wrong-graph-modality {image,text,both}
defaults to both, expanding wrong_graph into wrong_image and wrong_text. Subconditions
have separate rows, scores and paired comparisons. Question and cyclically permuted
choices never change; only evidence changes. Wrong image uses the mapped graph PNG;
wrong text uses the mapped graph's identical shuffled DOT/text triple block. Prompts
retain the same reference wording; none has no graph reference.

Pairing uses the entire 50-graph cohort even for smoke/--limit. Sort graph IDs,
shuffle with random.Random(seed), reject fixed points and any pair where either
source question's gold person occurs among the shown graph's person nodes. First
prove a valid perfect matching exists; fail explicitly otherwise. Log mapping,
draws/redraws and seed in run.log/config. With seed 0: **78,055 redraws**, 78,056
candidate draws, no fixed points or gold-name collisions. A generous 10,000,000
draw limit fails explicitly as sampling exhaustion, not as proof of impossibility.
Distractor-name overlaps are allowed by the handout's gold-name rule; near-25%
wrong-graph accuracy is an experimental expectation, not an enforced result.

## Rendering and mandatory CPU gate

Primary --render-style training: unchanged dot engine, 200 DPI, 30 pt
Helvetica-Bold nodes, 24 pt Helvetica edges, ranksep .4, nodesep .5, uniform
light-blue #ADD8E6 nodes, no answer cues. Automatic LR/TB uses the EXISTING training
selector: larger LLaVA resize scale wins; TB wins ties. Saved PNG/DOT share the
selected orientation. Preserve all nodes/triples and omit disconnected-answer
clusters. DOT edges and order-preserving kg_text are checked against the same
seeded shuffle. Compact (18/14 pt, ranksep .7, fixed LR, white fill) remains optional,
in another OUT; it is skipped by default because earlier edges failed the 10 px gate.
Do not change fonts, sizes or pixel bounds to repair a failed gate.

Use existing myvenv for Graphviz/Pillow and .venv_llava_cpu for pinned native
processor assets; never mix environments. New render/audit output is
outputs/synth_mcq_name_grounded_2026-10-10. Earlier render sets are untouched.

```bash
cd /home/henrique-leonel/home2/ba/bachelorArbeit
export CPU_PY="$PWD/myvenv/bin/python"
export ASSETS_LOCAL="$PWD/outputs/synth_mcq_name_grounded_2026-10-10"
"$CPU_PY" scripts/run_synth_mcq.py \
  --dataset-path ../knowledge_graphs_50_name_grounded_100_questions.json \
  --out "$ASSETS_LOCAL" --render-only --render-style training --seed 0
"$CPU_PY" scripts/audit_synth_mcq_images.py --measure-fonts-only --out "$ASSETS_LOCAL/audit/font_references"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv_llava_cpu/bin/python -u scripts/audit_synth_mcq_images.py \
  --dataset-path ../knowledge_graphs_50_name_grounded_100_questions.json \
  --renders "$ASSETS_LOCAL" --synthetic-only \
  --qwen-processor-assets outputs/qwen_stage1_assets/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --font-references "$ASSETS_LOCAL/audit/font_references/font_references.json" --out "$ASSETS_LOCAL/audit"
```

The audit uses processor-owned resize logic with min_pixels **262144**,
max_pixels **1310720**, pinned transformers **5.16.1**; it loads no model weights.
Measure independent H glyphs in the actual node AND edge fonts, following unchanged
image_audit.py: threshold ink at 128. Reference caps at 200 DPI: **60 px for
30 pt Helvetica-Bold**, **48 px for 24 pt Helvetica**. Scale each by the actual
vertical resize factor, not a nominal token budget. **1224** was one old OBQA
image's realised token count, not a fixed count. Each graph's original/resized
size, grid, realised tokens, node/edge caps, engine and selected orientation are
in audit/synthetic_training_qwen_processor.csv and .jsonl. Summary and hashes are
in audit/qwen_processor_audit_config.json; gate counts are in legibility_gate.json.
These are reference-cap estimates, not direct measurements of every label's ink.

If any graph's node or edge cap is below 10 px, the audit writes all reports then
exits 2. Stop, report failed IDs/counts, and wait for the user's decision. No font
or size adjustment is permitted. Future GPU inference requires --image-audit
with a passing legibility_gate.json matching data, render manifest, fonts, style,
50 images and pixel bounds; this extra guard implements the requested stop rule.

## Model loading and row identity

Qwen/Qwen3-VL-8B-Instruct pinned revision
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; bf16, SDPA, native user-only chat template,
batch 1, greedy generation, max_new_tokens **64**. Require existing transformers
5.16.1 / peft 0.21.2; do not change versions. Use the complete checkpoint-000450
directory with checkpoint_manifest.json, run_config.json, projector.pt and adapter/.
Base and tuned load sequential pristine models. Base loading never invokes PEFT;
tuned restores LoRA plus all 4 visual mergers via exactly:

```python
from llava_adapter import load_adapter
model, provenance = load_adapter(model, args.adapter, args.model_id, args.revision)
```

Log base no-PEFT assertion, restored LoRA/merger counts >0, peak VRAM and first-position
A-D logits. Delete model/processor, gc.collect(), torch.cuda.empty_cache() between
loads. No disable_adapter path. None of this loading logic or parser was changed.
Row identity includes prompt style, model, condition, graph, QUESTION ID and permutation.
Record question IDs/count, derived relation pair, shown_graph_id, render style/font,
model revision, parse tier, raw response, scores and first-position logits/probabilities.
Rows flush immediately. --resume spans both model loads and growing --limit; reject
changed data/code/render/budget/style/permutations/checkpoint or a torn/duplicate row.
--prompt-style retains the old single-style spelling; --prompt-styles accepts obqa
or obqa,needed in one invocation/config. The recommended Pod run uses **obqa only**,
because the previous run found wording did not matter. Needed remains available.

## Final prompt bodies

These are the exact inherited wording and separators, before Qwen's native chat
template. `{triples}` is the shuffled, parenthesized triple block, one per line.
The inherited graph-reference sentences include a trailing ASCII space; for
image it separates the sentence from `Question:`, and for text it precedes the
first newline. `<image>` becomes the native vision block.

Image:

```text
<image>
The image represents a knowledge graph relevant to the question, which may or may not be useful. Question: {question}
Choices:
A. {node_A}
B. {node_B}
C. {node_C}
D. {node_D}
Answer with the correct option's letter.
```

Text:

```text
The following knowledge graph is relevant to the question, which may or may not be useful. 
{triples}

Question: {question}
Choices:
A. {node_A}
B. {node_B}
C. {node_C}
D. {node_D}
Answer with the correct option's letter.
```

None:

```text
Question: {question}
Choices:
A. {node_A}
B. {node_B}
C. {node_C}
D. {node_D}
Answer with the correct option's letter.
```

Prompt bodies for all four permutations are generated during local rendering and
saved in `render_manifest.json`; the GPU runner reads them without importing the
Graphviz renderer or requiring the Python graphviz package on the pod.
The needed-style reference sentences are exactly:

```text
The image represents a knowledge graph relevant to the question, which contains the information needed to answer the question. 
The following knowledge graph is relevant to the question, which contains the information needed to answer the question. 
```

Style replacement is limited to the reference prefix; it never rewrites a
matching phrase inside the question or triples. Existing renders/manifests can
be reused; prompt bodies remain stored as obqa and are transformed at inference.

## Full grid and prepared Pod scripts (do not run during CPU work)

- 100 questions × 4 permutations × 2 prompt styles × 5 conditions × 2 models = **8000 rows**.
- --prompt-styles obqa: 100 × 4 × 1 × 5 × 2 = **4000 rows**; recommended.
- Smoke --limit 2 (graphs = 4 questions), --n-perms 1: **40 rows** with obqa,
  **80 rows** with both prompt styles. Every condition and both models are exercised.

Repo equivalents of /workspace/smoke.sh and /workspace/full.sh are
scripts/synth_mcq_name_grounded_smoke.sh and scripts/synth_mcq_name_grounded_full.sh.
They use audited training assets, offline cached weights, full checkpoint ADAPTER,
all five cells, sequential pristine loads, resume and run.log + /workspace/logs links.
They reject a failed gate. Smoke/full use distinct OUTs because n_perms differs.
Compact is never launched by these scripts. No concurrent writers to one OUT.

Only after the image gate passes, upload to an existing approved Pod:

```bash
export POD_SSH='USER@HOST'
export POD_SSH_PORT='PORT'
rsync -av -e "ssh -p $POD_SSH_PORT" scripts/run_synth_mcq.py scripts/analyze_synth_mcq.py scripts/synth_mcq_common.py "$POD_SSH:/workspace/bachelorArbeit/scripts/"
rsync -av -e "ssh -p $POD_SSH_PORT" experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/prompt_common.py "$POD_SSH:/workspace/bachelorArbeit/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/"
rsync -av -e "ssh -p $POD_SSH_PORT" ../knowledge_graphs_50_name_grounded_100_questions.json "$POD_SSH:/workspace/"
# Existing /workspace/outputs parent must be present.
rsync -av -e "ssh -p $POD_SSH_PORT" "$ASSETS_LOCAL/" "$POD_SSH:/workspace/outputs/synth_mcq_name_grounded_assets/"
rsync -av -e "ssh -p $POD_SSH_PORT" scripts/synth_mcq_name_grounded_smoke.sh "$POD_SSH:/workspace/smoke.sh"
rsync -av -e "ssh -p $POD_SSH_PORT" scripts/synth_mcq_name_grounded_full.sh "$POD_SSH:/workspace/full.sh"
```

On Pod, enter tmux, export the full checkpoint directory, then smoke:

```bash
tmux new-session -A -s synth-mcq-name-grounded
```

Inside tmux:

```bash
export ADAPTER=/workspace/PATH_TO/checkpoint-000450
export PROMPT_STYLES=obqa
bash /workspace/smoke.sh
```

Review peak VRAM, per-cell timing, token counts/ceiling, parser tiers and the
same-input base/tuned logit sanity. Full needs 400 rows per model/condition per
prompt style. Estimate full time by sum(400 × smoke mean row seconds per cell),
plus measured loads; use a +50% termination guard. Calculate cost with the existing
Pod's actual rate. Per AGENTS.md, obtain full-run approval after smoke/fixes.
Smoke does not auto-start full. Prepared full command, only after those checks:

```bash
nohup bash /workspace/full.sh > /workspace/logs/synth_mcq_name_grounded_launcher.log 2>&1 &
echo "PID=$!"
```

Monitor from separate shells:

```bash
tail -f /workspace/logs/synth_mcq_name_grounded_smoke.log /workspace/logs/synth_mcq_name_grounded_full.log
```

```bash
watch -n 15 'wc -l /workspace/outputs/synth_mcq_name_grounded_full/predictions.jsonl; echo "expected: 4000 obqa, 8000 both"'
```

```bash
watch -n 30 nvidia-smi
```

Stop the Pod when inference finishes or fails. Download predictions/config/run.log,
solver/pairing/audit reports, manifests/graphs and analysis artifacts. Model weights
and checkpoints need not be downloaded.

## CPU analysis and interpretation

```bash
"$CPU_PY" scripts/analyze_synth_mcq.py \
  --pred outputs/NAME_GROUNDED_DOWNLOAD/predictions.jsonl \
  --out outputs/NAME_GROUNDED_ANALYSIS --n-perms 4
```

The output header states the headline is a graph-level bootstrap (50 clusters,
roughly +/-14 pp worst-case 95% uncertainty). Within each complete graph average
its two questions × four permutations; then average graphs. Resample graphs,
never individual questions/permutations. Incomplete graphs/cells are explicitly
excluded, including a completely missing second question. This is unchanged
bootstrap logic with the row population expanded to two questions per cluster.

summary.csv: both parsed-letter and raw first-position A-D argmax accuracies for
image/text/none/wrong_image/wrong_text, per model and prompt style. Parser remains
explicit_v1 (exact, sole_letter, option_prefix, explicit_answer, bold_letter, FAILED).
Failed parsing scores incorrect. Four-token softmax is conditional on bare A-D.
Parse-tier counts, parse failures, letter-position counts and plots remain.

per_graph.csv: each graph/model/condition/style/metric accuracy across its **8 rows**.
paired_comparisons.csv: tuned-base for every cell; existing image-text; image-none,
text-none; image/text against BOTH wrong subconditions, with graph bootstrap CIs
and improved/worsened/tied counts. A sign change in graph mean is the same as a
sign change in the sum of the two questions' permutation-averaged accuracies.
Permutation discordances remain descriptive, not independent McNemar trials.

question_type_breakdown.csv: sorted pair of non-createdby supporting relations,
**11 pairs**, explicitly labelled **descriptive, small n**. The metadata's 8-type
catalogue is not a per-question label and is not used to invent one. Report question,
graph and row counts, and accuracy, without type-specific headline intervals.
No unrequested subset analysis or fine-tuning is introduced.

none_style_agreement.json checks output/score/token/logit equality between prompt
styles by model/graph/QUESTION/permutation, excluding timing/memory/style label.
Both styles have **800 matched none pairs**. Obqa-only correctly reports no matched
pairs, not agreement. Real-model none/wrong-graph chance behavior is unverified.

## CPU tests

```bash
"$CPU_PY" -m pytest -q tests/test_synth_mcq.py tests/test_synth_mcq_name_grounded.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv_llava_cpu/bin/python -m pytest -q tests/synth_mcq_inference_cpu_checks.py
```

All **68 CPU tests passed**: **63 renderer/analysis checks in 53.14 s** and
**5 native tiny-Qwen checks in 4.61 s**. The 48 baseline tests are preserved unchanged;
20 new checks were added. All 10 documented Bash blocks and both launch scripts
passed syntax checks. New checks cover strict schema/counts,
no duplicate compatibility question, independent solver and deliberate mismatches,
derangement feasibility/collisions/reproducibility, dataset aliases/hash identity,
4000/8000 grids, wrong-evidence prompt integrity, known graph-cluster paired CIs,
missing whole questions, audit gate rejection, and two-question/two-style/five-cell
resume with CPU stand-ins. Full 4000/8000-row fake analysis is run separately and
marked FAKE; its accuracies are not model findings. Original training renderer,
loader/checkpoint and parser are unmodified. Local renderer source hashes and every
PNG/DOT/text file are verified before upload/inference.

## New-dataset audit result (2026-10-10)

Gate: **PASS**. Nodes below 10 px: **0/50**. Edges below 10 px: **0/50**.

Unchanged dot renderer selected **50 LR / 0 TB**. Fonts 30 pt Helvetica-Bold / 24 pt Helvetica, 200 DPI. Processor pixel bounds: 262144–1310720.

| Font | Cap min / median / max (px) | Below 10 px |
| --- | --- | --- |
| node | 13.239 / 15.160 / 17.618 | 0/50 (0%) |
| edge | 10.591 / 12.128 / 14.095 | 0/50 (0%) |

Tokens min / median / max: **1218 / 1248 / 1276**. Independent reference glyph heights: 60 px node, 48 px edge. These are reference-cap estimates scaled by actual processor output, not measured readability.

| Graph | Resized W×H | Tokens | Node cap px | Edge cap px | Engine | Orientation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| G01 | 1088×1184 | 1258 | 13.741 | 10.993 | dot | LR |
| G02 | 928×1376 | 1247 | 15.334 | 12.267 | dot | LR |
| G03 | 896×1408 | 1232 | 15.299 | 12.239 | dot | LR |
| G04 | 992×1280 | 1240 | 15.080 | 12.064 | dot | LR |
| G05 | 992×1312 | 1271 | 16.305 | 13.044 | dot | LR |
| G06 | 1024×1248 | 1248 | 13.887 | 11.110 | dot | LR |
| G07 | 992×1280 | 1240 | 13.434 | 10.747 | dot | LR |
| G08 | 960×1344 | 1260 | 17.172 | 13.738 | dot | LR |
| G09 | 1024×1248 | 1248 | 14.734 | 11.787 | dot | LR |
| G10 | 1024×1248 | 1248 | 13.422 | 10.737 | dot | LR |
| G11 | 992×1312 | 1271 | 16.875 | 13.500 | dot | LR |
| G12 | 896×1408 | 1232 | 17.171 | 13.737 | dot | LR |
| G13 | 896×1408 | 1232 | 15.955 | 12.764 | dot | LR |
| G14 | 1056×1184 | 1221 | 14.962 | 11.970 | dot | LR |
| G15 | 992×1280 | 1240 | 14.275 | 11.420 | dot | LR |
| G16 | 928×1376 | 1247 | 16.397 | 13.118 | dot | LR |
| G17 | 928×1408 | 1276 | 17.618 | 14.095 | dot | LR |
| G18 | 1056×1216 | 1254 | 13.353 | 10.682 | dot | LR |
| G19 | 1024×1248 | 1248 | 13.546 | 10.836 | dot | LR |
| G20 | 992×1312 | 1271 | 16.770 | 13.416 | dot | LR |
| G21 | 992×1280 | 1240 | 15.241 | 12.193 | dot | LR |
| G22 | 960×1312 | 1230 | 17.072 | 13.658 | dot | LR |
| G23 | 992×1280 | 1240 | 15.993 | 12.795 | dot | LR |
| G24 | 960×1344 | 1260 | 17.257 | 13.805 | dot | LR |
| G25 | 1024×1248 | 1248 | 13.239 | 10.591 | dot | LR |
| G26 | 992×1312 | 1271 | 13.962 | 11.170 | dot | LR |
| G27 | 1024×1248 | 1248 | 15.260 | 12.208 | dot | LR |
| G28 | 992×1280 | 1240 | 15.661 | 12.529 | dot | LR |
| G29 | 960×1344 | 1260 | 16.821 | 13.457 | dot | LR |
| G30 | 1056×1216 | 1254 | 13.614 | 10.892 | dot | LR |
| G31 | 992×1280 | 1240 | 16.678 | 13.342 | dot | LR |
| G32 | 960×1344 | 1260 | 16.350 | 13.080 | dot | LR |
| G33 | 992×1280 | 1240 | 16.502 | 13.202 | dot | LR |
| G34 | 864×1472 | 1242 | 16.493 | 13.194 | dot | LR |
| G35 | 1088×1184 | 1258 | 13.373 | 10.699 | dot | LR |
| G36 | 1088×1184 | 1258 | 13.797 | 11.037 | dot | LR |
| G37 | 800×1568 | 1225 | 14.348 | 11.478 | dot | LR |
| G38 | 992×1280 | 1240 | 14.178 | 11.342 | dot | LR |
| G39 | 992×1280 | 1240 | 14.257 | 11.405 | dot | LR |
| G40 | 992×1312 | 1271 | 16.980 | 13.584 | dot | LR |
| G41 | 1024×1248 | 1248 | 14.720 | 11.776 | dot | LR |
| G42 | 864×1472 | 1242 | 15.544 | 12.435 | dot | LR |
| G43 | 928×1376 | 1247 | 16.842 | 13.474 | dot | LR |
| G44 | 1024×1248 | 1248 | 14.778 | 11.822 | dot | LR |
| G45 | 992×1312 | 1271 | 15.650 | 12.520 | dot | LR |
| G46 | 1056×1216 | 1254 | 14.453 | 11.563 | dot | LR |
| G47 | 896×1440 | 1260 | 14.964 | 11.971 | dot | LR |
| G48 | 928×1344 | 1218 | 13.449 | 10.759 | dot | LR |
| G49 | 928×1408 | 1276 | 14.496 | 11.596 | dot | LR |
| G50 | 832×1536 | 1248 | 14.509 | 11.607 | dot | LR |

## Historical LLaVA audit stop (2026-10-10; superseded below)

At this earlier stop, the backend and Pod session scripts were pending the user's
decision: all **50/50** original synthetic graphs failed the then-proposed 10 px
gate for both fonts. No GPU run was performed. The accepted 2026-10-11 render
and replacement training-range gate are documented below; these historical
numbers do not gate the new LLaVA assets.

CPU audit command (exit code 2 denotes the reported gate failure):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv_llava_cpu/bin/python scripts/audit_synth_mcq_llava_images.py \
  --out outputs/synth_mcq_name_grounded_llava_audit_2026-10-10
```

The audit loads only the cached processor/config/tokenizer, pinned to
`2424fdd47412fccc66d91719126b420e9fbd7065`, with transformers 5.16.1. It reuses
`llava_common.configure_processor`, `prepare_inputs`, `image_diagnostics`, and
`llava_stage1_prompt.format_llava_prompt(..., 'llava_v1')`. This is the template
selected by the trained Stage 1 QA evaluation, rather than its CLI's `hf-chat`
default. Inference would reuse the existing `LlavaNextForConditionalGeneration`
load in `run_zero_shot_llava.py` and
`llava_adapter.load_adapter(model, checkpoint, model_id, revision)`.

The measured H references are each font's own glyph: Helvetica-Bold 30 pt = 60 px,
Helvetica 24 pt = 48 px at 200 DPI. The primary cap estimate uses the actual
unpadded content height observed from the processor's `_resize_for_patching`,
not the padded grid height. The separate 336 px base view is informational only.
No graphs or reference glyphs were re-rendered.

| Set/view | Node min / median / max px | Nodes below 10 px | Edge min / median / max px | Edges below 10 px |
| --- | --- | ---: | --- | ---: |
| Synthetic anyres | 6.149 / 7.921 / 8.756 | 50/50 | 4.919 / 6.337 / 7.005 | 50/50 |
| Synthetic base | 3.075 / 3.961 / 4.378 | 50/50 | 2.460 / 3.169 / 3.502 | 50/50 |
| Stage 1 sample anyres | 7.406 / 10.914 / 17.880 | 17/50 | 5.925 / 8.731 / 14.304 | 37/50 |
| Stage 1 sample base | 4.636 / 9.447 / 16.102 | 31/50 | 3.708 / 7.558 / 12.882 | 42/50 |

The reference sample is the first 50 distinct training images in ascending
statement-index order, verified against the training data manifest; it is
descriptive, not a whole-training-set result. Synthetic images all select the
672×672 grid (height×width). Synthetic vision tokens min / median / max:
1872 / 2352 / 2736. Training sample: 1744 / 2182 / 2928.
Per-image dimensions, grids, tokens and both fonts/views are saved in
`outputs/synth_mcq_name_grounded_llava_audit_2026-10-10/images.csv` and
`image_audit.md`. `legibility_gate.json` records the failure.

Dataset and combined PNG/DOT/TXT checksums match the handout. Pairing matches the
actual Qwen run config exactly; unchanged pairing-file SHA256:
`8260de2296f5f4680e8d439dd6bf9df629d15004ee2dba5f9751f26cb0d8c6ba`.
The fetched artifact has 448 LoRA tensors. Its composite adapter/projector SHA256,
computed using the existing loader's sorted path/NUL/file-SHA/newline method,
matches the saved trained evaluation:
`6864483da020dc07665ca91d611ea26421ce4e416413074000d8d1b556258c83`.

One additional deployment limitation must be resolved before writing the upload
plan: the laptop's fetched checkpoint contains only adapter files and projector,
without `checkpoint_manifest.json`, `run_config.json` or training state. The
existing `llava_adapter.verify_checkpoint` requires the complete manifest's file
set. Uploading this sparse directory alone would fail the mandated loader; the
complete checkpoint on the Pod volume has not been checked. No manifest was
fabricated and checkpoint validation was not weakened.

Pinned token variants (bare / SentencePiece): A 28741/330, B 28760/365,
C 28743/334, D 28757/384. These are tokenizer observations, not inference results.
No projector-versus-base or logits-versus-generation agreement was tested.
Training versions: torch 2.8.0+cu128, transformers 5.16.1, peft 0.21.2,
accelerate 1.15.0. CPU audit uses torch 2.8.0+cpu.

Pending run grid remains 4000 full rows or 40 smoke rows. Without a GPU smoke,
there is no measured runtime estimate: estimate full generation time as the sum
over all ten model/condition cells of 400 × that cell's mean smoke seconds per
row, including both wrong-graph cells, then add measured loading/other overhead.
No Pod session commands have been added after this blocking failure.

CPU checks after adding the audit: **68 passed in 53.32 s** (the existing 63 plus
five new audit tests), and **5 native inference CPU checks passed in 4.61 s**.
All 68 existing tests remain passing, for 73 CPU tests in total.

## Authorized CPU render sweep for LLaVA (2026-10-10)

The subsequent CPU sweep changes only new candidate renders. Original Qwen
PNG/DOT/TXT checksums remain unchanged. Full results and the font-effect plot:
`outputs/synth_mcq_llava_legibility_sweep_2026-10-10/report.md`.

The sweep requested 72 configurations: font pairs 30/24, 42/34, 60/48, 84/68 pt;
dot LR/TB at standard spacing (nodesep .5, ranksep .4) or tight spacing (.2, .15),
plus neato/fdp; natural shape or numeric Graphviz ratio 1/3 or 3. This local
Graphviz build exposes only dot. Thus **48 configurations / 2400 complete graph
layouts** succeeded; 24 non-dot configurations were unavailable. No dependency
changes were made. All 50 graphs are evaluated for every supported configuration,
with every node/triple/label retained and the same seeded triple order as Qwen.

The closest tested candidate reaching both training-reference medians uses:
**dot, fixed LR, 42/34 pt, nodesep .2, ranksep .15, natural aspect ratio, 200 DPI**.
Uniform light-blue fill is retained. Assets are separate under
`outputs/synth_mcq_llava_legibility_sweep_2026-10-10/candidates/dot_lr_tight_natural_n42_e34/`.
Its `variant_manifest.json` records settings, file hashes and per-graph processor
measurements; it is a sweep artifact, not the existing runner's render manifest.
No inference default or backend has been changed.

| Render | Node min / median / max px | Nodes <10 px | Edge min / median / max px | Edges <10 px |
| --- | --- | ---: | --- | ---: |
| Training reference (50 images) | 7.406 / 10.914 / 17.880 | 17/50 | 5.925 / 8.731 / 14.304 | 37/50 |
| Original synthetic | 6.149 / 7.921 / 8.756 | 50/50 | 4.919 / 6.337 / 7.005 | 50/50 |
| Tight LR 42/34 | 8.931 / 11.429 / 13.926 | 10/50 | 7.230 / 9.252 / 11.273 | 38/50 |
| Tight LR 84/68 | 9.603 / 12.374 / 15.026 | 5/50 | 7.878 / 10.151 / 12.327 | 23/50 |

The 42/34 candidate's median caps are 4.7% (nodes) and 6.0% (edges) above the
reference medians; all 50 graphs' estimates lie inside the reference sample's
observed node and edge ranges. Thirty graphs reach both reference medians
individually. Realised vision tokens min / median / max: **1948 / 2438 / 2928**.
This matches the reference range, while **38 graphs still fail the previous
10 px all-font gate**. Matching training medians and passing that gate are
separate criteria; no override was enabled.

Larger fonts have diminishing returns as they enlarge the canvas. Tight LR
spacing helps substantially, while forcing wide/tall aspect ratios reduced
the best attainable medians in this sweep. Screening used the processor-owned
resize-size calculation; all 150 finalist PNGs were then verified through
the actual pinned CPU processor, with exact dimension/resize agreement.
Own-font H glyphs were measured separately for nodes and edges.

Processed tile-grid views for G01, G25 and G50 are saved beside each finalist.
Visual review found dense crossings and overlapping at-location labels in G50.
Cap estimates alone do not establish task accuracy or absence of crowding.

CPU reproduction, using a fresh output directory:

```bash
SWEEP_OUT=outputs/synth_mcq_llava_legibility_sweep_repeat
myvenv/bin/python scripts/sweep_synth_mcq_llava_legibility.py --phase render --out "$SWEEP_OUT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv_llava_cpu/bin/python scripts/sweep_synth_mcq_llava_legibility.py --phase score --out "$SWEEP_OUT"
myvenv/bin/python scripts/sweep_synth_mcq_llava_legibility.py --phase png --out "$SWEEP_OUT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv_llava_cpu/bin/python scripts/sweep_synth_mcq_llava_legibility.py --phase audit --out "$SWEEP_OUT"
myvenv/bin/python scripts/sweep_synth_mcq_llava_legibility.py --phase report --out "$SWEEP_OUT"
```

Any later LLaVA experiment must use the same selected renders for base and tuned.
Its images would differ from Qwen's, so a comparison between model families
would also include the render difference. Backend/deployment work was still
pending at the sweep stage; its completed implementation follows below.
The sweep started no GPU or Pod jobs.

CPU tests: **74 passed in 53.81 s**, plus **5 native inference checks passed in
4.68 s**. All 73 existing checks remain passing, with six new sweep tests.

## Adopted LLaVA backend and session plan (2026-10-11)

The accepted LLaVA variant is `dot_lr_tight_natural_n42_e34`: dot, fixed LR,
42/34 pt, nodesep .2, ranksep .15, natural shape, 200 DPI, light-blue nodes.
All 50 new renders are in
`outputs/synth_mcq_name_grounded_llava_renders_2026-10-11/graphs/`.
Every supplied node, triple and label was verified; no disconnected-answer
cluster is added. All 50 DOT triple sequences and TXT bytes match the original
Qwen assets. All 50 PNGs also match the chosen sweep candidate byte for byte.
The Stage 1 renderer itself is unchanged.

Combined `sha256sum G*.png G*.dot G*.txt | sha256sum`, inside this new folder:
`2317d3b134e9390f987e4a233438617da8e119a0a288e242c21168e8689903e1`.
Dataset SHA256:
`9e0742bc624e1fb96a97e543fb6d2283cd3932ded03f9bfc084596dd541a9da2`.
Unchanged wrong-graph pairing SHA256:
`8260de2296f5f4680e8d439dd6bf9df629d15004ee2dba5f9751f26cb0d8c6ba`.
Both launch scripts check these three hashes, every render-file manifest hash,
and the reference-range gate before inference.

Qwen uses its existing 30/24 pt training renders and retains its results.
LLaVA uses these different 42/34 pt renders for both base and tuned. The
within-LLaVA tuned-versus-base contrast uses identical images. LLaVA image
accuracy is **not directly comparable** with Qwen image accuracy.

The entrypoint adds `--backend {qwen,llava}`, default `qwen`. The prior Qwen
implementation is frozen verbatim in `_synth_mcq_qwen_v4.py`; its hash is
`0a81473b66fc57df681db1cf3e58a3904cf81ca5d9126b7317a974d0eb87487a`.
For existing Qwen resume identities the historical `scripts/run_synth_mcq.py`
code-hash key identifies that frozen execution source; the dispatcher is
excluded. A CPU test compares default arguments, serialized run identity and
validation output byte for byte against the prior implementation. No Qwen
inference, render change, or result rewrite was performed.

The LLaVA branch reuses `run_zero_shot_llava.format_qa_prompt`, which calls
`llava_stage1_prompt.format_llava_prompt(..., 'llava_v1')`, together with
`llava_common.configure_processor`, `prepare_inputs` and `image_diagnostics`.
The existing QA eval constructs the native base inline; it has no separate
reusable base-loader function. This branch uses that same
`LlavaNextForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16,
device_map='auto').eval()` API/settings, pinned to
`llava-hf/llava-v1.6-mistral-7b-hf` revision
`2424fdd47412fccc66d91719126b420e9fbd7065`. `local_files_only=True` makes
missing cached resources fail instead of downloading them during a job.
Tuned loading is exactly:

```python
from llava_adapter import load_adapter
model, provenance = load_adapter(model, args.adapter, args.model_id, args.revision)
```

The strict loader is unchanged. Base is a pristine load with zero PEFT/LoRA
modules; tuned is a second pristine load plus all 448 LoRA tensors and the
checkpoint projector. Each projector tensor is checked against the checkpoint
and its loaded digest must differ from the pristine projector. Both
`LOAD SANITY` records include hashes/provenance; `BASE/TUNED LOGIT SANITY` requires
different A–D logits for the same input. Model/processor deletion, garbage
collection and `torch.cuda.empty_cache()` run between loads and log allocated
memory. These GPU checks are implemented but have not been executed locally.
`--resume` supports completing base first and adding tuned later.

The adapter/projector composite hash is computed by the existing strict
`adapter_provenance` method: sorted adapter/projector relative paths, NUL,
verified file SHA256, newline. The sparse laptop files reproduce
`6864483da020dc07665ca91d611ea26421ce4e416413074000d8d1b556258c83`, identical
to the saved trained QA dry-run provenance. This is the adapter/projector
composite, distinct from the training checkpoint's whole-tree checksum.
It does **not** make the sparse laptop directory a loadable full checkpoint.

### Forced-choice probe, smoke gate, and parsers (CPU fix, 2026-10-11)

`explicit_v1` stays the headline parser, unchanged. The first-position A–D
logits also stay; they probe an answer decision only when the reply starts
with a letter. A prose reply's first token (usually “The”) is not its answer.
The LLaVA-only `first_generated_letter=None` for “C.” / “B.” was a **bug**:
its parse-tier filter omitted `sole_letter`. It now comes from the actual
first generated token and the verified letter IDs, independently of parsing.
A reply starting with punctuation rather than a letter still has `None`.
Historical predictions are retained, not rewritten.

Each row now has one extra forward pass: append the teacher-forced assistant
prefix **`The correct answer is`** (without trailing space) to the original
input token IDs, extend its attention mask, preserve image tensors/sizes,
and score the next position. No prefix is added to normal generation.
Both probes use the maximum over the verified variants: A `28741/330`,
B `28760/365`, C `28743/334`, D `28757/384`. Predictions record
`prefix_letter_logits`, raw variant logits, `prefix_logit_argmax_letter`,
node/correctness, `prefix_token_ids`, `generated_token_ids`,
`generated_prefix_token_count`, and `prefix_generated_letter`. `config.json:prefix_probe` records the prefix
string, its native tokenizer IDs, the letter IDs and scoring rule.
The native CPU test verifies that appending those prefix IDs matches encoding
the assistant prompt plus a space and the prefix, without a second BOS/EOS.

The LLaVA 95% per-condition first-position gate is removed. The new
`llava_prefix_agreement_exact_ties_v2` gate qualifies replies beginning with the decoded prefix whose
**immediately next generated token** is one of
the eight verified A–D IDs. The runner finds the prefix boundary by decoding generated token prefixes;
the report checks the token immediately after that boundary using generated
IDs, rather than the headline parse tier or a letter found later in prose.
Different tokenizations of the same decoded prefix also qualify.
Require **100% pooled agreement**: the generated letter’s saved probe logit
must equal the maximum of the four saved probe logits **exactly**. Exact
ties count as agreement; no epsilon or tolerance is used. Keep **minimum
n=20** across both models and all five conditions. Twenty is half the fixed 40-row smoke: it prevents a handful of
rows passing this consistency check while allowing the base model's varied
answer formats. It is an engineering consistency check, not an accuracy or
statistical-confidence claim. Neither model nor condition has its own
minimum or pass threshold. Per-condition n/matches/fraction are informational.
The report also retains load/logit sanity, completeness, ten-cell timings,
peak memory and extrapolated full time. It adds per-model/condition headline
parse-tier counts, both parser summaries, and first/prefix probe accuracies.
The pooled/per-condition report gives `n`, `matches`, `tied_rows`, and
`non_tie_mismatches`. A tied row has multiple letters at the maximum; a
non-tie mismatch has a generated score below that maximum (a tie between
other letters does not excuse it). Missing probe data also fails. Pass only
if pooled n >= 20 and non_tie_mismatches == 0, with the existing completeness
and base/tuned sanity checks. Insufficient n or missing prefix data fails.
The Qwen gate and backend are unchanged.

### Post hoc exact-tie amendment and offline re-gating

The exact-tie gate rule is **post hoc, defined after the first prefix smoke**
in `synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11`. That smoke had
21/22 argmax-letter agreements. In its tuned/wrong_text/G01/Q51/perm0 row,
probe B=C=16.0, generated C, gold D. The original deterministic probe argmax
was B. The new consistency rule accepts C at the exact maximum, while its
probe accuracy remains incorrect.

`prefix_logit_argmax_letter` keeps the existing tie-break: **choose the first
maximum in A, B, C, D order**. This defines argmax accuracy and is unchanged.
`prefix_logit_tie` is a boolean per new/enriched row: true when two or more
**letters** share the maximum per-letter logit (not when a letter’s two token
variants tie). The analyzer reports tied-row counts beside unchanged argmax
accuracy in `llava_prefix_probe_summary.csv` and stdout, per model/condition
over all available probe rows. The smoke gate’s tied-row count includes only
qualifying generated-prefix rows.

The saved prefix smoke is **re-gateable offline, without another GPU run**:

```bash
# Laptop, repository root; no model/tokenizer/weights are loaded.
CUDA_VISIBLE_DEVICES='' myvenv/bin/python scripts/regate_synth_mcq_llava.py \
  --smoke-out outputs/synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11
```

Verified result: **n=22, matches=22, tied_rows=3, non_tie_mismatches=0,
passed=true**. The three ties include two that already agreed under the old
tie-break. The script preserves `predictions.jsonl`, `config.json`, and the
original failed `smoke_report.json` byte-for-byte. It writes
`smoke_report_regated.json` (new gate/accuracy/tie summaries and original
inference identity, timings, provenance) and `predictions_regated.jsonl`
(same 40 rows with derived `prefix_logit_tie` flags). It verifies row mappings,
argmax correctness and saved first-position sanity without loading models.
The report records source prediction/report hashes and re-gating code hashes.

**The updated full script accepts this re-gated report; a GPU smoke rerun is
not required for this exact-tie amendment.** It prefers
`$SMOKE_OUT/smoke_report_regated.json` when present, recomputes the gate from
original `predictions.jsonl`, checks its saved checksum, sanity and checkpoint,
and accepts only exact identity or the known first-prefix helper hash
`4f6f43b959384b130e826598081633f7c8353ef0292c131aa8906787a8685d66`
changing to this helper plus the gate protocol update. All other inference
code hashes, assets, dataset, settings and checkpoint must still match.
The saved inference identity is never rewritten to claim the new code ran
on the GPU. Unknown code drift still requires a fresh smoke. `--resume` of
the old inference run is not an offline re-gating command and still rejects
code identity drift. Separate full-run approval under `AGENTS.md` is required.

Transfer exactly these **six runtime/guide files** from the laptop repository
root (backend, parsers, loader, renders and pairing need no transfer):

```bash
export RSYNC_RSH="ssh -p $POD_PORT"
rsync -rlt --no-owner --no-group \
  scripts/synth_mcq_llava_common.py scripts/analyze_synth_mcq.py \
  scripts/regate_synth_mcq_llava.py scripts/synth_mcq_name_grounded_llava_full.sh \
  scripts/synth_mcq_name_grounded_llava_session_eval.sh scripts/SYNTH_MCQ.md \
  "root@$POD_HOST:/workspace/bachelorArbeit/scripts/"
# Copy the locally re-gated report and optional enriched predictions to the pod.
rsync -rlt --no-owner --no-group \
  outputs/synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11/smoke_report_regated.json \
  outputs/synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11/predictions_regated.jsonl \
  "root@$POD_HOST:/workspace/outputs/synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11/"
```

Keep `SMOKE_OUT=/workspace/outputs/synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11`
in any later approved full/eval tmux invocation. The original three files
must remain there; no overwrite of them is needed. The session-eval wrapper
also prefers the new report and checks the current gate protocol. This task
ran only CPU work; no transfers, pod operations, GPU jobs or full run occurred.

`explicit_v1_llava_fallback` is a **secondary, post hoc parser**: only when
`explicit_v1` returns FAILED, accept a unique uppercase `is <Letter>.`
(`is_letter`), then a reply naming exactly one distinct choice name
(`choice_name`, case-insensitive whole-name boundaries; repeated mentions of
that same name are allowed). Ambiguous letters/names fail. Both added tiers
were defined from the original **40 smoke rows** and are not independently
validated; name mentions can reflect explanation rather than a decision.
Never replace headline `explicit_v1` accuracy with fallback accuracy.
`llava_parser_summary.csv` and the smoke JSON give parsed/total answer rate,
correct/total accuracy (failed parses count wrong), and correct/parsed accuracy
for each model/condition and both parsers. The analyzer adds fallback and
`prefix_logit_argmax` graph-bootstrap metrics and paired comparisons only for
LLaVA; old LLaVA rows can be analyzed for fallback, but acquire no invented
prefix logits. It saves `llava_prefix_agreement.json` as a diagnostic.

The original first-position-only `synth_mcq_name_grounded_llava_smoke` has
**20 base + 20 tuned** rows: 18 tuned and 5 base
replies begin with the prefix in decoded text (23 textual candidates).
Token-level qualification and agreement cannot be recovered from those old
rows because they lack prefix logits and generated token IDs. They need a
fresh smoke; the old failed report is not converted into a passing report.
The four verbatim parser fixtures are in
`tests/fixtures/synth_mcq_llava_smoke_replies.jsonl` (base `is D. Matthew`,
base Samuel-name prose, tuned `The correct answer is B. Andrew.`, tuned `C.`).

The extra pass reprocesses the original context and image, with no KV-cache
reuse, and adds **zero rows**: still 40 smoke / 4000 full. No GPU measurement
was run for this fix. The old smoke averaged 0.3778 s/row. A provisional
budgeting assumption is **+0.05–0.20 s/row** (about +13–53%, or +200–800 s
for 4000 rows, excluding loading). This is a rough estimate for one extra
prefill, not a measured hardware result or approval to run. The next smoke
records synchronized `prefix_probe_elapsed_seconds` per row and
`prefix_probe_timing` / `estimated_full_prefix_probe_seconds` in the report;
its total cell timings already include this pass, so do not add it twice.
Use that fresh smoke's total estimate for any proposed full-run guard/cost.

Raw variant logits, per-letter maxima, token IDs, realised image tokens,
grids, per-row times and peak memory are recorded. The processor config
supplies the grid pinpoints; they are compared with the CPU audit, and every
image row's realised tokens/grid must match its shown graph's audit.
Text and none send no image.

### Historical first-prefix upgrade: transfer and fresh smoke rerun

This subsection describes upgrading first-position-only predictions to the
prefix probe. For the saved **prefix** smoke, use offline exact-tie re-gating
above; another GPU smoke is not required.

From the laptop repository root, transfer these **six changed runtime/guide
files** to the already configured pod. The dataset, renders, pairing, strict
loader, checkpoint and Qwen code need no replacement for this fix.
The environment/smoke scripts and existing dependencies are already on the
pod from the previous smoke. No transfer or GPU job was executed here.

```bash
export POD_HOST='REPLACE_WITH_APPROVED_POD_SSH_HOST'
export POD_PORT='REPLACE_WITH_APPROVED_POD_SSH_PORT'
export RSYNC_RSH="ssh -p $POD_PORT"
rsync -rlt --no-owner --no-group \
  scripts/synth_mcq_llava_backend.py scripts/synth_mcq_llava_common.py \
  scripts/analyze_synth_mcq.py scripts/synth_mcq_name_grounded_llava_full.sh \
  scripts/synth_mcq_name_grounded_llava_session_eval.sh scripts/SYNTH_MCQ.md \
  "root@$POD_HOST:/workspace/bachelorArbeit/scripts/"
# Optional: transfer CPU regression tests, preserving their fixture directory.
rsync -rlt --no-owner --no-group \
  tests/test_synth_mcq_llava_backend.py tests/test_synth_mcq_llava_probe.py \
  tests/synth_mcq_llava_inference_cpu_checks.py \
  "root@$POD_HOST:/workspace/bachelorArbeit/tests/"
rsync -rlt --no-owner --no-group tests/fixtures/synth_mcq_llava_smoke_replies.jsonl \
  "root@$POD_HOST:/workspace/bachelorArbeit/tests/fixtures/"
```

Use a **fresh output folder**. Do **not** resume the old smoke: the new v2
identity/code hashes reject it and old rows lack the probe. The wrapper still
passes `--resume`; in a fresh folder this starts from zero, and it can later
resume an interrupted v2 smoke only with identical code/settings.
The following is a prepared command for the separately authorized GPU smoke,
not a job executed as part of this CPU task. If the dated folder already
exists, choose a new suffix and use it consistently for OUT/SMOKE_OUT.

```bash
cd /workspace/bachelorArbeit
export SMOKE_OUT=/workspace/outputs/synth_mcq_name_grounded_llava_smoke_prefix_2026-10-11
export MAX_SECONDS=3600
mkdir -p /workspace/logs
test ! -e "$SMOKE_OUT" || { echo 'Choose a fresh SMOKE_OUT'; exit 2; }
tmux new-session -d -s llava-synth-smoke-prefix "env OUT='$SMOKE_OUT' MAX_SECONDS='$MAX_SECONDS' bash /workspace/bachelorArbeit/scripts/synth_mcq_name_grounded_llava_smoke.sh >/workspace/logs/llava_synth_smoke_prefix_session.log 2>&1"
tail -f /workspace/logs/llava_synth_smoke_prefix_session.log
# In another terminal:
watch -n 30 nvidia-smi
# After completion:
cat "$SMOKE_OUT/smoke_report.json"
wc -l "$SMOKE_OUT/predictions.jsonl"  # must be 40
```

For an interrupted v2 smoke, rerun the same tmux command with the same OUT,
without the fresh-folder existence check. For later approved full/eval jobs,
pass `SMOKE_OUT` explicitly in their tmux `env` invocation so they use the new
report. The full script recomputes the new gate from the 40 predictions,
checks the saved gate, logit sanity, identity and checkpoint; it still needs
separate full-run approval under `AGENTS.md`. Stop the pod after a finished
or failed authorized job; the job timeout alone does not stop billing.

### CPU audit and replacement gate

All 50 Stage 1 reference images are the same fixed sample as the first audit.
Their indices and hashes are recorded in the new audit, and were re-measured
through the actual native CPU LLaVA processor. Its grid-resized content view
is the primary measurement; the 336 px base-thumbnail view is informational.
Each font uses its own measured H glyph, including the 34 pt edge font.

| Set | Labels | Min / median / max px | Outside training range | Below 10 px, information only |
| --- | --- | --- | ---: | ---: |
| New synthetic | Nodes | 8.931 / 11.429 / 13.926 | 0/50 | 10/50 |
| New synthetic | Edges | 7.230 / 9.252 / 11.273 | 0/50 | 38/50 |
| Fixed Stage 1 reference | Nodes | 7.406 / 10.914 / 17.880 | — | 17/50 |
| Fixed Stage 1 reference | Edges | 5.925 / 8.731 / 14.304 | — | 37/50 |

The new gate **passes all 50 graphs**. The range is inclusive and is checked
for both fonts on every graph; the reference list, reference audit hash,
reference min/max, render checksum and manifest are validated. A failed gate
lists graphs and refuses image/wrong_image by default. Text/none remain
available. `--allow-failed-legibility` records the requested override, whether
it was exercised, and the failing count. Pod scripts use no override.
The old 10 px threshold is no longer a LLaVA gate. The Qwen audit is unchanged.

All 50 selected grids are 672×672; unpadded resized content dimensions vary.
Synthetic realised tokens min / median / max: **1948 / 2438 / 2928**.
Full per-graph table, reference sample and thumbnail measurements:
`outputs/synth_mcq_name_grounded_llava_renders_2026-10-11/audit/report.md`,
`audit/images.csv`, `audit/reference_images.json`, `audit/legibility_gate.json`.

Overlap flags by graph: **G50 flagged; G01–G49 not flagged by the proxy**.
The sweep's method was visual inspection of G01/G25/G50, with confirmed
overlapping at-location labels in G50. The new per-graph report supplements
that review with conservative Graphviz text-box intersections using draw-op
positions, widths and font heights. That proxy found no intersecting boxes;
G50 remains flagged from the visual review of byte-identical PNGs. This is not
an ink/OCR test and cannot establish absence of collisions after shrinkage.
Whitespace can produce false positives; crossings and label-node collisions
are outside this proxy. Overlap is informational, and is not a gate.

CPU reproduction uses a **fresh** output directory for render, then audit:

```bash
myvenv/bin/python scripts/prepare_synth_mcq_llava.py --phase render --out outputs/synth_mcq_llava_repeat
cp outputs/synth_mcq_name_grounded_2026-10-10/wrong_graph_pairing.json outputs/synth_mcq_llava_repeat/
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv_llava_cpu/bin/python scripts/prepare_synth_mcq_llava.py --phase audit --out outputs/synth_mcq_llava_repeat
```

### Pod pre-flight and transfers (commands prepared, not executed)

Use the already approved pod connection. Set its SSH hostname/port in a laptop
terminal; nothing below provisions or changes a pod. The first remote read
checks the old Qwen assets, new LLaVA assets, dataset and full checkpoint:

```bash
export POD_HOST='REPLACE_WITH_APPROVED_POD_SSH_HOST'
export POD_PORT='REPLACE_WITH_APPROVED_POD_SSH_PORT'
ssh -p "$POD_PORT" "root@$POD_HOST" 'for p in /workspace/outputs/synth_mcq_name_grounded_assets /workspace/outputs/synth_mcq_name_grounded_llava_assets /workspace/knowledge_graphs_50_name_grounded_100_questions.json /workspace/outputs/llava_stage1/train/checkpoint-000450; do if test -e "$p"; then echo "PRESENT $p"; else echo "MISSING $p"; fi; done'
ssh -p "$POD_PORT" "root@$POD_HOST" 'set -eu; c=/workspace/outputs/llava_stage1/train/checkpoint-000450; for f in checkpoint_manifest.json run_config.json training_state.pt projector.pt adapter/adapter_config.json adapter/adapter_model.safetensors; do test -f "$c/$f" || { echo "STOP: missing $c/$f"; exit 2; }; done'
```

If the checkpoint check fails, **stop**. The laptop copy lacks
`checkpoint_manifest.json`, `run_config.json` and `training_state.pt`.
Do not rsync that sparse directory as a full checkpoint, fabricate files, or
use a loose loader. A verified original complete checkpoint must exist on the
volume before proceeding. Its availability has not been checked remotely.
Missing dataset/new assets can be uploaded with these exact laptop commands
(from the repository root); the existing Qwen assets are not replaced:

```bash
export LOCAL_LLAVA_ASSETS="$PWD/outputs/synth_mcq_name_grounded_llava_renders_2026-10-11"
export REMOTE_LLAVA_ASSETS=/workspace/outputs/synth_mcq_name_grounded_llava_assets
export RSYNC_RSH="ssh -p $POD_PORT"
ssh -p "$POD_PORT" "root@$POD_HOST" 'mkdir -p /workspace/bachelorArbeit/scripts /workspace/bachelorArbeit/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts /workspace/bachelorArbeit/outputs/synth_mcq_name_grounded_llava_audit_2026-10-10 /workspace/outputs/synth_mcq_name_grounded_llava_assets/graphs /workspace/outputs/synth_mcq_name_grounded_llava_assets/audit'
rsync -rlt --no-owner --no-group ../knowledge_graphs_50_name_grounded_100_questions.json "root@$POD_HOST:/workspace/"
rsync -rlt --no-owner --no-group scripts/ "root@$POD_HOST:/workspace/bachelorArbeit/scripts/"
rsync -rlt --no-owner --no-group experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/ "root@$POD_HOST:/workspace/bachelorArbeit/experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/"
rsync -rlt --no-owner --no-group outputs/synth_mcq_name_grounded_llava_audit_2026-10-10/audit_config.json "root@$POD_HOST:/workspace/bachelorArbeit/outputs/synth_mcq_name_grounded_llava_audit_2026-10-10/"
rsync -rlt --no-owner --no-group "$LOCAL_LLAVA_ASSETS/render_manifest.json" "$LOCAL_LLAVA_ASSETS/graphs.sha256" "$LOCAL_LLAVA_ASSETS/wrong_graph_pairing.json" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/"
rsync -rlt --no-owner --no-group "$LOCAL_LLAVA_ASSETS/audit/" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/audit/"
seq -w 1 50 | xargs -P4 -I{} bash -c 'rsync -rlt --no-owner --no-group "$LOCAL_LLAVA_ASSETS/graphs/G$1.png" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/graphs/"' _ {}
rsync -rlt --no-owner --no-group --include='G*.dot' --include='G*.txt' --exclude='*' "$LOCAL_LLAVA_ASSETS/graphs/" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/graphs/"
```

The PNG payload is 83,609,647 bytes. Each parallel connection sends one graph;
`-P4` addresses the slow single-connection uplink. SVG/layout JSON files are
CPU audit artifacts and need not be uploaded. No `fake_analysis_*` directories
are included. Final dry-runs **must list nothing**; any listed transfer requires
fixing and repeating the check. Root-directory mtimes are omitted from this
verification; each file's contents/mtime is checked and the runner verifies
SHA256 independently:

```bash
rsync -n -i -rlt --omit-dir-times --no-owner --no-group --include='G*.png' --include='G*.dot' --include='G*.txt' --exclude='*' "$LOCAL_LLAVA_ASSETS/graphs/" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/graphs/"
rsync -n -i -rlt --omit-dir-times --no-owner --no-group "$LOCAL_LLAVA_ASSETS/render_manifest.json" "$LOCAL_LLAVA_ASSETS/graphs.sha256" "$LOCAL_LLAVA_ASSETS/wrong_graph_pairing.json" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/"
rsync -n -i -rlt --omit-dir-times --no-owner --no-group "$LOCAL_LLAVA_ASSETS/audit/" "root@$POD_HOST:$REMOTE_LLAVA_ASSETS/audit/"
rsync -n -i -rlt --omit-dir-times --no-owner --no-group ../knowledge_graphs_50_name_grounded_100_questions.json "root@$POD_HOST:/workspace/"
```

On the pod, run the strict **CPU pre-flight**, not inference:

```bash
cd /workspace/bachelorArbeit
mkdir -p /workspace/logs
bash scripts/synth_mcq_name_grounded_llava_preflight.sh > /workspace/logs/synth_mcq_llava_preflight.log 2>&1
cat /workspace/logs/synth_mcq_llava_preflight.log
```

This calls the unchanged `llava_adapter.adapter_provenance` / strict manifest
checks, validates the recorded composite hash and recipe, and requires the
training versions: `/workspace/venvs/venv_train/bin/python`, torch
`2.8.0+cu128`, transformers `5.16.1`, peft `0.21.2`, accelerate `1.15.0`.
It also checks all cached pinned model shards and the native processor/tokenizer
without loading model weights. If cache/venv/version checks fail, stop and use
the established LLaVA setup runbook; do not silently substitute dependencies.
For the unchanged planned evaluation, its original data, dry reports, probe
report and `/workspace/outputs/llava_stage1/execution_plan.json` must also be
present as specified in `scripts/LLAVA_STAGE1_TRAINING.md`. On a fresh volume,
complete that runbook's pre-flight before authorizing the session.

### Three separate session jobs

Each launch below is manual, in its own tmux session and log. Each script exits
without starting the next. Approval is required before any full run, as stated
in `AGENTS.md`; a process timeout does not stop billing. The separately approved
pod termination guard must cover the session's measured estimate plus 50%.
The initial smoke ceiling below is a job timeout only, not an elapsed-time
prediction or pod provisioning instruction.

(a) Synthetic smoke: 2 graphs × 2 questions × 1 permutation × 5 conditions ×
2 models = **40 rows**. After the authorized smoke, inspect `LOAD SANITY`,
`BASE/TUNED LOGIT SANITY`, peak memory, the ten cell timings and agreement
fractions. Stop on any failed check; user reviews the results before step (b).

```bash
cd /workspace/bachelorArbeit
mkdir -p /workspace/logs
export MAX_SECONDS=3600
tmux new-session -d -s llava-synth-smoke "env MAX_SECONDS='$MAX_SECONDS' bash /workspace/bachelorArbeit/scripts/synth_mcq_name_grounded_llava_smoke.sh >/workspace/logs/llava_synth_smoke_session.log 2>&1"
tail -f /workspace/logs/llava_synth_smoke_session.log
# In another terminal:
watch -n 30 nvidia-smi
cat /workspace/outputs/synth_mcq_name_grounded_llava_smoke/smoke_report.json
```

(b) After review and approval, the planned full LLaVA Stage 1 + QA evaluation
is delegated to the **unchanged** `llava_stage1_pod_job.sh eval`, with
`RUN_TAG=llava_stage1`, `EVAL_TAG=trained`, `EVAL_SCOPE=all`, and `MAX_SECONDS`
computed from `execution_plan.json: jobs.eval.guard_hours × 3600`, rounded up.
No example counts, eval conditions, diagnostic arms or token limits are edited.
One handout/code discrepancy: the existing all-scope LLaVA job actually
contains **900 Stage 1 main + 900 Stage 1 prefix + 2000 QA = 3800 rows**,
rather than the handout's “900 + 2000”. The saved dry report and execution plan
already include that prefix arm and its timing. This wrapper preserves the
existing job exactly and adds pre-flight and smoke-pass checks only. No full
job is launched here; review this discrepancy before approving the session.

```bash
tmux new-session -d -s llava-eval "bash /workspace/bachelorArbeit/scripts/synth_mcq_name_grounded_llava_session_eval.sh >/workspace/logs/llava_stage1_trained_eval.log 2>&1"
tail -f /workspace/logs/llava_stage1_trained_eval.log
# In another terminal:
watch -n 30 nvidia-smi
```

(c) After the planned evaluation and separate approval, synthetic full uses
**100 × 4 × 5 × 2 = 4000 rows**, `obqa` only, 64 generated tokens, the same
`explicit_v1` parser, cyclic permutations and graph-level bootstrap.
Estimate generation time as the sum over all ten model/condition cells of
**400 × mean seconds per row**, then add measured base+tuned loading time.
The smoke JSON records this estimate, all cell means, and peak VRAM. It is an
estimate from only four rows per cell, not a verified runtime. Include slow
wrong_image/wrong_text rows: Qwen's were roughly 1–1.5 s from explanations up
to the token limit; fast image/none rows alone are not representative.
No numerical LLaVA runtime or GPU cost is available until the pod smoke.

```bash
export MAX_SECONDS=$(/workspace/venvs/venv_train/bin/python -c 'import json,math; r=json.load(open("/workspace/outputs/synth_mcq_name_grounded_llava_smoke/smoke_report.json")); assert r["passed"]; print(math.ceil(r["estimated_full_total_seconds"]*1.5))')
# Launch only after the user has approved this full job and the session guard.
tmux new-session -d -s llava-synth-full "env MAX_SECONDS='$MAX_SECONDS' bash /workspace/bachelorArbeit/scripts/synth_mcq_name_grounded_llava_full.sh >/workspace/logs/llava_synth_full_session.log 2>&1"
tail -f /workspace/logs/llava_synth_full_session.log
# In another terminal:
watch -n 30 nvidia-smi
```

All synthetic launches include `--resume`; repeat the same script/OUT after an
interruption only if the v2 code/settings identity is identical. For this probe
upgrade, use the fresh folder described above. The full script rejects drift from the passing smoke's code,
assets, checkpoint and settings. Outputs have distinct `..._llava_smoke` and
`..._llava_full` folders and a `graphs` symlink to the audited assets. Every
prediction row records `backend=llava`, model revision, `render_variant`, both
font sizes, prompt style and shown graph. The existing analysis format uses
`render_style=training` as the broad style family; `render_variant` records
these actual fixed LR 42/34 settings. Config records the complete settings.

Download results without following/downloading the `graphs` symlink. Analyze
on the laptop using the existing graph-bootstrap analyzer:

```bash
rsync -rlt --no-owner --no-group --exclude='/graphs' "root@$POD_HOST:/workspace/outputs/synth_mcq_name_grounded_llava_smoke/" outputs/synth_mcq_name_grounded_llava_smoke/
rsync -rlt --no-owner --no-group --exclude='/graphs' "root@$POD_HOST:/workspace/outputs/synth_mcq_name_grounded_llava_full/" outputs/synth_mcq_name_grounded_llava_full/
myvenv/bin/python scripts/analyze_synth_mcq.py --pred outputs/synth_mcq_name_grounded_llava_full/predictions.jsonl --out outputs/synth_mcq_name_grounded_llava_full/analysis --n-perms 4
```

The analyzer reports parsed-letter and logit accuracy, parse tiers, all five
conditions, paired graph bootstrap CIs, improved/worsened/tied graph counts,
per-graph CSV and descriptive relation-pair breakdown. It currently identifies
models only as `base`/`tuned` and requires a single font/style per input, so it
does **not** support concatenated Qwen/LLaVA families or a combined-model table.
Analyze them separately. The LLaVA-only metrics added above also respect the
image-comparability caveat.

Original adoption CPU validation (2026-10-11): **98 passed in 53.92 s**, plus **10 native
inference CPU checks passed in 5.05 s**: **108 total**, retaining all 79 prior
checks and adding 29. The native suite uses tiny random CPU models and cached
processors/tokenizers, never pretrained model weights. All five new shell
scripts pass `bash -n`; Python syntax and `git diff --check` pass. Logs are in
`outputs/synth_mcq_name_grounded_llava_renders_2026-10-11/cpu_tests.log` and
`native_cpu_tests.log`. CPU validation prints the solver distribution
`2,2,1: 100`, zero mismatches, and exact expected row counts 40/4000.

Prefix-gate fix CPU validation (2026-10-11): **117 contract tests + 11 native
CPU checks = 128 passed**, retaining the 108 original tests and adding 20.
Complete output, saved-smoke parser tables, file manifest and runtime caveats
are in `outputs/synth_mcq_llava_gate_fix_cpu_2026-10-11/report.md`; test logs
are `cpu_tests.log` and `native_cpu_tests.log` in that folder. The actual
smoke must still be rerun on the pod to verify prefix agreement and GPU time.

Exact-tie amendment CPU validation (2026-10-11): **123 contract tests + 11
native checks = 134 passed**, including all 128 checks from the prefix fix.
The six new regressions cover the real tie, a clear-gap failure, tied maxima
that exclude the generated letter, fixed argmax order, narrow identity
compatibility, offline/full-guard acceptance, and analyzer tie reporting.
The combined offline/full-guard test verifies originals stay byte-identical
and prefix accuracy/time estimates remain unchanged. Full output and review
are in `outputs/synth_mcq_llava_exact_tie_cpu_2026-10-11/report.md`.
