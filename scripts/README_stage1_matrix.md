# Stage 1 pruning-condition matrix

This matrix measures zero-shot image graph comprehension for DeepSeek-OCR-2,
Qwen3-VL-8B-Instruct, and Gemma-3-12B-IT. It does not run Stage 2, fine-tuning, or
LLaVA. Adding LLaVA requires a separately validated evaluator/environment; it is
not included merely because it is mentioned as a GraphVis backbone.

## Verified evaluator contracts

| Model | Existing evaluator | Fixed model-specific settings |
| --- | --- | --- |
| DeepSeek | `experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py` | revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`; evaluator constants `base_size=1024`, `image_size=768`, `crop_mode=True`; eager attention, safetensors, bfloat16; remote-code ceiling 8192 |
| Qwen | `experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_qwen.py` | revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; `--min-pixels 262144 --max-pixels 1310720 --max-new-tokens 1024`; SDPA |
| Gemma | `scripts/eval_gemma3_stage1.py` | revision `96b6f1eccf38110c56df3a15bffe176da04bfd80`; eager, bfloat16, dynamic cache, 1024 tokens, pan-and-scan with three explicit crop controls |

`score_stage1.TASK_SETS['extended']` contains all nine requested tasks.
`--expected-count` is measured per dataset, never assumed to be 3000 or 9×N.
The existing `run_config.condition` is the input modality (`image`), so matrix
runs add `pruning_condition` and `pruning`. All three evaluators already score
against their own `--graph-metadata`; the matrix additionally locks its path and
contents. Prediction filenames are `predictions_<model>_<task_type>.jsonl`.

Only Gemma requires `--preflight-report`. Each Gemma cell runs triple-listing
preflight on the first min(50, N) graphs under both crop settings. Full inference
requires the selected setting to reach raw macro gold-node recall ≥0.95.
Preflight predictions now support `--resume` as well as main predictions; a
requeued task resumes both crop trials without repeating completed items.
Failed gates block full inference and are reported as illegible.

Qwen and DeepSeek now accept `--seed 13` and call `transformers.set_seed`. Matrix
provenance is added by their existing shared `write_run_config`, before any
prediction is emitted. Non-matrix runs retain their existing provenance behavior.

## CPU build and dry run

Run from the project checkout on COMA with the **login-node `venv_gen` Python**.
The launcher passes that interpreter's absolute path to sbatch; it does not try
to resolve `venv_qwen` on the login node.

```bash
venv_gen/bin/python scripts/build_pruning_matrix.py \
  --conditions baseline,core_neighbor --limit 5 --dry-run
```

`--dry-run` generates real CPU datasets and diagnostics, then prints the array
submission and every evaluator invocation without submitting. Use fresh
`--out-root` and `--results-root` paths for subsequent builds. The default dataset
root is `outputs/pruning_matrix`, result root `outputs/stage1_matrix`, N=150.
Every existing selected condition/result directory is a hard collision, even if
empty; no directory is deleted or reused. A failed build leaves its partial
artifacts for inspection. No incomplete manifest is published.

The ordered, data-driven defaults are `baseline`, `nodes40`,
`nodes40_edges_off`, `core_neighbor`, `caps_off`, `unpruned`, and
`legacy_18_30_5`. Use `--conditions a,b` for a subset. `--conditions-file FILE.json`
can supply an ordered object mapping condition names to pruning overrides.
Only `max_nodes`, `max_edges`, `max_degree`, `bridge_rule`, and `lifelines` are
allowed; overrides of seed, rendering, prompts or slice are rejected. The
`baseline` name cannot be assigned different pruning settings.

All conditions use the handout's fixed rendering settings, test slice starting
at 0, seed 13, extended tasks, and per-task balance. Uncapped Graphviz rendering
can be extremely slow. An optional `--generation-timeout SECONDS` fails the build
and terminates the generator plus its Graphviz process; it never skips a
condition, changes the renderer, or substitutes metadata-only output.

Each condition includes:

- Generated PNGs, Stage 1/2 JSONL from the existing generator, and graph metadata.
  Stage 2 files are incidental generator output; no Stage 2 inference is launched.
- `pruning_stats.json`, validated to contain exactly one expected configuration.
- `image_audit.json` measured with node font size 18 and DPI 200. The default
  minimum label threshold is 10 px; a global `--min-label-px` is locked into the
  manifest and must apply to every condition.
- `build_config.json`, recording the generator command and source fingerprints.

The manifest has one line per model × condition. Its image root is the condition
directory, **not** its `test` directory: generated image paths already start with
`test/images/`. It locks data, image and diagnostic hashes; fixed settings;
model-specific decoding and crop controls; and source-code hashes. Changed
inputs or source code are rejected on launch, resume and collection.

### Prompt control

A fixed seed alone does not keep generator prompt selection fixed: graph-dependent
choices and eligibility consume different RNG draws. The default
`--prompt-policy fixed-template` therefore selects an **existing, unchanged**
prompt template by a hash of `(seed, statement_idx, task_type)` after generation.
The generator, renderer, pruning function and template text are not modified.
All conditions apply this same transformation. Gold answers and graph-dependent
sampled node/pair targets stay as generated, so target-specific prompt text can
still differ. `prompt_comparison.json` lists every paired-key difference.

This controls wording, not the identity of graph-dependent targets. If the
experiment requires verbatim identical prompts including targets, use
`--prompt-policy identical`: it rejects a build with any paired-key prompt
difference rather than hiding the confound. It does not invent common targets
or change the handout's eligibility/intersection definition.

## Supply validated Gemma crop controls

The repo has no recorded validated COMA values for these three controls. The
builder accepts flags or `PAN_AND_SCAN_*` environment variables, but does not
invent defaults. A build without them is valid for CPU/dry-run inspection;
**submission and execution are blocked** until all three are resolved.

To build a ready matrix, export the validated values first:

```bash
export PAN_AND_SCAN_MIN_CROP_SIZE=<validated_integer>
export PAN_AND_SCAN_MAX_NUM_CROPS=<validated_integer>
export PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE=<validated_ratio>
venv_gen/bin/python scripts/build_pruning_matrix.py --limit 150
```

To resolve an already-built, not-yet-started matrix without rerendering:

```bash
venv_gen/bin/python scripts/build_pruning_matrix.py \
  --manifest outputs/pruning_matrix/manifest.jsonl \
  --finalize-manifest outputs/pruning_matrix/manifest.ready.jsonl --dry-run
```

This writes a new immutable manifest using the explicit crop controls. Existing
result directories or a destination manifest are errors. Once a cell has
started, do not change its matrix settings; build a fresh experiment instead.

## Slurm launch and monitoring

```bash
venv_gen/bin/python scripts/build_pruning_matrix.py \
  --manifest outputs/pruning_matrix/manifest.ready.jsonl --submit --concurrency 3
```

The launcher passes `--array=1-<line_count>%3` to sbatch. Slurm cannot interpolate
a manifest length in a `#SBATCH` directive, so the array bounds are supplied by
the launcher, not hardcoded. Each task dispatches exactly one manifest line to
its existing evaluator inside
`/storage/home/hleonel/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif`, using absolute
`/storage/home/hleonel/venv_ocr2/bin/python` or
`/storage/home/hleonel/venv_qwen/bin/python`. The cache is
`/storage/home/hleonel/.cache/huggingface`; offline flags and
`local_files_only=True` prevent downloads. Versions and cached safetensors are
checked before model loading. `--approve-prompts` is explicitly logged and
passed non-interactively, along with `--resume` and the cell's record count.

The launcher prints the actual job ID in:

```bash
squeue -u "$USER"
scontrol show job <jobid>_<arrayidx>  # COMA has no sacct
tail -f ~/outputs/slurm-prunmat-<jobid>_<arrayidx>.out
venv_gen/bin/python scripts/build_pruning_matrix.py \
  --manifest outputs/pruning_matrix/manifest.ready.jsonl --progress
```

Progress sums the actual per-task prediction files. A cell cannot finish
successfully unless its realised unique prediction keys exactly equal its
input keys, its config matches pruning/model/data settings, and scored output
exists. An OS file lock prevents duplicate concurrent execution and releases
on process death, allowing a requeue.

## Collection and figures (login node)

```bash
venv_gen/bin/python scripts/collect_stage1_matrix.py \
  --manifest outputs/pruning_matrix/manifest.ready.jsonl \
  --output-dir outputs/stage1_matrix/collected
```

The collector accepts missing/partial runs and never drops them from the declared
matrix. It rescales no scores and rescores raw predictions with the shared
`score_record`/`aggregate_task` against each cell's own gold. An absent cell makes
that model's prediction intersection empty; an incomplete intersection is
explicitly provisional. Generated eligible-key intersection and inference
completion are reported separately. Unique-to-condition counts are eligibility
statistics, not job progress.

Outputs:

- `long.csv/json`: every numeric scorer leaf, for `intersection` and `full`
  scopes, item counts, metric-specific denominators, interpretation status, raw
  unmasked values, and per-condition graph/legibility diagnostics.
- `wide.csv/json`: model × task × scope × metric, with condition columns and
  adjacent item-count/status columns.
- `coverage.csv/json`: expected/realised/paired counts, unique and nonshared
  eligible items, per-task unique counts, and graph diagnostics.
- `macro.csv/json`: unweighted mean of nine explicitly named primary scorer
  keys; unavailable when any task has no items. Neighbor listing uses F1, so
  this is labelled a primary-score mean rather than inventing a universal
  accuracy metric.
- `summary.md` and fixed-width console tables, including baseline deltas with
  denominators. Console tables split into blocks at `--terminal-width`.
- `legibility_scatter`, `task_comparison`, `retention_trend`, each as PNG and PDF.
  Style configuration, palette, hatches and save helper come from
  `plot_results.py`. PNGs use 300 dpi; PDFs retain vector graphics. Fixed PDF
  timestamps make regeneration reproducible. Each figure states its scope,
  scorer keys and item denominators. Missing/illegible cells never become zero.

A condition with any image below the fixed audit threshold is conservatively
masked as `n/a (illegible)`, with the number of failing images reported. Raw
scores remain auditable in `unmasked_value`, not presented as evidence about
KG content. DeepSeek additionally gets a 768/896 label-size proxy; this is not
a simulation of its crop pipeline. Gemma's preflight is separate from that
whole-image audit estimate. Overlap is not measured by the existing audit and
is recorded as unavailable, never zero.

Matplotlib is used only by this CPU collector. If absent from `venv_gen`, the
collector reports the missing dependency after writing tables; it does not
install packages or touch either GPU environment. `--no-plots` explicitly opts
out of derived figures.
