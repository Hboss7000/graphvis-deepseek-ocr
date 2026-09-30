# Extended Stage 1 scoring

> Superseded for default extraction by [verbose-answer extraction](../2026-09-11_stage1_answer_extraction/README.md). Use `--extractor legacy` to reproduce the scoring behavior documented below; `span` is now the default.

The shared scorer, validation, previews, resume bookkeeping and output-file
selection support `--task-set paper` (default) and `--task-set extended`.
`TASK_TYPES` remains an alias for the six paper tasks, so callers importing it
and completed six-file runs retain their contract. Qwen, DeepSeek-OCR-2 and
Gemma now score the extended tasks inline and on completion. The generator,
existing six scoring branches, normalization ladder and prompt bodies were not
changed for this work.

An input task outside the selected set fails validation before model loading,
with the task name and a suggestion to use `--task-set extended`. Each runner
also checks the task explicitly inside its inference loop. Expected-count
validation remains explicit: the reported extended test split has **4496 rows,
not 4500**. Other slices/configurations can have different eligibility counts.

## Metrics and record schema

`gold` remains the natural-language answer string. A new task's structured input
`gold` dict is retained as `structured_gold` in predictions/scored rows and used
directly for scoring. Earlier Gemma predictions marked `scoring_status: pending`
can be rescored without rerunning inference; the returned scored rows drop the
stale pending marker.

All three tasks use the existing `raw`, `basic`, `annotation_stripped` tiers:

- `relation_identification`: `relation_correct` on each scored row; aggregate
  tier `accuracy` and `correct_count`, plus `per_relation`. Extracted relation
  strings are lowercased and stripped of whitespace, accepting both `atlocation`
  and `at location`. Gold's raw key is retained as `gold_relation`.
- `neighbor_listing`: existing node-list parsing after removing a neighbour
  answer lead-in. The primary metrics are aggregate `mean_precision`,
  `mean_recall`, `mean_f1`, including `by_degree` groups. Per-row `set_metrics`
  retains set diagnostics; `is_correct` denotes full F1 only to support the
  shared failure-case selector, not to replace F1 as the metric.
- `shortest_path_listing`: ordered `path_exact` membership against every stored
  shortest path, plus `path_partial.hop_count_correct` and
  `path_partial.node_overlap_f1`. Arrows (`->`, `→`), commas and numbered lists
  are accepted; commas within `[A,B]` suffixes stay inside labels. Aggregate
  tier fields are `path_exact`, `hop_count_correct`, `mean_overlap_f1`.
  `paths_truncated_count` identifies incomplete gold-path sets. Repeated nodes
  remain in the sequence for exact/hop scoring; overlap F1 compares node sets.

New scoring notes appear only for extended metrics. Paper metrics do not gain
extra keys. New-task prediction files may be empty if no input graph is eligible;
all selected-task files are still created by the runners.

## Running and rescoring

Use the existing model environments and prompt-review flags. For example:

```bash
# Qwen; the corresponding DeepSeek runner accepts the same task-set/count flags.
/path/to/venv_qwen/bin/python \
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_qwen.py \
  --input-jsonl /path/to/extended/stage1_graph_comprehension_0_500.jsonl \
  --graph-metadata /path/to/extended/graph_metadata_0_500.jsonl \
  --image-root /path/to/extended \
  --output-dir /path/to/new/qwen_results \
  --task-set extended --expected-count 4496 --preview-only
```

After reviewing prompts, replace `--preview-only` with `--approve-prompts`.
Both existing Qwen/DeepSeek sbatch wrappers accept `TASK_SET`, `EXPECTED_COUNT`
and `INPUT_JSONL` environment overrides, retaining paper/3000 as defaults.
Point `STAGE1_ROOT` at the matching image/metadata root as well.

```bash
TASK_SET=extended EXPECTED_COUNT=4496 \
  STAGE1_ROOT=/path/to/extended \
  INPUT_JSONL=/path/to/extended/test/stage1_graph_comprehension_0_500.jsonl \
  sbatch experiments/2026-09-04_stage1_graph_comprehension_zero_shot/slurm/run_stage1_qwen.sbatch
```

Use a distinct results directory for the extended run; incompatible manifests
are rejected rather than mixing paper and extended outputs. The wrappers now
also accept `RESULTS_DIR` for this purpose. The Gemma wrapper already accepts
`TASK_SET` and `STAGE1_EXPECTED_COUNT`.

Offline scoring is identical across backbones; choose the matching model name:

```bash
myvenv/bin/python \
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/score_stage1.py \
  --input-jsonl /path/to/extended/stage1_graph_comprehension_0_500.jsonl \
  --graph-metadata /path/to/extended/graph_metadata_0_500.jsonl \
  --predictions-dir /path/to/new/qwen_results \
  --model-name qwen --task-set extended
```

To rescore existing unscored Gemma files, use `--model-name gemma3` and their
matching source JSONL. Do not resume an old unscored Gemma run into a new scoring
manifest; offline rescoring preserves the inference outputs.

## Verification

Both completed cluster runs were rescored with the updated scorer in the
original Apptainer/venv environment. Their source-input and graph-metadata
checksums matched each run's manifest. The metrics were **byte-identical** for
both models, 3000 records each:

| Model | Saved and rescored metrics SHA-256 |
| --- | --- |
| Qwen | `e121f2691acfe93c9a7adf903fed52e3e27056b217bf4c6d646529294b1e369c` |
| DeepSeek | `42a758ec5bc8ffc2ba63574d536067ed2ed29250efc8d27b27b2f1d126149a98` |

The exact scorer hash and comparison results are in `paper_regression.json`.
Copied inputs/predictions and the returned cluster metrics are preserved locally
in `outputs/stage1_scoring_regression_2026-09-11/`. The new scorer was streamed to
an isolated Python namespace for read-only scoring; production cluster scripts,
inputs and metrics were not overwritten. No GPU job was submitted.

Local Python 3.14 changes the last decimal places of some float aggregates even
with the pre-change scorer. The completed-run tests therefore check the cached
original-environment byte comparison and allow only `1e-14` absolute float drift
when additionally rescoring locally. Counts, structure and other values remain
exact. A mixed-response synthetic fixture is also checked byte-for-byte against
golden metrics captured with the pre-change local scorer.

```bash
myvenv/bin/python -m pytest -q tests
```

Tests cover relation aliases, all normalization tiers, neighbour partial credit,
alternative shortest paths and wrong paths with correct lengths, truncated gold,
the gold-key collision, all-nine-task CLI scoring, early runner rejection,
mocked nine-task inference/resume for Qwen/DeepSeek/Gemma, paper fixtures and both
completed-run regressions. The two completed-run tests skip when the optional
local copies are absent; normal fixture tests require no GPU or cluster access.
