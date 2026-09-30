# GraphVis Dataset Generation

`generate_graphvis_datasets.py` creates GraphVis-style OBQA artifacts:

- rendered visual KG PNGs
- Stage 1 graph-comprehension JSONL
- Stage 2 OBQA JSONL
- graph metadata JSONL for inspection/debugging

The current implementation uses the working OBQA interpretation from this project:
QA-GNN stores one graph per `(question, answer choice)`, so the script merges all
four answer-choice graphs into one question-level image. This union-of-four step
is not explicitly specified in the GraphVis paper.

## Smoke test

Run a small sample from the project root:

```bash
bachelorArbeit/myvenv/bin/python bachelorArbeit/scripts/generate_graphvis_datasets.py \
  --split train \
  --start 0 \
  --limit 5 \
  --tasks-per-graph 6
```

Outputs are written to:

```text
outputs/graphvis_obqa/<split>/
```

## Full split generation

```bash
bachelorArbeit/myvenv/bin/python bachelorArbeit/scripts/generate_graphvis_datasets.py \
  --split train \
  --start 0 \
  --limit 4957 \
  --tasks-per-graph 6

bachelorArbeit/myvenv/bin/python bachelorArbeit/scripts/generate_graphvis_datasets.py \
  --split dev \
  --start 0 \
  --limit 500 \
  --tasks-per-graph 6

bachelorArbeit/myvenv/bin/python bachelorArbeit/scripts/generate_graphvis_datasets.py \
  --split test \
  --start 0 \
  --limit 500 \
  --tasks-per-graph 6
```

By default, `relatedto` edges are labeled as `related to` so Stage 1 triple-listing
answers match the visible graph. Use `--hide-relatedto-labels` only for cleaner
Stage 2-only rendering experiments.

## Pruning ablations

Both `generate_graphvis_datasets.py` and `legibility_sweep.py` accept:

| Flag | Default | Behavior |
| --- | --- | --- |
| `--max-nodes` | `18` | Node budget; `0` disables it. |
| `--max-edges` | `60` | Edge budget; `0` disables it. |
| `--max-degree` | `0` | Degree cap; `0` disables it. |
| `--bridge-rule` | `qa-bridge` | Filler eligibility, described below. |
| `--no-lifelines` | off | Remove lifeline exemptions from the degree cap, making it hard. |

All caps reject negative values. **Compatibility change:** `--max-edges 0`
previously kept zero edges; it now means unlimited edges. Default graph content,
images, and Stage 1/2 records are unchanged; metadata gains provenance fields.

Filler rules apply after core question/answer selection:

- `qa-bridge`: adjacent to at least one question node and one answer node.
- `core-neighbor`: adjacent to at least one question or answer node.
- `any`: every remaining node is eligible, including isolated nodes.
- `none`: no filler nodes.

Fillers rank by descending number of core neighbors, then concept ID. Only `any`
adds descending total degree before the concept-ID tie-break. When the core
exceeds a positive node budget, the existing question-first truncation still
applies. With `--max-nodes 0`, all core nodes and all eligible fillers are kept;
combining it with `--bridge-rule any` retains every merged node internally.
The existing visibility policy still omits isolated filler/question nodes from
`visible_nodes`; disconnected answers remain visible.

Lifelines are exempt from positive degree caps by default. `--no-lifelines`
removes that exemption while preserving lifeline selection and its priority in
edge-budget truncation. It has no effect on graph content if the degree cap is
disabled. Disabling all caps still preserves the per-pair relation deduplication,
so the result is not the raw merged union graph.

Each graph's `pruning` metadata records the caps as supplied, `bridge_rule`,
`lifelines`, `core_size` (before truncation), `core_truncated`, `bridges_added`
(selected fillers before edge pruning), `edges_before_truncation`, and
`edges_after`. `pruning_stats.py` groups by all five configuration values; for
older metadata, missing rule/lifeline fields mean `qa-bridge`/`true`.

Run each condition into a separate output directory (existing files are guarded):

```bash
myvenv/bin/python scripts/generate_graphvis_datasets.py --split test --limit 50 \
  --out-dir outputs/prune_ablation/baseline
myvenv/bin/python scripts/generate_graphvis_datasets.py --split test --limit 50 \
  --max-nodes 0 --max-edges 0 --out-dir outputs/prune_ablation/nocaps
myvenv/bin/python scripts/generate_graphvis_datasets.py --split test --limit 50 \
  --bridge-rule core-neighbor --out-dir outputs/prune_ablation/coreneighbor
myvenv/bin/python scripts/pruning_stats.py \
  --old outputs/prune_ablation/baseline/test --new outputs/prune_ablation/nocaps/test \
  --output outputs/prune_ablation/baseline_vs_nocaps.json
```

Separate follow-up: `eval_gemma3_stage1.py` and `eval_gemma3_stage2.py` still stamp
`18/30/5` as pruning provenance. They should read the actual values from
`graph_metadata`; this ablation change does not modify those evaluators.

## Stage 1 pruning-condition matrix

See [the matrix workflow](README_stage1_matrix.md) for controlled CPU dataset
builds, a manifest-driven COMA job array using the existing three evaluators,
and paired-intersection tables, diagnostics and figures.
