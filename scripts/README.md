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
