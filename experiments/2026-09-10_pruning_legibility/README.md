# Pruning and legibility validation

The generator now defaults to `--max-degree 0 --max-edges 60 --max-nodes 18`.
Zero disables degree capping; positive values retain the existing soft-cap and
lifeline behavior. Pruning metadata records the actual limits, the edge count
immediately before `max_edges` truncation, and the final edge count. Truncation
prints both counts. Existing output paths are rejected instead of overwritten.

The user explicitly expanded scope to fix upstream determinism. Core and bridge
ranking, relation/direction ties, lifeline selection, degree-cap ordering and
edge-budget truncation now have stable CID/edge-tuple tie-breaks. Selection rules
and relation priorities are otherwise unchanged. This makes pruning stable
across hash seeds, but **old flags cannot guarantee historical byte identity
where the previous implementation resolved ties by iteration order**. Metadata
also gains the required `pruning` block. The old-cap compatibility test covers a
graph without ambiguous ties; the existing paper-record regression still passes.

Rendering defaults remain 18/14-point node/edge fonts, nodesep 0.5, ranksep 0.7,
LR, 200 DPI, with `size` and `ratio` absent. The seven new rendering flags are
opt-in. Default PNG bytes match the frozen pre-change renderer on the same
synthetic graph in the local Graphviz installation. Stage 1/2 builders, node
styling and choice-graph merging remain unchanged.

## Controlled local runs

All runs below use test statements 0–19. Existing data in
`outputs/graphvis_obqa/test` was left untouched. New data and full JSON reports
are under `outputs/pruning_legibility_2026-09-10/`:

- `degree5_edges30`: deterministic pruning, old limits, default rendering.
- `degree0_edges30`: removes only the cap relative to the preceding run.
- `sweep/LR_font18_ranksep0.7`: uncapped, 60 edges, default rendering.
- `sweep/`: 12 render configurations of exactly the same 20 uncapped graphs.
- `cap_only_comparison.json`, `edge_budget_comparison.json`,
  `historical_vs_new_defaults.json`, and `default_render_audit.json`.

| Configuration | Correct option degree zero | Correct concept nodes degree zero | Correct option reachable | Best distractor option degree zero | Best distractor option reachable | Mean edges | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical saved split | 25% | 50.9% | 65% | 5% | 95% | 22.00 | Unknown |
| Deterministic, degree 5 / edges 30 | 30% | 50.9% | 60% | 5% | 95% | 22.15 | 2/20 |
| Deterministic, degree 0 / edges 30 | 30% | 50.9% | 60% | 5% | 95% | 24.15 | 9/20 |
| Deterministic, degree 0 / edges 60 | 30% | 50.9% | 60% | 5% | 95% | 27.60 | 0/20 |

**Cap removal did not improve zero-degree or reachability fractions on this
20-question slice.** The historical-to-new difference cannot be attributed to
the cap alone, because deterministic tie-breaking also changes retained graphs.
The new default's maximum visible-node degree is 14, versus 7 under the
positive soft cap in the controlled old-limit run. The cap can still be exceeded
by lifelines when enabled; this behavior was deliberately preserved.

The earlier approximately 22% statistic was question/option-level (maximum
degree across a choice's concepts), not a fraction over all concept nodes. The
report exposes both denominators. It selects the best distractor by maximum
option degree, then shortest reachable distance, then choice letter. Options
without any mapped concepts are separately counted; the legacy option-level
zero-degree fraction includes them as zero. Distances are undirected and may be
zero when an answer concept is itself a question concept. Old metadata cannot
establish whether truncation occurred: edge count equal to a cap is not proof.

## Rendering sweep

The default audit estimates cap height by measuring the ink of a borderless
Graphviz `H` rendered in Helvetica-Bold at the configured font/DPI, then scaling
by `896 / max(width, height)`. This is a geometric estimate for an
aspect-preserving fit, not an OCR score or a model of pan-and-scan crops. It
assumes no extra Graphviz whole-graph shrinkage; if using a `size` cap, pass its
actual `--graph-scale` to the audit or treat the result as an upper bound.

| Configuration | Images below 10 px | Median estimated cap px |
| --- | ---: | ---: |
| LR / 18 / 0.7 (default) | 17/20 | 7.69 |
| LR / 18 / 0.4 | 15/20 | 8.35 |
| LR / 24 / 0.7 | 13/20 | 9.04 |
| LR / 24 / 0.4 | 12/20 | 9.73 |
| LR / 30 / 0.7 | 9/20 | 10.14 |
| LR / 30 / 0.4 | 6/20 | 10.82 |
| TB / 18 / 0.7 | 14/20 | 7.73 |
| TB / 18 / 0.4 | 14/20 | 7.73 |
| TB / 24 / 0.7 | 11/20 | 9.71 |
| TB / 24 / 0.4 | 11/20 | 9.73 |
| TB / 30 / 0.7 | 5/20 | 11.64 |
| TB / 30 / 0.4 | 5/20 | 11.64 |

Ranking uses the fewest below-threshold images, then the largest median cap
height, then configuration name for deterministic ties. The selected preflight
candidate is **`--rankdir TB --node-fontsize 30 --ranksep 0.4`**, not a confirmed
Gemma readability winner. All 12 configurations have byte-identical Stage 1,
Stage 2 and graph-metadata JSONL files; only PNGs differ. Complete comparison
tables are in `sweep/comparison.{csv,json,md}`.

No Gemma preflight was run here. This 20-item sweep is too short for the default
50-graph gate; regenerate at least 50 graphs with the candidate settings in a
new directory, then run `eval_gemma3_stage1.py --preflight 50` in the GPU
environment. The previous Gemma notes document the unavailable cluster access.
A larger font's estimated improvement must be confirmed by gold-node recall.

## Commands

```bash
# Part A: identical rendering, two pruning configurations, fresh output paths.
myvenv/bin/python scripts/generate_graphvis_datasets.py --split test --limit 20 \
  --max-degree 5 --max-edges 30 --out-dir outputs/pruning_compare/degree5
myvenv/bin/python scripts/generate_graphvis_datasets.py --split test --limit 20 \
  --max-degree 0 --max-edges 30 --out-dir outputs/pruning_compare/degree0
myvenv/bin/python scripts/pruning_stats.py \
  --old outputs/pruning_compare/degree5/test --new outputs/pruning_compare/degree0/test \
  --output outputs/pruning_compare/comparison.json

# Part B: fixed pruned graphs, the requested 12-layout grid.
myvenv/bin/python scripts/legibility_sweep.py --out-dir outputs/legibility_sweep_new
myvenv/bin/python scripts/image_audit.py --split outputs/legibility_sweep_new/TB_font30_ranksep0.4/test \
  --node-fontsize 30 --output outputs/legibility_sweep_new/winner_audit.json

myvenv/bin/python -m pytest -q tests
```

Validation: 36 tests pass, including degree-9 cap disabling, branch exclusion,
hash-seed determinism with relation/direction ties and truncation, metadata,
default PNG byte equality, rendering-attribute isolation, statistics denominators
and measured reference cap height. A real rerun into an existing output path was
rejected and its record checksum stayed unchanged.

Only validation slices were generated. Full historical splits were preserved;
full replacements and zero-shot reruns remain future experimental work.
