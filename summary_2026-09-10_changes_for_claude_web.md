# Repo update summary — 2026-09-10

Context for a fresh Claude web conversation: three implementation handouts were
executed in this session against the GraphVis/OBQA dataset-generation and
Gemma-3 evaluation pipeline. This file summarizes what actually changed on
disk, what was intentionally left untouched, and what remains outstanding.
Paths are relative to the repo root (`bachelorArbeit/`).

---

## 1. Three new Stage 1 graph-comprehension tasks

**File:** `scripts/generate_graphvis_datasets.py` (modified, +211/-19 lines)
**Tests:** `tests/test_stage1_new_tasks.py`

The original six Stage 1 task types (`node_description`, `node_number`,
`edge_number`, `triple_listing`, `highest_node_degree`, `node_degree`) are
joined by three new, opt-in task types built only from the visible, pruned
graph:

- **`relation_identification`** — asks what relation connects two nodes on an
  edge; `gold` carries the raw relation id, its text form, and both node
  labels.
- **`neighbor_listing`** — asks for all nodes directly adjacent to a chosen
  node; `gold` carries the node, its sorted neighbor labels, and degree.
- **`shortest_path_listing`** — asks for the shortest path between two
  non-adjacent nodes; computed via BFS over the pruned undirected adjacency,
  capped at 32 enumerated paths (`gold.paths_truncated` flags the cap,
  conservatively including the case where the count exactly equals it).

Supporting logic added: an `adjacency()` helper over pruned edges, a
Levenshtein-based `_too_similar()` filter (post-suffix-stripped) to avoid
asking about two near-identical labels, and a 20% keep-rate for `relatedto`
edges in relation tasks when other relation types are available (mirrors the
paper's de-emphasis of the generic relation).

**CLI additions:**
- `--stage1-task-set {paper,extended}` (default `paper`) — `extended` adds the
  three new task types to the eligible pool.
- `--stage1-balance {pool,per-task}` (default `pool`) — `per-task` emits every
  eligible task per graph instead of sampling down to `--tasks-per-graph`.

**Compatibility decision (flagged, then confirmed by the user):** the
existing `label_for_node()` renders plain names with no `[A]`-style
choice-letter suffix. Adding suffixes to disambiguate choices would have
required touching rendering and previously-generated records, which the
handout explicitly protected. **Decision: preserve current rendered labels
as-is** — no suffix was added, `label_for_node` and image rendering are
byte-identical for existing inputs.

New records carry a structured `gold` field (dict) in addition to the
existing free-text `answer` string; paper-set records are unaffected (no
`gold` key is added unless the task produced one).

**Validation:** all 14 tests in this handout's suite pass, including a
byte-for-byte regression test of paper-mode output and a hash-seed
determinism check for task generation. The pre-existing `prune_graph`
hash-order nondeterminism was explicitly left untouched in this handout (it
was fixed later, see §3).

---

## 2. Gemma 3 evaluation scripts (new)

**New files:**
- `scripts/gemma3_common.py` — shared model loading, processor setup/
  verification, structured message construction, greedy inference, legibility
  report check.
- `scripts/eval_gemma3_stage1.py` — mirrors the image-only Qwen Stage 1
  runner.
- `scripts/eval_gemma3_stage2.py` — mirrors the four-condition Qwen Stage 2
  runner.
- `scripts/compare_gemma3_runs.py` — comparison tooling (e.g. eager vs. SDPA
  attention, repeat-run equality).
- `scripts/run_gemma3.sbatch` — Slurm wrapper with `preflight`/`sanity`/
  `stage1`/`stage2` phases.
- `scripts/README_gemma3.md` — full run instructions (see below).
- `tests/test_gemma3_evaluation.py`.

**Design decision — two files, not one file with two modes.** The repo
already keeps Qwen Stage 1 and Stage 2 evaluators separate, with materially
different contracts:
- Stage 2 (Qwen): four conditions, `--output-jsonl`, option scoring, 64-token
  default.
- Stage 1 (Qwen): image-only, `--output-dir`, per-task scoring, 1024-token
  default, scorer supports only the original six task types.

Mirroring the split (rather than unifying into one file with two disjoint arg
sets) avoids a redesign dressed up as consolidation. Common logic
(model/processor loading, message building) was factored into
`gemma3_common.py` instead.

**Task-set flag, not a hard-coded six.** Stage 1 exposes
`--task-set {paper,extended}` (default `paper`), mirroring the generator's new
flag from §1, instead of hard-coding to the six scorable tasks. `paper` runs
the existing unmodified scorer automatically. `extended` accepts all nine
generator task types, requires `--expected-count` (new task eligibility
varies by graph), preserves existing scores for the six paper tasks, and
emits the three new tasks with `scoring_status` (and `structured_gold` when
present in the input) instead of inventing scores — aggregate scoring for the
new tasks is deliberately deferred, no scorer changes were made.

Stage 2 retains all original fields/order (`image`, `text`, `text_noref`,
`kg_text`, `--output-jsonl`, option extractor, 64-token default) and appends
`vision_tokens_per_item` per record; Stage 1 appends the same field, logging
zero for any condition that never loads an image.

**Resolution controls — explicitly not equivalent to Qwen's.** Qwen exposes
`--min-pixels`/`--max-pixels` for its dynamic-resolution tokenizer. Gemma 3
has a fixed 896×896 encoder plus optional crops — there is no semantic
mapping between the two. Decision: **drop the Qwen pixel flags, add Gemma
pan-and-scan controls instead** (`--pan-and-scan` on by default,
`--no-pan-and-scan`, `--pan-and-scan-min-crop-size`,
`--pan-and-scan-max-num-crops`, `--pan-and-scan-min-ratio-to-activate`), and
record in the manifest that resolution control is **not comparable across
backbones**. Consequence recorded for later cross-model comparison: vision-
token budgets cannot be equalized via a shared flag across Qwen/DeepSeek/
Gemma — token-matched comparisons must match *realized* per-item token counts
post hoc, which is why `vision_tokens_per_item` was added to the record
schema (not the manifest).

**Environment:** targets
`/storage/home/hleonel/venv_qwen/bin/python` inside the existing
`pytorch_2.8.0-cuda12.6-cudnn9-devel.sif` container (no new venv). Defaults:
`google/gemma-3-12b-it`, bf16, eager attention, batch size 1, seed 13, static
cache; 4B/27B multimodal checkpoints selectable via `--model-id`, text-only 1B
family rejected. Requires a pinned full commit SHA (`MODEL_REVISION`) resolved
online once, then run fully offline (`HF_HUB_OFFLINE=1`); the loader verifies
every weight shard from the cached index before loading, and rejects unknown
processor APIs by inspecting the installed slow processor's signatures
(`use_fast=False`, left padding).

**Preflight/sanity gating before any full run:**
- `PHASE=preflight` runs both pan-and-scan settings on the same N graphs
  (default 50, `--preflight N` for a specific count), reports the existing
  scorer's raw gold-node recall for each, and requires raw macro recall
  ≥0.95 (`--preflight-min-recall`, explicit operational threshold, not a
  measured Qwen baseline) for the selected setting before allowing a full
  run.
- `PHASE=sanity` runs eager-repeat and eager-vs-SDPA comparisons on the same
  50 items, comparing all prediction fields except `timestamp_utc`; any
  mismatch exits unsuccessfully. Attention stays eager by default; SDPA is
  never silently selected.

**Validation:** 28 tests pass (contract tests: exact prompt-body parity vs.
Qwen prompt builders, both crop settings, failed-gate rejection, actual token
counting/prompt slicing through a fake processor/model, unmodified paper
scoring, deferred extended scoring, resume behavior, a 50-item synthetic
Stage 2 run accepted by the unmodified scorer).

**Explicitly not done / not claimed:** no real GPU run. The development
workspace has no torch/transformers install or Gemma cache, and the
configured cluster SSH connection timed out during this session. No Gemma
weights were downloaded, no Slurm job was submitted, and **no Gemma accuracy,
eager-vs-SDPA equality, or attention-equivalence result is claimed.** Full run
instructions are in `scripts/README_gemma3.md`.

---

## 3. Loosened pruning + measured legibility evaluation

**File:** `scripts/generate_graphvis_datasets.py` (same file as §1, part of
the same diff)
**New files:** `scripts/pruning_stats.py`, `scripts/legibility_sweep.py`,
`scripts/image_audit.py`
**Tests:** `tests/test_pruning_legibility.py`,
`tests/fixtures/legacy_pruning_render.py` (frozen pre-change renderer used for
byte-equality regression)
**Results/PR writeup:** `experiments/2026-09-10_pruning_legibility/README.md`,
`experiments/2026-09-10_pruning_legibility/PR_DESCRIPTION.md`

### Pruning changes
- New defaults: `--max-degree 0` (was 5; **0 disables the soft degree cap
  entirely**), `--max-edges 60` (was 30), `--max-nodes 18` (unchanged).
  Positive `--max-degree` values retain the previous soft-cap and lifeline
  (guaranteed-connecting-edge) behavior exactly.
- `prune_graph()` metadata now includes a `pruning` block: actual
  `max_nodes`/`max_edges`/`max_degree` limits, `edges_before_truncation`, and
  `edges_after`. Truncation also prints both counts to stdout.
- The generator now **refuses to overwrite an existing output split** —
  before writing, it checks all planned JSONL/PNG/graph-JSONL paths for the
  requested `--start`/`--limit` range and raises `FileExistsError` on the
  first collision, forcing a new `--out-dir`.

### Upstream determinism fix (scope expanded at the user's explicit request)
The handout's requested truncation tie-break alone would have left two other
tie-break paths (`best_for_pair` edge selection and lifeline selection)
still dependent on Python's hash/iteration order, meaning historical output
could still vary across `PYTHONHASHSEED` values even with unchanged flags.
**Decision, given to the user as an explicit tradeoff and resolved as
"expand scope":** core/bridge node ranking, relation/direction ties, lifeline
selection, degree-cap edge ordering, and edge-budget truncation all now use
stable `(cid, ...)` / edge-tuple tie-breaks instead of relying on iteration
order. Selection rules and relation priorities themselves are unchanged.

**Consequence recorded explicitly:** this makes pruning reproducible across
hash seeds going forward, but **old flags do not guarantee byte-identical
reproduction of historical output** wherever the previous implementation
happened to resolve a tie by (nondeterministic) iteration order — verified to
actually differ under `--max-degree 5 --max-edges 30` on statements with
ambiguous ties. The old-cap compatibility test covers a tie-free graph; the
pre-existing paper-record byte-regression test still passes unchanged.

### Rendering / legibility
- Seven new **opt-in** rendering flags, all defaulting to the historical
  values so default PNG output is byte-identical to the frozen pre-change
  renderer on an unchanged graph: `--node-fontsize` (18), `--edge-fontsize`
  (14), `--nodesep` (0.5), `--ranksep` (0.7), `--graph-size` (absent),
  `--graph-ratio` (absent), `--rankdir {LR,TB}` (LR).
- `scripts/pruning_stats.py` — split connectivity statistics (correct-option/
  concept-node zero-degree fractions, reachability, mean edges, truncation
  counts), reporting both the legacy question/option-level denominator and a
  full concept-node-level denominator explicitly, since they differ.
- `scripts/image_audit.py` — estimates rendered node-label cap height in
  pixels by measuring the ink extent of a borderless Graphviz `H` glyph
  rendered in Helvetica-Bold at the configured font/DPI, scaled by
  `896 / max(width, height)` (the eventual model input size). This is a
  geometric estimate for aspect-preserving resize, **not OCR and not a
  pan-and-scan crop model**; it assumes no additional whole-graph shrinkage
  from a `size` cap.
- `scripts/legibility_sweep.py` — renders a fixed set of already-pruned
  graphs across a 12-configuration grid (rankdir × fontsize × ranksep) so
  that Stage 1/Stage 2/graph-metadata JSONL content is byte-identical across
  the sweep and only the PNGs differ; outputs `sweep/comparison.{csv,json,md}`.

### Controlled local validation (20 test-split statements, indices 0–19)
Full tables are in `experiments/2026-09-10_pruning_legibility/README.md`.
Headline numbers:

| Configuration | Correct-option zero-degree | Mean edges | Truncation |
|---|---:|---:|---:|
| Historical saved split | 25% | 22.00 | unknown |
| Deterministic, degree 5 / edges 30 | 30% | 22.15 | 2/20 |
| Deterministic, degree 0 / edges 30 | 30% | 24.15 | 9/20 |
| Deterministic, degree 0 / edges 60 (new default) | 30% | 27.60 | 0/20 |

**Cap removal alone did not improve zero-degree or reachability fractions**
on this slice; the historical-vs-new gap is confounded with the determinism
fix, since both change which edges survive a tie.

Legibility sweep: default rendering (LR/18pt/ranksep 0.7) put 17/20 images'
estimated cap height below 10px; the best candidate found,
`--rankdir TB --node-fontsize 30 --ranksep 0.4`, reduced that to 5/20. This is
a **preflight candidate, not a confirmed readability win** — it has not been
checked against actual Gemma gold-node recall.

**Validation:** 36 tests pass, including degree-cap disabling, branch
exclusion, hash-seed determinism under both `PYTHONHASHSEED=0` and `1` with
relation/direction ties and truncation present, metadata correctness, default
PNG byte-equality against the frozen legacy renderer, rendering-attribute
isolation (PNG-only diffs across the sweep), statistics denominators, and
measured reference cap height. A real rerun into an existing output directory
was rejected as expected, and its record checksum was unchanged.

**Explicitly not done:** no full split regeneration, no new zero-shot scoring
runs under the new regime, no Gemma preflight (blocked on the same cluster
SSH timeout as §2). Existing saved splits under `outputs/graphvis_obqa/` were
left completely untouched; all new data lives under
`outputs/pruning_legibility_2026-09-10/`.

---

## Cross-cutting notes for follow-up work

- **Reproducing historical scores:** because of the determinism fix in §3,
  regenerating any split with the *old* flags (`--max-degree 5 --max-edges
  30`) is not guaranteed to reproduce historical files byte-for-byte where
  the legacy implementation broke a tie by iteration order. Full
  historical-regime reruns should be diffed against saved splits, not
  assumed identical.
- **Cross-backbone vision-token comparability (Qwen/DeepSeek/Gemma):** there
  is intentionally no shared flag to equalize vision-token budget. Any
  token-matched analysis must be done post hoc from `vision_tokens_per_item`
  in the per-item output records.
- **Blocking dependency shared by §2 and §3:** cluster SSH access was
  unavailable during this session, so no GPU-side result (Gemma accuracy,
  attention equivalence, or gold-node recall for the new rendering candidate)
  has been produced yet. Both are ready to run once cluster access is
  restored — see `scripts/README_gemma3.md` for the exact `sbatch` commands.
