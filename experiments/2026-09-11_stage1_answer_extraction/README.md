# Answer extraction from verbose Stage 1 responses

The shared Stage 1 scorer now defaults to `--extractor span`. The previous
extractors remain available through `--extractor legacy`; their metrics schema
and serialization are preserved. No prompts, inference code, dataset generator,
gold answers, or normalization tiers were changed for this work.

The complete results are in [comparison.md](comparison.md). They cover both
models on the six-task September 4 run (3,000 responses each) and the nine-task
40-node run (4,496 responses each), with all records included in denominators.

## Offline use

From the repository root, using any existing Python environment:

```bash
myvenv/bin/python scripts/rescore_stage1.py \
  --predictions-dir /path/to/existing/qwen \
  --model-name qwen \
  --input-jsonl /path/to/stage1_graph_comprehension_0_500.jsonl \
  --graph-metadata /path/to/graph_metadata_0_500.jsonl \
  --task-set extended \
  --extractor span \
  --output /path/to/new/rescoring-directory
```

Use `--task-set paper` for the September 4 files. The entry point requires only
the Python standard library and refuses an existing output directory. It reads
stored `raw_response` values and never writes into the source prediction files.
It computes both variants for comparison, writing the selected variant to
`metrics_MODEL.json`, plus separate `metrics_MODEL_legacy.json` and
`metrics_MODEL_span.json`, comparison JSON/Markdown, scored JSONL records, and
an input/scorer checksum manifest. `--extractor legacy` leaves the selected
metrics file in the historical schema without new fields.

## Extraction and metric definitions

`answer_span` removes markdown emphasis and selects the suffix after the last
answer marker, otherwise the last nonempty line, otherwise the whole response.
The markers are Final Answer, Answer:, Therefore, In conclusion, So the, and
the answer is. Numeric parsing selects the last integer in that span, falling
back to the full response only when the span contains no integer. Maximum degree
uses degree cues, then the existing gold-sentence degree pattern, with bare
integer answers also accepted; numeric node names are not a fallback degree.

Explicit sets retain their list/tuple region, including lists split by headings
and lists followed by explanatory paragraphs. This structure-preserving extension
to the generic last-line rule is necessary for the actual `looking`/`look`
response, which has two paragraphs after its correct bullet. The last explicit
answer marker still limits the selected region. Star/dash/numbered bullets and
comma lists are accepted. Triples also accept three backtick-quoted components
on a list line. Relation labels and stopword-only node candidates are removed;
the relation component of a triple is retained. The existing `raw`, `basic`, and
`annotation_stripped` tiers are unchanged.

Relation extraction searches the 17 ConceptNet labels used by the dataset
renderer, with lowercase alphabetic equivalence. The last-ending match wins;
longer overlapping negative labels win over their contained positive labels.
An unmatched span gives `parsed_relation: null`. Relation accuracy retains
per-relation and per-tier fields, and adds `relatedto_excluded` with its own
record count, strict accuracy and containment.

Paths accept arrows, commas and numbered lists. The last complete source-to-target
chain is preferred across traces; a final numbered sequence can supersede an
earlier trace. Relation labels are removed from extracted nodes. Cycles are
retained for exact/hop scoring and flagged in `degenerate_output`, with
`degenerate_output_fraction` in aggregates. Exact correctness still means
membership in stored `gold.paths`; no graph search creates extra gold paths.

`strict_accuracy` aliases the strict numeric, relation or path metric. For
maximum degree it is degree-only; `strict_joint_accuracy` also requires the
single selected node name, accepting metadata-derived ties. Existing name/joint
metrics remain available. Set `strict` contains the three existing normalization
tiers with precision, recall and F1. MAE and signed-error distributions use
strict parsed numeric values, including maximum degree.

`lenient_containment` is a diagnostic upper bound on what the response mentions,
**not accuracy or evidence of commitment to an answer**. It is separate from
strict scores and documented in every span metrics file's `scoring_notes`:

- Numeric: the gold integer occurs as a standalone token anywhere.
- Relation: the gold relation occurs anywhere, including candidate or negated mentions.
- Node sets: fraction of gold members mentioned, with phrase boundaries.
- Triples: fraction of gold triples whose three components are each mentioned;
  this does not require the response to assert that triple.
- Paths: any stored gold sequence appears in order, allowing intervening text.

Raw is the headline tier; set/path containment also has the other two tiers.
Containment is not a mathematical bound on set F1 (it measures recall), or on
permissive integer extraction from a non-standalone number. Differences between
these unlike metrics should not be interpreted as an exact amount of format loss.

## Reproducibility and validation

The 40-node prediction files and their source JSONL/metadata were copied read-only
from `coma-cluster`, under `~/outputs/experiments/stage1_nodes40/` and the input
locations in each run manifest. Both source checksums match both model manifests.
September 4 inputs and predictions use the existing local regression copies.
All four legacy metrics files reproduce their saved originals byte-for-byte;
checksums are recorded in [regression.json](regression.json).

The completed runs used left-to-right floating-point summation. An explicit
left-to-right loop in `_mean` preserves that behavior on local Python 3.14,
whose built-in float `sum` otherwise changes the final decimal places. This
allows direct byte regression, replacing reliance on cached historical checks.

```bash
myvenv/bin/python -m pytest -q \
  tests/test_stage1_span_scoring.py \
  tests/test_stage1_extended_scoring.py \
  tests/test_stage1_new_tasks.py
```

All 66 targeted tests pass, including direct byte comparisons for both completed
September 4 models, requested extraction examples, overlapping relation labels,
normalization tiers, cycles, final comma/numbered paths, immutable source files,
and refusal to reuse an output directory.

The full suite also exposed four existing failures in
`test_inference_slices_prompt_and_counts_realised_tokens`: its Gemma fixture
lacks `cache_implementation`. Those same four failures were reproduced with the
pre-change scorer loaded; inference code and those fixtures were left untouched.
