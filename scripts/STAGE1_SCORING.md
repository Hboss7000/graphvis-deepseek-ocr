# Corrected Stage 1 scoring (`audit_v2`)

`audit_v2` is the default scorer in the Qwen, DeepSeek, LLaVA and Gemma Stage 1
runners, shared run completion, and pruning-matrix collection. Generation
settings and prompts are unchanged. `--extractor legacy`, `span`,
`span_extended`, or `template_v1` selects a historical implementation.
The original `score_stage1.py` and `score_stage1_template_v1.py` remain frozen;
the historical standalone scorer and rescore commands keep their old defaults.

Use `stage1_scoring.score_record`, `aggregate_task`, and `score_files` for
version-aware code. The corrected parser exposes `extract(task, response)`;
it receives neither gold answers nor graph metadata. Metadata is used only
after extraction to score the selected claim.

## Changes

- Counts and degrees come from asserted totals or a bare final integer.
  Candidate list indices, per-node edge subtotals, in/out-degree fragments,
  and explicit retractions without replacement do not supply an answer.
- Neighbor/node lists retain literal labels, including relation names,
  stopwords, numeric labels, quoted commas and renderer annotations. Query
  prefixes and explanatory relation suffixes are excluded.
- Highest-degree extraction selects a node-degree pair. `strict_accuracy`
  requires both a valid maximum-degree node and its correct degree. Ties
  accept any maximum-degree node. `degree_accuracy` and
  `strict_degree_accuracy` retain the degree-only diagnostic.
- Triples support parenthesized tuples, three quoted/backtick fields,
  hyphen-separated triples, and known predicate-separated list entries.
  Closed malformed tuples add false positives. Incomplete tails are reported
  separately; missing text is never reconstructed.
- Relation assertions take precedence over labels on other edges in an
  explanation. Distinct unselected candidate labels are ambiguous. Asserted
  unknown labels and explicit `NO_EDGE` remain wrong claims rather than being
  replaced with a vocabulary label that matches gold.
- Paths preserve the asserted node order and literal labels. Extraction does
  not choose traces by gold endpoints or solve the graph. Explicit absence is
  represented as an empty claimed path; missing answers remain uncommitted.
- Rows report `extracted_claim`, `extraction_decision`, `parser_rule`,
  `answer_span`, and `scorer_version`. Missing claims never receive exact
  credit. An explicitly empty neighbor list against empty gold receives F1=1;
  an absent answer against that gold receives F1=0.

Raw, basic and annotation-stripped normalization remain separately reported.
Containment measures are diagnostics and do not determine correctness.

## Offline rescoring

Run from the `bachelorArbeit` project root, on CPU:

```bash
python3 scripts/rescore_stage1_audit_v2.py \
  --predictions-dir outputs/fullrun_2026-10-04_B_results/qwen/stage1 \
  --model-name qwen \
  --input-jsonl outputs/fullrun_2026-10-04_B/test/stage1_subset100.jsonl \
  --graph-metadata outputs/fullrun_2026-10-04_B/test/graph_metadata_0_500.jsonl \
  --task-set extended \
  --output outputs/my_new_stage1_rescore
```

The output directory must be new. Input responses, data and source modules
are hashed and checked for changes. Scores are recomputed from raw responses;
stored prediction-score fields are replaced. The command writes corrected
rows, metrics for all three recent scorer versions, comparisons and a source
manifest. Highest-degree comparisons use joint accuracy for every version.

## Evidence and limits

`outputs/stage1_audit_v2_2026-10-09/REPORT.md` records the CPU rescoring and
regression results. The parser agrees with 1,786/1,793 accepted sealed Qwen
Stage 1 claims (99.61%), versus 1,542/1,793 for `template_v1` using the same
claim-value comparison. This corpus informed development; it is not held-out
validation or evidence of that error rate on other models. Seven prior reader
disagreements were excluded. Seven remaining parser/reader claim differences
and eleven decision-label differences are retained for inspection.

The earlier audit's LLaVA validation failure remains recorded. This code
regression check does not repair or certify those reader packets. Prior
checkpoints, seals, snapshots and frozen scorer versions are preserved.
