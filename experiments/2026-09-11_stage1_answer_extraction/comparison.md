# Legacy vs span extraction: completed Stage 1 runs

CPU-only rescoring of **14,992 stored responses**. All task records are included, including unparsed responses and token-ceiling hits. Scores use raw normalization; values below are percentages (F1 multiplied by 100). **Lenient containment is not accuracy.**

## September 4: six tasks, 3,000 responses per model

| Task | Model | N | Strict metric | Legacy | Span | Lenient containment |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| node_description | Qwen | 500 | macro F1 | 58.02 | 96.64 | 99.33 |
| node_description | DeepSeek | 500 | macro F1 | 13.19 | 3.30 | 45.51 |
| node_number | Qwen | 500 | accuracy | 10.40 | 80.20 | 83.80 |
| node_number | DeepSeek | 500 | accuracy | 0.60 | 0.60 | 1.20 |
| edge_number | Qwen | 500 | accuracy | 0.60 | 29.60 | 66.60 |
| edge_number | DeepSeek | 500 | accuracy | 4.20 | 4.20 | 4.20 |
| triple_listing | Qwen | 500 | macro F1 | 0.95 | 48.45 | 98.16 |
| triple_listing | DeepSeek | 500 | macro F1 | 0.00 | 0.00 | 40.91 |
| highest_node_degree | Qwen | 500 | degree accuracy | 26.80 | 20.40 | 86.80 |
| highest_node_degree | DeepSeek | 500 | degree accuracy | 1.00 | 0.00 | 1.60 |
| node_degree | Qwen | 500 | accuracy | 48.80 | 61.60 | 87.20 |
| node_degree | DeepSeek | 500 | accuracy | 22.20 | 22.20 | 21.80 |

## 40 nodes: nine tasks, 4,496 responses per model

| Task | Model | N | Strict metric | Legacy | Span | Lenient containment |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| node_description | Qwen | 500 | macro F1 | 64.66 | 94.08 | 95.94 |
| node_description | DeepSeek | 500 | macro F1 | 11.76 | 1.69 | 24.20 |
| node_number | Qwen | 500 | accuracy | 1.80 | 23.60 | 27.80 |
| node_number | DeepSeek | 500 | accuracy | 1.80 | 1.60 | 1.80 |
| edge_number | Qwen | 500 | accuracy | 0.60 | 4.60 | 30.60 |
| edge_number | DeepSeek | 500 | accuracy | 0.40 | 0.40 | 0.80 |
| triple_listing | Qwen | 500 | macro F1 | 0.57 | 14.22 | 69.37 |
| triple_listing | DeepSeek | 500 | macro F1 | 0.00 | 0.00 | 12.21 |
| highest_node_degree | Qwen | 500 | degree accuracy | 3.80 | 3.60 | 44.40 |
| highest_node_degree | DeepSeek | 500 | degree accuracy | 0.20 | 0.00 | 0.20 |
| node_degree | Qwen | 500 | accuracy | 32.40 | 32.80 | 81.80 |
| node_degree | DeepSeek | 500 | accuracy | 22.60 | 22.60 | 22.60 |
| relation_identification | Qwen | 499 | accuracy | 0.00 | 53.11 | 81.36 |
| relation_identification | DeepSeek | 499 | accuracy | 7.41 | 20.84 | 26.45 |
| neighbor_listing | Qwen | 500 | macro F1 | 0.85 | 16.17 | 82.25 |
| neighbor_listing | DeepSeek | 500 | macro F1 | 5.59 | 5.16 | 15.62 |
| shortest_path_listing | Qwen | 497 | path exact | 0.00 | 22.13 | 41.25 |
| shortest_path_listing | DeepSeek | 497 | path exact | 0.00 | 0.20 | 2.82 |

## Cross-backbone differences on the 40-node run

Qwen minus DeepSeek, in percentage points of the displayed strict metric. The change is caused by extraction/scoring policy on identical responses; it does not establish the models’ underlying comprehension accuracy.

| Task | Legacy gap | Span gap | Change |
| --- | ---: | ---: | ---: |
| node_description | +52.90 | +92.39 | +39.50 |
| node_number | +0.00 | +22.00 | +22.00 |
| edge_number | +0.20 | +4.20 | +4.00 |
| triple_listing | +0.57 | +14.22 | +13.65 |
| highest_node_degree | +3.60 | +3.60 | +0.00 |
| node_degree | +9.80 | +10.20 | +0.40 |
| relation_identification | -7.41 | +32.26 | +39.68 |
| neighbor_listing | -4.74 | +11.01 | +15.75 |
| shortest_path_listing | +0.00 | +21.93 | +21.93 |

## Relation frequency and path diagnostics

| Model | Relation strict, all (N=499) | Strict, excluding relatedto | Excluded-bucket N | Containment, excluding relatedto | Cyclic paths (N=497) |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen | 53.11 | 53.57 | 420 | 82.14 | 0.60 |
| deepseek | 20.84 | 16.67 | 420 | 19.29 | 24.55 |

## Highest-degree joint correctness

The main tables compare degree accuracy. Joint accuracy below also requires the node name. Legacy accepted any valid maximum-degree name mentioned in the response; span requires its selected name.

| Dataset | Model | Legacy joint | Span joint |
| --- | --- | ---: | ---: |
| paper | qwen | 26.80 | 8.00 |
| paper | deepseek | 0.40 | 0.00 |
| nodes40 | qwen | 3.80 | 1.80 |
| nodes40 | deepseek | 0.00 | 0.00 |

## Interpretation and artifacts

Qwen’s 40-node relation score changes from 0/499 to 265/499; its containment is 406/499. Boundary-aware vocabulary matching gives 406 mentions rather than the handout’s preliminary 410. The neighbor example with `looking` and `look`, including its trailing commentary, now has strict precision and recall of 1.0.

Strict extraction is deliberately conservative: responses ending in commentary without an answer marker can still lose the asserted relation, while unstructured multiline node output falls back to the last line. This lowers DeepSeek node-description F1. Degree-cue extraction also rejects ambiguous numeric node names. These decreases are retained, not hidden. The table therefore measures the specified extraction policy, not a guarantee of recovering every intended answer.

For sets, containment measures gold recall and is not directly comparable to F1 as a numerical loss bound. Triple containment can be high when the right components are scattered across incorrect triples. Numeric containment requires standalone integer tokens, whereas strict integer parsing can extract a digit sequence from a decimal or attached token. None of these containment values should be quoted as accuracy.

All four legacy metrics files are byte-identical to their saved originals. [regression.json](regression.json) records original and rescored hashes, scorer hash, interpreter version and source hashes. [README.md](README.md) documents the extraction rules, CLI and validation.

Full metrics, per-record strict extractions/cycle flags, comparisons and manifests are in:

- [paper/qwen metrics](../../outputs/stage1_answer_extraction_2026-09-11_verified/paper/qwen/metrics_qwen.json) and [comparison](../../outputs/stage1_answer_extraction_2026-09-11_verified/paper/qwen/comparison.md).
- [paper/deepseek metrics](../../outputs/stage1_answer_extraction_2026-09-11_verified/paper/deepseek/metrics_deepseek.json) and [comparison](../../outputs/stage1_answer_extraction_2026-09-11_verified/paper/deepseek/comparison.md).
- [nodes40/qwen metrics](../../outputs/stage1_answer_extraction_2026-09-11_verified/nodes40/qwen/metrics_qwen.json) and [comparison](../../outputs/stage1_answer_extraction_2026-09-11_verified/nodes40/qwen/comparison.md).
- [nodes40/deepseek metrics](../../outputs/stage1_answer_extraction_2026-09-11_verified/nodes40/deepseek/metrics_deepseek.json) and [comparison](../../outputs/stage1_answer_extraction_2026-09-11_verified/nodes40/deepseek/comparison.md).
