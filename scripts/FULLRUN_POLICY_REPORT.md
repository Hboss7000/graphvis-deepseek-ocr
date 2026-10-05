# Full-run CPU policy report — 2026-10-05

No GPU job, paid infrastructure action or model-weight load was performed. User pushes and launches separately. Current decisions: uniform blue, QA64→every-ceiling512, Stage1=1024, nonblocking Gemma D6. Gate is per model.

## llava

Gate: **PASSED**. Resume proof: PASSED.

| job | coverage | unparsed (new parser) | ceilings | seconds/item | peak GiB |
|---|---:|---:|---:|---:|---:|
| stage1 | 45/45 | 0 | 14 | 5.261007 | 14.668 |
| qa_image | 5/5 | 0 | 0 | 0.411024 | 14.689 |
| qa_text_noref | 5/5 | 0 | 0 | 0.137079 | 14.125 |
| qa_text | 5/5 | 1 | 2 | 0.480629 | 14.130 |
| qa_kg_text | 5/5 | 0 | 1 | 0.291954 | 14.200 |

Strict estimate 1.498681 h + capped rescue allowance 0.681524 h = **2.180204 h / $4.56**. With 50%: **3.270307 h / $6.83**.

Stage 1 ceilings by task: `{"edge_number": 1, "highest_node_degree": 1, "neighbor_listing": 2, "node_degree": 0, "node_description": 4, "node_number": 1, "relation_identification": 0, "shortest_path_listing": 1, "triple_listing": 4}`.

50-graph blue − legacy paired basic recall: **-0.015022470**, PASSED. uniform: 32/50 ceilings, legacy: 28/50 ceilings, white: 32/50 ceilings. No regeneration.

Rescue forecast by condition (whole reruns, up to 512 tokens; n=5 rates are uncertain):

| condition | smoke ceilings | measured tokens/s | estimated rescue count / 500 | rescue allowance h |
|---|---:|---:|---:|---:|
| image | 0/5 | 10.949 | 0 | 0.000000 |
| kg_text | 1/5 | 60.020 | 100 | 0.236957 |
| text | 2/5 | 63.982 | 200 | 0.444567 |
| text_noref | 0/5 | 39.862 | 0 | 0.000000 |

TPS denominator: image: generation_elapsed_seconds, kg_text: generation_elapsed_seconds, text: generation_elapsed_seconds, text_noref: generation_elapsed_seconds.

## qwen

Gate: **PASSED**. Resume proof: PASSED.

| job | coverage | unparsed (new parser) | ceilings | seconds/item | peak GiB |
|---|---:|---:|---:|---:|---:|
| stage1 | 45/45 | 0 | 4 | 5.389315 | 16.745 |
| qa_image | 5/5 | 1 | 1 | 0.674398 | 16.766 |
| qa_text_noref | 5/5 | 1 | 1 | 0.341577 | 16.371 |
| qa_text | 5/5 | 0 | 0 | 0.167755 | 16.375 |
| qa_kg_text | 5/5 | 2 | 2 | 0.977309 | 16.432 |

Strict estimate 1.647473 h + capped rescue allowance 1.683028 h = **3.330501 h / $6.96**. With 50%: **4.995752 h / $10.44**.

Stage 1 ceilings by task: `{"edge_number": 3, "highest_node_degree": 1, "neighbor_listing": 0, "node_degree": 0, "node_description": 0, "node_number": 0, "relation_identification": 0, "shortest_path_listing": 0, "triple_listing": 0}`.

Rescue forecast by condition (whole reruns, up to 512 tokens; n=5 rates are uncertain):

| condition | smoke ceilings | measured tokens/s | estimated rescue count / 500 | rescue allowance h |
|---|---:|---:|---:|---:|
| image | 1/5 | 27.877 | 100 | 0.510183 |
| kg_text | 2/5 | 31.106 | 200 | 0.914441 |
| text | 0/5 | 40.535 | 0 | 0.000000 |
| text_noref | 1/5 | 55.039 | 100 | 0.258403 |

TPS denominator: image: item_elapsed_seconds (includes preprocessing), kg_text: item_elapsed_seconds (includes preprocessing), text: item_elapsed_seconds (includes preprocessing), text_noref: item_elapsed_seconds (includes preprocessing).

## deepseek

Gate: **PASSED**. Resume proof: PASSED.

| job | coverage | unparsed (new parser) | ceilings | seconds/item | peak GiB |
|---|---:|---:|---:|---:|---:|
| stage1 | 45/45 | 0 | 0 | 1.847185 | 7.613 |
| qa_image | 5/5 | 0 | 0 | 0.539235 | 7.691 |
| qa_text_noref | 5/5 | 0 | 0 | 0.115124 | 6.517 |
| qa_text | 5/5 | 0 | 0 | 0.132635 | 6.536 |
| qa_kg_text | 5/5 | 0 | 0 | 0.119626 | 6.785 |

Strict estimate 0.587716 h + capped rescue allowance 0.000000 h = **0.587716 h / $1.23**. With 50%: **0.881574 h / $1.84**.

Stage 1 ceilings by task: `{"edge_number": 0, "highest_node_degree": 0, "neighbor_listing": 0, "node_degree": 0, "node_description": 0, "node_number": 0, "relation_identification": 0, "shortest_path_listing": 0, "triple_listing": 0}`.

Rescue forecast by condition (whole reruns, up to 512 tokens; n=5 rates are uncertain):

| condition | smoke ceilings | measured tokens/s | estimated rescue count / 500 | rescue allowance h |
|---|---:|---:|---:|---:|
| image | 0/5 | 3.341 | 0 | 0.000000 |
| kg_text | 0/5 | 16.730 | 0 | 0.000000 |
| text | 0/5 | 30.171 | 0 | 0.000000 |
| text_noref | 0/5 | 17.380 | 0 | 0.000000 |

TPS denominator: image: generation_elapsed_seconds, kg_text: generation_elapsed_seconds, text: generation_elapsed_seconds, text_noref: generation_elapsed_seconds.

## gemma

Gate: **PASSED**. Resume proof: PASSED.

| job | coverage | unparsed (new parser) | ceilings | seconds/item | peak GiB |
|---|---:|---:|---:|---:|---:|
| stage1 | 45/45 | 0 | 0 | 4.579006 | 32.964 |
| qa_image | 5/5 | 3 | 3 | 2.046052 | 32.964 |
| qa_text_noref | 5/5 | 4 | 3 | 1.247737 | 22.768 |
| qa_text | 5/5 | 1 | 1 | 0.451880 | 22.775 |
| qa_kg_text | 5/5 | 5 | 5 | 1.749479 | 22.899 |

Strict estimate 1.907967 h + capped rescue allowance 5.740865 h = **7.648832 h / $15.99**. With 50%: **11.473248 h / $23.98**.

Stage 1 ceilings by task: `{"edge_number": 0, "highest_node_degree": 0, "neighbor_listing": 0, "node_degree": 0, "node_description": 0, "node_number": 0, "relation_identification": 0, "shortest_path_listing": 0, "triple_listing": 0}`.

The original three saved QA image rows exactly match the first three final predictions. `resume_verified.json` was reconstructed in the fetched directory; prediction/config/status files were preserved. Old n=5 ceiling-only sanity failures are superseded by the fixed descriptive policy. Legibility D6 remains nonblocking. The full launcher still runs the requested fresh five-question QA/resume smoke before automatically continuing.

Rescue forecast by condition (whole reruns, up to 512 tokens; n=5 rates are uncertain):

| condition | smoke ceilings | measured tokens/s | estimated rescue count / 500 | rescue allowance h |
|---|---:|---:|---:|---:|
| image | 3/5 | 19.354 | 300 | 2.204500 |
| kg_text | 5/5 | 36.582 | 500 | 1.943866 |
| text | 1/5 | 32.309 | 100 | 0.440188 |
| text_noref | 3/5 | 37.027 | 300 | 1.152311 |

TPS denominator: image: item_elapsed_seconds (includes preprocessing), kg_text: item_elapsed_seconds (includes preprocessing), text: item_elapsed_seconds (includes preprocessing), text_noref: item_elapsed_seconds (includes preprocessing).

## Total allowance

Without startup: 13.747253 h / $28.73; with 50% 20.620880 h / $43.10. With the unmeasured ~3 min Gemma startup allowance: **13.797253 h / $28.84**, or **20.695880 h / $43.25** with margin. No observed DeepSeek ceilings is not a guarantee of zero full rescues. Actual early stopping, loading and prefill costs remain unknown.

## Stage 1 processor lengths

900 records / 100 graphs, pinned LLaVA-NeXT processor/tokenizer, CPU only. Percentiles: `{"0": 1784, "100": 3137, "25": 1982.0, "50": 2192.0, "75": 2381.25, "90": 2766.0, "95": 2824.949999999999, "99": 2975.04}`; maximum **3137**; **613/900 exceed 2048**. Includes exact image-token expansion, prompt, gold answer and EOS; no truncation. Original GraphVis training may use a different collator/image encoder path, which still needs validation before fine-tuning.

## Evidence and outstanding GPU work

Gate/estimate source artifact: `outputs/fullrun_policy_2026-10-05_verified/`. Per-record/task token lengths: `outputs/fullrun_policy_2026-10-05/stage1_lengths.json`. All gates validate model/revision, settings, prompt/input/image hashes, coverage, duplicates, measured timings/VRAM and resume evidence; historical reused file hashes are checked per model. JSON gates retain exact file paths/hashes.

The five previously FAILED responses tested were Gemma image/kg_text idx8 and Qwen image idx8, kg_text idx7/8: all capped explanations and still unparsed. New explicit parsing can additionally reject previously accepted prose/prefix answers; the table above shows the new counts. Strict and rescue full predictions, 512 prefix results, extended accuracies/paired tests and 30 extended spot-check answers are **not available yet**. The code will compute/print them after real runs; CPU mocks are not GPU dry runs.

No historical Qwen/Gemma 500-question text runs are available locally. After the full DeepSeek run, `compare_fullrun_deepseek.py` still compares its strict-64 text/text_noref outputs with the old local run and lists differences plus unknown historical settings, without causal conclusions.

Node-count tercile cutpoints are **17 and 18**, with 229 low / 120 middle / 151 high graphs; ties stay together. Correct-option reachability strata are 390 reachable / 110 unreachable. These strata are fixed across every model/condition/version.
