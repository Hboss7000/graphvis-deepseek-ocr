# Stage 1 graph-comprehension zero-shot evaluation (RQ1)

This experiment measures whether DeepSeek-OCR-2 and Qwen3-VL-8B-Instruct can
read the rendered knowledge graphs themselves. It evaluates the six Stage 1
tasks in the image condition only. It is additive: the renderer, pruning code,
shared `prompt_common.py`, and the 2026-08-25 experiment are unchanged.

## Audited input

The generated Stage 1 file is:

`outputs/graphvis_obqa/test/stage1_graph_comprehension_0_500.jsonl`

It contains 3,000 records, with 500 records for each of:

- `node_description`
- `node_number`
- `edge_number`
- `triple_listing`
- `highest_node_degree`
- `node_degree`

Every `(statement_idx, task_type)` pair is unique. Every Stage 1 record has the
same image as the matching Stage 2 and metadata record. The three files were
written within milliseconds of one another. Shell history records the common
generation command:

```bash
python scripts/generate_graphvis_datasets.py \
  --split test --start 0 --limit 500 --tasks-per-graph 6 \
  --data-root data_preprocessed_release \
  --out-dir outputs/graphvis_obqa --seed 13
```

Audited SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `stage1_graph_comprehension_0_500.jsonl` | `e86ee6ff6fcd73dd493f04ce481dbea598a49c0312f4dfefe2c166166b53d1c5` |
| `stage2_obqa_0_500.jsonl` | `1e1a360a274e0735251bb87fbeafc9ad37dd6248cca64e6af25d560b7db2fa3a` |
| `graph_metadata_0_500.jsonl` | `01049cd8b4bfde4008c43aa1119d7eb5da1898d019db1d9cb83e713f4a7bb4dc` |

The runners validate the Stage 1 count, unique composite keys, task names,
metadata coverage, and image parity again before preview or inference.

## Models and generation

| Model | Immutable revision | Output budget |
| --- | --- | ---: |
| `deepseek-ai/DeepSeek-OCR-2` | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | remote-code maximum 8,192 |
| `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 1,024 new tokens |

DeepSeek receives `<image>\n` followed verbatim by the Stage 1 prompt through
the same remote-code `model.infer` path as the earlier experiment. Qwen receives
one image content item plus the verbatim prompt through its chat template. Both
use greedy decoding. Qwen retains the earlier pinned 262,144–1,310,720 dynamic
pixel range; this is not visually equivalent to DeepSeek's `base_size=1024`,
`image_size=768`, `crop_mode=True` path.

## Answer-format conditions

All Stage 1 runners accept `--answer-format none|constrained` and default to
`none`, which reproduces the original prompts. `constrained` appends a
task-specific formatting suffix at inference time. The source JSONL `prompt`
field is never modified, so the suffix cannot enter generated training data.
The exact suffix map, separator, selected mode, effective prompt SHA-256, and
expected input split are recorded in `run_config.json`.

The constrained condition is defined only for the six paper tasks. It allows
reasoning but requires the final `Answer:` line/block documented in
`stage1_common.py`. It intentionally fails for the three extended tasks until
their formats are specified. Use `--expected-split train|dev|test` (default
`test`) to make pilot/test separation explicit; prompt development must use
train or dev and be frozen before loading test records.

## COMA runs

Both jobs default to a safe preview. DeepSeek prints the exact raw prompt passed
to `model.infer` for the first record of each task and exits before importing
PyTorch or loading the model. Qwen loads only its processor, prints the six
chat-template prompts and the first-image budget, and exits before loading the
model.

```bash
sbatch experiments/2026-09-04_stage1_graph_comprehension_zero_shot/slurm/run_stage1_deepseek.sbatch
sbatch experiments/2026-09-04_stage1_graph_comprehension_zero_shot/slurm/run_stage1_qwen.sbatch
```

After reviewing each preview log, submit the restart-safe full jobs:

```bash
sbatch --export=ALL,APPROVE_PROMPTS=1 \
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/slurm/run_stage1_deepseek.sbatch
sbatch --export=ALL,APPROVE_PROMPTS=1 \
  experiments/2026-09-04_stage1_graph_comprehension_zero_shot/slurm/run_stage1_qwen.sbatch
```

Both full paths pass `--resume`. Qwen uses
`HF_HOME=${HOME}/nobackup/huggingface`; the current Hub CLI executable is `hf`,
not `huggingface-cli`.

## Outputs

Each model writes into its own results directory:

```text
predictions_<model>_<task>.jsonl
metrics_<model>.json
failure_cases_<model>.md
run_config.json
```

Every prediction contains the source identifiers and prompt, gold and raw
response, parsed task-specific fields, strict correctness flags, generated
token count, and `hit_token_ceiling`. The metrics report the ceiling count and
fraction separately for every task. For DeepSeek, token counting is necessarily
approximate because the validated remote `infer` method returns decoded text;
Qwen's count comes directly from generated token IDs.

The runners score automatically after all 3,000 predictions are present. To
rescore an already completed directory independently:

```bash
python3 experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/score_stage1.py \
  --input-jsonl outputs/graphvis_obqa/test/stage1_graph_comprehension_0_500.jsonl \
  --graph-metadata outputs/graphvis_obqa/test/graph_metadata_0_500.jsonl \
  --predictions-dir /path/to/model/results \
  --model-name deepseek
```

Use `--model-name qwen` for the Qwen output directory.

## Scoring decision table

Numeric tasks report exact accuracy, MAE over parseable responses, signed error
distribution, and gold-value distribution. `highest_node_degree` separately
reports degree accuracy, tie-aware name accuracy, and joint accuracy. Valid tied
names are recomputed from metadata, while the target degree is checked against
the stored gold.

Node and triple sets are scored under all three requested variants:

1. raw component strings;
2. lowercase, collapsed whitespace, and stripped trailing punctuation;
3. basic normalization plus removal of trailing `[A]` or `[B,C]` annotations.

Each variant includes exact set equality and macro/micro precision, recall, and
F1. Triple metrics require all three tuple components to match and additionally
report gold-node component recall anywhere in the response. No fuzzy match ever
contributes to a score. The failure report separately lists every unmatched
node-description prediction within Levenshtein distance 2 of a gold node.

## Known maximum-degree confound

The recomputed `highest_node_degree` gold distribution is:

```text
1: 4, 2: 23, 3: 47, 4: 52, 5: 369, 6: 4, 7: 1
```

Thus 369/500 (73.8%) targets equal the expected `max_degree=5` pruning cap. The
metric JSON flags this pile-up explicitly. Degree-only performance on this task
must not be interpreted as unconstrained graph comprehension; a constant answer
of 5 already obtains 73.8% degree accuracy.
