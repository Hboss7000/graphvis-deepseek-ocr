# Gemma 3 zero-shot evaluation

`eval_gemma3_stage1.py` mirrors the image-only Stage 1 Qwen runner;
`eval_gemma3_stage2.py` mirrors the four-condition Stage 2 runner.
Both import the existing prompt builders and scorers. `gemma3_common.py`
contains stock Transformers loading, processor verification, structured messages,
greedy inference and the legibility-report check. No generator, Qwen runner or
scorer was changed for this evaluation implementation.

Stage 1 defaults to `--task-set paper`, 1024 new tokens, and the existing six
per-task prediction files, named `predictions_gemma3_<task>.jsonl`. The shared
Stage 1 scorer runs automatically on completion. Use a paper input JSONL with
this setting; an extended input is rejected rather than silently filtered.

`--task-set extended` accepts all nine generator task types. Specify
`--expected-count` as the actual input row count: new task eligibility varies by
graph. All nine tasks are now scored by the shared Stage 1 scorer. New-task records
preserve the input structured payload in `structured_gold`; the existing `gold`
field remains the answer string. Relation accuracy, neighbour F1, and exact/partial
shortest-path metrics are aggregated automatically. Earlier unscored Gemma files
can be scored offline with `score_stage1.py --task-set extended`.

Stage 2 retains `image`, `text`, `text_noref`, `kg_text`, `--output-jsonl`, the
option extractor, prediction field order and the 64-token default. All original
prediction fields retain their order; `image_views` and `image_soft_tokens` are appended.
Stage 1 appends the same field. Text conditions log zero and never load an image.

## Resolution and prompt comparability

There are deliberately no `--min-pixels` or `--max-pixels` flags. Gemma's fixed
896×896 encoder plus crops cannot map to Qwen's dynamic-resolution controls or
DeepSeek's vision configuration. A common flag cannot equalise vision-token
budgets across the three backbones. Any token-matched comparison must match
**realised per-item counts post hoc**, using the logs available for each model.
The manifest records this limitation explicitly.

`--pan-and-scan` is on by default; `--no-pan-and-scan` disables it. Optional
controls are:

- `--pan-and-scan-min-crop-size`
- `--pan-and-scan-max-num-crops`
- `--pan-and-scan-min-ratio-to-activate`

Omitted controls use the pinned processor's actual defaults, which are recorded
in the manifest. Before using controls, the loader inspects the installed slow
image processor's `__init__` and `preprocess` signatures and rejects unsupported
APIs. It pins `use_fast=False`, left padding and checks the processor resolution.
The API and token-count implementation were checked against the
[official Gemma documentation](https://huggingface.co/docs/transformers/model_doc/gemma3)
and [Transformers 4.57.1 processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/processing_gemma3.py).
The actual cluster environment is checked at runtime, not assumed to match that
source version.

The runners use the same prompt body functions as Qwen. The shared `<image>\n`
placeholder becomes a structured image block; no Gemma image or turn tokens are
assembled manually. There is no explicit system instruction, matching the Qwen
runners. The shared message helper supports a system role if one is needed later.

Reuse the existing 200-DPI graph renders for every run. Neither evaluator
rerenders or externally resizes them. The inherited source-image manifest blocks
are provenance expectations from Qwen, not independent verification of renderer
settings. The legibility report hashes its actual source images and rejects
changed images when reused.

## Environment and immutable offline cache

Use `/storage/home/hleonel/venv_qwen/bin/python` inside the existing
`pytorch_2.8.0-cuda12.6-cudnn9-devel.sif` container. Do not create another venv or
use `venv_ocr2`. Defaults are `google/gemma-3-12b-it`, bf16, eager attention,
batch size 1, seed 13 and static cache. The 4B and 27B multimodal checkpoints can
be selected through `--model-id`; the text-only 1B family is rejected.

Accept the Gemma licence on the Hugging Face account used for downloads. On the
login node, use that account's existing authenticated Hub setup to resolve and
prefetch a specific commit. Do not put access tokens in scripts or logs:

```bash
export HF_HOME="$HOME/nobackup/huggingface"
export MODEL_ID=google/gemma-3-12b-it
# Online login-node step. Copy the returned full SHA into MODEL_REVISION.
/storage/home/hleonel/venv_qwen/bin/python -c \
  'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["MODEL_ID"]).sha)'
export MODEL_REVISION=<full-40-character-commit>
/storage/home/hleonel/venv_qwen/bin/hf download "$MODEL_ID" --revision "$MODEL_REVISION"
```

Run these commands in the same Apptainer environment if the login node requires
it. Both loaders require the full hash and pass it to `from_pretrained`; they
never resolve a moving branch implicitly. The manifest includes both the
reference `model_revision` field and the requested `revision` field.

Before submitting, test processor access offline on the login node:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/storage/home/hleonel/venv_qwen/bin/python -c \
  'import os; from transformers import AutoProcessor; AutoProcessor.from_pretrained(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], padding_side="left", use_fast=False, local_files_only=True)'
```

A full `hf download` must have completed: processor access alone does not prove
that all weight shards are present. `run_gemma3.sbatch` additionally checks every
weight shard named by the cached index before loading the model. It explicitly
exports `HF_HOME`, `HF_HUB_OFFLINE=1` and their Apptainer equivalents so cache
misses fail rather than attempting network access on compute nodes.

## Preflight and attention checks before full runs

Run from the repository root. The sbatch file mirrors the Qwen job structure,
uses the absolute venv interpreter, and requests one GPU. Select an H100/H200
queue using the site's usual submission options.

```bash
# The wrapper defaults to processor/prompt preview, as the Qwen wrappers do.
PHASE=preflight sbatch scripts/run_gemma3.sbatch

# After reviewing the previews, run both crop settings on the same 50 graphs.
PHASE=preflight APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch

# Run text/eager, text/eager again, and text/sdpa on the same first 50 items.
PHASE=sanity APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch
```

`RESULTS_DIR`, `DATA_ROOT`, `STAGE1_JSONL`, `STAGE2_JSONL`, `GRAPH_METADATA`,
`STAGE1_EXPECTED_COUNT` and `STAGE2_EXPECTED_COUNT` can override the Qwen-style
paths/defaults in the wrapper. `--expected-count` validates the input file's row
count before preflight selection. `--preflight` without a number means 50;
`--preflight N` chooses the first N unique triple-listing graphs in index order.
Every preflight invocation runs **both** settings, writes separate prediction
files/manifests, and prints recall for each setting. Use a fresh preflight output
directory; partial preflights are not resumed.

The report is `RESULTS_DIR/preflight/preflight_report.json`. It uses the existing
scorer's raw gold-node component recall, reports all scorer normalization
variants, and passes on raw macro recall ≥0.95 by default. **0.95 is an explicit
operational threshold, not a measured Qwen baseline.** Adjust
`--preflight-min-recall` (wrapper: `PREFLIGHT_MIN_RECALL`) if the agreed comparison
criterion differs. A failure for the selected setting exits unsuccessfully after
reporting both settings. Full image inference requires a passing report for the
selected crop setting, with matching checkpoint, attention implementation, crop
kwargs, metadata and image source. A failed no-crop ablation does not prevent
using a passing pan-and-scan configuration.

The sanity phase writes three prediction files, runs the unmodified Stage 2
scorer, and writes `sanity/compare_eager_repeat.json` and
`sanity/compare_sdpa.json`. The comparison checks all prediction fields except
`timestamp_utc`, which necessarily changes. It validates matching model,
revision, prompt-body hash and generation settings before comparing. Differences
are reported with item IDs and both responses; a mismatch exits unsuccessfully.
The wrapper refuses a full run until these two 50-item reports pass. Attention
remains eager by default; SDPA is never selected silently.

```bash
PHASE=stage1 APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch
PHASE=stage2 CONDITION=image APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch
PHASE=stage2 CONDITION=text APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch
PHASE=stage2 CONDITION=text_noref APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch
PHASE=stage2 CONDITION=kg_text APPROVE_PROMPTS=1 sbatch scripts/run_gemma3.sbatch
```

For extended Stage 1, set `TASK_SET=extended`, point `STAGE1_JSONL` at the extended
input and set its actual `STAGE1_EXPECTED_COUNT`. Use a separate `RESULTS_DIR` or
invoke the Python runner with a fresh output directory and the existing matching
`--preflight-report`. Do not mix configurations in one result directory.

Use `scontrol show job <job-id>` for job status; these scripts do not use `sacct`.

## Validation delivered with the code

```bash
myvenv/bin/python -m pytest -q tests/test_gemma3_evaluation.py tests/test_stage1_new_tasks.py
bash -n scripts/run_gemma3.sbatch
```

The local contract tests cover exact prompt-body parity, both crop settings,
failed-gate rejection, actual token counting and prompt slicing through a fake
processor/model, unchanged paper scoring, extended scoring, resume,
and a 50-item synthetic Stage 2 run accepted by the unmodified scorer. Fake-model
attention/repeat equality is only a test of the comparison machinery.

**Real Gemma inference acceptance remains pending.** This development workspace
has no torch/transformers installation, Gemma cache or cluster venv. The configured
cluster SSH connection timed out. No Gemma weights were downloaded, no Slurm job
was submitted, and no real recall, eager-vs-SDPA equality or GPU repeatability
result is claimed. Run the two cluster phases above and inspect their reports
before reporting image-condition accuracy.
