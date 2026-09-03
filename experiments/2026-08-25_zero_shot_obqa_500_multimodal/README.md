# Zero-shot OBQA: 500 questions with graph figures and text

## Purpose

Measure the **unfine-tuned** DeepSeek-OCR-2 model on 500 OpenBookQA test
questions when it receives both:

1. the rendered GraphVis knowledge-graph figure; and
2. the question and four answer choices as text.

This is a zero-shot multimodal baseline.  It does **not** train, adapt, or
otherwise change the model weights.

## Experimental definition

| Item | Value |
| --- | --- |
| Dataset | QA-GNN-preprocessed OpenBookQA test split |
| Questions | 500 (`statement_idx` 0--499) |
| Model | DeepSeek-OCR-2 base checkpoint (set `MODEL_ID` when running) |
| Input | One rendered union-of-four-choice graph PNG + question + choices |
| Output | One option letter: `A`, `B`, `C`, or `D` |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=8192` |
| Metric | Exact option-letter accuracy |
| Fine-tuning | None |

The graph generator must be run **without** `--reveal-correct-answer`.
Record the model revision/commit, container image, GPU type, and Slurm job ID in
[`run_notes.md`](run_notes.md) after the run.  These details make the result
traceable in the thesis.

## 1. Create the fixed 500-question input

From the repository root, run:

```bash
python scripts/generate_graphvis_datasets.py \
  --split test \
  --start 0 \
  --limit 500 \
  --tasks-per-graph 6 \
  --out-dir outputs/graphvis_obqa
```

This creates the evaluation input at:

```text
outputs/graphvis_obqa/test/stage2_obqa_0_500.jsonl
```

It is intentional that the generator also creates Stage 1 artifacts; they are
not used by this zero-shot test.  Before starting inference, retain the
generated JSONL and images unchanged so a rerun uses exactly the same inputs.

## 2. Run on the cluster

Edit the clearly marked settings at the top of
[`slurm/run_inference.sbatch`](slurm/run_inference.sbatch), especially the
Apptainer image and `MODEL_ID`.  Set `MODEL_REVISION` to a commit hash whenever
the checkpoint is downloaded from Hugging Face.  The runner uses the documented
Transformers image-text-to-text interface for DeepSeek-OCR-2, so the container
must have a recent Transformers release that includes this model.  Then submit
it from the repository root:

```bash
mkdir -p outputs/experiments/2026-08-25_zero_shot_obqa_500_multimodal
sbatch experiments/2026-08-25_zero_shot_obqa_500_multimodal/slurm/run_inference.sbatch
```

The inference runner writes one JSON object per question to
`outputs/experiments/2026-08-25_zero_shot_obqa_500_multimodal/predictions.jsonl`.
It is restart-safe: submitting the same command again skips completed
`statement_idx` values.  Do not mix outputs from different model revisions or
prompting settings in the same file; use a new results directory for a new
condition.

## Qwen3-VL environment and run

Qwen uses a separate virtual environment because it requires
`transformers>=4.57.0`, while the DeepSeek runner requires exactly 4.46.3.
Create the environment on COMA, but do not run these setup commands from an
inference job:

```bash
apptainer exec --nv --bind /storage/nobackup ~/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif \
  python -m venv --system-site-packages ~/venv_qwen
~/venv_qwen/bin/pip install "transformers>=4.57.0" pillow packaging
~/venv_qwen/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
~/venv_qwen/bin/python -c "from transformers import Qwen3VLForConditionalGeneration; print('ok')"
```

The container supplies `torch==2.8.0+cu126`. If pip attempts to install or
replace Torch, install Transformers with `--no-deps` and add only its missing
pure-Python dependencies manually. The version check above must still report
`2.8.0+cu126` after setup.

Download the checkpoint into nobackup storage and copy the immutable commit
printed in the job log:

```bash
sbatch experiments/2026-08-25_zero_shot_obqa_500_multimodal/slurm/download_qwen.sbatch
```

Then submit the default prompt/visual-budget preview. It loads only the
processor, not the 8B model:

```bash
MODEL_REVISION=<resolved-commit> \
  sbatch experiments/2026-08-25_zero_shot_obqa_500_multimodal/slurm/run_inference_qwen.sbatch
```

After reviewing the preview, run the `image` and `text_noref` conditions:

```bash
MODEL_REVISION=<resolved-commit> APPROVE_PROMPT_DIFF=1 \
  sbatch experiments/2026-08-25_zero_shot_obqa_500_multimodal/slurm/run_inference_qwen.sbatch
```

Qwen defaults to a pinned 256--1280 visual-token-equivalent pixel range
(`262144 <= pixels <= 1310720`) and 64 maximum new tokens. These settings are
recorded in each `run_config.json`. They intentionally differ from
DeepSeek-OCR-2's `base_size=1024`, `image_size=768`, `crop_mode=True`, and
remote-code maximum of 8192 new tokens; the two visual and decoding budgets are
not equivalent.

At the Qwen-formatted level, removing the vision-token segment from `image`
yields `text`; removing `IMAGE_REFERENCE_SENTENCE` then yields `text_noref`.
The image prompt cannot equal `text_noref` after removing only vision tokens,
because the shared image body intentionally retains that reference sentence.

The prompt-body dump is model-independent and can be regenerated with:

```bash
python experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/prompt_common.py \
  --input-jsonl outputs/graphvis_obqa/test/stage2_obqa_0_500.jsonl \
  --graph-metadata outputs/graphvis_obqa/test/graph_metadata_0_500.jsonl \
  --dump-prompt-bodies /tmp/obqa_prompt_bodies.jsonl
```

The Part A extraction gate serialized one JSON string per physical line in
`statement_idx` order, with conditions ordered as `image`, `text`,
`text_noref`, `kg_text`. The 2,000-row dump was byte-identical before and after
the refactor, with SHA-256
`8669ab4a02035d1ce754366278a7a95cf417e0d10f3447eb097fd51ff76f4e41`.

For two run configs from the same condition, verify cross-model body identity
with:

```bash
python experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/compare_run_configs.py \
  path/to/deepseek/run_config.json path/to/qwen/run_config.json
```

## 3. Score and check the result

```bash
python experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/score_predictions.py \
  --predictions outputs/experiments/2026-08-25_zero_shot_obqa_500_multimodal/predictions.jsonl \
  --expected-count 500 \
  --metrics-out outputs/experiments/2026-08-25_zero_shot_obqa_500_multimodal/metrics.json
```

Only report the accuracy after the command confirms that there are 500 unique
predictions and no malformed/missing option letters.  Keep `predictions.jsonl`,
`metrics.json`, the Slurm log, and completed `run_notes.md` together as the
experiment record.

## Important comparability rule

If this result is compared with the earlier **text-only** run, keep fixed: the
same 500 question indices, model checkpoint/revision, prompt wording (apart
from adding the figure), decoding settings, and answer parser.  The only
intended change is the presence of the graph figure.
