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
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=8` |
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
