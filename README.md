# GraphVis → DeepSeek-OCR-2

Bachelor's thesis project (TU Munich, Data Engineering) investigating whether the **GraphVis** curriculum fine-tuning method ([Deng et al., NeurIPS 2024](https://arxiv.org/abs/2402.16631)) transfers to **DeepSeek-OCR-2** as a backbone vision-language model, evaluated on **OpenBookQA (OBQA)**.

Supervised by Yuchen.

## Research questions

- **Main RQ** — Does GraphVis-style curriculum fine-tuning transfer to DeepSeek-OCR-2 on OBQA?
- **RQ1** — What is the base model's performance on six graph-comprehension tasks (zero-shot)?
- **RQ2** — How much does Stage 1 (graph comprehension) fine-tuning improve graph interpretation on those tasks?

## Method overview

The original GraphVis pipeline retrieves a ConceptNet subgraph relevant to a question, renders it as an image via Graphviz, and fine-tunes an LVLM in two curriculum stages:

1. **Stage 1 — Graph comprehension fine-tuning**: self-supervised tasks (node count, edge count, node degree, highest-degree node, node listing, triple listing) teach the model to read the rendered graph image itself.
2. **Stage 2 — KG-enhanced QA fine-tuning**: the rendered subgraph is paired with the original QA question, teaching the model to *use* the graph to answer.

This repo adapts that pipeline to DeepSeek-OCR-2 as the backbone and to QA-GNN-preprocessed OBQA data as input.

## Repository structure

```
.
├── generate_graphvis_datasets.py   # main dataset generation pipeline (Stage 0–2)
├── convert_one.py                  # single QA-GNN .pk entry → JSONL graph object
├── render_paths.py                 # single-entry graph filtering + Graphviz rendering
├── render_question_unified.py      # union-of-4-choices merged graph rendering
├── CLAUDE.md                       # project context for Claude Code sessions
└── README.md
```

## Pipeline stages (dataset generation)

- **Stage 0** — Load QA-GNN preprocessed `.pk` files (COO adjacency matrices; `concepts`/`qmask`/`amask`; 4 entries per question at indices `4i…4i+3`).
- **Stage 1a** — Union-of-four choice-subgraph aggregation (`merge_choice_graphs`). ⚠️ This is a working interpretation of how to combine the four per-choice subgraphs into one image, not an explicit claim from the paper.
- **Stage 1b** — Bridge-node filtering (by default, keep fillers connecting a question node to an answer node); Q/A nodes form the core, subject to the node budget.
- **Stage 1c** — Pruning heuristics (defaults: `max_nodes=18`, `max_edges=60`, `max_degree=0`, relation-priority ordering). Each cap accepts `0` to disable it; filler eligibility and lifeline exemptions are configurable. ⚠️ Codex-invented, not paper-grounded — documented as such in the thesis. See [pruning ablation flags](scripts/README.md#pruning-ablations). **Compatibility change:** `--max-edges 0` now means unlimited edges, previously zero edges.
- **Stage 1d** — Graphviz rendering, with an answer-leak fix gated behind `--reveal-correct-answer`.
- **Stage 2** — KG-enhanced QA record construction (`build_stage2_record`).

## Known open issues

1. Missing `pruning_note` in metadata documenting the non-paper-grounded pruning heuristics (an `aggregation_note` exists, but no analogous one for pruning).
2. The Gemma Stage 1/2 evaluators still hardcode the old `18/30/5` pruning provenance; a separate fix should read it from graph metadata.
3. Positive degree caps are soft by default because of lifeline exemptions; use `--no-lifelines` for a hard cap.

## Status

- [x] Dataset generation pipeline built and analyzed
- [x] Model loaded on TUM COMA cluster (H200 NVL, `bfloat16`, `eager` attention)
- [ ] Live inference on a rendered graph image confirmed
- [x] Choice-letter label leak fixed
- [ ] Full training split generated
- [ ] Stage 1 fine-tuning run
- [ ] Stage 2 fine-tuning run

## Data setup

`scripts/generate_graphvis_datasets.py` expects the following files, relative to `--data-root` (default `data_preprocessed_release/`):

```
data_preprocessed_release/
├── cpnet/
│   └── concept.txt
└── obqa/
    ├── graph/
    │   └── {split}.graph.adj.pk        # split = train | dev | test
    └── statement/
        └── {split}.statement.jsonl
```

This is the standard output layout of QA-GNN's own preprocessing pipeline ([Yasunaga et al., NAACL 2021](https://github.com/michiyasunaga/qagnn)).

**Automated setup:**
```bash
./scripts/setup_data.sh
```
This clones the official QA-GNN repo, runs their `download_preprocessed_data.sh`, and copies the `cpnet/` and `obqa/` folders into `data_preprocessed_release/` at the project root. It does not re-host the data itself.

⚠️ Not yet verified end-to-end — QA-GNN's data is hosted externally (Stanford NLP group servers), so if the download step fails, check the [QA-GNN repo](https://github.com/michiyasunaga/qagnn) directly for current download instructions.

**Manual setup**, if the script doesn't work:
1. Clone https://github.com/michiyasunaga/qagnn
2. Run `./download_preprocessed_data.sh` inside it
3. Copy `data/cpnet/` and `data/obqa/` from that repo into `data_preprocessed_release/` here

Data is 17-relation-merged ConceptNet + QA-GNN-preprocessed OpenBookQA (`.pk` COO adjacency format, 4 graph entries per question — one per answer choice).

## Compute

TUM COMA GPU cluster (NVIDIA H200 NVL), Apptainer + PyTorch container, Slurm (24h job limit). Full-split generation is designed for array-job sharding via `--start`/`--limit` (slice offset/count, not start/end indices).

### Stage 1 constrained-output pilot

The constrained condition changes inference prompts only; it never changes training data.
Qwen3-VL and Gemma 3 use greedy decoding with an effective limit of 2048 new
tokens in that condition (1024 in the reproducible `none` condition). DeepSeek-OCR-2's
pinned remote `infer()` path remains asymmetric: it hard-codes 8192 new tokens and a
35-token no-repeat n-gram. Every runner records its effective limit in `run_config.json`.

The DeepSeek dev ablation includes completion-style prompts because the original
[DeepSeek-OCR report](https://arxiv.org/abs/2510.18234) describes next-token training
over OCR/image-text data without a separate instruction-alignment or SFT stage, while
the [DeepSeek-OCR 2 report](https://arxiv.org/abs/2601.20552) says it reuses the same
data sources and specializes the decoder with continued next-token training. This is a
rationale to test, not evidence that completion prompts will work on GraphVis. At pinned
revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`, `infer()` formats the prompt with
the `plain` conversation template; its separator is empty and its empty assistant turn
adds no suffix, so the model input ends exactly at the supplied completion prefix.

## Citation

```bibtex
@inproceedings{deng2024graphvis,
  title     = {GraphVis: Boosting LLMs with Visual Knowledge Graph Integration},
  author    = {Deng, Yihe and Ye, Chenchen and Huang, Zijie and Ma, Mingyu Derek and Kou, Yiwen and Wang, Wei},
  booktitle = {NeurIPS},
  year      = {2024}
}
```
