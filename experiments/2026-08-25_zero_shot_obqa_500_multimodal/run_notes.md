# Run notes

Fill this in immediately after submitting and completing the run.

| Field | Value |
| --- | --- |
| Date and time (Europe/Berlin) | |
| Researcher | |
| Git commit | |
| Slurm job ID | |
| GPU model and count | |
| Container image (including digest/tag) | |
| Python / PyTorch / Transformers versions | |
| Model ID and revision/commit | |
| Dataset input SHA-256 | `1e1a360a274e0735251bb87fbeafc9ad37dd6248cca64e6af25d560b7db2fa3a` |
| Prompt change from text-only baseline | For `kg_text`, prepend the metadata-derived KG triple block to the unchanged `text_noref` question/choices text. |
| Decoding | Greedy for both models; DeepSeek-OCR-2 maximum 8192 new tokens, Qwen3-VL maximum 64 new tokens by default. |
| Completed predictions | |
| Accuracy | |
| Failures / exclusions | |
| Notes | All 500 metadata records contain at least one edge. In `kg_text`, disconnected visible nodes are omitted because the serialized graph contains no incident triples for them; the image condition renders those nodes as dashed and unconnected. Qwen's pinned dynamic-resolution budget is not equivalent to DeepSeek's `base_size=1024`, `image_size=768`, `crop_mode=True`. Generation flags in `run_config.json` are expected settings from the README/shell history, not independently encoded provenance. |

Useful provenance commands:

```bash
git rev-parse HEAD
sha256sum outputs/graphvis_obqa/test/stage2_obqa_0_500.jsonl
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```
