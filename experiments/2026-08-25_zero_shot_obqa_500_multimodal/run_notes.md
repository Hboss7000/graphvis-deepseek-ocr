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
| Dataset input SHA-256 | |
| Prompt change from text-only baseline | Graph figure supplied before the unchanged question/choices text |
| Decoding | Greedy; maximum 8 new tokens |
| Completed predictions | |
| Accuracy | |
| Failures / exclusions | |
| Notes | |

Useful provenance commands:

```bash
git rev-parse HEAD
sha256sum outputs/graphvis_obqa/test/stage2_obqa_0_500.jsonl
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```
