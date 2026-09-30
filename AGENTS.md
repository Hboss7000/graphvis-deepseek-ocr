# AGENTS.md — BA thesis (GraphVis → DeepSeek-OCR-2 / Qwen3-VL / Gemma 3)

## Runpod cost rules (non-negotiable)

- NEVER create, start, resize, or modify a Pod or network volume without my explicit
  "yes" in the current message. Before asking, state: GPU type, data center,
  Secure/Community tier, $/hr, expected runtime, and estimated total cost.
- Use GPU Pods only. Never Serverless, never Flash, never multi-GPU.
- Default setup and training GPU: RTX PRO 6000 96GB. Use H100 only with my
  explicit approval.
- Every Pod gets an automatic termination guard: 4 h for setup Pods; for
  training/eval Pods, expected runtime + 50%, stated in the proposal.
- Before creating any Pod, list currently running Pods and flag any that are idle.
- When a job finishes or fails, tell me immediately and remind me to STOP the Pod.
- Never terminate a Pod or delete/shrink a network volume without my confirmation.
- Never create API keys, change billing settings, or enable auto-pay.

## Before any paid run

- Dry-run every training/eval script on a tiny subset first (e.g. 8 examples,
  2 optimizer steps). Report peak VRAM (`torch.cuda.max_memory_allocated`) and
  time per step, and extrapolate the full-run cost before I approve it.
- A dry run is only valid if it exercises the same code path as the full run
  (same model, same image settings, same collator). No shortcuts that skip the
  vision encoder or image loading.
- Never start a full run in the same step as fixing a bug. Fix → dry run → ask.

## Running jobs on a Pod

- Every long job runs inside `tmux` and writes a log file under
  `/workspace/logs/`. Always give me the launch command AND the monitoring
  command together (e.g. `tail -f <log>` and `watch -n 30 nvidia-smi`).
- All outputs, checkpoints, and caches go to `/workspace` (network volume),
  never to container disk (it is lost on termination).
- Save checkpoints often enough that a crash loses at most ~30 min of compute.
- `HF_HOME=/workspace/.cache/huggingface`. Download weights once; never
  re-download on each Pod start.
- Graphviz rendering and other CPU-only work happens on my laptop, not on a
  GPU Pod. Rendered images are rsynced up.

## Environments

- Two separate venvs under `/workspace/venvs/`, never mixed:
  - `venv_ocr2` — DeepSeek-OCR-2, `transformers==4.46.3`, `trust_remote_code=True`
  - `venv_qwen` — Qwen3-VL-8B-Instruct and Gemma 3 12B-it, `transformers==5.16.1`
- Pin model revisions (commit hashes). Never load a model without a pinned
  revision. TODO(Henrique): confirm these are the hashes used on COMA.
  - deepseek-ai/DeepSeek-OCR-2: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
  - Qwen/Qwen3-VL-8B-Instruct: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  - google/gemma-3-12b-it: `96b6f1eccf38110c56df3a15bffe176da04bfd80`
- If a dependency cannot be installed at the version validated on COMA
  (e.g. because of the Python version on the Pod image), stop and tell me.
  Do not silently upgrade or downgrade.

## Validated model settings (do not change without asking)

- DeepSeek-OCR-2: `base_size=1024`, `image_size=768`, `crop_mode=True`,
  eager attention.
- Gemma 3: keep the transformers 5.x patch in `gemma3_common.py` and set the
  pan-and-scan options explicitly; never rely on processor defaults.
- LoRA target modules are an experimental decision, not an implementation
  detail. Propose them; never pick them silently.

## General

- If something is ambiguous, ask. Do not guess on anything that costs money
  or changes experimental conditions.
- Report what you verified vs. what you assumed.
