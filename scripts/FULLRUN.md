# Full inference run — S-A, CPU preparation (2026-10-04)

Continue on `inference50-core-keep`. No GPU job, model download, pod access, or environment installation was performed during preparation. Stop at S-A; Henrique reviews before S0. Stop again after the rehearsal (S-B), each model (S-C), and consolidation (S-D).

## Prepared data and verified consistency

`outputs/fullrun_2026-10-04_B/` contains 500 uniform-style config-B images, 500 QA records, and `test/stage1_subset100.jsonl` with 900 records (100 graphs × all nine tasks; all selected graphs have eligible shortest paths). The superset is `subsets/obqa_test_subset100_seed13.json`: existing 50 plus `random.Random(13).sample(sorted(remaining_450), 50)`, sorted.

The 500-graph generator's Stage 1 file is deliberately **not** an inference input. It differs in 403/450 baseline records because its shared RNG consumes draws for other graphs. Changed fields are recorded per item in `consistency.json`. The composed input preserves the original 450 lines exactly and adds a separate seed-13 run over the extra 50 indices. `--reuse-render-from` validates the graph and reuses its images/metadata, without another Graphviz render. `stage1_provenance.json` records source paths, SHA-256, flags, base generator commit and exact working-source hash. The generator hash identifies the source even though generation preceded the final commit.

All original 50 image dimensions and orientations match config B. QA and full graph metadata match byte-for-byte, including pruning. The four data tests check explicit subset construction, byte-preserving composition, DOT node attribute equality, unchanged legacy defaults, identical JSONL/dimensions across styles, and rendering reuse.

Diagnostics:

| Measurement | Value |
|---|---:|
| Nodes, median / maximum | 18 / 43 |
| Edges, median / maximum | 23 / 30 |
| Edge-cap truncations | 146 / 500 (29.2%) |
| Correct-answer option degree zero | 13.4% |
| Correct-answer option reachable from question | 78.0% |
| Correct answer missing concept | 12 / 500 |
| Estimated LLaVA label height, median | 10.543 px |
| Images below estimated 10 px | 217 / 500 |
| LR / TB orientation | 129 / 371 |

The node count can exceed 18 because `core-policy keep` retains every core node. Label heights are geometric estimates, not measured reading accuracy. Detailed correct-answer/distractor node and option denominators are in `pruning_diagnostics.json`.

The new split is approximately 279 MB (decimal). All local outputs are approximately 2.61 GB; the **laptop** HF cache is 1.32 GB. Reserve 1 GB for the four runs' 11,600 predictions, scoring JSON and logs (at most 4,198,400 generated tokens). The **pod's** actual cache size/free space is unknown; measure it before S0. Preflight requires at least 5 GB free on the 100 GB workspace volume and complete pinned cached snapshots. Nothing downloads in these scripts.

## Historical regression evidence and limitations

`preparation_report.json` records the evidence. Both current text prompt-body hashes equal hashes reconstructed from the old 500-question input:

- `text`: `1adc53548a0d9d6006e5bf7c639d2192af4fa994c33789f9db95f7e57d117349`
- `text_noref`: `4157c560be865950ec0a45c05200875816d2148e22d96d2f8eb185c31f6fd58b`.

Only the old DeepSeek image/text/text_noref predictions are available locally (no kg_text); their revision matches the pinned revision. Their run_config is absent. The historical run notes document an 8192-token default, versus the new explicit 64-token cap; that is documented provenance, not verification of the old effective setting. Henrique confirmed the Qwen/Gemma runs are on unreachable COMA and explicitly waived that part of §3.4. Actual old-run prompt hashes/settings and prediction equivalence are therefore **not verified**. CPU prompt reconstruction does not establish GPU output equality. No historical predictions enter new results. After the new DeepSeek run, `compare_fullrun_deepseek.py` lists text/text_noref differences and unknown settings without drawing conclusions.

## Execution contract

The only shell path root is `WORKSPACE` (default `/workspace`). The existing clone is `$WORKSPACE/bachelorArbeit`; outputs stay in its `outputs/`, logs in `$WORKSPACE/logs`, caches in `$WORKSPACE/.cache`, temporary files in `$WORKSPACE/.cache/fullrun/tmp`. `HF_HOME=$WORKSPACE/.cache/huggingface`, both HF offline flags are enabled. No pod lifecycle commands are present. The shell must run in tmux.

Validated versions are pinned to `torch==2.8.0+cu128`, `transformers==5.16.1` in `venv_qwen` and `transformers==4.46.3` in `venv_ocr2`, taken from the available phase-1 GPU run configs. Missing/mismatched versions stop preflight; there is no install or fallback. All four model revisions are immutable 40-character commits; LLaVA is `2424fdd47412fccc66d91719126b420e9fbd7065`. Other revisions are those in AGENTS.md.

`fullrun_manual.sh MODEL` runs Stage 1, then QA image, text_noref, text, kg_text. Gemma first runs both crop variants of its real five-graph preflight. D6 explicitly allows a failed recall gate; the report remains required, its identity/images are checked, and subsequent run_config files record `legibility_confounded=true`. A technical failure that produces no valid report remains a job failure. Independent later jobs continue.

`SMOKE=1 ... all` runs all models sequentially, 45 Stage 1 items and five QA items per condition/model, both Gemma preflight crop variants, plus paired 20-graph LLaVA reading probes on uniform and legacy images. QA image runs first with `--limit 3`, then `--limit 5 --resume` against the **same five-record input and config**. It verifies the original three rows remain unchanged and exactly five distinct rows exist. Smoke lives only in `outputs/fullrun_2026-10-04_B_smoke`; full results live in `outputs/fullrun_2026-10-04_B_results`.

Before loading any weights, the parent checks every planned input/image hash, count and key, existing output/config/duplicate state, free space, imports, exact versions, one CUDA GPU, pinned snapshot config/tokenizer and every indexed safetensors shard. Every run_config contains all input hashes and code-file hashes. Unknown output directories are collisions; known compatible outputs resume. JSONL rows flush individually; corrupt/incomplete existing lines are rejected rather than discarded. A workspace-wide GPU lock rejects concurrent sessions; each model directory also has a resume lock.

Tripwires inspect the chronological first 20 predictions: >50% QA parse failure; >80% empty/one-token Stage 1 responses; >3× reference mean item time. References are the supplied Stage 1 3.5/5.2/4.2/1.7 s/item; these are explicitly provisional, conservative references for QA as no equivalent measured QA reference was supplied. S0 reports QA timing separately. Output-growth monitoring runs continuously from process launch, including loading: no growth for five minutes stops that job. Console chatter does not reset it. The watchdog signals the job process group and escalates if it cannot exit.

`MAX_HOURS` defaults to four hours **for the whole invocation**, including all models in a smoke invocation. On expiry it stops the current process, preserves flushed predictions, marks PAUSED and prints the rerun command instruction. **This does not stop pod billing.** One append-only log and one status file with attempt history exist per job. COMPLETED/FAILED/TRIPWIRE are printed immediately, with a STOP-Pod reminder.

`verify_fullrun.py` checks exact coverage, duplicate/misplaced/unexpected keys, input/config/model/version/decoding identity, common prompt hashes across available models, stored QA parsing against the shared parser, real time/VRAM fields and ceiling rates. Operational final sanity limits are 50% parse failure (QA), 50% token ceilings (every job), and the same first-20 tripwires. These are safety checks, not changed scoring rules. DeepSeek's existing token counts/ceiling detection remain approximate (response retokenization), as recorded by its runner.

The CPU stub tests exercise orchestration only. They **do not** constitute the real S0 rehearsal. Full model runs refuse to start until all four real smoke runs, resume proofs, both Gemma preflight arms and both LLaVA reading probes verify. The reading gate compares paired basic recall; a drop greater than 0.05 stops full execution. `raw` is also reported. S0 produces measured s/item, peak `torch.cuda.max_memory_allocated`, per-model extrapolated inference time/cost at $2.09/h and an explicit 50% overhead margin.

## Exact transfer and session commands

On the laptop, from this repository, enter the **current** pod SSH address. No stale IP is hard-coded. `IP` and `PORT` remain shell variables; use the same values for fetch-back. Henrique handles pod creation and stopping.

```bash
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
export IP PORT
WORKSPACE=/workspace
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"

# Publish the reviewed local commit so the existing pod clone can pull it.
git push origin inference50-core-keep
"${SSH[@]}" "root@$IP" "cd '$WORKSPACE/bachelorArbeit' && git pull --ff-only origin inference50-core-keep && git rev-parse HEAD"

rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" \
  outputs/fullrun_2026-10-04_B \
  outputs/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_tb30 \
  "root@$IP:$WORKSPACE/bachelorArbeit/outputs/"

"${SSH[@]}" "root@$IP" "df -h '$WORKSPACE'; du -sh '$WORKSPACE/.cache/huggingface' '$WORKSPACE/bachelorArbeit/outputs'"
"${SSH[@]}" "root@$IP"
```

On the pod, S0 (only after S-A review):

```bash
WORKSPACE=/workspace
cd "$WORKSPACE/bachelorArbeit"
tmux new-session -d -s fullrun-s0 "cd '$WORKSPACE/bachelorArbeit' && SMOKE=1 MAX_HOURS=4 bash scripts/fullrun_manual.sh all; read -r -p 'STOP the Pod after reviewing logs. Press Enter to close.'"
# Monitoring, paired with the launch:
watch -n 30 "nvidia-smi; tail -n 3 $WORKSPACE/logs/fullrun_2026-10-04_B_smoke/*/*.log"
# Detailed current job log, e.g.:
tail -F "$WORKSPACE/logs/fullrun_2026-10-04_B_smoke/llava/stage1.log"
# When S0 ends:
"$WORKSPACE/venvs/venv_qwen/bin/python" scripts/verify_fullrun.py all --smoke
"$WORKSPACE/venvs/venv_qwen/bin/python" scripts/fullrun_summary.py
# STOP the Pod. Review S-B, measured costs and reading gate before any full session.
```

After S-B approval, S1:

```bash
WORKSPACE=/workspace
cd "$WORKSPACE/bachelorArbeit"
tmux new-session -d -s fullrun-llava "cd '$WORKSPACE/bachelorArbeit' && MAX_HOURS=4 bash scripts/fullrun_manual.sh llava; read -r -p 'STOP the Pod after verification. Press Enter to close.'"
watch -n 30 "nvidia-smi; tail -n 3 $WORKSPACE/logs/fullrun_2026-10-04_B_results/llava/*.log"
"$WORKSPACE/venvs/venv_qwen/bin/python" scripts/verify_fullrun.py llava
# STOP the Pod and review S-C.
```

S2, after reviewing the previous session:

```bash
WORKSPACE=/workspace
cd "$WORKSPACE/bachelorArbeit"
tmux new-session -d -s fullrun-qwen "cd '$WORKSPACE/bachelorArbeit' && MAX_HOURS=4 bash scripts/fullrun_manual.sh qwen; read -r -p 'STOP the Pod after verification. Press Enter to close.'"
watch -n 30 "nvidia-smi; tail -n 3 $WORKSPACE/logs/fullrun_2026-10-04_B_results/qwen/*.log"
"$WORKSPACE/venvs/venv_qwen/bin/python" scripts/verify_fullrun.py qwen
# STOP the Pod and review S-C.
```

S3 runs Gemma first, then pauses for its S-C review before DeepSeek:

```bash
WORKSPACE=/workspace
cd "$WORKSPACE/bachelorArbeit"
tmux new-session -d -s fullrun-gemma "cd '$WORKSPACE/bachelorArbeit' && MAX_HOURS=4 bash scripts/fullrun_manual.sh gemma; read -r -p 'STOP the Pod after verification. Press Enter to close.'"
watch -n 30 "nvidia-smi; tail -n 3 $WORKSPACE/logs/fullrun_2026-10-04_B_results/gemma/*.log"
"$WORKSPACE/venvs/venv_qwen/bin/python" scripts/verify_fullrun.py gemma
# Review Gemma S-C before the following launch; STOP the Pod during any wait.
tmux new-session -d -s fullrun-deepseek "cd '$WORKSPACE/bachelorArbeit' && MAX_HOURS=4 bash scripts/fullrun_manual.sh deepseek; read -r -p 'STOP the Pod after verification. Press Enter to close.'"
watch -n 30 "nvidia-smi; tail -n 3 $WORKSPACE/logs/fullrun_2026-10-04_B_results/deepseek/*.log"
"$WORKSPACE/venvs/venv_qwen/bin/python" scripts/verify_fullrun.py deepseek
# Run this CPU comparison on the laptop after fetching results (the historical files are local):
# myvenv/bin/python scripts/compare_fullrun_deepseek.py --output-dir outputs/fullrun_2026-10-04_B_deepseek_comparison
# STOP the Pod and review S-C.
```

Resume a PAUSED invocation by rerunning its exact launch command after the old tmux session has closed (or run the same shell command in a new tmux session). Never delete partial prediction files to “fix” resume.

Fetch back on the laptop after every session. The fresh timestamped destination prevents overwriting earlier fetches:

```bash
WORKSPACE=/workspace
FETCH="$(mktemp -d outputs/fullrun_fetch_$(date +%Y%m%d_%H%M%S).XXXXXX)"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:$WORKSPACE/bachelorArbeit/outputs/fullrun_2026-10-04_B_smoke" "$FETCH/"
# Once full results exist:
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:$WORKSPACE/bachelorArbeit/outputs/fullrun_2026-10-04_B_results" "$FETCH/"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:$WORKSPACE/logs/fullrun_2026-10-04_B_*" "$FETCH/logs/"
```

For laptop verification, the result directories must be alongside the frozen inputs under the selected root's `outputs/`. Copy the frozen data/baseline and fetched smoke/results into a **fresh review root**, then run `verify_fullrun.py --root` against a checkout of the same commit there; code hashes must match. Alternatively fetch the run directories to this clone's `outputs/` only if those destinations do not already exist. Do not overwrite old runs.

## Local reproduction and checks

The build refuses an existing output root or an incompatible subset100 manifest; an identical committed subset100 manifest is reused without writing. Preserve generated data; do not rerun into this directory. A clean checkout with the original inputs can reproduce preparation:

```bash
myvenv/bin/python scripts/build_fullrun_data.py
myvenv/bin/python scripts/prepare_fullrun_inputs.py
myvenv/bin/python scripts/report_fullrun_preparation.py
myvenv/bin/python -m pytest tests -q
myvenv/bin/python scripts/fullrun_session.py all --smoke --plan
bash -n scripts/fullrun_manual.sh
```

No new venv is created. The existing laptop render environment is separate from the pinned inference venvs. Final statistical consolidation and plots are S-D work after verified results exist; no placeholder final results are produced during S-A.

Validation includes the working-tree suite and a separate export of exactly the staged files. The existing `stage1_paper_None.jsonl` fixture correction is included: its three `target_node` fields match the pre-task generator byte-for-byte. Other pre-existing edits are excluded from this change.
