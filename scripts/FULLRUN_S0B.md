# Session 0b: failed smoke jobs and three 50-graph reading arms

No pod or GPU work was performed locally. Henrique pushes this commit and runs the session on the existing approved pod. No full inference run is launched by these commands.

The fetched S0 files are under `outputs/fullrun_fetch_20261004_183521.qtJrSg/fullrun_2026-10-04_B_smoke/`, not the canonical local smoke path. Both reading predictions are complete (20 graphs each); the uniform job's FAILED status came from the old ceiling check, not missing predictions. [The rescore](FULLRUN_S0_PROBE_REPORT.md) gives every graph's basic/raw recall; [the CSV](fullrun_s0_probe_per_graph.csv) also gives per-graph precision, F1, ceilings and paired differences. SHA-256 sources are recorded locally in `outputs/fullrun_s0_probe_rescore_2026-10-04/report.json`.

| arm | tier | recall | precision | F1 | ceilings |
|---|---|---:|---:|---:|---:|
| uniform blue | basic | .533126 | .867835 | .634525 | 14/20 |
| uniform blue | raw | .495658 | .799978 | .591816 | 14/20 |
| legacy B | basic | .606940 | .851260 | .692257 | 10/20 |
| legacy B | raw | .540797 | .751981 | .614198 | 10/20 |

These are macro means of each graph's unchanged `span_extended` scorer metrics, not pooled node counts. Paired basic uniform − legacy is **−.073813871**, **5 better / 11 worse / 4 equal**, so the old 20-graph reading gate fails. Graph 43 has basic .705882 versus .764706 (−.058824); it is counted as worse. The approximation supplied by Henrique differs from this rescore. Raw difference is −.045139. No inference about causes is drawn.

Session 0b preserves the original S0 directories. It verifies and reuses LLaVA's standard Stage 1/four QA jobs, Qwen/Gemma Stage 1, both Gemma preflight crop arms, and LLaVA's existing three→five resume proof. It records every reused file hash and retains historical code hashes. Experimental fields must match exactly; only code hashes may differ because of these fixes. New outputs are isolated in `outputs/fullrun_2026-10-04_B_s0b/`; rerunning the command resumes these outputs and skips COMPLETED jobs. The original failed QA/DeepSeek jobs have no predictions to salvage: Qwen/Gemma failed while enriching their configs; DeepSeek failed in import preflight before weights.

New jobs: Qwen four QA conditions; Gemma four QA conditions using the original preflight report; DeepSeek Stage 1 and four QA conditions; LLaVA node_description on **all original 50** graphs for legacy B, uniform blue and uniform white. The three arms keep pinned model, prompts, seed, 1024-token cap, outline, dimensions, pruning, orientation and scoring. White changes only node fill to `#FFFFFF`. Its 450 Stage 1 records, 50 QA records and full metadata are byte-identical to the original subset; every PNG dimension matches blue and legacy. Rendering and input preparation happened on the laptop. Added data is approximately 27 MiB.

The new 50-graph probe supersedes the old 20-graph reading gate. Blue passes if paired macro **basic** recall (blue − legacy) is ≥ −.05. White never substitutes automatically. If blue fails, the report says to wait for Henrique's decision and full execution remains blocked. Reports contain basic/raw recall, precision/F1, numeric ceiling counts, every graph and all arm pairs. Reading ceilings and empty/parse outcomes are descriptive; integrity, time/stall guards, ordinary-job sanity, four-model coverage, all four resume proofs, and the two Gemma preflight paths remain required. The original failed Gemma readability threshold remains recorded and nonblocking under D6. Full-run launchers recompute the recovery gate if its source marker exists.

DeepSeek preflight now audits imports from the `auto_map` entry points and reachable local Python imports in its **actual pinned offline snapshot** (source hashes are printed to logs). Module-scope external imports are checked; conditional/deferred imports are listed separately so eager attention does not force Flash Attention. There is no requirements-derived `easydict` assertion, automatic install, dependency upgrade or model download. The snapshot is not available locally, so the actual dependency list and GPU execution are unverified here. Henrique is checking `easydict` separately.

Old phase-1 configs with flat QA settings and missing effective caps are normalized for schema tests. Missing historical package versions are explicitly stubbed in tests; those tests do not establish the unknown historical version. New Qwen/Gemma QA writers explicitly record greedy generation. Tests do not load weights or act as the required real GPU rehearsal.

## Exact commands

On the laptop after Henrique pushes, with the current existing pod's SSH IP/port:

```bash
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT" "root@$IP" \
  'cd /workspace/bachelorArbeit && git pull --ff-only origin inference50-core-keep && git rev-parse HEAD'
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" \
  outputs/fullrun_s0b_inputs_2026-10-04 \
  outputs/fullrun_2026-10-04_B_uniform_white50 \
  "root@$IP:/workspace/bachelorArbeit/outputs/"
```

On the pod, preview (CPU only), then launch in tmux and monitor:

```bash
cd /workspace/bachelorArbeit
/workspace/venvs/venv_qwen/bin/python scripts/fullrun_s0b.py --plan
mkdir -p /workspace/logs/fullrun_2026-10-04_B_s0b
tmux new-session -d -s fullrun-s0b 'cd /workspace/bachelorArbeit && MAX_HOURS=1.5 bash scripts/fullrun_s0b_manual.sh > /workspace/logs/fullrun_2026-10-04_B_s0b/console.log 2>&1; read -r -p "Session ended: STOP the Pod. Press Enter to close."'
tail -F /workspace/logs/fullrun_2026-10-04_B_s0b/console.log
watch -n 30 nvidia-smi
```

The default source on the pod is `/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_smoke`; do not move or alter it. If its path differs, pass the actual in-repository path via `--source-smoke-root PATH` in both preview and launch. Do not use the laptop's fetched path unless that is also the pod's real path.

After interruption, use **the same tmux launch command** after the existing session exits (or a different tmux session name). It resumes compatible partial jobs and verifies completed jobs; do not pull additional code mid-session because frozen contracts intentionally reject changed code. Gate-only rerun, with no GPU load:

```bash
cd /workspace/bachelorArbeit
/workspace/venvs/venv_qwen/bin/python scripts/fullrun_s0b.py --gate-only \
  > /workspace/logs/fullrun_2026-10-04_B_s0b/gate.log 2>&1
cat /workspace/logs/fullrun_2026-10-04_B_s0b/gate.log
```

Budget: observed old reading means are about 10.14 s/graph blue and 8.14 legacy. Extrapolating 50 each gives 15.2 minutes; white has no measurements, and assuming blue-like timing adds 8.4 minutes. Failed-job reruns, model loading and overhead are additional and unmeasured. The **90-minute whole-session cap** is a guard, not a runtime promise; at the supplied $2.09/h its capped compute allowance is $3.14. Keep the pod's existing automatic termination guard valid for the approved runtime plus 50% (135 minutes for a 90-minute allowance). This script stops inference on its deadline; **it does not stop pod billing or provision/change a guard**. Stop the pod promptly when the session finishes or fails.

Fetch to a fresh laptop directory, never onto old evidence:

```bash
FETCH_0B="$(mktemp -d "$PWD/outputs/fullrun_s0b_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_s0b" "$FETCH_0B/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:/workspace/logs/fullrun_2026-10-04_B_s0b" "$FETCH_0B/logs/"
```

Optional reproduction of the white render (only into a **fresh** directory; the prepared output already exists):

```bash
myvenv/bin/python scripts/generate_graphvis_datasets.py \
  --split test --seed 13 --stage1-task-set extended --stage1-balance per-task \
  --core-policy keep --max-nodes 18 --max-edges 30 --max-degree 0 --bridge-rule qa-bridge \
  --rankdir TB --node-fontsize 30 --edge-fontsize 24 --ranksep .4 --auto-orient llava \
  --wrap-labels 0 --edge-label-style plain --node-style uniform --node-fill '#FFFFFF' \
  --indices-file subsets/obqa_test_subset50_seed13.json \
  --out-dir outputs/fullrun_white50_reproduction
myvenv/bin/python scripts/prepare_fullrun_s0b.py
```

The preparation command verifies/freeze-records the canonical prepared white50 path; it does not replace it with a reproduction directory.
