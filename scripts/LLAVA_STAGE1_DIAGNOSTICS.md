# LLaVA Stage 1 diagnostic arms, before training

Run the blocks independently on the **laptop from this repository**, after Henrique pushes the five commits. Use the already approved single RTX PRO 6000 96 GB GPU Pod, pinned cache and network volume. These blocks do not create, start, resize, modify, stop or terminate infrastructure. Keep its approved automatic termination guard active. Every GPU process runs inside tmux, writes to `/workspace/logs/`, and stops at its expected-runtime + 50% process limit; this does not stop billing. STOP the Pod after completion/failure and fetching.

| Label | Prompt template | Assistant prefix |
|---|---|---|
| P1 | hf-chat | gold-template |
| P2 | llava_v1 | none |
| P3 | llava_v1 | gold-template |

The default runner remains hf-chat/none. Diagnostics use greedy decoding, seed 13, 1024 **new** tokens, the pinned LLaVA-NeXT Mistral model/revision and its main-run anyres images/processor. User-turn text is unchanged; the template and prefix change only the surrounding conversation and assistant side. Stored primed responses are exactly prefix + decoded continuation, including continuation whitespace; generated-token counts/ceilings count continuation only. The scorer is unchanged. Node-degree uses Henrique's requested `with the name` wording; the other prefixes follow the generator. Question node names are the only variable prefix content; no gold or metadata is passed to the prefix builder. The 900-answer leak test covers every task.

Pilot: the lowest 20 statement indices in the frozen main 100-graph subset, all four numeric tasks (80 examples/arm, 240 total). Pilot dry run: the first two of those graphs, all four tasks (8/arm). Full: the original `stage1_subset100.jsonl`, 100 graphs × 9 tasks (900) for one explicitly chosen arm. Full dry run: the first graph × all 9 tasks, including image loading and the vision encoder, same processor and full scoring path. Each arm and phase has a separate directory under `outputs/llava_stage1_diagnostics_2026-10-05/`. Contracts/configs reject resume drift and unknown output collisions. Flushed rows resume individually.

The manifest records every user-turn hash and verifies each against the main input. Pilot aggregate hashes cover only pilot rows; they equal the corresponding main subset, not the entire 900-row aggregate. Full aggregate hashes equal the main run. Frozen input, metadata and image hashes are checked before loading weights. Versions are exactly torch 2.8.0+cu128 / transformers 5.16.1; pinned weights are offline. No installs or downloads are performed.

Measured evidence: the completed main run's 100 rows/task in `outputs/fullrun_2026-10-04_B_results/llava/stage1`. Numeric s/item: node_number 5.4100, edge_number 3.7328, node_degree 1.1355, highest_node_degree 2.2708. Across all nine tasks: 47.2246 s/graph. Loading: 29.7353 s per arm. Peak allocated VRAM: 14.829 GiB.

| Job | Initial estimate including loading | Process guard (+50%) |
|---|---:|---:|
| Pilot dry, all 3 arms | 2.74 min | 4.12 min |
| Pilot, all 3 arms | 14.04 min | 21.07 min |
| Full dry, one arm | 1.28 min | 1.93 min |
| Full, one arm | 79.20 min | 118.82 min |

These initial estimates use measured **main-run** speeds; diagnostic response lengths can differ. The actual pilot/full launch replaces its estimate/guard with same-arm dry-run measurements. The dry verification blocks print peak `torch.cuda.max_memory_allocated`, seconds/item and per task, extrapolated runtime, and cost at the actual Pod hourly rate you enter. Review these before explicitly executing the next launch block. GPU experiments have not been run by this implementation. A timeout preserves rows; rerun the same launch block after the tmux session exits. Use a different approved `--max-hours` only after reviewing measured timing; no flag changes experimental settings.

Only elapsed-time, output-stall (five minutes), coverage, config, identity and measurement checks guard these diagnostic jobs. A one-token continuation can be a valid primed numeric answer, so the main fullrun's empty/one-token quality tripwire does not apply. Empty/unparsed responses remain wrong in exact accuracy and appear in answer rate; no result is filtered out. Mean absolute error uses parsed answers only. Highest-node-degree numeric exact accuracy uses `parsed_degree`; the unchanged scorer separately retains name/joint accuracy. Lenient containment remains a diagnostic measure, not accuracy.

## 1. Prepare on the laptop, pull pushed code, upload frozen inputs

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
NAME=llava_stage1_diagnostics_2026-10-05
EXPECTED="$(git rev-parse HEAD)"
REMOTE="$(git ls-remote origin refs/heads/inference50-core-keep | cut -f1)"
test "$EXPECTED" = "$REMOTE" || { echo 'Push the reviewed commits first.'; exit 1; }
mkdir -p "outputs/$NAME"
python3 scripts/llava_stage1_jobs.py prepare > "outputs/$NAME/preparation_console.json"
python3 scripts/llava_stage1_jobs.py estimate --phase pilot
python3 scripts/llava_stage1_jobs.py estimate --phase full --arm P1
"${SSH[@]}" "root@$IP" "cd /workspace/bachelorArbeit && git pull --ff-only origin inference50-core-keep && test \"\$(git rev-parse HEAD)\" = '$EXPECTED'"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" outputs/fullrun_2026-10-04_B "root@$IP:/workspace/bachelorArbeit/outputs/"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" "outputs/$NAME" "root@$IP:/workspace/bachelorArbeit/outputs/"
```

## 2. Pilot dry run: all three arms; monitor included

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
NAME=llava_stage1_diagnostics_2026-10-05
"${SSH[@]}" "root@$IP" "mkdir -p /workspace/logs/$NAME && tmux new-session -d -s llava-pilot-dry \"cd /workspace/bachelorArbeit && bash scripts/llava_stage1_jobs.sh pilot-dry all >> /workspace/logs/$NAME/pilot-dry_console.log 2>&1\""
"${SSH[@]}" -tt "root@$IP" "watch -n 10 \"tail -n 12 /workspace/logs/$NAME/pilot-dry_console.log; tail -n 4 /workspace/logs/$NAME/pilot-dry_*.log; nvidia-smi\""
```

## 3. Fetch and verify pilot dry runs on the laptop; review VRAM/runtime/cost

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
NAME=llava_stage1_diagnostics_2026-10-05
read -rp 'Actual current Pod price in USD/hour: ' RATE
mkdir -p "$PWD/outputs"
FETCH="$(mktemp -d "$PWD/outputs/llava_pilot-dry_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/$NAME" "$FETCH/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/$NAME" "$FETCH/logs/"
rsync -rlt "$FETCH/$NAME/" "$PWD/outputs/$NAME/"
python3 scripts/llava_stage1_jobs.py verify --phase pilot-dry --arm all --hourly-rate "$RATE"
python3 scripts/llava_stage1_summary.py --phase pilot-dry
printf 'Evidence fetched to %s\nSTOP the Pod after your session; billing continues.\n' "$FETCH"
```

## 4. Explicitly launch the three pilot arms after reviewing dry runs

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
NAME=llava_stage1_diagnostics_2026-10-05
"${SSH[@]}" "root@$IP" "mkdir -p /workspace/logs/$NAME && tmux new-session -d -s llava-pilot \"cd /workspace/bachelorArbeit && bash scripts/llava_stage1_jobs.sh pilot all >> /workspace/logs/$NAME/pilot_console.log 2>&1\""
"${SSH[@]}" -tt "root@$IP" "watch -n 10 \"tail -n 12 /workspace/logs/$NAME/pilot_console.log; tail -n 4 /workspace/logs/$NAME/pilot_*.log; nvidia-smi\""
```

## 5. Fetch and print the pilot comparison on the laptop

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
NAME=llava_stage1_diagnostics_2026-10-05
mkdir -p "$PWD/outputs"
FETCH="$(mktemp -d "$PWD/outputs/llava_pilot_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/$NAME" "$FETCH/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/$NAME" "$FETCH/logs/"
rsync -rlt "$FETCH/$NAME/" "$PWD/outputs/$NAME/"
python3 scripts/llava_stage1_jobs.py verify --phase pilot --arm all
python3 scripts/llava_stage1_summary.py --phase pilot
printf 'Evidence fetched to %s\nSTOP the Pod after your session; billing continues.\n' "$FETCH"
```

## 6. Full dry run for the chosen arm: all nine tasks

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
read -rp 'Chosen arm (P1 / P2 / P3): ' ARM
case "$ARM" in P1|P2|P3) ;; *) echo 'Invalid arm'; exit 2;; esac
NAME=llava_stage1_diagnostics_2026-10-05
"${SSH[@]}" "root@$IP" "mkdir -p /workspace/logs/$NAME && tmux new-session -d -s llava-full-dry-$ARM \"cd /workspace/bachelorArbeit && bash scripts/llava_stage1_jobs.sh full-dry $ARM >> /workspace/logs/$NAME/full-dry_console.log 2>&1\""
"${SSH[@]}" -tt "root@$IP" "watch -n 10 \"tail -n 12 /workspace/logs/$NAME/full-dry_console.log; tail -n 4 /workspace/logs/$NAME/full-dry_*.log; nvidia-smi\""
```

## 7. Fetch and verify full dry run on the laptop; review VRAM/runtime/cost

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
NAME=llava_stage1_diagnostics_2026-10-05
read -rp 'Chosen arm (P1 / P2 / P3): ' ARM
case "$ARM" in P1|P2|P3) ;; *) echo 'Invalid arm'; exit 2;; esac
read -rp 'Actual current Pod price in USD/hour: ' RATE
mkdir -p "$PWD/outputs"
FETCH="$(mktemp -d "$PWD/outputs/llava_full-dry_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/$NAME" "$FETCH/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/$NAME" "$FETCH/logs/"
rsync -rlt "$FETCH/$NAME/" "$PWD/outputs/$NAME/"
python3 scripts/llava_stage1_jobs.py verify --phase full-dry --arm $ARM --hourly-rate "$RATE"
printf 'Evidence fetched to %s\nSTOP the Pod after your session; billing continues.\n' "$FETCH"
```

## 8. Explicitly launch the chosen full arm after reviewing its dry run

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
read -rp 'Chosen arm (P1 / P2 / P3): ' ARM
case "$ARM" in P1|P2|P3) ;; *) echo 'Invalid arm'; exit 2;; esac
NAME=llava_stage1_diagnostics_2026-10-05
"${SSH[@]}" "root@$IP" "mkdir -p /workspace/logs/$NAME && tmux new-session -d -s llava-full-$ARM \"cd /workspace/bachelorArbeit && bash scripts/llava_stage1_jobs.sh full $ARM >> /workspace/logs/$NAME/full_console.log 2>&1\""
"${SSH[@]}" -tt "root@$IP" "watch -n 10 \"tail -n 12 /workspace/logs/$NAME/full_console.log; tail -n 4 /workspace/logs/$NAME/full_*.log; nvidia-smi\""
```

## 9. Fetch and verify the full result; print unchanged scorer metrics

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
NAME=llava_stage1_diagnostics_2026-10-05
read -rp 'Chosen arm (P1 / P2 / P3): ' ARM
case "$ARM" in P1|P2|P3) ;; *) echo 'Invalid arm'; exit 2;; esac
mkdir -p "$PWD/outputs"
FETCH="$(mktemp -d "$PWD/outputs/llava_full_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/$NAME" "$FETCH/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/$NAME" "$FETCH/logs/"
rsync -rlt "$FETCH/$NAME/" "$PWD/outputs/$NAME/"
python3 scripts/llava_stage1_jobs.py verify --phase full --arm $ARM
cat "outputs/$NAME/full/$ARM/metrics_llava.json"
printf 'Evidence fetched to %s\nSTOP the Pod after your session; billing continues.\n' "$FETCH"
```

Monitoring is included with every launch. Ctrl-C ends the watch and leaves inference in tmux. Each fetch creates a fresh evidence directory and mirrors only diagnostic outputs into the laptop's diagnostic tree; main-run results stay in their own directories. On failure, fetch anyway and STOP the Pod. For an independent watch, rerun only the second SSH line of the relevant launch block after defining its variables, or rerun the whole block once its old tmux session has ended to resume.
