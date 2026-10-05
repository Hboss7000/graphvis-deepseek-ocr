# Gemma S3: resume only strict-64 qa_kg_text

Commit and push the reviewed changes to `inference50-core-keep` before block 1 (Henrique pushes). Run each block independently on the **laptop from this repository**. The existing Pod must already be running. These commands do not create, start, stop, or modify infrastructure. Keep its existing automatic termination guard active.

The explicit job-scoped flag disables only the first-20 QA parse-failure tripwire. The original contract and first 20 prediction rows remain unchanged; the original config is backed up byte-for-byte. `tripwire_overridden=true` appears in the current config, status and verification. `tripwire_override.json` records the exception, original config/contract/prediction hashes and row count; `code_attempts/` records current code hashes and new prediction rows link to their attempt. Only code hashes may differ; inputs, data provenance, prompts and experimental settings must match. All other guards stay active. Existing Gemma startup smoke evidence is verified without launching more smoke/preflight/other jobs.

The fetched 20 rows measured 1.3081 s/item and 22.91 GiB peak allocated VRAM. The remaining 480 imply about 10.5 minutes of inference; allowing roughly 30 seconds loading and 50% margin gives the 16.5-minute process limit below (`--max-hours 0.275`). This estimate is based on existing GPU evidence, not a new GPU dry run. At the previously documented $2.09/h rate, the estimate is about $0.38 before margin / $0.57 with margin; the current Pod price is not verified. The process limit does **not** stop billing. STOP the Pod after completion/failure and fetching.

1. Pull the pushed commit and check that the new CLI exists.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
EXPECTED="$(git rev-parse HEAD)"
REMOTE="$(git ls-remote origin refs/heads/inference50-core-keep | cut -f1)"
test "$EXPECTED" = "$REMOTE" || { echo 'Commit and push the reviewed changes first.'; exit 1; }
"${SSH[@]}" "root@$IP" "cd /workspace/bachelorArbeit && git pull --ff-only origin inference50-core-keep && test \"\$(git rev-parse HEAD)\" = '$EXPECTED' && /workspace/venvs/venv_qwen/bin/python scripts/fullrun_session.py --help | grep -- --override-qa-parse-tripwire"
```

2. Resume only the interrupted job in tmux. Monitoring is included in this block and also stands alone in block 3. Ctrl-C stops the watch, leaving the job in tmux.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
"${SSH[@]}" "root@$IP" 'mkdir -p /workspace/logs/fullrun_2026-10-04_B_results && tmux new-session -d -s gemma-kg-resume "cd /workspace/bachelorArbeit && MAX_HOURS=0.275 bash scripts/fullrun_manual.sh gemma --job qa_kg_text --override-qa-parse-tripwire >> /workspace/logs/fullrun_2026-10-04_B_results/gemma_kg_resume_console.log 2>&1"'
"${SSH[@]}" -tt "root@$IP" 'watch -n 10 "/workspace/venvs/venv_qwen/bin/python /workspace/bachelorArbeit/scripts/fullrun_monitor.py --gpu; tail -n 8 /workspace/logs/fullrun_2026-10-04_B_results/gemma_kg_resume_console.log"'
```

3. Monitor independently (counter, GPU, latest console status). Detailed evaluator log: `/workspace/logs/fullrun_2026-10-04_B_results/gemma/qa_kg_text.log`.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
"${SSH[@]}" -tt "root@$IP" 'watch -n 10 "/workspace/venvs/venv_qwen/bin/python /workspace/bachelorArbeit/scripts/fullrun_monitor.py --gpu; tail -n 8 /workspace/logs/fullrun_2026-10-04_B_results/gemma_kg_resume_console.log"'
```

4. Verify the completed job, then all Gemma strict results. The first command isolates this recovery; the second also checks the other existing jobs. Both show the override and strict/parsed-only QA accuracy. An incomplete job fails verification.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
"${SSH[@]}" "root@$IP" 'cd /workspace/bachelorArbeit && /workspace/venvs/venv_qwen/bin/python scripts/verify_fullrun.py gemma --job qa_kg_text && /workspace/venvs/venv_qwen/bin/python scripts/verify_fullrun.py gemma'
```

5. Fetch all Gemma results and logs into a fresh directory, including original config, override audit and code attempts. Fetch even after a failure to preserve evidence.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
mkdir -p "$PWD/outputs"
FETCH="$(mktemp -d "$PWD/outputs/fullrun_S3_kg_resume_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_results/gemma" "$FETCH/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/fullrun_2026-10-04_B_results" "$FETCH/logs/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_gemma_start_smoke" "$FETCH/"
printf 'Fetched to %s\nSTOP the Pod; billing continues.\n' "$FETCH"
```

Strict accuracy continues to use every answer in the denominator, counting unparsed answers as wrong. Parsed-only accuracy excludes unparsed answers and is secondary. Consolidation marks the overridden strict condition explicitly. The optional 512-token protocol is unchanged; this recovery launches strict-64 only.
