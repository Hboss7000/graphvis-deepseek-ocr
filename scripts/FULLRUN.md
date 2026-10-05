# Full inference run — S-A, CPU preparation (2026-10-04)

For recovery of the fetched S0 failures and the new three-arm 50-graph reading gate, use [Session 0b](FULLRUN_S0B.md). Reading-probe parse/ceiling outcomes are descriptive; their quality gate uses paired basic recall only. Ordinary jobs retain the sanity limits below. Once session 0b has a source marker, full launchers require its recomputed gate.

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

`verify_fullrun.py` checks exact coverage, duplicate/misplaced/unexpected keys, input/config/model/version/decoding identity, common prompt hashes across available models, stored QA parsing against the shared parser, real time/VRAM fields and ceiling rates. Token ceilings are descriptive; parse failure applies only as the first-20 tripwire (never to n<20). These are safety checks, not changed scoring rules. DeepSeek's existing token counts/ceiling detection remain approximate (response retokenization), as recorded by its runner.

The CPU stub tests exercise orchestration only. They **do not** constitute the real S0 rehearsal. Each full model run checks its own real smoke/resume evidence; LLaVA additionally checks the 50-graph reading gate and Gemma retains its nonblocking preflight. The reading gate compares paired basic recall; a drop greater than 0.05 stops full execution. `raw` is also reported. S0 produces measured s/item, peak `torch.cuda.max_memory_allocated`, per-model extrapolated inference time/cost at $2.09/h and an explicit 50% overhead margin.

## Full-run decisions fixed on 2026-10-05

Uniform **light blue** is selected, with no regeneration. The all-50 reading comparison passed: paired basic blue − legacy is approximately −0.015. White remains a diagnostic arm. Every model starts all 500 QA answers per condition at **64 tokens**, then reruns **every** capped answer at **512**, including already parsed/correct answers, in `rescue_512/CONDITION`. Primary QA results replace only the capped strict responses with their reruns; strict-64 is reported alongside. Stage 1 remains **1024** with no rescue. Ceiling counts/rates are descriptive everywhere and Stage 1 reports them by task. QA parse failures only trip at n≥20 (>50% in the first 20); no final parse-rate gate is applied to five-item smoke results. Coverage, duplicates, identity/settings, missing measurements and frozen hashes remain hard checks. Time, output-growth and empty Stage 1 guards remain operational tripwires.

The launch gate is **per model**. `fullrun_gate.py llava` reads only LLaVA smoke/resume evidence and its 50-graph probe; missing or failed Gemma/Qwen/DeepSeek does not block LLaVA. Qwen and DeepSeek require their own smoke and resume proofs. Gemma begins its full session with a fresh five-question QA smoke (all four conditions, image three→five resume proof; allow approximately three minutes), then automatically continues only if its own gate passes. Both Gemma legibility paths remain recorded and nonblocking under D6. Its full-session Stage 1 preflight still runs both crop paths and labels failed legibility as confounded.

The recovered Gemma `qa_image/resume_verified.json` was reconstructed locally from fetched `resume_first3.json` and the final five predictions: all original three rows match exactly. No predictions/configs/statuses were rewritten. Resume proof creation now precedes final verification/sanity checks. Historical code hashes remain historical; experimental settings/input hashes are validated, and original reused evidence checksums are checked per model.

Long prose answers require an unambiguous explicit `answer is X`, `correct answer: X`, or bolded letter. Explicit answer statements outrank bolded references; contradictory explicit conclusions stay unparsed. A sole letter/letter with punctuation or a single short option-label/choice line remains accepted. Prose letter mentions have no fallback. `Option A is wrong… so the answer is D` parses as D. Historical predictions are re-scored under this parser without editing their saved parse fields; additional old prefix-plus-explanation answers may now be unparsed.

All new full QA outputs record **actual generated token IDs**, including special tokens, for every item (all four model paths, including DeepSeek image/text generation). Rescue checks every selected item's first 64 IDs against strict IDs, not decoded-string prefixes or retokenized text. Missing/short IDs and differences are explicitly reported in `prefix_report.json`; no selected item is omitted. Greedy decoding is a protocol expectation, not a claim that CUDA reruns are guaranteed bitwise identical. DeepSeek ceiling detection remains approximate because its flag uses response retokenization; prefix verification still uses actual IDs. A mismatch remains visible in the primary report and requires review; no response is silently discarded.

`fullrun_manual.sh MODEL` automatically appends the rescue using the same whole-session time limit. Repeating the same session command resumes completed strict and rescue jobs. `fullrun_rescue.py MODEL --verify-only` is a CPU audit. `fullrun_consolidate.py MODEL --output-dir FRESH_DIR` validates strict/rescue coverage and writes strict-64 and extended-512 tables: accuracy (unparsed wrong), accuracy among parsed, parse-failure/ceiling rates and exact paired McNemar image-vs-text tests, for all three textual conditions. Graph strata use undirected correct-option reachability and all-500 nearest-rank node-count terciles with cutpoints 17/18 (229 low, 120 middle, 151 high) and ties kept together; condition metrics and paired differences are reported per stratum. Reachability matches preparation diagnostics. `fullrun_consolidate.py all` performs S-D and prints 30 extended answers round-robin across available models/conditions for spot checks; each model rescue also prints up to 30 across results available so far. Full-run predictions do not exist yet, so no extended accuracies/answers are fabricated.

### Measured CPU report and budget

[Per-model gate/estimate report](FULLRUN_POLICY_REPORT.md) records all fetched checks, source hashes, counts and estimates. All four fetched gates pass under the new rules; Gemma still runs its requested fresh startup smoke. Strict estimates are LLaVA 1.499 h, Qwen 1.647 h, DeepSeek .588 h and Gemma 1.908 h (Stage 1 4.579 s/item). Rescue allowances extrapolate the five-item ceiling rates to 500 per condition and use measured tokens/second; **each rerun is charged up to all 512 tokens**, not just the extra 448. These capped allowances are not predictions that every rerun reaches 512. Rates use generation elapsed time where available, otherwise measured item elapsed time including preprocessing. Loading/prefill overhead and full-dataset ceiling frequencies remain uncertain.

| model | strict h | rescue allowance h | total h | $2.09/h | total +50% h | cost +50% |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA | 1.499 | .682 | 2.180 | $4.56 | 3.270 | $6.83 |
| Qwen | 1.647 | 1.683 | 3.331 | $6.96 | 4.996 | $10.44 |
| DeepSeek | .588 | .000 | .588 | $1.23 | .882 | $1.84 |
| Gemma | 1.908 | 5.741 | 7.649 | $15.99 | 11.473 | $23.98 |
| total | 5.642 | 8.105 | 13.747 | $28.73 | 20.621 | $43.10 |

Gemma's unmeasured requested ~3 minute startup allowance adds .05 h / $0.1045 before margin, .075 h / $0.15675 with margin: all-in allowance **13.797 h / $28.84**, or **20.696 h / $43.25** with 50%. DeepSeek had zero observed ceilings in its tiny smoke; this is not a guarantee of zero full-run rescues. Gemma's capped estimate exceeds the old four-hour guard; its actual later session budget/termination guard must be reviewed rather than silently reusing four hours.

The nonblocking Stage 1 length audit used the pinned LLaVA-NeXT processor/tokenizer (transformers 5.16.1; temporary CPU-only torch 2.8.0+cpu/torchvision .23.0+cpu tooling, no model weights). Combined expanded image + chat prompt + gold + EOS: p25 **1982**, median **2192**, p75 **2381.25**, p90 **2766**, p95 **2824.95**, p99 **2975.04**, maximum **3137**; **613/900 exceed 2048**. Exact per-record/task counts are in `outputs/fullrun_policy_2026-10-05/stage1_lengths.json`. The first record of each of 100 graphs was checked against full processor output and image expansion reused across its nine tasks. These are the current NeXT chat/anyres lengths; verify the original GraphVis training collator/image path before fine-tuning. No model_max_length, image setting, prompt or data was changed.

## First full session: LLaVA

Run the following **separate blocks on the laptop from this repository**. Each defines its own connection/fetch variables; no block relies on the previous block's shell state. `watch` is intentionally its own block; press Ctrl-C to return before proceeding. SSH uses the current existing pod, never an old hard-coded address. Henrique pushes; these instructions do not push or provision automatically. The existing single RTX PRO 6000 96 GB pod/cache and network volume must already be approved and available. Before launch, verify its approved automatic termination guard covers the LLaVA expected 2.18 h plus 50% (about **3 h 16 min**); this code does not create/start/modify pods or guards. The launcher has a 3.3 h whole-session stop limit, which stops inference but not billing.

### 1. Push check and code update

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
EXPECTED="$(git rev-parse HEAD)"
REMOTE="$(git ls-remote origin refs/heads/inference50-core-keep | cut -f1)"
if [[ "$EXPECTED" != "$REMOTE" ]]; then echo 'Push your reviewed commit first; remote HEAD differs.'; exit 1; fi
"${SSH[@]}" "root@$IP" "cd /workspace/bachelorArbeit && git pull --ff-only origin inference50-core-keep && test \"\$(git rev-parse HEAD)\" = '$EXPECTED'"
```

### 2. Transfer only missing prepared files, then verify LLaVA gate

The full blue tree already had to be present for smoke input hashes. `--ignore-existing` fills missing files; it never replaces evidence. No white regeneration, model download or transfer of irrelevant models is needed. A mismatched existing file fails the gate rather than being overwritten.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" \
  outputs/fullrun_2026-10-04_B "root@$IP:/workspace/bachelorArbeit/outputs/"
"${SSH[@]}" "root@$IP" 'cd /workspace/bachelorArbeit && /workspace/venvs/venv_qwen/bin/python scripts/fullrun_gate.py llava'
```

### 3. Launch strict run and automatic rescue in tmux

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
"${SSH[@]}" "root@$IP" 'mkdir -p /workspace/logs/fullrun_2026-10-04_B_results; tmux new-session -d -s fullrun-llava "cd /workspace/bachelorArbeit && MAX_HOURS=3.3 bash scripts/fullrun_manual.sh llava > /workspace/logs/fullrun_2026-10-04_B_results/llava_console.log 2>&1; read -r -p \"Session ended: STOP the Pod. Press Enter to close.\""'
```

### 4. Single watch line (current job's log path, counter and GPU)

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
"${SSH[@]}" -tt "root@$IP" 'watch -n 10 "/workspace/venvs/venv_qwen/bin/python /workspace/bachelorArbeit/scripts/fullrun_monitor.py --gpu"'
```

### 5. Verify strict, rescue and consolidated primary tables

Wait until the session completes. The rescue already ran automatically in block 3; `--verify-only` performs no inference and checks every prefix. If interrupted, rerun block 3 after the old tmux session exits (use a new session name if it still exists). This resumes strict/rescue rather than launching another full run.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
"${SSH[@]}" "root@$IP" 'cd /workspace/bachelorArbeit && /workspace/venvs/venv_qwen/bin/python scripts/verify_fullrun.py llava && /workspace/venvs/venv_qwen/bin/python scripts/fullrun_rescue.py llava --verify-only && /workspace/venvs/venv_qwen/bin/python scripts/fullrun_consolidate.py llava --output-dir outputs/fullrun_2026-10-04_B_consolidated'
```

### 6. Fetch to a fresh directory

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$(mktemp -d "$PWD/outputs/fullrun_llava_fetch_XXXXXXXX")"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_results/llava" "$FETCH/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_consolidated/llava" "$FETCH/consolidated/"
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" \
  "root@$IP:/workspace/logs/fullrun_2026-10-04_B_results" "$FETCH/logs/"
printf 'Fetched results: %s\n' "$FETCH"
```

### 7. Terminate the pod after successful fetch and review

This last block is the user's explicit termination action. It requires an already authenticated current `runpodctl`; no API key is created. Confirm the pod ID and that `/workspace` is the retained network volume. `pod delete` terminates the pod; it does not delete a network volume. Syntax was checked against [Runpod's generated CLI documentation](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_pod_delete.md). If the CLI is unavailable, use the Runpod console's Terminate action; do not treat SSH shutdown as stopping billing.

```bash
set -euo pipefail
read -rp 'Pod SSH IP (for your record): ' IP
read -rp 'Pod SSH port (for your record): ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
read -rp 'Successful local fetch directory: ' FETCH
test -d "$FETCH"
read -rp 'Exact pod ID to terminate: ' POD_ID
read -rp "Type TERMINATE $POD_ID to confirm termination: " CONFIRM
if [[ "$CONFIRM" != "TERMINATE $POD_ID" ]]; then echo 'Termination not confirmed.'; exit 1; fi
runpodctl pod delete --help >/dev/null
runpodctl pod delete "$POD_ID"
```
