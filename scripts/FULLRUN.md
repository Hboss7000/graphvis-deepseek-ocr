# Full inference run — S-A, CPU preparation (2026-10-04)

For the interrupted Gemma S3 `qa_kg_text` job, use the [job-scoped parse override and five self-contained recovery blocks](FULLRUN_GEMMA_KG_RESUME.md). All other tripwires and strict-64 scoring remain active.

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

Uniform **light blue** is selected, with no regeneration. The all-50 reading comparison passed: paired basic blue − legacy is approximately −0.015. White remains a diagnostic arm. Every model runs all 500 QA answers per condition at **64 tokens** and stops at strict-64. Rescue is a separate optional later session, never part of a full session. If requested, every capped answer reruns at **512**, including already parsed/correct answers, in `rescue_512/CONDITION`. The headline is extended only when **all four models** have completed verified rescues; otherwise strict-64. Both tables are always reported; an unavailable extended table is explicitly labelled unavailable with null metrics. Stage 1 remains **1024** with no rescue. Ceiling counts/rates are descriptive everywhere and Stage 1 reports them by task. QA parse failures only trip at n≥20 (>50% in the first 20); no final parse-rate gate is applied to five-item smoke results. Coverage, duplicates, identity/settings, missing measurements and frozen hashes remain hard checks. Time, output-growth and empty Stage 1 guards remain operational tripwires.

The launch gate is **per model**. `fullrun_gate.py llava` reads only LLaVA smoke/resume evidence and its 50-graph probe; missing or failed Gemma/Qwen/DeepSeek does not block LLaVA. Qwen and DeepSeek require their own smoke and resume proofs. Gemma begins its full session with a fresh five-question QA smoke (all four conditions, image three→five resume proof; allow approximately three minutes), then automatically continues only if its own gate passes. Both Gemma legibility paths remain recorded and nonblocking under D6. Its full-session Stage 1 preflight still runs both crop paths and labels failed legibility as confounded.

The recovered Gemma `qa_image/resume_verified.json` was reconstructed locally from fetched `resume_first3.json` and the final five predictions: all original three rows match exactly. No predictions/configs/statuses were rewritten. Resume proof creation now precedes final verification/sanity checks. Historical code hashes remain historical; experimental settings/input hashes are validated, and original reused evidence checksums are checked per model.

Long prose answers require an unambiguous explicit `answer is X`, `correct answer: X`, or bolded letter. Explicit answer statements outrank bolded references; contradictory explicit conclusions stay unparsed. A sole letter/letter with punctuation or a single short option-label/choice line remains accepted. Prose letter mentions have no fallback. `Option A is wrong… so the answer is D` parses as D. Historical predictions are re-scored under this parser without editing their saved parse fields; additional old prefix-plus-explanation answers may now be unparsed.

All new full QA outputs record **actual generated token IDs**, including special tokens, for every item (all four model paths, including DeepSeek image/text generation). Rescue checks every selected item's first 64 IDs against strict IDs, not decoded-string prefixes or retokenized text. Missing/short IDs and differences are explicitly reported in `prefix_report.json`; no selected item is omitted. Greedy decoding is a protocol expectation, not a claim that CUDA reruns are guaranteed bitwise identical. DeepSeek ceiling detection remains approximate because its flag uses response retokenization; prefix verification still uses actual IDs. Every mismatch remains visible in the prefix report and prevents that rescue from qualifying for the extended headline; no response is silently discarded.

`fullrun_manual.sh MODEL` runs strict-64 only and resumes completed strict jobs. `fullrun_strict_session.sh S1|S2|S3` groups LLaVA, Qwen + DeepSeek sequentially, and Gemma respectively, with a shared session time allowance. Gemma starts its own fresh five-question QA smoke and resume proof before continuing automatically. LLaVA needs only its own smoke and the completed probe gate.

Optional later rescue: run `bash scripts/fullrun_rescue.sh MODEL` inside tmux on the approved same RTX PRO 6000 GPU type, with `MAX_HOURS` set from the later rescue plan. Default strict launchers never invoke it. It selects every saved strict ceiling, writes a separate labelled directory and resumes completed items. Rescue can also resume after a code-only update: its initial contract stays fixed, `rescue_512/code_attempts/` records each invocation’s actual code hashes, and newly generated rows link to that attempt. Strict contracts/configs and prediction hashes are preserved; strict code hashes and current rescue code hashes are recorded separately. A newer rescue commit is allowed, but model/revision, full input hashes, prompt hashes, seed, greedy decoding, versions and actual GPU type must match. The only allowed generation change is 64 → 512. Full QA records seed 13 and actual GPU type; Qwen/DeepSeek initialize the recorded seed in the shared full-run runtime. Rescue rejects source runs lacking this provenance or exact token IDs. `fullrun_rescue.py MODEL --verify-only` is a CPU prefix audit.

After fetching each strict model, run on the laptop: `myvenv/bin/python scripts/fullrun_rescue_plan.py MODEL --model-dir FETCH/MODEL --output FRESH_JSON`. This reads only saved predictions, reports actual capped counts and measured tokens/s per condition, rescue hours/cost at $2.09/h, and ten capped answer excerpts (200 characters). Strict-64 censors completion lengths: its default estimate is the observed token replay **lower bound**, not a claimed completion-time estimate. Supply `--mean-rescue-tokens N` (64–512) for an explicit mean-length assumption at measured speed. Loading/prefill changes remain unmeasured; DeepSeek ceiling detection is labelled approximate. No worst-case 512 workload is substituted by default.

`fullrun_consolidate.py MODEL --output-dir FRESH_DIR` verifies archived strict settings while permitting newer analysis code, then writes both tables, all condition metrics, exact paired McNemar image-vs-text tests, and reachability/node-tercile strata. The extended table is complete only after that model's rescue is verified; otherwise its metrics are explicitly null/unavailable. The headline decision checks **all four models**, even for a single-model report. Node cutpoints stay 17/18 (229/120/151); reachable graphs 390/110. `all` prints 30 available answers mixed across models/conditions, preferring rescued answers. No full results or rescued answers exist locally yet.

### Strict session estimates

Measured s0b time/item extrapolated to 900 Stage 1 rows plus 500 answers per QA condition; excludes rescue, loading and transfer. S3 adds the requested unmeasured three-minute startup allowance.

| session | models | expected h | cost at $2.09/h | h with 50% margin | cost with margin |
|---|---|---:|---:|---:|---:|
| S1 | LLaVA | 1.499 | $3.13 | 2.248 | $4.70 |
| S2 | Qwen + DeepSeek | 2.235 | $4.67 | 3.353 | $7.01 |
| S3 | Gemma + fresh smoke | 1.958 | $4.09 | 2.937 | $6.14 |
| total | all four | 5.692 | $11.90 | 8.538 | $17.84 |

The grouped local job limits round up to 2.25 / 3.36 / 2.94 hours. These stop inference, **not billing**. Confirm the pod's already approved automatic termination guard covers the chosen session; no command here provisions or modifies infrastructure. Earlier automatic-rescue allowances in the October 5 policy report are superseded by this strict-only plan.

The nonblocking Stage 1 length audit used the pinned LLaVA-NeXT processor/tokenizer (transformers 5.16.1; temporary CPU-only torch 2.8.0+cpu/torchvision .23.0+cpu tooling, no model weights). Combined expanded image + chat prompt + gold + EOS: p25 **1982**, median **2192**, p75 **2381.25**, p90 **2766**, p95 **2824.95**, p99 **2975.04**, maximum **3137**; **613/900 exceed 2048**. Exact per-record/task counts are in `outputs/fullrun_policy_2026-10-05/stage1_lengths.json`. The first record of each of 100 graphs was checked against full processor output and image expansion reused across its nine tasks. These are the current NeXT chat/anyres lengths; verify the original GraphVis training collator/image path before fine-tuning. No model_max_length, image setting, prompt or data was changed.

## Strict sessions S1 / S2 / S3

Run each separate block on the laptop from this repository. Each prompts for the current IP/port and (where needed) S1, S2 or S3; no shell state carries between blocks. S1 is LLaVA (1.499 h), S2 Qwen then DeepSeek (2.235 h), S3 Gemma with fresh smoke (1.958 h). Use the existing approved single RTX PRO 6000 96 GB pod/cache/network volume. Henrique pushes. The watch block stands alone: Ctrl-C before the next block. After each session: verify strict results, fetch everything, then explicitly terminate. No rescue runs in these blocks.

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

### 2. Transfer missing inputs and evidence

`--ignore-existing` fills missing files without replacing evidence. Existing mismatches fail verification. The historical smoke evidence is fetched locally already; transfer also fills in the recovered Gemma resume proof. No rendering or model download occurs.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" outputs/fullrun_2026-10-04_B "root@$IP:/workspace/bachelorArbeit/outputs/"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" outputs/fullrun_fetch_20261004_183521.qtJrSg/fullrun_2026-10-04_B_smoke "root@$IP:/workspace/bachelorArbeit/outputs/"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" outputs/fullrun_s0b_fetch_jcjYPY6T/fullrun_2026-10-04_B_s0b "root@$IP:/workspace/bachelorArbeit/outputs/"
rsync -rltvz --ignore-existing --no-owner --no-group -e "$RSYNC_SSH" outputs/fullrun_s0b_inputs_2026-10-04 "root@$IP:/workspace/bachelorArbeit/outputs/"
```

### 3. Launch the chosen strict session in tmux

S2 shares its 3.36-hour limit across both models; a failed Qwen job stops the group and preserves completed output for resume. S3 runs Gemma's own fresh smoke first. Repeat this block after the old tmux session exits to resume; it never schedules rescue.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
read -rp 'Session (S1 LLaVA / S2 Qwen+DeepSeek / S3 Gemma): ' SESSION
case "$SESSION" in S1|S2|S3) ;; *) echo 'Invalid session'; exit 2;; esac
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
"${SSH[@]}" "root@$IP" "mkdir -p /workspace/logs/fullrun_2026-10-04_B_results; tmux new-session -d -s fullrun-$SESSION \"cd /workspace/bachelorArbeit && bash scripts/fullrun_strict_session.sh $SESSION > /workspace/logs/fullrun_2026-10-04_B_results/${SESSION}_console.log 2>&1; read -r -p \\\"Session ended: STOP the Pod. Press Enter to close.\\\"\""
```

### 4. Single watch line (current log path, counter and GPU)

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
"${SSH[@]}" -tt "root@$IP" 'watch -n 10 "/workspace/venvs/venv_qwen/bin/python /workspace/bachelorArbeit/scripts/fullrun_monitor.py --gpu"'
```

### 5. Verify strict results

Wait for completion. No rescue or extended inference is requested. Missing results fail verification. For laptop consolidation, first create `outputs/fullrun_2026-10-04_B_results/` and copy fetched model directories there with `rsync -rlt --ignore-existing FETCH/MODEL outputs/fullrun_2026-10-04_B_results/` (replace FETCH/MODEL with the actual directory); the strict table remains the headline until all four models have verified rescues.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
read -rp 'Completed session (S1 / S2 / S3): ' SESSION
case "$SESSION" in S1) MODELS='llava';; S2) MODELS='qwen deepseek';; S3) MODELS='gemma';; *) exit 2;; esac
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$PWD/outputs"
for MODEL in $MODELS; do
  "${SSH[@]}" "root@$IP" "cd /workspace/bachelorArbeit && /workspace/venvs/venv_qwen/bin/python scripts/verify_fullrun.py $MODEL"
done
```

### 6. Fetch all session outputs and logs to a fresh directory

This includes predictions, actual token IDs, configs, contracts, statuses, preflight files and Gemma's fresh smoke/resume proof. Keep this directory; the CPU planner takes `--model-dir "$FETCH/MODEL"`. The full input tree is already on the laptop and its hashes are recorded in every contract.

```bash
set -euo pipefail
read -rp 'Current pod SSH IP: ' IP
read -rp 'Current pod SSH port: ' PORT
read -rp 'Completed session (S1 / S2 / S3): ' SESSION
case "$SESSION" in S1) MODELS='llava';; S2) MODELS='qwen deepseek';; S3) MODELS='gemma';; *) exit 2;; esac
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -p "$PORT")
RSYNC_SSH="ssh -i $HOME/.ssh/id_ed25519 -p $PORT"
FETCH="$(mktemp -d "$PWD/outputs/fullrun_${SESSION}_fetch_XXXXXXXX")"
for MODEL in $MODELS; do
  rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_results/$MODEL" "$FETCH/"
done
rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/fullrun_2026-10-04_B_results" "$FETCH/logs/"
if [[ "$SESSION" == S3 ]]; then
  rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/bachelorArbeit/outputs/fullrun_2026-10-04_B_gemma_start_smoke" "$FETCH/"
  rsync -rltvz --no-owner --no-group -e "$RSYNC_SSH" "root@$IP:/workspace/logs/fullrun_2026-10-04_B_gemma_start_smoke" "$FETCH/logs/"
fi
printf 'Fetched everything: %s\n' "$FETCH"
```

### 7. Terminate after successful fetch and review

This is the user's explicit termination action, using an already authenticated `runpodctl`. No API keys are created. Confirm `/workspace` is the retained network volume. [Official CLI syntax](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_pod_delete.md): pod deletion terminates the pod; it does not delete the network volume. If unavailable, use the console Terminate action.

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
