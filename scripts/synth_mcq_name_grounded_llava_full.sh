#!/usr/bin/env bash
# Prepared full run: requires a passing smoke and separate user approval before launch.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/synth_mcq_name_grounded_llava_env.sh"
[[ -n "${TMUX:-}" ]] || { echo 'Use the documented tmux launch command.' >&2; exit 2; }
: "${MAX_SECONDS:?Set the approved full-run ceiling from the smoke estimate plus 50%}"
SMOKE_OUT=${SMOKE_OUT:-$WORKSPACE/outputs/synth_mcq_name_grounded_llava_smoke}
SMOKE_REPORT=${SMOKE_REPORT:-$SMOKE_OUT/smoke_report_regated.json}
[[ -f "$SMOKE_REPORT" ]] || SMOKE_REPORT="$SMOKE_OUT/smoke_report.json"
OUT=${OUT:-$WORKSPACE/outputs/synth_mcq_name_grounded_llava_full}
mkdir -p "$OUT"
ln -sfn "$OUT/run.log" "$WORKSPACE/logs/synth_mcq_name_grounded_llava_full.log"
exec > >(tee -a "$OUT/run.log") 2>&1
exec 9>>"$WORKSPACE/.fullrun_gpu.lock"
flock -n 9 || { echo 'Another GPU session holds the workspace lock.'; exit 2; }
trap 'rc=$?; echo "Synthetic full ended ($rc): fetch outputs and STOP the pod if finished or failed."' EXIT
"$PY" -u scripts/run_synth_mcq.py "${LLAVA_ARGS[@]}" --preflight
"$PY" - "$SMOKE_REPORT" "$DATASET_PATH" "$ASSETS" "$PAIRING" "$AUDIT_GATE" "$ADAPTER" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from synth_mcq_llava_backend import parse_args, run_identity, strict_artifact
from synth_mcq_llava_common import verify_assets, smoke_report, compatible_smoke_identity
from synth_mcq_common import read_predictions, sha256_file
from analyze_synth_mcq import validate_rows
report = json.load(open(sys.argv[1]))
assert report['passed'] and report['complete_smoke'], 'STOP: smoke failed/incomplete'
rows = read_predictions(Path(sys.argv[1]).parent / 'predictions.jsonl')
if 'offline_regate' in report:
    assert report['offline_regate']['source_predictions_sha256'] == sha256_file(Path(sys.argv[1]).parent / 'predictions.jsonl'), 'STOP: predictions changed after offline re-gating'
validate_rows(rows)
recomputed = smoke_report(rows)
assert recomputed['passed'], 'STOP: new pooled prefix gate failed/incomplete'
assert report.get('prefix_agreement') == recomputed['prefix_agreement'], 'STOP: prefix gate missing or inconsistent'
assert report.get('first_position_sanity', {}).get('differ'), 'STOP: base/tuned logit sanity failed'
args = parse_args(['--dataset-path',sys.argv[2],'--assets',sys.argv[3],'--pairing',sys.argv[4],
                   '--image-audit',sys.argv[5],'--adapter',sys.argv[6],'--validate-only','--n-perms','1'])
_, checksum = verify_assets(args.dataset_path, args.assets, args.pairing)
assert compatible_smoke_identity(report['identity'], run_identity(args, checksum)), 'STOP: code/assets/settings differ from smoke; repeat smoke'
assert report['adapter_sha256'] == strict_artifact(args.adapter)['sha256'], 'STOP: checkpoint differs from smoke'
print('Passing smoke: estimated full seconds (all ten cells plus loading):', report['estimated_full_total_seconds'])
PY
timeout --signal=INT --kill-after=120s "$MAX_SECONDS" "$PY" -u scripts/run_synth_mcq.py \
  "${LLAVA_ARGS[@]}" --out "$OUT" --n-perms 4
"$PY" - "$OUT/predictions.jsonl" <<'PY'
import sys
from pathlib import Path
count = len(Path(sys.argv[1]).read_text().splitlines())
assert count == 4000, f'Expected 4000 rows, found {count}'
print('COMPLETE: 4000 rows')
PY
