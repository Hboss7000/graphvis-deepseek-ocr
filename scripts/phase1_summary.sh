#!/usr/bin/env bash
# Read-only phase-1 report.  Run on the pod after phase1_manual.sh completes.
set -Eeuo pipefail

readonly WORKSPACE=/workspace
readonly REPO_ROOT=/workspace/bachelorArbeit
readonly OUTPUTS_ROOT="${REPO_ROOT}/outputs"
readonly DRY_ROOT="${OUTPUTS_ROOT}/phase1_dryrun"
readonly PROBE_ROOT="${OUTPUTS_ROOT}/phase1_reading_probe"
readonly LOG_ROOT="${WORKSPACE}/logs/phase1"
readonly PYTHON="${WORKSPACE}/venvs/venv_qwen/bin/python"

[[ -x "${PYTHON}" ]] || { echo "ERROR: missing ${PYTHON}" >&2; exit 2; }

"${PYTHON}" - "${DRY_ROOT}" "${PROBE_ROOT}" "${LOG_ROOT}" <<'PY'
"""Summarize runner JSONL metrics and the dedicated per-job GPU sampler."""
import csv
import json
import statistics
import sys
from pathlib import Path

dry, probe, logs = map(Path, sys.argv[1:])

jobs = {
    'DeepSeek Stage 1': (dry / 'runs/deepseek_stage1', 'deepseek_stage1_8g'),
    'Qwen Stage 1': (dry / 'runs/qwen_stage1', 'qwen_stage1_8g'),
    'LLaVA Stage 1': (dry / 'runs/llava_stage1', 'llava_stage1_8g'),
    'Gemma Stage 1': (dry / 'runs/gemma_stage1', 'gemma_stage1_8g'),
    'DeepSeek QA': (dry / 'runs/deepseek_qa', 'deepseek_qa_8g'),
    'Qwen QA': (dry / 'runs/qwen_qa', 'qwen_qa_8g'),
    'LLaVA QA': (dry / 'runs/llava_qa', 'llava_qa_8g'),
    'Gemma QA': (dry / 'runs/gemma_qa', 'gemma_qa_8g'),
}

def read_rows(root):
    rows = []
    for path in sorted(root.rglob('*.jsonl')):
        if path.name.startswith('graph_metadata'):
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows

def gpu_peak(label):
    path = logs / f'{label}.gpu.csv'
    if not path.exists():
        return None
    values = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row['memory_used_mib']))
            except (KeyError, TypeError, ValueError):
                pass
    return max(values) if values else None

def status(label):
    path = dry / 'status' / f'{label}.json'
    return json.loads(path.read_text()) if path.exists() else None

print('| runner | extractor | records | s/item | peak VRAM | ceiling hits | parse failures | exit |')
print('|---|---|---:|---:|---:|---:|---:|---:|')
total_projected_seconds = 0.0
projection_available = True
for name, (root, label) in jobs.items():
    rows = read_rows(root) if root.exists() else []
    config_path = root / 'run_config.json'
    metric_files = sorted(root.glob('metrics_*.json')) if root.exists() else []
    if config_path.exists():
        extractor = json.loads(config_path.read_text()).get('extractor')
    elif metric_files:
        extractor = json.loads(metric_files[0].read_text()).get('scoring_notes', {}).get('extractor')
    else:
        extractor = None
    extractor = extractor or ('n/a' if ' QA' in name else 'unknown')
    timing = [float(row['generation_elapsed_seconds']) for row in rows
              if row.get('generation_elapsed_seconds') is not None]
    run = status(label)
    if timing:
        seconds_per_item = statistics.mean(timing)
        timing_text = f'{seconds_per_item:.2f}'
    elif run and rows:
        # Gemma runners do not currently emit per-item timing. This is clearly
        # labelled wall time, rather than pretending it is generation-only time.
        seconds_per_item = float(run['wall_seconds']) / len(rows)
        timing_text = f'{seconds_per_item:.2f} wall'
    else:
        seconds_per_item = None
        timing_text = 'n/a'
        projection_available = False
    torch_peaks = [int(row['peak_memory_allocated_bytes']) for row in rows
                   if row.get('peak_memory_allocated_bytes') is not None]
    peak = (max(torch_peaks) / 2**20) if torch_peaks else gpu_peak(label)
    peak_text = f'{peak:.0f} MiB' if peak is not None else 'n/a'
    ceilings = sum(bool(row.get('hit_token_ceiling')) for row in rows)
    # QA emits a parse tier; Stage 1 does not. Empty responses are the only
    # runner-independent Stage-1 parse failure signal available in JSONL.
    parse_failures = sum(row.get('parse_tier') == 'FAILED' or
                         ('parse_tier' not in row and not str(row.get('raw_response', '')).strip())
                         for row in rows)
    exit_code = run['exit_code'] if run else 'n/a'
print(f'| {name} | {extractor} | {len(rows)} | {timing_text} | {peak_text} | {ceilings} | {parse_failures} | {exit_code} |')
    # The full split has 50 question graphs: 450 extended Stage-1 records and
    # 50 QA records. For timed runners, retain one model-load wall cost and
    # scale only subsequent item generation. For Gemma's wall-only fallback,
    # use a conservative linear projection.
    target = 450 if 'Stage 1' in name else 50
    if seconds_per_item is not None and rows:
        if timing and run:
            total_projected_seconds += float(run['wall_seconds']) + seconds_per_item * (target - len(rows))
        else:
        total_projected_seconds += seconds_per_item * target

print('\nRecorded effective max_new_tokens from each runner config:')
for name, (root, _label) in jobs.items():
    path = root / 'run_config.json'
    if not path.is_file():
        candidates = sorted(root.glob('*/run_config.json')) if root.is_dir() else []
        path = candidates[0] if candidates else path
    if not path.is_file():
        print(f'  {name}: missing run_config.json')
        continue
    config = json.loads(path.read_text())
    effective = config.get('effective_max_new_tokens')
    if effective is None:
        effective = config.get('generation', {}).get('max_new_tokens')
    if effective is None:
        effective = config.get('max_new_tokens')
    expected = 1024 if 'Stage 1' in name else 64
    state = 'OK' if effective == expected else f'EXPECTED {expected}'
    print(f'  {name}: {effective} ({state}) — {path}')

report = probe / 'runs/reading_probe_report.json'
print('\nReading-probe raw macro gold-node recall:')
if report.exists():
    arms = json.loads(report.read_text())['arms']
    for arm, value in arms.items():
        print(f"  {arm}: {value['raw_macro_gold_node_recall']:.4f} ({value['n']} graphs)")
else:
    print(f'  unavailable: missing {report}')

if projection_available:
    hours = total_projected_seconds / 3600
    print('\nProjected full run (all four models; Stage 1=450 records/model, QA=50/model):')
    print(f'  estimated GPU time: {hours:.2f} h')
    print(f'  estimated H100-equivalent cost at $2.09/h: ${hours * 2.09:.2f}')
    print('  Gemma uses a conservative wall-time/item fallback because its current runners do not emit per-item elapsed seconds.')
else:
    print('\nFull-run estimate unavailable: at least one runner did not produce records/timing.')
PY
