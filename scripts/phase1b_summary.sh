#!/usr/bin/env bash
# Summarize phase-1b QA parse tiers and LLaVA reading-probe measures.
set -Eeuo pipefail
readonly REPO_ROOT=/workspace/bachelorArbeit
readonly OUTPUTS_ROOT="${REPO_ROOT}/outputs"
readonly PHASE1B_ROOT="${OUTPUTS_ROOT}/phase1b"
readonly PYTHON=/workspace/venvs/venv_qwen/bin/python
"${PYTHON}" - "${PHASE1B_ROOT}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1])
print('| QA runner/condition | records | parse failures | failure rate | parse tiers |')
print('|---|---:|---:|---:|---|')
for label in ('llava_image','llava_text','llava_text_noref','llava_kg_text','deepseek_image'):
    path=root/'qa'/label/'predictions.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    tiers=Counter(row.get('parse_tier','MISSING') for row in rows)
    failures=tiers.get('FAILED',0)
    rate=failures/len(rows) if rows else None
    print(f'| {label} | {len(rows)} | {failures} | {rate:.3f} | {dict(sorted(tiers.items()))} |' if rows else f'| {label} | 0 | n/a | n/a | missing output |')

report=root/'reading_probe'/'reading_probe_report.json'
print('\n| Probe arm | raw macro recall | raw macro precision | raw macro F1 | intrusion raw items | intrusion distinct/graph |')
print('|---|---:|---:|---:|---:|---:|')
if report.exists():
    for arm, metrics in json.loads(report.read_text())['arms'].items():
        intrusion=metrics['edge_label_intrusion']
        raw_rate=intrusion['raw_predicted_items']['rate']
        distinct_rate=intrusion['distinct_predicted_items_per_graph']['rate']
        print(f"| {arm} | {metrics['raw_macro_gold_node_recall']:.4f} | {metrics['raw_macro_precision']:.4f} | {metrics['raw_macro_f1']:.4f} | {raw_rate:.4f} | {distinct_rate:.4f} |")
else:
    print(f'Probe report missing: {report}')
PY
