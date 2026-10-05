#!/usr/bin/env python3
"""CPU rehearsal gate and estimates from measured real inference, never stub results."""
import argparse
import json
from pathlib import Path
from fullrun_common import *
from verify_fullrun import verify_model, print_table
from score_stage1 import score_record


def rehearsal_report(root):
    recovery = root / 'outputs' / (DATA_NAME + '_s0b') / 'source.json'
    if recovery.exists():
        from fullrun_s0b import gate
        return gate(root, root / json.loads(recovery.read_text())['source_smoke_root'])
    from fullrun_gate import model_gate
    reports={m:model_gate(root,m) for m in MODELS}
    errors=[m+': '+e for m,r in reports.items() for e in r['errors']]
    return {'models':reports,'errors':errors,'probe':reports['llava']['probe'],'passed':not errors}


def print_report(report):
    for result in report['models'].values():
        print_table(result)
        if 'estimate' in result:
            print('Estimate: ' + json.dumps(result['estimate'], sort_keys=True))
    print('Reading probe (basic headline, raw alongside): ' + json.dumps(
        {k:v for k,v in report['probe'].items() if k not in ('per_graph','sources')}, sort_keys=True))
    for error in report['errors']:
        print('FAILED: ' + error)
    print('COMPLETED rehearsal gate; STOP the Pod and review S-B before full runs.' if report['passed'] else 'Some model gates failed; only that model is blocked. Review its per-model report. STOP the Pod.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    report = rehearsal_report(args.root)
    print_report(report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
