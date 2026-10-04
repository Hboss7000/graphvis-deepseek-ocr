#!/usr/bin/env python3
"""CPU rehearsal gate and estimates from measured real inference, never stub results."""
import argparse
import json
from pathlib import Path
from fullrun_common import *
from verify_fullrun import verify_model, print_table
from score_stage1 import score_record


def rehearsal_report(root):
    base = root / 'outputs' / (DATA_NAME + '_smoke')
    report = {'models': {}, 'errors': [], 'probe': {}, 'passed': False}
    for model in MODELS:
        verified = verify_model(root, model, smoke=True)
        report['models'][model] = verified
        if not verified['passed']:
            report['errors'].append(model + ': smoke verification failed')
        try:
            proof = json.loads((base / model / 'qa_image/resume_verified.json').read_text())
            if proof['first_count'] != 3 or proof['final_count'] != 5:
                raise ValueError('wrong resume counts')
            for key, name in [('first3_sha256', 'resume_first3.json'), ('predictions_sha256', 'predictions.jsonl')]:
                if proof[key] != sha(base / model / 'qa_image' / name):
                    raise ValueError('resume proof hash mismatch')
        except (OSError, ValueError, KeyError) as exc:
            report['errors'].append(f'{model}: resume rehearsal proof missing/invalid: {exc}')
        jobs = {j['job']: j for j in verified['jobs']}
        if len(jobs) == 5:
            full_specs = specs(root, model, smoke=False)
            seconds = sum(jobs[s['label']]['seconds_per_item'] * s['record_count'] for s in full_specs)
            verified['estimate'] = {'inference_seconds': seconds, 'inference_hours': seconds / 3600,
                'inference_cost_usd_at_2_09_per_hour': seconds / 3600 * 2.09,
                'with_50_percent_margin_hours': seconds / 3600 * 1.5,
                'with_50_percent_margin_cost_usd': seconds / 3600 * 1.5 * 2.09,
                'note': 'Measured item time extrapolation; 50% margin for loading/overhead, not a billing guarantee.'}
    try:
        from fullrun_session import auxiliary_specs
        data = root / 'outputs' / DATA_NAME
        manifest = json.loads((data / 'inputs_manifest.json').read_text())
        records = {r['statement_idx']: r for r in read_rows(root / manifest['probe_input'])}
        meta = {r['statement_idx']: r for r in read_rows(data / 'test/graph_metadata_0_500.jsonl')}
        for spec in auxiliary_specs(root, 'llava', True):
            directory = base / 'llava' / spec['label']
            verify_job(directory, spec)
            values = {tier: [] for tier in ('basic', 'raw')}
            for row in prediction_rows(directory, spec):
                scored = score_record(records[row['statement_idx']], row['raw_response'], meta[row['statement_idx']], extractor='span_extended')
                for tier in values:
                    values[tier].append(scored['set_metrics'][tier]['recall'])
            report['probe'][spec['label']] = {tier: sum(v) / len(v) for tier, v in values.items()}
        drop = report['probe']['probe_legacy']['basic'] - report['probe']['probe_uniform']['basic']
        report['probe'].update(basic_recall_drop=drop, allowed_drop=.05, passed=drop <= .05)
        if drop > .05:
            report['errors'].append('Uniform reading-probe basic recall fell by more than 5 percentage points')
        # Preflight must have run both real crop paths, regardless of recall gate outcome.
        preflight_spec, = auxiliary_specs(root, 'gemma', True)
        for crop in ('pan_and_scan', 'no_pan_and_scan'):
            verify_job(base / 'gemma/preflight' / crop, preflight_spec)
        gemma_report = json.loads((base / 'gemma/preflight/preflight_report.json').read_text())
        report['gemma_preflight_passed'] = gemma_report['settings']['pan_and_scan']['passed']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['errors'].append(f'Probe/preflight verification: {exc}')
    report['passed'] = not report['errors']
    return report


def print_report(report):
    for result in report['models'].values():
        print_table(result)
        if 'estimate' in result:
            print('Estimate: ' + json.dumps(result['estimate'], sort_keys=True))
    print('Reading probe (basic headline, raw alongside): ' + json.dumps(report['probe'], sort_keys=True))
    for error in report['errors']:
        print('FAILED: ' + error)
    print('COMPLETED rehearsal gate; STOP the Pod and review S-B before full runs.' if report['passed'] else 'FAILED rehearsal gate; do not launch full runs. STOP the Pod.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    report = rehearsal_report(args.root)
    print_report(report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
