#!/usr/bin/env python3
"""CPU verification of exact fullrun coverage, frozen identity and sanity bounds."""
import argparse
import json
from pathlib import Path
from fullrun_common import ROOT, DATA_NAME, MODELS, specs, verify_job, validate_frozen_files, saved_spec


def verify_model(root, model, smoke=False):
    mode = 'smoke' if smoke else 'results'
    base = root / 'outputs' / (DATA_NAME + '_' + mode)
    report = {'model': model, 'mode': mode, 'jobs': [], 'errors': []}
    for spec in specs(root, model, smoke):
        try:
            spec = saved_spec(root, base / model / spec["label"], spec)
            result = verify_job(base / model / spec['label'], spec)
            status = base / model / 'status' / (spec['label'] + '.json')
            if not status.exists() or json.loads(status.read_text())['state'] != 'COMPLETED':
                raise ValueError('Missing/non-COMPLETED job status')
            for peer in MODELS:
                path = base / peer / spec['label'] / 'run_config.json'
                if path.exists():
                    other = json.loads(path.read_text())
                    if other.get('prompt_bodies_sha256') != spec['prompt_bodies_sha256']:
                        raise ValueError(f'Prompt hash differs across models: {peer}')
            report['jobs'].append(result)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            report['errors'].append(f'{spec["label"]}: {exc}')
    report['passed'] = not report['errors']
    return report


def print_table(report):
    print(f'\n{report["model"]} ({report["mode"]})')
    print('job                 rows/expected    FAILED  ceilings    s/item    peak GiB')
    for job in report['jobs']:
        print(f'{job["job"]:20} {job["rows"]:4}/{job["expected"]:<8} {job["FAILED"]:7} {job["ceilings"]:9} '
              f'{job["seconds_per_item"]:9.3f} {job["peak_vram_bytes"]/1024**3:10.2f}')
        if job['parse_tiers']:
            print('  parse tiers: ' + json.dumps(job['parse_tiers'], sort_keys=True))
        if job.get('ceilings_per_task'):
            print('  ceilings per task: ' + json.dumps(job['ceilings_per_task'],sort_keys=True))
        if job['legibility_confounded']:
            print('  Gemma: legibility-confounded (failed preflight recorded)')
    for error in report['errors']:
        print('FAILED: ' + error)
    print('COMPLETED verification' if report['passed'] else 'FAILED verification')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=(*MODELS, 'all'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    ok = True
    for model in MODELS if args.model == 'all' else [args.model]:
        result = verify_model(args.root, model, args.smoke)
        print_table(result)
        ok &= result['passed']
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
