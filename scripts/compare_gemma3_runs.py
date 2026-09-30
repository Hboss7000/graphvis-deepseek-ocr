#!/usr/bin/env python3
"""Compare realised predictions; timestamps intentionally differ between runs."""
import argparse
import json
from pathlib import Path


def compare(left, right, expected_count):
    def indexed(path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        keys = [(r['statement_idx'], r.get('task_type')) for r in rows]
        if len(rows) != expected_count or len(set(keys)) != len(keys):
            raise ValueError(f'{path}: expected {expected_count} unique items')
        return dict(zip(keys, rows))
    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise ValueError('Runs contain different item slices')
    configs = [json.loads((path.parent / 'run_config.json').read_text()) for path in (left, right)]
    for field in ('model_id', 'model_revision', 'condition', 'prompt_bodies_sha256',
                  'max_new_tokens', 'seed', 'dtype', 'pan_and_scan', 'pan_and_scan_kwargs'):
        if configs[0].get(field) != configs[1].get(field):
            raise ValueError(f'Run configurations differ in {field}; comparison is confounded')
    differences = []
    for key in sorted(a):
        fields = (a[key].keys() | b[key].keys()) - {'timestamp_utc'}
        changed = [field for field in sorted(fields) if a[key].get(field) != b[key].get(field)]
        if changed:
            differences.append({'statement_idx': key[0], 'task_type': key[1],
                                'fields': changed, 'left_response': a[key]['raw_response'],
                                'right_response': b[key]['raw_response']})
    return {'left': str(left), 'right': str(right), 'item_count': len(a),
            'model_id': configs[0]['model_id'], 'revision': configs[0]['model_revision'],
            'attention_implementations': [c['attn_implementation'] for c in configs],
            'prompt_bodies_sha256': configs[0]['prompt_bodies_sha256'],
            'ignored_fields': ['timestamp_utc'], 'different_items': len(differences),
            'identical': not differences, 'differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=50)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.left, args.right, args.expected_count)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['identical']:
        raise SystemExit('Predictions differ; inspect the report before full runs')


if __name__ == '__main__':
    main()
