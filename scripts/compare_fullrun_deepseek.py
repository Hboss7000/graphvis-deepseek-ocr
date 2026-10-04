#!/usr/bin/env python3
"""After the full run, list historical DeepSeek differences without attribution."""
import argparse
import csv
import json
from pathlib import Path
from fullrun_common import ROOT, DATA_NAME, read_rows, sha
from verify_fullrun import verify_model


def compare(old_rows, new_rows):
    def indexed(rows):
        result = {r['statement_idx']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate historical/new prediction keys')
        if set(result) != set(range(500)):
            raise ValueError('Expected all 500 statement indices')
        return result
    old, new = indexed(old_rows), indexed(new_rows)
    differences = []
    fields = ('raw_response', 'predicted_option', 'is_correct', 'gold_option', 'model_id', 'model_revision')
    for idx in range(500):
        for field in fields:
            if old[idx].get(field) != new[idx].get(field):
                differences.append({'statement_idx': idx, 'field': field,
                                    'old': old[idx].get(field), 'new': new[idx].get(field)})
    return differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, required=True, help='Fresh report directory; existing directories refused')
    args = p.parse_args()
    verified = verify_model(args.root, 'deepseek')
    if not verified['passed']:
        raise ValueError('Verify the complete new DeepSeek model session before historical comparison')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = {'conditions': {}, 'unknown_historical_settings': [
        'Effective max_new_tokens (8192 is documented, not verified)',
        'Effective decoding settings, including greedy configuration and no_repeat_ngram_size',
        'Actual prompt-body hashes used by the historical run',
        'Torch/Transformers versions and hardware/runtime configuration'],
        'limitations': 'No historical run_config.json. Qwen/Gemma historical runs are on unreachable COMA and unavailable. '
                       'Only DeepSeek text/text_noref are compared. No attribution or conclusions are drawn.'}
    differences = []
    for condition in ('text', 'text_noref'):
        old = args.root / f'outputs/2026-08-25_zero_shot_obqa_500/predictions_{condition}.jsonl'
        new = args.root / 'outputs' / (DATA_NAME + '_results') / 'deepseek' / ('qa_' + condition) / 'predictions.jsonl'
        diff = compare(read_rows(old), read_rows(new))
        differences.extend(dict(condition=condition, **r) for r in diff)
        report['conditions'][condition] = {
            'old_path': str(old), 'old_sha256': sha(old), 'new_path': str(new), 'new_sha256': sha(new),
            'n': 500, 'differing_items': len({r['statement_idx'] for r in diff}),
            'differing_fields': len(diff),
            'new_run_config': json.loads((new.parent / 'run_config.json').read_text())}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'differences.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'statement_idx', 'field', 'old', 'new'])
        writer.writeheader()
        writer.writerows(differences)
    print(f'COMPLETED historical difference listing: {args.output_dir}; no conclusions drawn')


if __name__ == '__main__':
    main()
