#!/usr/bin/env python3
"""Rescore paired reading arms; ceilings are descriptive, the gate uses basic recall only."""
from __future__ import annotations
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from fullrun_common import read_rows, sha, frozen_json
from score_stage1 import score_record


def paired_report(records_path, metadata_path, arms, expected_count):
    records = {r['statement_idx']: r for r in read_rows(records_path)}
    if len(records) != expected_count:
        raise ValueError(f'Expected {expected_count} source graphs, found {len(records)}')
    meta = {r['statement_idx']: r for r in read_rows(metadata_path)}
    report = {'extractor': 'span_extended', 'headline_tier': 'basic', 'n': expected_count,
              'sources': {str(records_path): sha(records_path), str(metadata_path): sha(metadata_path)},
              'arms': {}, 'per_graph': [], 'pairs': {}}
    scored = {}
    for label, path in arms.items():
        rows = read_rows(path)
        by_idx = {r['statement_idx']: r for r in rows}
        if len(rows) != len(by_idx) or set(by_idx) != set(records):
            raise ValueError(f'{label}: missing, duplicate or unexpected graph indices: {path}')
        scored[label] = {}
        for idx, row in by_idx.items():
            if row['task_type'] != 'node_description' or not isinstance(row.get('raw_response'), str):
                raise ValueError(f'{label}: invalid reading prediction at {idx}')
            if row.get('gold') != records[idx]['answer'] or row.get('prompt') != records[idx]['prompt']:
                raise ValueError(f'{label}: source gold/prompt differs at {idx}')
            value = score_record(records[idx], row['raw_response'], meta[idx], extractor='span_extended')
            scored[label][idx] = {f'{tier}_{metric}': value['set_metrics'][tier][metric]
                                 for tier in ('basic', 'raw') for metric in ('recall', 'precision', 'f1')}
            scored[label][idx]['ceiling'] = int(bool(row['hit_token_ceiling']))
        report['arms'][label] = {'path': str(path), 'sha256': sha(path),
            'ceilings': sum(s['ceiling'] for s in scored[label].values()),
            **{k: sum(s[k] for s in scored[label].values()) / expected_count
               for k in next(iter(scored[label].values())) if k != 'ceiling'}}
    pairs = [(a, b) for a, b in combinations(arms, 2)]
    for idx in sorted(records):
        row = {'statement_idx': idx}
        for label in arms:
            row.update({f'{label}_{key}': value for key, value in scored[label][idx].items()})
        for a, b in pairs:
            for tier in ('basic', 'raw'):
                row[f'{a}_minus_{b}_{tier}_recall'] = scored[a][idx][f'{tier}_recall'] - scored[b][idx][f'{tier}_recall']
        report['per_graph'].append(row)
    for a, b in pairs:
        pair = {}
        for tier in ('basic', 'raw'):
            values = [scored[a][idx][f'{tier}_recall'] - scored[b][idx][f'{tier}_recall'] for idx in records]
            pair[tier] = {'mean_recall_difference': sum(values) / expected_count,
                          'better': sum(v > 1e-12 for v in values), 'worse': sum(v < -1e-12 for v in values),
                          'equal': sum(abs(v) <= 1e-12 for v in values)}
        report['pairs'][f'{a}_minus_{b}'] = pair
    if 'uniform' in arms and 'legacy' in arms:
        values = [scored['uniform'][i]['basic_recall'] - scored['legacy'][i]['basic_recall'] for i in records]
        delta = sum(values) / expected_count
        report['gate'] = {'metric': 'paired macro basic recall: uniform - legacy',
                          'difference': delta, 'minimum': -.05, 'passed': delta >= -.05 - 1e-12,
                          'rule': 'Ceilings, precision, F1 and raw recall are reported; only paired basic recall decides.'}
    return report


def write_report(report, output):
    if output.exists():
        raise FileExistsError(f'Fresh report directory required: {output}')
    output.mkdir(parents=True)
    frozen_json(output / 'report.json', report)
    with (output / 'per_graph.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['per_graph'][0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(report['per_graph'])
    text = ['| arm | tier | recall | precision | F1 | ceilings |', '|---|---|---:|---:|---:|---:|']
    for label, arm in report['arms'].items():
        for tier in ('basic', 'raw'):
            text.append(f'| {label} | {tier} | {arm[tier+"_recall"]:.6f} | {arm[tier+"_precision"]:.6f} | {arm[tier+"_f1"]:.6f} | {arm["ceilings"]}/{report["n"]} |')
    if 'gate' in report:
        g = report['gate']
        text.append(f'\nGate: uniform − legacy basic recall = {g["difference"]:.9f}; {"PASSED" if g["passed"] else "FAILED"}.')
    text.extend(['\n| graph | ' + ' | '.join(f'{a} basic / raw recall' for a in report['arms']) + ' | uniform − legacy basic / raw |',
                 '|---:|' + '---:|' * (len(report['arms']) + 1)])
    for row in report['per_graph']:
        values = [f'{row[a+"_basic_recall"]:.6f} / {row[a+"_raw_recall"]:.6f}' for a in report['arms']]
        delta = f'{row["uniform_basic_recall"]-row["legacy_basic_recall"]:+.6f} / {row["uniform_raw_recall"]-row["legacy_raw_recall"]:+.6f}' if 'uniform' in report['arms'] and 'legacy' in report['arms'] else ''
        text.append(f'| {row["statement_idx"]} | ' + ' | '.join(values) + f' | {delta} |')
    (output / 'report.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--graph-metadata', type=Path, required=True)
    p.add_argument('--arm', action='append', required=True, help='LABEL=predictions JSONL')
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    arms = {}
    for arg in args.arm:
        label, sep, path = arg.partition('=')
        if not sep or label in arms:
            raise ValueError('Arms must have distinct labels and paths')
        arms[label] = Path(path)
    write_report(paired_report(args.records, args.graph_metadata, arms, args.expected_count), args.output_dir)


if __name__ == '__main__':
    main()
