#!/usr/bin/env python3
"""Rescore saved Stage 1 answers on CPU with audit_v2, preserving every input."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0, str(SCORER_DIR))
import score_stage1 as base
import stage1_scoring as scoring


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def headline(task, metrics):
    if task in ('node_description', 'triple_listing'):
        return 'macro_f1', metrics['strict']['raw']['macro_f1']
    if task == 'neighbor_listing':
        return 'macro_f1', metrics['strict']['raw']['mean_f1']
    if task == 'highest_node_degree':
        return 'joint_accuracy', metrics['strict_joint_accuracy']
    return 'strict_accuracy', metrics['strict_accuracy']


def rescore(args):
    if args.output.exists():
        raise FileExistsError(f'Output must be a new directory: {args.output}')
    inputs = [args.input_jsonl, args.graph_metadata] + [
        args.predictions_dir / f'predictions_{args.model_name}_{task}.jsonl'
        for task in base.TASK_SETS[args.task_set]]
    sources = [SCORER_DIR / name for name in ('score_stage1.py', 'score_stage1_template_v1.py',
               'score_stage1_audit_v2.py', 'stage1_scoring.py')] + [Path(__file__)]
    hashes = {str(p.resolve()): sha(p) for p in inputs + sources}
    results = {}; corrected = None
    for version in ('span_extended', 'template_v1', 'audit_v2'):
        options = SimpleNamespace(**{**vars(args), 'extractor': version})
        results[version], rows = scoring.score_files(options)
        if version == 'audit_v2': corrected = rows
    if hashes != {str(p.resolve()): sha(p) for p in inputs + sources}:
        raise RuntimeError('Inputs or scorer changed while rescoring; no output written.')
    comparison = []
    for task in corrected:
        metric, value = headline(task, results['audit_v2']['tasks'][task])
        row = {'task': task, 'n': len(corrected[task]), 'metric': metric,
               **{v: headline(task, m['tasks'][task])[1] for v,m in results.items()}}
        if task == 'highest_node_degree':
            row.update({v + '_degree_accuracy': m['tasks'][task]['degree_accuracy'] for v,m in results.items()})
        comparison.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    for version, metrics in results.items():
        base.write_json(args.output / f'metrics_{args.model_name}_{version}.json', metrics)
    base.write_json(args.output / f'metrics_{args.model_name}.json', results['audit_v2'])
    for task, rows in corrected.items():
        with (args.output / f'scored_{args.model_name}_{task}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    base.write_json(args.output / 'comparison.json', comparison)
    manifest = {'scorer_version': 'audit_v2', 'python_version': sys.version,
                'input_and_source_sha256': hashes, 'inputs_unchanged': True,
                'highest_degree_headline': 'Joint name and degree; degree accuracy retained separately.'}
    base.write_json(args.output / 'rescore_manifest.json', manifest)
    lines = ['# Stage 1 corrected scoring', '',
             'Raw normalization. Highest-degree comparisons use joint node-and-degree accuracy for every version.', '',
             '| Task | N | Metric | Span extended | Template v1 | Audit v2 |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for row in comparison:
        lines.append('| ' + ' | '.join([row['task'],str(row['n']),row['metric']] + [f"{row[v]:.6f}" if row[v] is not None else 'n/a' for v in results]) + ' |')
    (args.output / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--graph-metadata', type=Path, required=True)
    parser.add_argument('--task-set', choices=base.TASK_SETS, default='extended')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rescore(args)
    print(f'Wrote corrected scores, comparisons and provenance to {args.output}')


if __name__ == '__main__':
    main()
