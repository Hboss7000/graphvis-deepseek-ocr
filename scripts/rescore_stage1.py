#!/usr/bin/env python3
"""CPU-only rescoring of stored Stage 1 responses into a new directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SCORER_DIR = Path(__file__).resolve().parents[1] / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0, str(SCORER_DIR))
import score_stage1 as scorer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_rows(legacy: dict, span: dict) -> list[dict]:
    rows = []
    for task, old in legacy['tasks'].items():
        new = span['tasks'][task]
        if task in scorer.NUMERIC_TASKS:
            label, before, after = 'accuracy', old['exact_accuracy'], new['strict_accuracy']
        elif task == 'highest_node_degree':
            label, before, after = 'degree accuracy', old['degree_accuracy'], new['strict_accuracy']
        elif task == 'relation_identification':
            label, before, after = 'accuracy', old['raw']['accuracy'], new['strict_accuracy']
        elif task == 'shortest_path_listing':
            label, before, after = 'path exact', old['raw']['path_exact'], new['raw']['path_exact']
        elif task == 'neighbor_listing':
            label, before, after = 'macro F1', old['raw']['mean_f1'], new['raw']['mean_f1']
        else:
            label, before, after = 'macro F1', old['set_metrics']['raw']['macro_f1'], new['set_metrics']['raw']['macro_f1']
        rows.append({'task': task, 'record_count': new['record_count'], 'metric': label,
                     'legacy': before, 'span': after, 'lenient_containment': new['lenient_containment']})
    return rows


def comparison_markdown(model: str, rows: list[dict]) -> str:
    lines = [f'# Answer extraction comparison: {model}', '',
             'Raw normalization; all records included. Containment is a diagnostic upper bound, **not accuracy**. '
             'For sets it measures gold-member recall, so its difference from F1 is not a pure extraction loss.', '',
             '| Task | N | Strict metric | Legacy | Span | Lenient containment |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    def fmt(value):
        return 'n/a' if value is None else f'{value:.4f}'
    for row in rows:
        lines.append(f"| {row['task']} | {row['record_count']} | {row['metric']} | "
                     f"{fmt(row['legacy'])} | {fmt(row['span'])} | {fmt(row['lenient_containment'])} |")
    return '\n'.join(lines) + '\n'


def rescore(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f'Output must be a new directory: {args.output}')
    inputs = [args.input_jsonl, args.graph_metadata] + [
        args.predictions_dir / f'predictions_{args.model_name}_{task}.jsonl'
        for task in scorer.TASK_SETS[args.task_set]]
    before = {str(path.resolve()): sha256(path) for path in inputs}
    results = {}
    scored_rows = None
    for extractor in ('legacy', 'span'):
        options = SimpleNamespace(**{**vars(args), 'extractor': extractor})
        metrics, rows = scorer.score_files(options)
        results[extractor] = metrics
        if extractor == args.extractor:
            scored_rows = rows
    if before != {str(path.resolve()): sha256(path) for path in inputs}:
        raise RuntimeError('Scoring inputs changed during rescoring; no results written.')
    comparison = comparison_rows(results['legacy'], results['span'])
    baseline = args.predictions_dir / f'metrics_{args.model_name}.json'
    encoded_legacy = (json.dumps(results['legacy'], indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode()
    manifest = {'extractor': args.extractor, 'python_version': sys.version,
                'scorer_sha256': sha256(Path(scorer.__file__)), 'input_sha256': before,
                'legacy_metrics_sha256': hashlib.sha256(encoded_legacy).hexdigest(),
                'saved_metrics_sha256': sha256(baseline) if baseline.exists() else None,
                'legacy_byte_identical_to_saved': encoded_legacy == baseline.read_bytes() if baseline.exists() else None}
    args.output.mkdir(parents=True, exist_ok=False)
    scorer.write_json(args.output / f'metrics_{args.model_name}.json', results[args.extractor])
    # Store both variants independently so the comparison remains auditable.
    for extractor, metrics in results.items():
        scorer.write_json(args.output / f'metrics_{args.model_name}_{extractor}.json', metrics)
    scorer.write_json(args.output / 'rescore_manifest.json', manifest)
    scorer.write_json(args.output / 'comparison.json', {'model_name': args.model_name, 'rows': comparison})
    (args.output / 'comparison.md').write_text(comparison_markdown(args.model_name, comparison), encoding='utf-8')
    for task, rows in scored_rows.items():
        with (args.output / f'scored_{args.model_name}_{task}.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--graph-metadata', type=Path, required=True)
    parser.add_argument('--task-set', choices=scorer.TASK_SETS, default='paper')
    parser.add_argument('--extractor', choices=('legacy', 'span'), default='span')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rescore(args)
    print(f'Wrote metrics, scored records and extractor comparison to {args.output}')


if __name__ == '__main__':
    main()
