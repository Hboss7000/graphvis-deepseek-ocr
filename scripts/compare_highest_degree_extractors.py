#!/usr/bin/env python3
"""Compare the historical span scorer with opt-in extended degree extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCORER_DIR = Path(__file__).resolve().parents[1] / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0, str(SCORER_DIR))
import score_stage1 as scorer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def build_report(input_path: Path, metadata_path: Path, predictions_dir: Path,
                 model: str) -> tuple[str, dict]:
    records = {int(row['statement_idx']): row for row in read_jsonl(input_path)
               if row['task_type'] == 'highest_node_degree'}
    metadata = {int(row['statement_idx']): row for row in read_jsonl(metadata_path)}
    predictions = {int(row['statement_idx']): row for row in read_jsonl(
        predictions_dir / f'predictions_{model}_highest_node_degree.jsonl')}
    if set(records) != set(predictions):
        raise ValueError(f'{model}: highest-degree predictions do not match input keys')
    rows_by_extractor = {'span': [], 'span_extended': []}
    differences = []
    for idx in sorted(records):
        response = predictions[idx].get('raw_response', '')
        scored = {}
        for extractor in rows_by_extractor:
            scored[extractor] = scorer.score_record(records[idx], response, metadata[idx], extractor)
            rows_by_extractor[extractor].append(scored[extractor])
        old, new = scored['span'], scored['span_extended']
        if any(old[key] != new[key] for key in ('parsed_degree', 'parsed_name', 'degree_correct', 'name_correct_raw', 'is_correct')):
            differences.append((idx, response, old, new))
    aggregates = {extractor: scorer.aggregate_task('highest_node_degree', rows, extractor)
                  for extractor, rows in rows_by_extractor.items()}
    def pct(value):
        return 'n/a' if value is None else f'{value:.3f}'
    lines = [f'## {model}', '',
             '| Extractor | Degree accuracy | Node-only name accuracy (raw; ties accepted) | Joint accuracy |',
             '|---|---:|---:|---:|']
    for extractor, metrics in aggregates.items():
        lines.append(f"| {extractor} | {pct(metrics['degree_accuracy'])} | "
                     f"{pct(metrics['raw']['name_accuracy'])} | {pct(metrics['raw']['both_accuracy'])} |")
    lines += ['', f'Changed items: {len(differences)} of {len(records)}.', '']
    if differences:
        lines += ['| statement_idx | Old span: degree / name / degree-correct / name-correct | '
                  'New span_extended: degree / name / degree-correct / name-correct | Response |',
                  '|---:|---|---|---|']
        for idx, response, old, new in differences:
            compact_response = ' '.join(str(response).split()).replace('|', '\\|')
            if len(compact_response) > 220:
                compact_response = compact_response[:217] + '…'
            old_value = f"{old['parsed_degree']} / {old['parsed_name']} / {old['degree_correct']} / {old['name_correct_raw']}"
            new_value = f"{new['parsed_degree']} / {new['parsed_name']} / {new['degree_correct']} / {new['name_correct_raw']}"
            lines.append(f'| {idx} | {old_value} | {new_value} | {compact_response} |')
    else:
        lines.append('No parsed-field or correctness differences.')
    return '\n'.join(lines) + '\n', {
        'model': model, 'n': len(records), 'changed_items': [
            {'statement_idx': idx, 'response': response,
             'span': {k: old[k] for k in ('parsed_degree', 'parsed_name', 'degree_correct', 'name_correct_raw', 'is_correct')},
             'span_extended': {k: new[k] for k in ('parsed_degree', 'parsed_name', 'degree_correct', 'name_correct_raw', 'is_correct')}}
            for idx, response, old, new in differences],
        'metrics': {extractor: {
            'degree_accuracy': value['degree_accuracy'],
            'name_accuracy_raw': value['raw']['name_accuracy'],
            'joint_accuracy_raw': value['raw']['both_accuracy'],
        } for extractor, value in aggregates.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--graph-metadata', type=Path, required=True)
    parser.add_argument('--predictions-root', type=Path, required=True,
                        help='Directory containing one <model>_stage1 subdirectory per model.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, data = [], {}
    for model, directory in (('deepseek', 'deepseek_stage1'), ('qwen', 'qwen_stage1'),
                             ('gemma3', 'gemma_stage1'), ('llava', 'llava_stage1')):
        section, payload = build_report(args.input_jsonl, args.graph_metadata,
                                        args.predictions_root / directory, model)
        reports.append(section)
        data[model] = payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('# Highest-node-degree extractor comparison\n\n' + '\n'.join(reports), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
