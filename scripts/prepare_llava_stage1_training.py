#!/usr/bin/env python3
"""Laptop-only deterministic Stage 1 selection, rendering and token diagnostics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION, configure_processor, image_token_id
from llava_stage1_training import training_text

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('node_description', 'node_degree', 'highest_node_degree',
         'node_number', 'edge_number', 'triple_listing')
RENDER_FLAGS = ['--core-policy', 'keep', '--max-nodes', '18', '--max-edges', '30',
                '--bridge-rule', 'qa-bridge', '--rankdir', 'TB', '--node-fontsize', '30',
                '--edge-fontsize', '24', '--ranksep', '0.4', '--auto-orient', 'llava',
                '--node-style', 'uniform', '--node-fill', '#ADD8E6',
                '--stage1-task-set', 'paper', '--stage1-balance', 'per-task', '--seed', '13']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def export_reports(args):
    """Keep reviewable manifests/summaries in git; images/items stay in outputs."""
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for name in ('training.json', 'validation.json', 'data_manifest.json', 'render_diagnostics.json'):
        path = args.out_dir / name
        if path.exists():
            (args.report_dir / name).write_bytes(path.read_bytes())
    path = args.out_dir / 'token_diagnostics.json'
    if path.exists():
        value = json.loads(path.read_text())
        for subset in value['subsets'].values():
            subset.pop('items')
        write_json(args.report_dir / 'token_diagnostics_summary.json', value)


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError('Empty distribution')
    def percentile(p):
        position = (len(values) - 1) * p / 100
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {'n': len(values), 'mean': sum(values) / len(values),
            'percentiles': {str(p): percentile(p) for p in (0, 25, 50, 75, 90, 95, 99, 100)}}


def select(statements, tests, n_train=1200, n_validation=100):
    indexed = {str(row['id']): i for i, row in enumerate(statements)}
    if len(indexed) != len(statements):
        raise ValueError('Duplicate train question IDs')
    test_ids = {str(row['id']) for row in tests}
    rng = random.Random(13)
    train_ids = rng.sample(sorted(indexed), n_train)
    val_ids = rng.sample(sorted(set(indexed) - set(train_ids)), n_validation)
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & test_ids:
        raise ValueError('Question ID overlap between train/validation/test')
    return {name: sorted(indexed[qid] for qid in ids)
            for name, ids in [('training', train_ids), ('validation', val_ids)]}


def generate(args):
    if args.out_dir.exists():
        raise FileExistsError(f'Use a fresh output directory: {args.out_dir}')
    statements_path = args.data_root / 'obqa/statement/train.statement.jsonl'
    tests_path = args.data_root / 'obqa/statement/test.statement.jsonl'
    statements, tests = rows(statements_path), rows(tests_path)
    selected = select(statements, tests)
    args.out_dir.mkdir(parents=True)
    sources = {str(path.relative_to(args.data_root)): sha(path) for path in (
        statements_path, tests_path, args.data_root / 'obqa/graph/train.graph.adj.pk',
        args.data_root / 'cpnet/concept.txt')}
    generator = ROOT / 'scripts/generate_graphvis_datasets.py'
    manifest = {'model_id': DEFAULT_MODEL_ID, 'revision': DEFAULT_REVISION,
                'seed': 13, 'selection': 'Random(13).sample(sorted(train_question_ids),1200); '
                'same RNG.sample(sorted(remaining_question_ids),100); render in statement-index order',
                'test_exclusion': 'question IDs against the complete QA-GNN OBQA test split',
                'test_questions_checked': len(tests), 'overlap': 0, 'sources_sha256': sources,
                'tasks': list(TASKS), 'held_out_tasks': ['relation_identification', 'neighbor_listing',
                                                       'shortest_path_listing'],
                'render_flags': RENDER_FLAGS, 'generator_sha256': sha(generator),
                'preparation_script_sha256': sha(Path(__file__)),
                'template_script_sha256': sha(ROOT / 'scripts/llava_stage1_prompt.py'),
                'training_primitives_sha256': sha(ROOT / 'scripts/llava_stage1_training.py'),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'subsets': {}}
    for name, indices in selected.items():
        subset = {'split': 'train', 'seed': 13, 'source_n': len(statements), 'n': len(indices),
                  'indices': indices, 'question_ids': [str(statements[i]['id']) for i in indices],
                  'sha256_of_statement_file': sha(statements_path)}
        subset_path = args.out_dir / f'{name}.json'
        write_json(subset_path, subset)
        command = [sys.executable, str(generator), '--split', 'train', '--data-root', str(args.data_root),
                   '--out-dir', str(args.out_dir), '--indices-file', str(subset_path), *RENDER_FLAGS]
        print(f'Rendering {name}: {len(indices)} graphs', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        record_path = args.out_dir / f'train/stage1_graph_comprehension_{name}.jsonl'
        records = rows(record_path)
        counts = Counter(row['task_type'] for row in records)
        if counts != Counter({task: len(indices) for task in TASKS}):
            raise ValueError(f'Missing/extra paper tasks: {counts}')
        expected = {(idx, task) for idx in indices for task in TASKS}
        actual = [(row['statement_idx'], row['task_type']) for row in records]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Missing, duplicate or unexpected example keys')
        manifest['subsets'][name] = {'manifest': subset_path.name, 'sha256': sha(subset_path),
                                     'records': str(record_path.relative_to(args.out_dir)),
                                     'records_sha256': sha(record_path), 'counts_per_task': dict(counts)}
    manifest['files_sha256'] = {str(path.relative_to(args.out_dir)): sha(path)
                               for path in sorted(args.out_dir.rglob('*')) if path.is_file()}
    write_json(args.out_dir / 'data_manifest.json', manifest)
    basic = {name: {'answer_characters': distribution([len(r['answer']) for r in rows(
                    args.out_dir / subset['records'])]),
                    'example_prompts': {task: next(r for r in rows(args.out_dir / subset['records'])
                                                    if r['task_type'] == task) for task in TASKS}}
             for name, subset in manifest['subsets'].items()}
    write_json(args.out_dir / 'render_diagnostics.json', basic)
    export_reports(args)
    print('Rendering complete; manifest SHA256:', sha(args.out_dir / 'data_manifest.json'), flush=True)


def audit(args):
    import transformers
    from PIL import Image
    from transformers import LlavaNextConfig, LlavaNextProcessor
    if transformers.__version__ != '5.16.1':
        raise ValueError('Requires transformers==5.16.1; do not substitute')
    manifest = json.loads((args.out_dir / 'data_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if sha(args.out_dir / name) != digest:
            raise ValueError(f'Data hash changed: {name}')
    processor = LlavaNextProcessor.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION,
                                                   local_files_only=True)
    config = LlavaNextConfig.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION,
                                             local_files_only=True)
    configure_processor(processor, config)
    processor.tokenizer.padding_side = 'right'
    report = {'model_id': DEFAULT_MODEL_ID, 'revision': DEFAULT_REVISION, 'template': 'llava_v1',
              'transformers': transformers.__version__, 'model_max_length': 4096,
              'data_manifest_sha256': sha(args.out_dir / 'data_manifest.json'), 'subsets': {}}
    overlength = []
    for name, subset in manifest['subsets'].items():
        image_counts, items, per_task = {}, [], defaultdict(list)
        for row in rows(args.out_dir / subset['records']):
            prompt, full = training_text(processor, row)
            plain = processor.tokenizer(full, truncation=False)['input_ids']
            prefix = processor.tokenizer(prompt, truncation=False)['input_ids']
            if plain[:len(prefix)] != prefix or plain[-1] != processor.tokenizer.eos_token_id:
                raise ValueError('Answer boundary or final EOS mismatch')
            if plain.count(image_token_id(processor)) != 1:
                raise ValueError('Expected exactly one unexpanded image marker')
            idx = row['statement_idx']
            if idx not in image_counts:
                with Image.open(args.out_dir / row['image']) as image:
                    encoded = processor(text=full, images=image.convert('RGB'), return_tensors='np',
                                        truncation=False)
                ids = encoded['input_ids'][0].tolist()
                count = ids.count(image_token_id(processor))
                expected_ids = [token for x in plain for token in
                                ([x] * count if x == image_token_id(processor) else [x])]
                if ids != expected_ids:
                    raise ValueError('Full processor differs from tokenizer image replacement')
                image_counts[idx] = count
            total = len(plain) + image_counts[idx] - 1
            item = {'statement_idx': idx, 'task_type': row['task_type'], 'total_tokens': total,
                    'answer_eos_tokens': len(plain) - len(prefix), 'image_tokens': image_counts[idx]}
            if total > 4096:
                overlength.append({'subset': name, **item})
            items.append(item)
            per_task[row['task_type']].append(item)
        report['subsets'][name] = {
            'total_tokens': distribution([r['total_tokens'] for r in items]),
            'answer_eos_tokens': distribution([r['answer_eos_tokens'] for r in items]),
            'exceed_2048': sum(r['total_tokens'] > 2048 for r in items),
            'exceed_4096': sum(r['total_tokens'] > 4096 for r in items),
            'per_task': {task: {'total_tokens': distribution([r['total_tokens'] for r in group]),
                                 'answer_eos_tokens': distribution([r['answer_eos_tokens'] for r in group])}
                         for task, group in per_task.items()}, 'items': items}
    report['overlength_examples'] = overlength
    report['method'] = 'Full pinned processor on one record per graph; exact image replacement checked; '
    report['method'] += 'same template/tokenizer on every record. No truncation or dropped examples.'
    write_json(args.out_dir / 'token_diagnostics.json', report)
    export_reports(args)
    print(json.dumps({name: {k: v for k, v in data.items() if k not in ('items', 'per_task')}
                      for name, data in report['subsets'].items()}, indent=2), flush=True)
    if overlength:
        raise ValueError(f'{len(overlength)} examples exceed 4096; no experimental settings changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data_preprocessed_release')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path,
                        default=ROOT / 'experiments/2026-10-06_llava_stage1_training')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only and args.render_only:
        parser.error('Choose one mode')
    if not args.audit_only:
        generate(args)
    if not args.render_only:
        audit(args)


if __name__ == '__main__':
    main()
