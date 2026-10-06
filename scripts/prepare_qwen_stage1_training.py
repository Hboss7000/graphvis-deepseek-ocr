#!/usr/bin/env python3
"""CPU audit of reused manifests using only pinned Qwen processor assets."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from llava_stage1_training import file_sha
from prepare_llava_stage1_training import distribution
from qwen_stage1_training import (pinned_processor, training_text, Stage1Collator,
    DEFAULT_MODEL_ID, DEFAULT_REVISION, TEMPLATE, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)


def audit(data_dir, assets, output):
    import torch
    import transformers
    if transformers.__version__ != '5.16.1':
        raise ValueError('Requires the unchanged transformers==5.16.1')
    torch.set_num_threads(1)
    asset_manifest = json.loads((assets / 'processor_assets_manifest.json').read_text())
    if (asset_manifest['model_id'], asset_manifest['revision'], asset_manifest['weights_downloaded']) != (
            DEFAULT_MODEL_ID, DEFAULT_REVISION, False):
        raise ValueError('Processor asset provenance differs')
    for name, digest in asset_manifest['files_sha256'].items():
        if file_sha(assets / name) != digest:
            raise ValueError('Processor asset changed: ' + name)
    manifest_path = data_dir / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_sha(data_dir / name) != digest:
            raise ValueError('Reused data changed: ' + name)
    processor, api = pinned_processor(assets)
    collator = Stage1Collator(processor, data_dir)
    report = {'model_id': DEFAULT_MODEL_ID, 'revision': DEFAULT_REVISION, 'template': TEMPLATE,
        'default_system_prompt_injected': False, 'model_max_length': 4096,
        'min_pixels': DEFAULT_MIN_PIXELS, 'max_pixels': DEFAULT_MAX_PIXELS,
        'processor_pixel_budget_api': api, 'processor_assets': asset_manifest,
        'data_manifest_sha256': file_sha(manifest_path), 'overlength_examples': 0,
        'truncation': False, 'subsets': {},
        'audit_method': 'Actual pinned processor and collator per rendered image; exact native tokenizer/image-grid expansion per task; no truncation or model weights.'}
    for subset, info in manifest['subsets'].items():
        rows = [json.loads(x) for x in (data_dir / info['records']).read_text().splitlines() if x]
        counts, items = {}, []
        for i, row in enumerate(rows):
            if row['image'] not in counts:
                batch = collator([row])
                counts[row['image']] = int((batch['input_ids'] == processor.image_token_id).sum())
            count = counts[row['image']]
            prefix, full = training_text(processor, row)
            plain = processor.tokenizer(full, truncation=False)['input_ids']
            prefix_ids = processor.tokenizer(prefix, truncation=False)['input_ids']
            if plain[:len(prefix_ids)] != prefix_ids or plain.count(processor.image_token_id) != 1:
                raise ValueError('Native answer boundary/image marker differs')
            if plain[-1] != processor.tokenizer.eos_token_id:
                raise ValueError('Missing end token')
            total = len(plain) + count - 1
            answer_tokens = len(plain) - len(prefix_ids)
            if total > 4096:
                report['overlength_examples'] += 1
            items.append({'statement_idx': row['statement_idx'], 'task_type': row['task_type'],
                'total_tokens': total, 'answer_eos_tokens': answer_tokens, 'image_tokens': count})
            if i % 600 == 0:
                print(f'{subset}: {i}/{len(rows)}', flush=True)
        by_task = defaultdict(list)
        for item in items:
            by_task[item['task_type']].append(item)
        summarize = lambda seq: {k: distribution([r[k] for r in seq]) for k in
                                ('total_tokens', 'answer_eos_tokens', 'image_tokens')}
        report['subsets'][subset] = {**summarize(items), 'items': items,
            'exceed_4096': sum(r['total_tokens'] > 4096 for r in items),
            'tasks': {task: summarize(seq) for task, seq in by_task.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if report['overlength_examples']:
        raise ValueError(f"Zero-overlength assertion failed: {report['overlength_examples']}")
    print(json.dumps({**{k: v for k, v in report.items() if k != 'subsets'},
        'subsets': {k: {n: v for n, v in s.items() if n != 'items'} for k, s in report['subsets'].items()}}, indent=2))
    print('AUDIT SHA256:', file_sha(output), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--processor-assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.data_dir, args.processor_assets, args.output)


if __name__ == '__main__':
    main()
