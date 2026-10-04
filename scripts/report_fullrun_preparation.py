#!/usr/bin/env python3
"""CPU-only data, disk, and historical prompt regression evidence for S-A."""
from collections import Counter
import json
from pathlib import Path
import shutil
from fullrun_common import ROOT, DATA_NAME, MODELS, specs, read_rows, sha, frozen_json, qa_hash


def tree_bytes(path):
    if not path.exists():
        return None
    # Count unique underlying files once (HF snapshots symlink to blobs).
    seen, total = set(), 0
    for file in path.rglob('*'):
        if file.is_file():
            stat = file.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def main():
    root = ROOT
    data = root / 'outputs' / DATA_NAME
    qa = read_rows(data / 'test/stage2_obqa_0_500.jsonl')
    meta = {r['statement_idx']: r for r in read_rows(data / 'test/graph_metadata_0_500.jsonl')}
    old_input = root / 'outputs/graphvis_obqa/test/stage2_obqa_0_500.jsonl'
    old_rows = read_rows(old_input)
    reconstructed = {c: {'current': qa_hash(qa, meta, c),
                         'old_source_reconstructed': qa_hash(old_rows, meta, c) if old_rows else None}
                     for c in ('text', 'text_noref')}
    old_configs = []
    for parent in (root / 'outputs', root / 'experiments'):
        for path in parent.rglob('run_config.json'):
            config = json.loads(path.read_text())
            if config.get('condition') in ('text', 'text_noref') and config.get('model_id') in [MODELS[m][0] for m in ('qwen','gemma','deepseek')]:
                old_configs.append((path, config))
    regression = []
    for model in ('qwen','gemma','deepseek'):
        for condition in ('text','text_noref'):
            evidence = {'model': model, 'condition': condition, **reconstructed[condition],
                        'new_generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 64},
                        'new_revision': MODELS[model][1], 'prediction_equality': 'NOT_VERIFIED: GPU inference is out of scope',
                        'old_config': None, 'verified_old_run_prompt_hash': False}
            for path, config in old_configs:
                if config['model_id'] == MODELS[model][0] and config['condition'] == condition:
                    evidence.update(old_config=str(path.relative_to(root)), old_generation=config.get('generation'),
                                    old_revision=config.get('model_revision'), old_prompt_hash=config.get('prompt_bodies_sha256'),
                                    verified_old_run_prompt_hash=config.get('prompt_bodies_sha256') == evidence['current'])
            if model == 'deepseek':
                path = root / f'outputs/2026-08-25_zero_shot_obqa_500/predictions_{condition}.jsonl'
                rows = read_rows(path)
                evidence.update(old_predictions=str(path.relative_to(root)), old_predictions_sha256=sha(path),
                                old_prediction_count=len(rows), old_prediction_revisions=sorted({r['model_revision'] for r in rows}),
                                old_token_limit='8192 documented in run_notes.md; not independently verified without run_config',
                                token_limit_difference='documented 8192 -> 64; historical prediction equality not expected')
            elif evidence['old_config'] is None:
                evidence['missing'] = '500-question historical predictions and run_config unavailable locally'
            regression.append(evidence)
    audit = json.loads((data / 'image_audit.json').read_text())
    pruning = json.loads((data / 'pruning_diagnostics.json').read_text())
    # Budget output storage using worst-case generated text plus generous metadata/log overhead.
    items_per_model = 900 + 4 * 500
    generated_token_budget = 4 * (900 * 1024 + 2000 * 64)
    reserve = 1_000_000_000
    local_cache = Path.home() / '.cache/huggingface'
    report = {
        'counts': {'qa': len(qa), 'stage1': len(read_rows(data / 'test/stage1_subset100.jsonl'))},
        'pruning': pruning, 'image_audit_summary': audit['summary'],
        'orientations': json.loads((data / 'orientation_counts.json').read_text()),
        'regression': regression,
        'disk': {'volume_capacity_bytes': 100_000_000_000, 'new_split_bytes': tree_bytes(data),
                 'all_local_outputs_bytes': tree_bytes(root / 'outputs'),
                 'local_hf_cache_path': str(local_cache), 'local_hf_cache_bytes': tree_bytes(local_cache),
                 'pod_hf_cache_bytes': None, 'pod_free_bytes': None,
                 'note': 'Pod cache/free space unknown: no pod access. Preflight checks actual volume free space before weights load.',
                 'planned_prediction_count_all_models': 4 * items_per_model,
                 'maximum_generated_tokens_all_models': generated_token_budget,
                 'new_run_outputs_reserved_bytes': reserve,
                 'estimate_note': '1 GB output/log reserve; 4.2M max generated tokens at 8 bytes/token is ~34 MB raw text, plus scoring JSON and logs. No new weights downloaded.',
                 'local_free_bytes': shutil.disk_usage(root).free},
        'not_verified': ['Real GPU smoke and peak VRAM/timing', 'Pod cache and free disk',
                         'Historical Qwen/Gemma text predictions and old run configs',
                         'Historical DeepSeek run_config and actual old generation settings'],
    }
    frozen_json(data / 'preparation_report.json', report)
    print(json.dumps({'counts': report['counts'], 'disk': report['disk'], 'regression': regression}, indent=2))


if __name__ == '__main__':
    main()
