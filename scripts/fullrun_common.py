"""CPU contracts shared by the fullrun launcher, verifier and rehearsal summary."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
S1 = 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
QA = 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'
for directory in (ROOT / QA, ROOT / S1):
    sys.path.insert(0, str(directory))
from stage1_common import prompt_bodies_sha256 as stage_hash
from score_stage1 import TASK_SETS
from prompt_common import prompt_bodies_sha256 as qa_hash, parse_answer

MODELS = {
    'llava': ('llava-hf/llava-v1.6-mistral-7b-hf', '2424fdd47412fccc66d91719126b420e9fbd7065'),
    'qwen': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    'gemma': ('google/gemma-3-12b-it', '96b6f1eccf38110c56df3a15bffe176da04bfd80'),
    'deepseek': ('deepseek-ai/DeepSeek-OCR-2', 'aaa02f3811945a91062062994c5c4a3f4c0af2b0'),
}
MODEL_NAMES = dict(llava='llava', qwen='qwen', gemma='gemma3', deepseek='deepseek')
CONDITIONS = ('image', 'text_noref', 'text', 'kg_text')
SECONDS = dict(llava=3.5, qwen=5.2, gemma=4.2, deepseek=1.7)
DATA_NAME = 'fullrun_2026-10-04_B'
TASKS = TASK_SETS['extended']
# Operational sanity thresholds; no scorer or prompt semantics change.
MAX_PARSE_FAILURE = 0.5
MAX_CEILING = 0.5


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_rows(path, growing=False):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        if not growing:
            raise ValueError(f'Incomplete JSONL tail: {path}')
        raw = raw[:raw.rfind(b'\n') + 1]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def frozen_json(path, data):
    text = json.dumps(data, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Collision/incompatible resume: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            f.write(text)


def ordered_stage(rows):
    order = {task: i for i, task in enumerate(TASKS)}
    return sorted(rows, key=lambda r: (r['statement_idx'], order[r['task_type']]))


def specs(root, model, smoke=False, data_name=DATA_NAME):
    data = root / 'outputs' / data_name
    manifest = json.loads((data / 'inputs_manifest.json').read_text())
    mode = 'smoke' if smoke else 'full'
    prefix = f'outputs/{data_name}'
    metadata = prefix + '/test/graph_metadata_0_500.jsonl'
    meta = {r['statement_idx']: r for r in read_rows(root / metadata)}
    code_paths = sorted(set(
        list((root / 'scripts').glob('fullrun_*.py'))
        + [root / 'scripts/verify_fullrun.py', root / 'scripts/report_fullrun_probes.py', root / 'scripts/gemma3_common.py', root / 'scripts/llava_common.py',
           root / 'scripts/eval_gemma3_stage1.py', root / 'scripts/eval_gemma3_stage2.py']
        + list((root / S1).glob('*.py')) + list((root / QA).glob('*.py'))))
    code_files = {str(path.relative_to(root)): sha(path) for path in code_paths}
    jobs = []
    for stage, condition in [('stage1', 'image')] + [('qa', c) for c in CONDITIONS]:
        source = manifest['selections'][mode][stage]
        rows = read_rows(root / source)
        if stage == 'stage1':
            rows = ordered_stage(rows)
        jobs.append({
            'schema': 1, 'model': model, 'stage': stage, 'condition': condition,
            'label': 'stage1' if stage == 'stage1' else 'qa_' + condition,
            'mode': mode, 'data_name': data_name,
            'model_id': MODELS[model][0], 'model_revision': MODELS[model][1],
            'effective_max_new_tokens': 1024 if stage == 'stage1' else 64,
            'versions': {'torch': '2.8.0+cu128', 'transformers': '4.46.3' if model == 'deepseek' else '5.16.1'},
            'input_jsonl': source, 'graph_metadata': metadata, 'image_root': prefix,
            'record_count': len(rows),
            'expected_keys': [[r['statement_idx'], r['task_type'] if stage == 'stage1' else 'obqa_answer'] for r in rows],
            'prompt_bodies_sha256': stage_hash(rows) if stage == 'stage1' else qa_hash(rows, meta, condition),
            'input_files': manifest['files'], 'data_provenance': manifest['provenance'],
            'code_files': code_files,
            'seconds_per_item_reference': SECONDS[model],
            'reference_note': 'Stage 1 dry-run s/item; conservative provisional QA watchdog reference. Rehearsal reports QA separately.',
        })
    return jobs


def prediction_files(directory, spec):
    if spec['stage'] == 'stage1':
        return [directory / f'predictions_{MODEL_NAMES[spec["model"]]}_{task}.jsonl' for task in TASKS]
    return [directory / 'predictions.jsonl']


def prediction_rows(directory, spec, growing=False):
    rows = []
    unknown = set(directory.glob('predictions*.jsonl')) - set(prediction_files(directory, spec))
    if unknown:
        raise ValueError(f'Unexpected prediction files: {sorted(map(str, unknown))}')
    for path in prediction_files(directory, spec):
        for row in read_rows(path, growing):
            if spec['stage'] == 'stage1' and path.name != f'predictions_{MODEL_NAMES[spec["model"]]}_{row["task_type"]}.jsonl':
                raise ValueError(f'Wrong task file: {path}')
            rows.append(row)
    return rows


def row_key(row, stage):
    return (int(row['statement_idx']), row['task_type'] if stage == 'stage1' else 'obqa_answer')


def validate_rows(rows, spec, complete=False):
    keys = [row_key(r, spec['stage']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate prediction keys')
    expected = {tuple(k) for k in spec['expected_keys']}
    if not set(keys) <= expected:
        raise ValueError('Unexpected prediction keys')
    if complete and set(keys) != expected:
        raise ValueError(f'Missing predictions: {len(expected - set(keys))}')
    for row in rows:
        if row.get('model_revision') != spec['model_revision'] or row.get('model_id') != spec['model_id']:
            raise ValueError('Prediction model/revision mismatch')
        if not isinstance(row.get('raw_response'), str):
            raise ValueError('Missing raw response')
    return keys


def validate_frozen_files(root, spec):
    for name, expected in {**spec['input_files'], **spec.get('code_files', {})}.items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Missing/changed input: {path}')


def tripwire(rows, stage, reference, reading_probe=False):
    if len(rows) < 20:
        return None
    first = rows[:20]
    if not reading_probe and stage == 'qa' and sum(parse_answer(r['raw_response'], 4)[0] == 'FAILED' for r in first) / 20 > MAX_PARSE_FAILURE:
        return 'more than 50% QA parse failures in first 20'
    if not reading_probe and stage == 'stage1' and sum(not r['raw_response'].strip() or r.get('generated_token_count', 0) <= 1 for r in first) / 20 > .8:
        return 'more than 80% empty/one-token Stage 1 responses in first 20'
    times = [r.get('item_elapsed_seconds') for r in first]
    if all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times) and sum(times) / 20 > reference * 3:
        return 'first 20 items more than 3x slower than reference'
    return None


def verify_job(directory, spec, complete=True):
    rows = prediction_rows(directory, spec)
    validate_rows(rows, spec, complete)
    config_path = directory / 'run_config.json'
    if not config_path.exists():
        raise ValueError(f'Missing run_config: {directory}')
    config = json.loads(config_path.read_text())
    if config.get('fullrun') != spec:
        raise ValueError('Fullrun contract differs from current frozen inputs')
    for name in ('effective_max_new_tokens', 'model_id', 'model_revision', 'prompt_bodies_sha256'):
        if config.get(name) != spec[name]:
            raise ValueError(f'Incorrect {name}')
    if config['generation'].get('max_new_tokens') != spec['effective_max_new_tokens'] or config['generation'].get('do_sample') is not False or config['generation'].get('num_beams') != 1:
        raise ValueError('Incorrect generation settings')
    for name, value in spec['versions'].items():
        if config.get(name + '_version') != value:
            raise ValueError('Version mismatch')
    for key in ('input_jsonl', 'graph_metadata'):
        if key in spec and config.get(key, {}).get('sha256') != spec['input_files'][spec[key]]:
            raise ValueError(f'Incorrect input hash: {key}')
    for row in rows:
        for field in ('item_elapsed_seconds', 'peak_memory_allocated_bytes', 'generated_token_count', 'hit_token_ceiling'):
            if field not in row:
                raise ValueError(f'Missing measurement: {field}')
        if (not math.isfinite(row['item_elapsed_seconds']) or row['item_elapsed_seconds'] <= 0
                or not math.isfinite(row['peak_memory_allocated_bytes']) or row['peak_memory_allocated_bytes'] <= 0):
            raise ValueError('Invalid measured time/VRAM')
        if spec['stage'] == 'qa':
            parsed, tier = parse_answer(row['raw_response'], 4)
            if row.get('parse_tier') != tier or row.get('predicted_option') != (None if parsed == 'FAILED' else parsed):
                raise ValueError('QA stored parse differs from shared scorer')
    failures = sum(parse_answer(r['raw_response'], 4)[0] == 'FAILED' for r in rows) if spec['stage'] == 'qa' else 0
    ceiling = sum(bool(r['hit_token_ceiling']) for r in rows)
    reading_probe = spec['label'].startswith('probe_')
    wire = tripwire(sorted(rows, key=lambda r: r.get('timestamp_utc', '')), spec['stage'], spec['seconds_per_item_reference'], reading_probe=reading_probe)
    if wire:
        raise ValueError('TRIPWIRE: ' + wire)
    if not reading_probe and rows and (failures / len(rows) > MAX_PARSE_FAILURE or ceiling / len(rows) > MAX_CEILING):
        raise ValueError(f'Parse/ceiling sanity threshold exceeded: FAILED={failures}, ceiling={ceiling}/{len(rows)}')
    return {'job': spec['label'], 'rows': len(rows), 'expected': spec['record_count'], 'FAILED': failures,
            'ceilings': ceiling, 'parse_tiers': dict(Counter(r.get('parse_tier') for r in rows)) if spec['stage'] == 'qa' else {},
            'seconds_per_item': sum(r['item_elapsed_seconds'] for r in rows) / len(rows) if rows else None,
            'peak_vram_bytes': max((r['peak_memory_allocated_bytes'] for r in rows), default=0),
            'legibility_confounded': config.get('legibility_confounded', False)}
