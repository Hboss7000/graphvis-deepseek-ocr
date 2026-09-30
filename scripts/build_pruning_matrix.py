#!/usr/bin/env python3
"""Build paired pruning datasets; launch or execute manifest-indexed Slurm cells.

--dry-run builds real CPU datasets and prints every command, without submission.
--manifest existing.jsonl --dry-run only validates/prints an existing matrix.
GPU inference always dispatches to the existing per-model evaluators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = Path('experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts')
STAGE2 = Path('experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts')
sys.path.insert(0, str(ROOT / STAGE1))
from score_stage1 import EXTENDED_TASK_TYPES, read_jsonl

PRUNING_DEFAULTS = dict(max_nodes=18, max_edges=60, max_degree=0, bridge_rule='qa-bridge', lifelines=True)
CONDITIONS = {
    'baseline': {}, 'nodes40': {'max_nodes': 40},
    'nodes40_edges_off': {'max_nodes': 40, 'max_edges': 0},
    'core_neighbor': {'bridge_rule': 'core-neighbor'},
    'caps_off': {'max_nodes': 0, 'max_edges': 0},
    'unpruned': {'max_nodes': 0, 'max_edges': 0, 'bridge_rule': 'any'},
    'legacy_18_30_5': {'max_nodes': 18, 'max_edges': 30, 'max_degree': 5},
}
FIXED = dict(split='test', start=0, seed=13, stage1_task_set='extended', stage1_balance='per-task',
             engine='dot', dpi=200, rankdir='LR', node_fontsize=18, edge_fontsize=14,
             nodesep=0.5, ranksep=0.7, disconnected_rows=3,
             hide_relatedto_labels=False, reveal_correct_answer=False)
MODELS = {
    'deepseek': dict(evaluator=str(STAGE1 / 'run_stage1_deepseek.py'),
        model_id='deepseek-ai/DeepSeek-OCR-2', revision='aaa02f3811945a91062062994c5c4a3f4c0af2b0',
        python='/storage/home/hleonel/venv_ocr2/bin/python', transformers='4.46.3',
        max_new_tokens=8192, attention='eager', base_size=1024, image_size=768, crop_mode=True),
    'qwen': dict(evaluator=str(STAGE1 / 'run_stage1_qwen.py'),
        model_id='Qwen/Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        python='/storage/home/hleonel/venv_qwen/bin/python', transformers='5.16.1',
        max_new_tokens=1024, attention='sdpa', min_pixels=262144, max_pixels=1310720),
    'gemma3': dict(evaluator='scripts/eval_gemma3_stage1.py',
        model_id='google/gemma-3-12b-it', revision='96b6f1eccf38110c56df3a15bffe176da04bfd80',
        python='/storage/home/hleonel/venv_qwen/bin/python', transformers='5.16.1',
        max_new_tokens=1024, attention='eager', cache_implementation='dynamic', pan_and_scan=True),
}
CONTAINER = '/storage/home/hleonel/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif'
HF_CACHE = '/storage/home/hleonel/.cache/huggingface'
CROP_KEYS = ('pan_and_scan_min_crop_size', 'pan_and_scan_max_num_crops', 'pan_and_scan_min_ratio_to_activate')
SOURCE_FILES = [Path('scripts') / name for name in (
    'build_pruning_matrix.py', 'generate_graphvis_datasets.py', 'gemma3_common.py',
    'eval_gemma3_stage1.py', 'pruning_stats.py', 'image_audit.py')]
SOURCE_FILES += [STAGE1 / name for name in ('score_stage1.py', 'stage1_common.py',
                                          'run_stage1_deepseek.py', 'run_stage1_qwen.py')]
SOURCE_FILES += [STAGE2 / name for name in ('prompt_common.py', 'run_zero_shot_qwen.py')]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def condition_settings(raw):
    result = {}
    for name, overrides in raw.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
            raise ValueError(f'Unsafe condition name: {name}')
        if not isinstance(overrides, dict) or set(overrides) - PRUNING_DEFAULTS.keys():
            raise ValueError(f'{name}: conditions may override ONLY pruning flags: {overrides}')
        settings = {**PRUNING_DEFAULTS, **overrides}
        for key in ('max_nodes', 'max_edges', 'max_degree'):
            if type(settings[key]) is not int or settings[key] < 0:
                raise ValueError(f'{name}: invalid {key}')
        if settings['bridge_rule'] not in ('qa-bridge', 'core-neighbor', 'any', 'none'):
            raise ValueError(f'{name}: invalid bridge_rule')
        if type(settings['lifelines']) is not bool:
            raise ValueError(f'{name}: lifelines must be boolean')
        result[name] = settings
    if 'baseline' in result and result['baseline'] != PRUNING_DEFAULTS:
        raise ValueError('baseline must use the current default pruning configuration')
    return result


def cli_flags(settings):
    flags = []
    for key, value in settings.items():
        if key == 'lifelines':
            if not value:
                flags.append('--no-lifelines')
        elif isinstance(value, bool):
            if value:
                flags.append('--' + key.replace('_', '-'))
        else:
            flags += ['--' + key.replace('_', '-'), str(value)]
    return flags


def print_command(command):
    print(shlex.join(map(str, command)), flush=True)


def check_call(command, **kwargs):
    print_command(command)
    timeout = kwargs.pop('timeout', None)
    with subprocess.Popen(list(map(str, command)), start_new_session=True, **kwargs) as process:
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def image_digest(metadata, image_root):
    return sha_object([(r['statement_idx'], r['image'], sha_file(Path(image_root) / r['image']))
                       for r in sorted(metadata, key=lambda r: r['statement_idx'])])


def validate_cell_data(cell):
    records = read_jsonl(Path(cell['input_jsonl']))
    metadata = read_jsonl(Path(cell['graph_metadata']))
    keys = {(r['statement_idx'], r['task_type']) for r in records}
    if not records or len(records) != cell['expected_count'] or len(keys) != len(records):
        raise ValueError(f"{cell['cell_id']}: expected_count or unique keys differ")
    if len(metadata) != cell['generation']['limit']:
        raise ValueError('Metadata count differs from declared slice')
    if {r['statement_idx'] for r in metadata} != set(range(cell['generation']['limit'])):
        raise ValueError(f"{cell['cell_id']}: statement slice differs")
    for row in metadata:
        if {k: row['pruning'][k] for k in PRUNING_DEFAULTS} != cell['pruning_flags']:
            raise ValueError(f"{cell['cell_id']}: metadata pruning differs from manifest")
    for key, path in cell['diagnostics'].items():
        if sha_file(path) != cell['diagnostics_sha256'][key]:
            raise ValueError(f'Changed diagnostic: {path}')
    for key in ('input_jsonl', 'graph_metadata'):
        if sha_file(cell[key]) != cell[key + '_sha256']:
            raise ValueError(f"{cell['cell_id']}: changed {key}")
    if image_digest(metadata, cell['image_root']) != cell['images_sha256']:
        raise ValueError(f"{cell['cell_id']}: images changed")
    return records, metadata


def load_manifest(path, check_sources=False):
    cells = read_jsonl(Path(path))
    if not cells:
        raise ValueError('Empty matrix manifest')
    seen = set()
    reference = cells[0]
    per_model, per_condition = {}, {}
    for cell in cells:
        if cell['cell_id'] in seen:
            raise ValueError('Duplicate matrix cell')
        seen.add(cell['cell_id'])
        if cell['cell_id'] != cell['model'] + '__' + cell['condition']:
            raise ValueError('Cell identity does not match model/condition')
        if cell['contract_sha256'] != sha_object({k: v for k, v in cell.items() if k != 'contract_sha256'}):
            raise ValueError(f"Manifest contract changed: {cell['cell_id']}")
        if cell['generation'] != reference['generation'] or cell['source_sha256'] != reference['source_sha256']:
            raise ValueError('Rendering, slice, seed or source code drift between conditions')
        if {k: cell['generation'][k] for k in FIXED} != FIXED:
            raise ValueError('Matrix must use the declared fixed generation settings')
        model = cell['model_settings']
        if any(model.get(k) != v for k, v in MODELS[cell['model']].items()):
            raise ValueError('Model/environment settings differ from the verified setup')
        condition_settings({cell['condition']: cell['pruning_flags']})
        if model != per_model.setdefault(cell['model'], model):
            raise ValueError('Model settings drift between conditions')
        settings = {k: cell[k] for k in ('input_jsonl', 'graph_metadata', 'image_root', 'expected_count',
                                        'pruning_flags', 'input_jsonl_sha256', 'graph_metadata_sha256',
                                        'images_sha256', 'diagnostics', 'diagnostics_sha256')}
        if settings != per_condition.setdefault(cell['condition'], settings):
            raise ValueError('Models received different condition data')
    if len(seen) != len(per_model) * len(per_condition):
        raise ValueError('Manifest must contain the full selected model × condition product')
    if check_sources:
        for file, digest in reference['source_sha256'].items():
            if sha_file(ROOT / file) != digest:
                raise ValueError(f'Source changed since build: {file}; build a new matrix')
    return cells


def evaluator_command(cell, preflight=False, preflight_dir=None, report=None):
    model = cell['model_settings']
    command = [model['python'], model['evaluator'], '--model-id', model['model_id'],
               '--revision', model['revision'], '--input-jsonl', cell['input_jsonl'],
               '--graph-metadata', cell['graph_metadata'], '--image-root', cell['image_root'],
               '--output-dir', str(preflight_dir if preflight else cell['output_dir']),
               '--expected-count', str(cell['expected_count']), '--task-set', 'extended',
               '--approve-prompts', '--resume', '--seed', '13']
    if cell['model'] == 'qwen':
        command += cli_flags({k: model[k] for k in ('min_pixels', 'max_pixels', 'max_new_tokens')})
    if cell['model'] == 'gemma3':
        command += ['--attn-impl', model['attention'], '--pan-and-scan', '--cache-implementation',
                    model['cache_implementation'], '--max-new-tokens', str(model['max_new_tokens'])]
        command += cli_flags({k: model[k] if model[k] is not None else '<REQUIRED_' + k.upper() + '>' for k in CROP_KEYS})
        if preflight:
            command += ['--preflight', str(cell['preflight_graphs']), '--preflight-min-recall', '0.95']
        else:
            command += ['--preflight-report', str(report)]
    return command


def container_command(command):
    return ['apptainer', 'exec', '--nv', '--bind', '/storage/nobackup', CONTAINER, *command]


def run_config_fields(cell):
    """Stable fields also used by the evaluator's shared config writer."""
    return dict(pruning_condition=cell['condition'], pruning=cell['pruning_flags'],
                matrix_contract_sha256=cell['contract_sha256'], matrix_generation=cell['generation'],
                graph_metadata_manifest_path_sha256=sha_object(cell['graph_metadata']))


def validate_run_config(cell, config, preflight=False):
    for key, value in run_config_fields(cell).items():
        if config.get(key) != value:
            raise ValueError(f"{cell['cell_id']}: run_config differs in {key}")
    if config.get('condition') != 'image':
        raise ValueError('Run modality must remain image')
    for key in ('input_jsonl', 'graph_metadata'):
        if config.get(key, {}).get('sha256') != cell[key + '_sha256']:
            raise ValueError(f"{cell['cell_id']}: wrong {key} used for scoring")
    model = cell['model_settings']
    if config.get('model_revision') != model['revision'] or config.get('model_id') != model['model_id']:
        raise ValueError('Run model/revision differs from manifest')
    if config.get('graph_metadata', {}).get('path_sha256') != sha_object(config.get('graph_metadata', {}).get('path')):
        raise ValueError('Graph metadata path hash differs')
    if config.get('seed') != 13 or config.get('task_set') != 'extended':
        raise ValueError('Run seed/task set differs from manifest')
    gen = config.get('generation', {})
    if (gen.get('do_sample'), gen.get('num_beams'), gen.get('max_new_tokens')) != (False, 1, model['max_new_tokens']):
        raise ValueError('Run decoding differs from manifest')
    if config.get('attention_implementation') != model['attention']:
        raise ValueError('Run attention differs from manifest')
    if cell['model'] == 'gemma3':
        if (config.get('pan_and_scan_kwargs') != {k: model[k] for k in CROP_KEYS}
                or (not preflight and config.get('pan_and_scan') is not True)
                or config.get('cache_implementation') != model['cache_implementation']):
            raise ValueError('Gemma crop/cache settings differ')
    else:
        keys = ('base_size', 'image_size', 'crop_mode') if cell['model'] == 'deepseek' else ('min_pixels', 'max_pixels')
        if any(config.get('image_processing', {}).get(k) != model[k] for k in keys):
            raise ValueError('Model image budget differs')


def prediction_rows(cell, complete=False):
    records = read_jsonl(Path(cell['input_jsonl']))
    expected = {(r['statement_idx'], r['task_type']) for r in records}
    result = {}
    for task in EXTENDED_TASK_TYPES:
        path = Path(cell['output_dir']) / f"predictions_{cell['model']}_{task}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = (row['statement_idx'], row['task_type'])
            if key in result or key not in expected or key[1] != task:
                raise ValueError(f'Unexpected, duplicated or misplaced prediction: {path}: {key}')
            result[key] = row
    if complete and set(result) != expected:
        raise ValueError(f"{cell['cell_id']}: realised {len(result)} != expected {cell['expected_count']}")
    return result


def require_model_controls(cells):
    for cell in cells:
        if cell['model'] == 'gemma3' and any(
                cell['model_settings'].get(k) is None or cell['model_settings'][k] <= 0 for k in CROP_KEYS):
            raise ValueError('Gemma crop values are unresolved. Supply the validated values with '
                             '--manifest ... --finalize-manifest NEW.jsonl and all three crop flags.')


def run_cell(cell):
    require_model_controls([cell])
    validate_cell_data(cell)
    output = Path(cell['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    # flock is released even on a hard-killed/requeued job; never remove its inode.
    lock = (output / '.matrix.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'Cell is already running: {output}')
    try:
        old_config = output / 'run_config.json'
        if old_config.exists():
            validate_run_config(cell, json.loads(old_config.read_text()))
        elif prediction_rows(cell):
            raise ValueError('Cannot resume predictions without their run_config.json')
        env = os.environ.copy()
        runtime = dict(HF_HOME=HF_CACHE, HF_HUB_CACHE=HF_CACHE + '/hub', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', PYTHONHASHSEED='13', STAGE1_MATRIX_PHASE='inference',
                       STAGE1_MATRIX_CELL_JSON=json.dumps(cell, sort_keys=True))
        for key, value in runtime.items():
            env[key] = value
            env['APPTAINERENV_' + key] = value
        model = cell['model_settings']
        # local_files_only is explicit even though offline mode is also exported.
        check = """import json, pathlib, sys, transformers, torch
from huggingface_hub import snapshot_download
assert transformers.__version__ == sys.argv[3], (transformers.__version__, sys.argv[3])
if sys.argv[1] == 'deepseek-ai/DeepSeek-OCR-2':
    assert torch.__version__ == '2.8.0+cu126', torch.__version__
else:
    assert sys.version_info[:3] == (3, 11, 13), sys.version
p = pathlib.Path(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
index = p / 'model.safetensors.index.json'
weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
assert all((p / f).is_file() and (p / f).stat().st_size for f in weights), 'Incomplete offline weight cache'
print('Offline cache verified:', p)
"""
        check_call(container_command([model['python'], '-c', check, model['model_id'],
                                      model['revision'], model['transformers']]), env=env)
        print('Prompt approval: --approve-prompts is passed explicitly and non-interactively.', flush=True)
        report = None
        if cell['model'] == 'gemma3':
            preflights = output / 'preflight'
            preflights.mkdir(exist_ok=True)
            reports = sorted(preflights.glob('attempt_*/preflight_report.json'))
            if reports:
                report = reports[-1]
                if not json.loads(report.read_text())['settings']['pan_and_scan']['passed']:
                    raise RuntimeError(f'Gemma preflight failed; full inference blocked: {report}')
            else:
                attempts = sorted(preflights.glob('attempt_*'))
                attempt = attempts[-1] if attempts else preflights / 'attempt_0001'
                check_call(container_command(evaluator_command(cell, True, attempt)),
                           env={**env, 'APPTAINERENV_STAGE1_MATRIX_PHASE': 'preflight'})
                report = attempt / 'preflight_report.json'
        check_call(container_command(evaluator_command(cell, report=report)), env=env)
        validate_run_config(cell, json.loads(old_config.read_text()))
        prediction_rows(cell, complete=True)
        if not (output / f"metrics_{cell['model']}.json").is_file():
            raise FileNotFoundError('Evaluator did not write scored metrics')
        write_json(output / 'matrix_complete.json', {'contract_sha256': cell['contract_sha256'],
                                                   'expected_count': cell['expected_count']})
    finally:
        lock.close()


def launch(manifest, cells, concurrency, dry_run):
    command = ['sbatch', '--parsable', f'--array=1-{len(cells)}%{concurrency}',
               'scripts/run_stage1_matrix.sbatch', str(Path(manifest).resolve()), str(ROOT),
               str(Path(sys.executable).absolute())]
    print_command(command)
    if dry_run:
        job = '<jobid>'
        for idx, cell in enumerate(cells, 1):
            print(f"Array task {idx}: {cell['cell_id']} ({cell['expected_count']} items)")
            report = Path(cell['output_dir']) / 'preflight/attempt_0001/preflight_report.json'
            if cell['model'] == 'gemma3':
                print_command(container_command(evaluator_command(cell, True, report.parent)))
            print_command(container_command(evaluator_command(cell, report=report)))
    else:
        require_model_controls(cells)
        for cell in cells:
            validate_cell_data(cell)
        Path('/storage/home/hleonel/outputs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(f'Unexpected sbatch job ID: {job}')
        print('Submitted array', job)
    print('squeue -u "$USER"')
    print(f'scontrol show job {job}_<arrayidx>  # sacct is unavailable on COMA')
    print(f'tail -f ~/outputs/slurm-prunmat-{job}_<arrayidx>.out')
    print(f'{shlex.quote(sys.executable)} scripts/build_pruning_matrix.py --manifest '
          f'{shlex.quote(str(manifest))} --progress')


def build(args):
    import generate_graphvis_datasets as gen
    from image_audit import audit_split
    from pruning_stats import summarize
    raw = json.loads(args.conditions_file.read_text()) if args.conditions_file else CONDITIONS
    settings = condition_settings(raw)
    names = args.conditions.split(',') if args.conditions else list(settings)
    if not names or len(set(names)) != len(names) or set(names) - settings.keys():
        raise ValueError(f'Invalid condition subset: {names}')
    # Always retain the declaration order, independent of CLI comma order.
    names = [name for name in settings if name in names]
    if args.limit <= 0:
        raise ValueError('--limit must be positive')
    source_statements = read_jsonl(args.data_root / 'obqa/statement/test.statement.jsonl')
    if args.limit > len(source_statements):
        raise ValueError('Requested paired slice exceeds available statements')
    for path in [args.out_root / 'manifest.jsonl', *(args.out_root / n for n in names),
                 *(args.results_root / m / n for m in MODELS for n in names)]:
        if path.exists():
            raise FileExistsError(f'Matrix output collision: {path}; choose fresh output roots')
    crops = {k: getattr(args, k) for k in CROP_KEYS}
    if any(value is not None and value <= 0 for value in crops.values()):
        raise ValueError('Gemma crop controls must be positive')
    if any(value is None for value in crops.values()):
        print('Gemma crop controls are unresolved: this manifest cannot be submitted until --finalize-manifest.', flush=True)
    generation = {**FIXED, 'limit': args.limit, 'prompt_policy': args.prompt_policy, 'min_label_px': args.min_label_px}
    sources = {str(p): sha_file(ROOT / p) for p in SOURCE_FILES}
    args.out_root.mkdir(parents=True, exist_ok=True)
    cells = []
    for name in names:
        output = args.out_root / name
        command = [sys.executable, 'scripts/generate_graphvis_datasets.py', '--data-root', str(args.data_root),
                   '--out-dir', str(output), *cli_flags({**FIXED, 'limit': args.limit}),
                   *cli_flags(settings[name])]
        check_call(command, env={**os.environ, 'PYTHONHASHSEED': '13'}, timeout=args.generation_timeout)
        split = output / 'test'
        input_path = split / f'stage1_graph_comprehension_0_{args.limit}.jsonl'
        meta_path = split / f'graph_metadata_0_{args.limit}.jsonl'
        records, metadata = read_jsonl(input_path), read_jsonl(meta_path)
        if args.prompt_policy == 'fixed-template':
            normalize_prompts(records, gen)
            gen.write_jsonl(input_path, records)
        stats = summarize(metadata)
        expected_bucket = json.dumps(settings[name], sort_keys=True)
        if stats['pruning_configurations'] != {expected_bucket: args.limit}:
            raise ValueError(f'{name}: expected exactly one matching pruning configuration bucket')
        write_json(output / 'pruning_stats.json', stats)
        audit = audit_split(split, output, node_fontsize=FIXED['node_fontsize'], dpi=FIXED['dpi'],
                             min_label_px=args.min_label_px)
        write_json(output / 'image_audit.json', audit)
        diagnostics = {'pruning_stats': str(output / 'pruning_stats.json'),
                       'image_audit': str(output / 'image_audit.json')}
        write_json(output / 'build_config.json', {'generation': generation, 'pruning': settings[name],
                                                'generator_command': command, 'source_sha256': sources})
        for model, defaults in MODELS.items():
            model_settings = {**defaults, **(crops if model == 'gemma3' else {})}
            cell = dict(cell_id=f'{model}__{name}', model=model, condition=name,
                        pruning_flags=settings[name], generation=generation, model_settings=model_settings,
                        source_sha256=sources, input_jsonl=str(input_path), graph_metadata=str(meta_path),
                        image_root=str(output), expected_count=len(records),
                        output_dir=str(args.results_root / model / name), diagnostics=diagnostics,
                        diagnostics_sha256={k: sha_file(p) for k, p in diagnostics.items()},
                        input_jsonl_sha256=sha_file(input_path), graph_metadata_sha256=sha_file(meta_path),
                        images_sha256=image_digest(metadata, output),
                        preflight_graphs=min(50, sum(r['task_type'] == 'triple_listing' for r in records)))
            cell['contract_sha256'] = sha_object(cell)
            validate_cell_data(cell)
            cells.append(cell)
    check_prompts(cells, args.out_root / 'prompt_comparison.json')
    manifest = args.out_root / 'manifest.jsonl'
    with manifest.open('x') as handle:
        for cell in cells:
            handle.write(json.dumps(cell, sort_keys=True) + '\n')
    return manifest, load_manifest(manifest, check_sources=True)


def normalize_prompts(records, gen):
    """Choose existing template text by task/index, independent of graph RNG use.

    Graph-specific target slots are extracted from structured gold/the original
    generated degree prompt, then retained. Neither templates nor gold change.
    """
    templates = {
        'node_description': gen.NODE_DESCRIPTION_PROMPTS, 'node_number': gen.NODE_NUMBER_PROMPTS,
        'edge_number': gen.EDGE_NUMBER_PROMPTS, 'triple_listing': gen.TRIPLE_LISTING_PROMPTS,
        'highest_node_degree': gen.HIGHEST_DEGREE_PROMPTS, 'node_degree': gen.NODE_DEGREE_PROMPTS,
        'relation_identification': gen.RELATION_IDENTIFICATION_PROMPTS,
        'neighbor_listing': gen.NEIGHBOR_LISTING_PROMPTS, 'shortest_path_listing': gen.SHORTEST_PATH_PROMPTS,
    }
    for row in records:
        task = row['task_type']
        choices = templates[task]
        slot = int(sha_object([13, row['statement_idx'], task]), 16) % len(choices)
        fields = dict(row.get('gold', {}))
        if task == 'node_degree':
            for template in choices:
                pattern = re.escape(template).replace(re.escape('{node}'), '(?P<node>.+)')
                match = re.fullmatch(pattern, row['prompt'])
                if match:
                    fields['node'] = match['node']
                    break
            else:
                raise ValueError('Generated node-degree prompt does not match any existing template')
        row['prompt'] = choices[slot].format(**fields)


def check_prompts(cells, output):
    by_condition = {cell['condition']: read_jsonl(Path(cell['input_jsonl'])) for cell in cells}
    indexed = {c: {(r['statement_idx'], r['task_type']): r for r in rows} for c, rows in by_condition.items()}
    paired = set.intersection(*(set(rows) for rows in indexed.values()))
    changed = [dict(statement_idx=k[0], task_type=k[1], prompts={c: rows[k]['prompt'] for c, rows in indexed.items()})
               for k in sorted(paired) if len({rows[k]['prompt'] for rows in indexed.values()}) > 1]
    write_json(output, {'paired_keys': len(paired), 'different_prompt_count': len(changed), 'differences': changed})
    if cells[0]['generation']['prompt_policy'] == 'identical' and changed:
        raise ValueError(f'Prompt drift in {len(changed)} paired keys; inspect {output}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, default=Path('outputs/pruning_matrix'))
    parser.add_argument('--results-root', type=Path, default=Path('outputs/stage1_matrix'))
    parser.add_argument('--data-root', type=Path, default=Path('data_preprocessed_release'))
    parser.add_argument('--conditions')
    parser.add_argument('--conditions-file', type=Path)
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--min-label-px', type=float, default=10)
    parser.add_argument('--generation-timeout', type=float, help='Optional seconds per condition; fail, never skip')
    parser.add_argument('--prompt-policy', choices=('identical', 'fixed-template'), default='fixed-template')
    for key in CROP_KEYS:
        parser.add_argument('--' + key.replace('_', '-'), type=float if 'ratio' in key else int,
                            default=os.environ.get(key.upper()))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--finalize-manifest', type=Path, help='Write a new manifest with explicit crop controls; never overwrite')
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--run-cell', type=int, help='1-based manifest index; used by sbatch')
    parser.add_argument('--progress', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    if args.concurrency < 1 or args.min_label_px <= 0:
        raise ValueError('Concurrency and label threshold must be positive')
    if args.dry_run and (args.submit or args.run_cell):
        raise ValueError('--dry-run cannot submit or execute inference')
    if args.manifest:
        manifest, cells = args.manifest, load_manifest(args.manifest, check_sources=not args.progress)
    else:
        if args.run_cell or args.progress:
            raise ValueError('--manifest is required')
        manifest, cells = build(args)
    if args.finalize_manifest:
        if not args.manifest:
            raise ValueError('--finalize-manifest requires --manifest')
        if any(Path(cell['output_dir']).exists() for cell in cells):
            raise FileExistsError('Cannot change a matrix contract after result directories exist')
        for cell in cells:
            if cell['model'] == 'gemma3':
                for key in CROP_KEYS:
                    value = getattr(args, key)
                    if value is not None:
                        cell['model_settings'][key] = value
            cell.pop('contract_sha256')
            cell['contract_sha256'] = sha_object(cell)
        require_model_controls(cells)
        with args.finalize_manifest.open('x') as handle:
            for cell in cells:
                handle.write(json.dumps(cell, sort_keys=True) + '\n')
        manifest = args.finalize_manifest
    if args.progress:
        for cell in cells:
            print(f"{cell['cell_id']}: {len(prediction_rows(cell))}/{cell['expected_count']}")
    elif args.run_cell is not None:
        if not 1 <= args.run_cell <= len(cells):
            raise ValueError('Array index outside manifest')
        run_cell(cells[args.run_cell - 1])
    elif args.dry_run or args.submit:
        launch(manifest, cells, args.concurrency, args.dry_run)
    else:
        print(f'Built {manifest}; use --manifest {manifest} --dry-run or --submit.')


if __name__ == '__main__':
    main()
