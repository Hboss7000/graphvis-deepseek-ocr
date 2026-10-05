"""Opt-in fullrun provenance and timing; standalone evaluator defaults are unchanged."""
from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path
import sys
from time import perf_counter

_started = None
_generated_ids = None


def generation_settings(config):
    """Normalize recorded decoding schemas, including the historical QA writers."""
    generation = config.get('generation', config.get('greedy_generation'))
    if generation is not None:
        return dict(generation)
    model = config.get('model_id')
    if model not in ('Qwen/Qwen3-VL-8B-Instruct', 'google/gemma-3-12b-it'):
        raise ValueError('Missing recorded generation settings for unrecognized runner')
    # These two legacy QA writers recorded the cap flat; their infer_one calls
    # explicitly pass do_sample=False and num_beams=1. New writers record all three.
    return {'do_sample': config.get('do_sample', False), 'num_beams': config.get('num_beams', 1),
            'max_new_tokens': config.get('effective_max_new_tokens', config.get('max_new_tokens'))}


def contract():
    path = os.environ.get('FULLRUN_CONTRACT')
    return json.loads(Path(path).read_text()) if path else None


def enrich_config(config):
    spec = contract()
    if spec is None:
        return config
    config = dict(config)
    for name in ('torch', 'transformers'):
        if name + '_version' not in config:
            version = getattr(sys.modules.get(name), '__version__', None)
            if version is None:
                raise ValueError(f'Cannot determine installed {name} version')
            config[name + '_version'] = version
    config['generation'] = generation_settings(config)
    if 'effective_max_new_tokens' not in config:
        config['effective_max_new_tokens'] = config['generation'].get('max_new_tokens')
    for name in ('model_id', 'model_revision', 'effective_max_new_tokens', 'prompt_bodies_sha256'):
        if config.get(name) != spec[name]:
            raise ValueError(f'Fullrun contract mismatch: {name}: {config.get(name)!r} != {spec[name]!r}')
    for name, version in spec['versions'].items():
        if config.get(name + '_version') != version:
            raise ValueError(f'Fullrun version mismatch: {name}')
    if config['generation'].get('do_sample') is not False or config['generation'].get('num_beams') != 1:
        raise ValueError('Fullrun requires greedy decoding')
    if config['generation'].get('max_new_tokens') != spec['effective_max_new_tokens']:
        raise ValueError('Fullrun generation token cap mismatch')
    for key in ('input_jsonl', 'graph_metadata'):
        if config[key]['sha256'] != spec['input_files'][spec[key]]:
            raise ValueError(f'Fullrun input mismatch: {key}')
    if spec.get('stage') == 'qa':
        config['parser_protocol'] = 'explicit_v1'
        if spec.get('mode') == 'full':
            torch = sys.modules.get('torch')
            seed = config.setdefault('seed', spec.get('rescue', {}).get('strict_execution', {}).get('seed', 13))
            random.seed(seed)
            numpy = sys.modules.get('numpy')
            if numpy is not None:
                numpy.random.seed(seed)
            if hasattr(torch, 'manual_seed'):
                torch.manual_seed(seed)
            # Recorded from the actual device, not inferred from a requirements file.
            config['gpu_type'] = torch.cuda.get_device_name(0)
            if 'RTX PRO 6000' not in config['gpu_type']:
                raise ValueError('Full QA requires RTX PRO 6000')
            source = spec.get('rescue', {}).get('strict_execution')
            if source:
                actual = {k:config.get(k) for k in source}
                if actual != source:
                    raise ValueError('Rescue seed/GPU/decoding differs from strict execution')
    contract_path = os.environ.get('FULLRUN_CONTRACT')
    override_path = Path(contract_path).parent / 'tripwire_override.json' if contract_path else None
    if override_path is not None and override_path.exists():
        from fullrun_override import allowed
        if not allowed(spec):
            raise ValueError('Invalid parse override scope')
        config['tripwire_overridden'] = True
    config['fullrun'] = spec
    config.pop('source_image_generation_flags', None)
    config.pop('source_image_generation_flags_provenance', None)
    config['source_image_generation'] = spec['data_provenance']
    config['legibility_confounded'] = bool(config.get('preflight_report', {}).get('legibility_confounded', False)) if config.get('preflight_report') else False
    return config


def begin_item():
    global _started, _generated_ids
    _generated_ids = None
    if not os.environ.get('FULLRUN_CONTRACT'):
        return
    torch = sys.modules['torch']
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _started = perf_counter()


def finish_item(row):
    if not os.environ.get('FULLRUN_CONTRACT'):
        return
    torch = sys.modules['torch']
    torch.cuda.synchronize()
    if contract().get('stage') == 'qa':
        if _generated_ids is None:
            raise ValueError('QA generation did not capture exact token IDs')
        row['generated_token_ids'] = _generated_ids
        attempt = os.environ.get('FULLRUN_CODE_ATTEMPT')
        if attempt:
            row['code_attempt'] = {'path': 'code_attempts/' + Path(attempt).name, 'sha256': hashlib.sha256(Path(attempt).read_bytes()).hexdigest()}
    row['item_elapsed_seconds'] = perf_counter() - _started
    row['peak_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())


def capture_token_ids(ids):
    """Record the actual generate suffix, including special tokens; no retokenization."""
    global _generated_ids
    spec = contract()
    if spec and spec.get('stage') == 'qa':
        _generated_ids = ids.detach().cpu().tolist()
        if _generated_ids and isinstance(_generated_ids[0], list):
            if len(_generated_ids) != 1: raise ValueError('QA must remain batch size one')
            _generated_ids = _generated_ids[0]


def selected_records(records, args):
    path = getattr(args, 'only_indices_json', None)
    if path is None: return records
    indices = json.loads(path.read_text())
    if not isinstance(indices,list) or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Rescue indices must be a unique integer list')
    wanted = set(indices)
    if not wanted <= {int(r['statement_idx']) for r in records}:
        raise ValueError('Rescue selection contains unknown records')
    return [r for r in records if int(r['statement_idx']) in wanted]
