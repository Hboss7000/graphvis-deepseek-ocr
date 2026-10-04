"""Opt-in fullrun provenance and timing; standalone evaluator defaults are unchanged."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

_started = None


def contract():
    path = os.environ.get('FULLRUN_CONTRACT')
    return json.loads(Path(path).read_text()) if path else None


def enrich_config(config):
    spec = contract()
    if spec is None:
        return config
    config = dict(config)
    for name in ('torch', 'transformers'):
        config.setdefault(name + '_version', sys.modules[name].__version__)
    for name in ('model_id', 'model_revision', 'effective_max_new_tokens', 'prompt_bodies_sha256'):
        if config.get(name) != spec[name]:
            raise ValueError(f'Fullrun contract mismatch: {name}: {config.get(name)!r} != {spec[name]!r}')
    for name, version in spec['versions'].items():
        if config.get(name + '_version') != version:
            raise ValueError(f'Fullrun version mismatch: {name}')
    if config['generation'].get('do_sample') is not False or config['generation'].get('num_beams') != 1:
        raise ValueError('Fullrun requires greedy decoding')
    for key in ('input_jsonl', 'graph_metadata'):
        if config[key]['sha256'] != spec['input_files'][spec[key]]:
            raise ValueError(f'Fullrun input mismatch: {key}')
    config['fullrun'] = spec
    config.pop('source_image_generation_flags', None)
    config.pop('source_image_generation_flags_provenance', None)
    config['source_image_generation'] = spec['data_provenance']
    config['legibility_confounded'] = bool(config.get('preflight_report', {}).get('legibility_confounded', False)) if config.get('preflight_report') else False
    return config


def begin_item():
    global _started
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
    row['item_elapsed_seconds'] = perf_counter() - _started
    row['peak_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
