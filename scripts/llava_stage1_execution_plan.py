#!/usr/bin/env python3
"""Laptop-only hours/cost/guard extrapolation from fetched real GPU evidence."""
import argparse
import csv
import json
from pathlib import Path


def plan(probe, smoke=None, evaluation_dry=None):
    if not probe.get('passed'):
        raise ValueError('A successful measured real-model dry report is required')
    if probe.get('dtype') != 'bf16' or 'RTX PRO 6000' not in probe.get('gpu_type', ''):
        raise ValueError('Speed evidence must come from the real bf16 GPU path')
    probe_wall = sum(r['wall_seconds'] for r in probe['batch_probe_attempts'])
    overhead = probe['batch_probe_attempts'][-1]['wall_seconds']
    seconds_per_step = probe['seconds_per_optimizer_step']
    estimates = {'smoke': 20 * seconds_per_step + 2 * overhead}
    notes = ['Smoke estimate uses measured dry optimizer steps plus two complete passing-dry wall times for load/save/reload overhead.']
    if smoke:
        if not smoke.get('passed') or smoke['per_device_batch_size'] != probe['selected_batch']:
            raise ValueError('Smoke does not match selected dry batch')
        matched = ('model_id', 'model_revision', 'data_manifest_sha256', 'token_diagnostics_sha256',
                   'dtype', 'gpu_type', 'lora_lr', 'projector_lr', 'code_sha256')
        if (any(smoke.get(key) != probe.get(key) for key in matched)
                or smoke.get('mode') != 'smoke' or smoke.get('step') != 20
                or not smoke.get('resume_verified')):
            raise ValueError('Smoke recipe/model/code differs from the measured dry run')
        # 450 steps; validation at 100/200/300/400/450. Smoke validates 64 records once.
        validation_seconds = smoke.get('validation_seconds', 0)
        checkpoint_seconds = smoke.get('checkpoint_seconds', 0)
        estimates['train'] = (smoke['full_training_compute_hours'] * 3600
            + 5 * validation_seconds * 600 / smoke['validation_examples'] + 5 * checkpoint_seconds + 2 * overhead)
        notes.append('Training estimate adds five 600-record validations scaled from smoke, checkpoint time and load overhead; wall-time checkpoint saves may add more.')
    if evaluation_dry:
        if not evaluation_dry.get('passed') or evaluation_dry.get('phase') != 'dry':
            raise ValueError('Evaluation cost requires successful same-adapter dry evidence')
        estimates['eval'] = evaluation_dry['full_extrapolated_seconds']
        notes.append('Evaluation estimate scales the same adapter/template/scorer dry predictions to 900 Stage 1 rows per arm and 500 QA rows per condition.')
    return {'measured_probe_wall_seconds': probe_wall, 'jobs': {
        name: {'expected_hours': seconds / 3600, 'estimated_cost_usd': seconds / 3600 * 2.09,
               'guard_hours': seconds / 3600 * 1.5, 'guard_cost_usd': seconds / 3600 * 1.5 * 2.09}
        for name, seconds in estimates.items()}, 'usd_per_hour': 2.09,
        'notes': notes, 'no_pod_mutations': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-report', type=Path, required=True)
    p.add_argument('--smoke-report', type=Path)
    p.add_argument('--evaluation-dry-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    load = lambda path: json.loads(path.read_text()) if path else None
    result = plan(load(args.probe_report), load(args.smoke_report), load(args.evaluation_dry_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
