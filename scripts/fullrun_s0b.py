#!/usr/bin/env python3
"""Targeted smoke recovery: immutable S0 reuse, failed jobs, and three 50-graph arms."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time
from fullrun_common import *
from fullrun_session import auxiliary_specs, run_model
from report_fullrun_probes import paired_report, write_report

INPUTS = 'outputs/fullrun_s0b_inputs_2026-10-04'


def probe_specs(root):
    path = root / INPUTS / 'manifest.json'
    manifest = json.loads(path.read_text())
    rows = ordered_stage(read_rows(root / manifest['records']))
    if len(rows) != 50 or any(r['task_type'] != 'node_description' for r in rows):
        raise ValueError('The new reading probe must contain all 50 node_description records')
    original = read_rows(root / 'outputs' / DATA_NAME / 'test/stage1_subset100.jsonl')
    original50 = json.loads((root / 'subsets/obqa_test_subset50_seed13.json').read_text())['indices']
    expected = ordered_stage([r for r in original if r['task_type'] == 'node_description' and r['statement_idx'] in original50])
    if rows != expected:
        raise ValueError('Probe does not preserve the original 50 prompts and gold records')
    result = []
    for arm, images in manifest['arms'].items():
        spec = copy.deepcopy(specs(root, 'llava', True)[0])
        spec.update(label='probe_' + arm + '50', input_jsonl=manifest['records'], record_count=50,
                    expected_keys=[[r['statement_idx'], r['task_type']] for r in rows],
                    prompt_bodies_sha256=stage_hash(rows), image_root=images)
        spec['input_files'].update(manifest['files'])
        spec['input_files'][str(path.relative_to(root))] = sha(path)
        spec['data_provenance'] = {'original': spec['data_provenance'], 'reading_arm': arm,
                                   'probe_manifest': str(path.relative_to(root)), 'node_fill': '#FFFFFF' if arm == 'white' else None}
        result.append(spec)
    return result


def historical_spec(directory, current):
    """Reuse evidence only if every experimental field matches; code hashes stay historical."""
    old = json.loads((directory / 'contract.json').read_text())
    if {k:v for k,v in old.items() if k != 'code_files'} != {k:v for k,v in current.items() if k != 'code_files'}:
        raise ValueError(f'Historical experimental contract mismatch: {directory}')
    return old


def recovery_jobs(root):
    return {'llava': probe_specs(root), 'qwen': specs(root, 'qwen', True)[1:],
            'gemma': specs(root, 'gemma', True)[1:], 'deepseek': specs(root, 'deepseek', True)}


def reused(root, source):
    entries = []
    for model in ('llava', 'qwen', 'gemma'):
        for current in specs(root, model, True) if model == 'llava' else specs(root, model, True)[:1]:
            directory = source / model / current['label']
            old = historical_spec(directory, current)
            validate_frozen_files(root, dict(old, code_files={}))
            verify_job(directory, old)
            state = json.loads((source / model / 'status' / (current['label'] + '.json')).read_text())['state']
            if state != 'COMPLETED':
                raise ValueError(f'Cannot reuse incomplete job: {directory}')
            entries.append({'model': model, 'label': current['label'], 'directory': str(directory.relative_to(root)),
                            'files': {str(p.relative_to(root)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()},
                            'historical_code_files': old['code_files'], 'current_code_files': current['code_files']})
    preflight, = auxiliary_specs(root, 'gemma', True)
    directory = source / 'gemma/preflight'
    old = historical_spec(directory, preflight)
    for crop in ('pan_and_scan', 'no_pan_and_scan'):
        verify_job(directory / crop, old)
    report = json.loads((directory / 'preflight_report.json').read_text())
    if 'pan_and_scan' not in report['settings']:
        raise ValueError('Incomplete Gemma preflight report')
    entries.append({'model':'gemma','label':'preflight','directory':str(directory.relative_to(root)),
                    'files':{str(p.relative_to(root)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    return entries


def proof(directory):
    evidence = json.loads((directory / 'resume_verified.json').read_text())
    if evidence['first_count'] != 3 or evidence['final_count'] != 5:
        raise ValueError('Wrong resume counts')
    for key, name in [('first3_sha256','resume_first3.json'), ('predictions_sha256','predictions.jsonl')]:
        if evidence[key] != sha(directory / name):
            raise ValueError('Resume proof hash mismatch')


def gate(root, source):
    from fullrun_gate import model_gate
    reports={m:model_gate(root,m,smoke_root=source) for m in MODELS}
    errors=[m+': '+e for m,r in reports.items() for e in r['errors']]
    return {'models':reports,'errors':errors,'probe':reports['llava']['probe'],
            'passed':not errors,'gate_scope':'per model; aggregate is a report only, never a launch prerequisite'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--workspace',type=Path,default=Path('/workspace'))
    p.add_argument('--source-smoke-root',type=Path)
    p.add_argument('--max-hours',type=float,default=1.5)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--gate-only',action='store_true')
    args=p.parse_args();args.root=args.root.resolve()
    source=(args.source_smoke_root or args.root/'outputs'/(DATA_NAME+'_smoke')).resolve()
    if not source.is_relative_to(args.root):
        p.error('Source smoke evidence must be under --root')
    if not math.isfinite(args.max_hours) or args.max_hours<=0:
        p.error('--max-hours must be finite and positive')
    args.smoke=True;args.recovery_jobs=recovery_jobs(args.root)
    args.gemma_preflight=source/'gemma/preflight/preflight_report.json'
    entries=reused(args.root,source)
    print('REUSE (read-only): '+', '.join(e['model']+'/'+e['label'] for e in entries),flush=True)
    base=args.root/'outputs'/(DATA_NAME+'_s0b')
    if not args.plan:
        if not args.gate_only:
            if not os.environ.get('TMUX') or not args.root.is_relative_to(args.workspace.resolve()):
                raise ValueError('Run inside tmux with repository and outputs under /workspace')
            for name in ('STAGE1_MATRIX_CELL_JSON','STAGE1_FROZEN_PROMPT_POLICY_JSON'):
                if os.environ.get(name): raise ValueError('Unexpected inherited experiment override: '+name)
            os.environ.update(HF_HOME=str(args.workspace/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            lock=(args.workspace/'.fullrun_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            frozen_json(base/'source.json',{'source_smoke_root':str(source.relative_to(args.root))})
            frozen_json(base/'reuse_manifest.json',entries)
    deadline=time.monotonic()+args.max_hours*3600
    ok=True
    if not args.gate_only:
        for model in ('qwen','gemma','deepseek','llava'):
            try: ok=run_model(args,model,deadline) and ok
            except Exception as exc:
                print(f'FAILED {model}: {exc}. STOP the Pod when session ends.',flush=True);ok=False
    if args.plan:return
    result=gate(args.root,source)
    from fullrun_summary import print_report
    print_report(result)
    if result['probe']:
        target=base/('probe_report_'+str(time.time_ns()))
        write_report(result['probe'],target)
        print('Per-graph three-arm report: '+str(target),flush=True)
    path=base/('gate_'+str(time.time_ns())+'.json');frozen_json(path,result)
    print('STOP the Pod. No full run is launched by session 0b.',flush=True)
    raise SystemExit(0 if ok and result['passed'] else 1)


if __name__=='__main__':main()
