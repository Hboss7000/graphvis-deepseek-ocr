#!/usr/bin/env python3
"""Sequential real evaluators with strict provenance, watchdogs and resumable status."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from fullrun_common import *


def auxiliary_specs(root, model, smoke, data_name=DATA_NAME):
    stage = specs(root, model, smoke, data_name)[0]
    if model == 'gemma':
        result = copy.deepcopy(stage)
        rows = [r for r in ordered_stage(read_rows(root / stage['input_jsonl'])) if r['task_type'] == 'triple_listing'][:5]
        result.update(label='preflight', record_count=5, prompt_bodies_sha256=stage_hash(rows),
                      expected_keys=[[r['statement_idx'], r['task_type']] for r in rows])
        return [result]
    if model == 'llava' and smoke:
        manifest = json.loads((root / 'outputs' / data_name / 'inputs_manifest.json').read_text())
        rows = ordered_stage(read_rows(root / manifest['probe_input']))
        result = []
        for style in ('uniform', 'legacy'):
            s = copy.deepcopy(stage)
            s.update(label='probe_' + style, input_jsonl=manifest['probe_input'], record_count=20,
                     expected_keys=[[r['statement_idx'], r['task_type']] for r in rows],
                     prompt_bodies_sha256=stage_hash(rows))
            if style == 'legacy':
                s['image_root'] = manifest['legacy_image_root']
            result.append(s)
        return result
    return []


def command(root, workspace, spec, directory, preflight_report=None, limit=None):
    model = spec['model']
    py = workspace / 'venvs' / ('venv_ocr2' if model == 'deepseek' else 'venv_qwen') / 'bin/python'
    if spec['stage'] == 'stage1':
        runner = 'scripts/eval_gemma3_stage1.py' if model == 'gemma' else f'{S1}/run_stage1_{model}.py'
    else:
        runner = 'scripts/eval_gemma3_stage2.py' if model == 'gemma' else f'{QA}/run_zero_shot{"" if model == "deepseek" else "_" + model}.py'
    count = len(read_rows(root / spec['input_jsonl']))
    cmd = [str(py), '-u', str(root / runner), '--model-id', spec['model_id'], '--revision', spec['model_revision'],
           '--input-jsonl', str(root / spec['input_jsonl']), '--graph-metadata', str(root / spec['graph_metadata']),
           '--image-root', str(root / spec['image_root']), '--expected-count', str(count),
           '--max-new-tokens', str(spec['effective_max_new_tokens']), '--resume']
    if spec['stage'] == 'stage1':
        cmd += ['--output-dir', str(directory), '--task-set', 'extended', '--extractor', 'span_extended',
                '--answer-format', 'none', '--expected-split', 'test', '--approve-prompts']
        if spec['label'].startswith('probe_'):
            cmd += ['--behavior-only']
        if spec['label'] == 'preflight':
            cmd += ['--preflight', '5', '--preflight-min-recall', '0.95']
    else:
        cmd += ['--output-jsonl', str(directory / 'predictions.jsonl'), '--condition', spec['condition'], '--approve-prompt-diff']
        if model == 'deepseek':
            cmd += ['--infer-output-dir', str(directory / 'infer')]
        if limit is not None:
            cmd += ['--limit', str(limit)]
    if model == 'qwen':
        cmd += ['--min-pixels', '262144', '--max-pixels', '1310720']
    if spec['stage'] == 'stage1' or model in ('llava', 'gemma'):
        cmd += ['--seed', '13']
    if model == 'gemma':
        cmd += ['--attn-impl', 'eager', '--pan-and-scan', '--pan-and-scan-min-crop-size', '256',
                '--pan-and-scan-max-num-crops', '4', '--pan-and-scan-min-ratio-to-activate', '1.2',
                '--cache-implementation', 'dynamic', '--batch-size', '1']
        if preflight_report:
            cmd += ['--preflight-report', str(preflight_report), '--allow-failed-preflight']
    return cmd


def validate_inputs(root, spec):
    validate_frozen_files(root, spec)
    rows = read_rows(root / spec['input_jsonl'])
    if spec['label'] == 'preflight':
        rows = [r for r in ordered_stage(rows) if r['task_type'] == 'triple_listing'][:5]
    if len(rows) != spec['record_count']:
        raise ValueError('Unexpected input record count')
    if {row_key(r, spec['stage']) for r in rows} != {tuple(k) for k in spec['expected_keys']}:
        raise ValueError('Unexpected input keys')


def hardware_preflight(workspace, model, versions):
    if shutil.disk_usage(workspace).free < 5_000_000_000:
        raise ValueError('Less than 5 GB free on workspace volume')
    py = workspace / 'venvs' / ('venv_ocr2' if model == 'deepseek' else 'venv_qwen') / 'bin/python'
    code = '''import importlib, json, sys, torch, transformers
from PIL import Image
expected=json.loads(sys.argv[1])
assert torch.__version__ == expected['torch'], ('torch', torch.__version__)
assert transformers.__version__ == expected['transformers'], ('transformers', transformers.__version__)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible CUDA GPU required'
assert 'RTX PRO 6000' in torch.cuda.get_device_name(0), 'Expected the approved RTX PRO 6000 setup'
for name in json.loads(sys.argv[2]): importlib.import_module(name)
print('COMPLETED imports/GPU/version preflight: ' + torch.cuda.get_device_name(0), flush=True)
'''
    imports = ['accelerate', 'safetensors', 'sentencepiece']
    if model == 'deepseek':
        imports += ['einops', 'addict', 'easydict']
    subprocess.run([str(py), '-c', code, json.dumps(versions), json.dumps(imports)], check=True, timeout=90)
    model_id, revision = MODELS[model]
    snapshot = workspace / '.cache/huggingface/hub' / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
    if not (snapshot / 'config.json').is_file():
        raise ValueError(f'Missing pinned cached config: {snapshot}')
    indices = list(snapshot.glob('*.safetensors.index.json'))
    weights = list(snapshot.glob('*.safetensors'))
    if indices:
        for index in indices:
            for shard in set(json.loads(index.read_text())['weight_map'].values()):
                if not (snapshot / shard).is_file() or (snapshot / shard).stat().st_size == 0:
                    raise ValueError(f'Missing cached weight shard: {snapshot / shard}')
    elif not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError(f'Missing pinned cached weights: {snapshot}')
    if not (snapshot / 'tokenizer_config.json').is_file():
        raise ValueError(f'Missing cached tokenizer: {snapshot}')
    if model == 'deepseek' and not list(snapshot.glob('modeling*.py')):
        raise ValueError('Missing pinned DeepSeek remote code')


def check_resume(directory, spec):
    if directory.exists() and any(directory.iterdir()) and not (directory / 'contract.json').exists():
        raise ValueError(f'Output collision, no fullrun contract: {directory}')
    if (directory / 'contract.json').exists() and json.loads((directory / 'contract.json').read_text()) != spec:
        raise ValueError(f'Incompatible resume contract: {directory}')
    targets = [directory / s for s in ('pan_and_scan', 'no_pan_and_scan')] if spec['label'] == 'preflight' else [directory]
    for target in targets:
        rows = prediction_rows(target, spec)
        validate_rows(rows, spec)
        if rows or (target / 'run_config.json').exists():
            # Reject config drift before weights load; completeness is checked after the job.
            verify_job(target, spec, complete=False)


def stop_process(process):
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            break
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass


def watch_job(cmd, directory, spec, log, deadline, poll_seconds=1, stall_seconds=300):
    """Watch flushed JSONL bytes, not console chatter; kill the whole job process group."""
    env = dict(os.environ, FULLRUN_CONTRACT=str(directory / 'contract.json'))
    print('Launch: ' + shlex.join(cmd), flush=True)
    print(f'Monitor: tail -F {shlex.quote(str(log))}\nGPU: watch -n 30 nvidia-smi', flush=True)
    targets = [directory / s for s in ('pan_and_scan', 'no_pan_and_scan')] if spec['label'] == 'preflight' else [directory]
    paths = [p for target in targets for p in prediction_files(target, spec)]
    sizes = {p: p.stat().st_size if p.exists() else 0 for p in paths}
    last_growth = time.monotonic()
    started = last_growth
    reason, state = None, 'FAILED'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', buffering=1) as handle:
        handle.write('\nLaunch: ' + shlex.join(cmd) + '\n')
        process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    state, reason = 'PAUSED', 'MAX_HOURS reached'
                    stop_process(process)
                    break
                current = {p: p.stat().st_size if p.exists() else 0 for p in paths}
                if any(current[p] > sizes[p] for p in paths):
                    last_growth = now
                sizes = current
                for target in targets:
                    rows = prediction_rows(target, spec, growing=process.poll() is None)
                    validate_rows(rows, spec)
                    rows.sort(key=lambda r: r.get('timestamp_utc', ''))
                    reason = tripwire(rows, spec['stage'], spec['seconds_per_item_reference'])
                    if reason:
                        break
                if not reason and now - last_growth >= stall_seconds:
                    reason = 'prediction files not growing for 5 minutes'
                if reason:
                    state = 'TRIPWIRE'
                    stop_process(process)
                    break
                code = process.poll()
                if code is not None:
                    # A failed Gemma recall threshold is recorded, but D6 permits continuing.
                    if spec['label'] == 'preflight' and (directory / 'preflight_report.json').exists():
                        for target in targets:
                            verify_job(target, spec)
                        state = 'COMPLETED'
                        reason = 'Gemma readability result recorded; failed recall gate is nonblocking under D6'
                    elif code == 0:
                        state = 'COMPLETED'
                    else:
                        state, reason = 'FAILED', f'exit code {code}'
                    break
                time.sleep(poll_seconds)
        except BaseException:
            stop_process(process)
            raise
    return {'state': state, 'reason': reason, 'wall_seconds': time.monotonic() - started,
            'exit_code': process.returncode, 'command': cmd}


def record_status(path, status):
    previous = json.loads(path.read_text()) if path.exists() else {'attempts': []}
    previous['attempts'].append(dict(status, timestamp=time.time()))
    previous['state'] = status['state']
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(previous, indent=2) + '\n')
    temp.replace(path)
    print(f'{status["state"]}: {path.stem}: {status.get("reason") or "job finished"}. STOP the Pod when your session ends; billing continues.', flush=True)


def run_model(args, model, deadline):
    jobs = specs(args.root, model, args.smoke)
    auxiliary = auxiliary_specs(args.root, model, args.smoke)
    sequence = auxiliary + jobs if model == 'gemma' else jobs + auxiliary
    mode = 'smoke' if args.smoke else 'results'
    base = args.root / 'outputs' / (DATA_NAME + '_' + mode) / model
    log_root = args.workspace / 'logs' / (DATA_NAME + '_' + mode) / model
    if args.plan:
        for spec in sequence:
            for limit in ([3, 5] if args.smoke and spec['label'] == 'qa_image' else [None]):
                print(shlex.join(command(args.root, args.workspace, spec, base / spec['label'], base / 'preflight/preflight_report.json' if model == 'gemma' and spec['label'] != 'preflight' else None, limit)))
        return True
    if base.exists() and any(p.name not in ('.lock', 'session.json') for p in base.iterdir()) and not (base / 'session.json').exists():
        raise ValueError(f'Model output collision: {base}')
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        frozen_json(base / 'session.json', {'model': model, 'mode': mode, 'data_name': DATA_NAME})
        # ALL planned inputs and existing output states are checked before any model loads.
        eligible = []
        failed = False
        for spec in sequence:
            try:
                validate_inputs(args.root, spec)
                check_resume(base / spec['label'], spec)
                eligible.append(spec)
            except Exception as exc:
                record_status(base / 'status' / (spec['label'] + '.json'),
                              {'state': 'FAILED', 'reason': 'preflight: ' + str(exc)})
                failed = True
        if not eligible:
            return False
        try:
            hardware_preflight(args.workspace, model, jobs[0]['versions'])
        except Exception as exc:
            for spec in eligible:
                record_status(base / 'status' / (spec['label'] + '.json'),
                              {'state': 'FAILED', 'reason': 'hardware preflight: ' + str(exc)})
            return False
        for spec in eligible:
            if time.monotonic() >= deadline:
                print('PAUSED - resume by rerunning the same command', flush=True)
                return False
            directory = base / spec['label']
            frozen_json(directory / 'contract.json', spec)
            status_file = base / 'status' / (spec['label'] + '.json')
            report = base / 'preflight/preflight_report.json' if model == 'gemma' and spec['label'] != 'preflight' else None
            try:
                if status_file.exists() and json.loads(status_file.read_text())['state'] == 'COMPLETED':
                    if spec['label'] == 'preflight':
                        for crop in ('pan_and_scan', 'no_pan_and_scan'):
                            verify_job(directory / crop, spec)
                        if not (directory / 'preflight_report.json').is_file():
                            raise ValueError('Missing Gemma preflight report')
                        print(f'COMPLETED: verified existing {model}/preflight', flush=True)
                        continue
                    else:
                        verify_job(directory, spec)
                        print(f'COMPLETED: verified existing {model}/{spec["label"]}', flush=True)
                        continue
                limits = [None]
                if args.smoke and spec['label'] == 'qa_image':
                    limits = [3, 5]
                status = None
                for limit in limits:
                    if limit == 3 and len(prediction_rows(directory, spec)) > 3:
                        # The first half was already proven in a previous interrupted invocation.
                        if not (directory / 'resume_first3.json').exists():
                            raise ValueError('Five-row smoke output without three-row resume evidence')
                        continue
                    cmd = command(args.root, args.workspace, spec, directory, report, limit)
                    status = watch_job(cmd, directory, spec, log_root / (spec['label'] + '.log'), deadline)
                    if status['state'] != 'COMPLETED':
                        break
                    if limit == 3:
                        rows = prediction_rows(directory, spec)
                        validate_rows(rows, spec)
                        expected3 = {tuple(k) for k in spec['expected_keys'][:3]}
                        if {row_key(r, spec['stage']) for r in rows} != expected3:
                            raise ValueError('Resume rehearsal first leg must contain exactly first three rows')
                        frozen_json(directory / 'resume_first3.json', {'rows': rows, 'command': cmd})
                    elif spec['label'] != 'preflight':
                        verify_job(directory, spec)
                        if limit == 5:
                            first = json.loads((directory / 'resume_first3.json').read_text())['rows']
                            all_rows = prediction_rows(directory, spec)
                            if all_rows[:3] != first:
                                raise ValueError('Resume modified the original three predictions')
                            frozen_json(directory / 'resume_verified.json', {'first_count': 3, 'final_count': 5,
                                        'first3_sha256': sha(directory / 'resume_first3.json'),
                                        'predictions_sha256': sha(directory / 'predictions.jsonl')})
                record_status(status_file, status)
                if status['state'] == 'PAUSED':
                    print('PAUSED - resume by rerunning the same command', flush=True)
                    return False
                failed |= status['state'] != 'COMPLETED'
            except Exception as exc:
                record_status(status_file, {'state': 'FAILED', 'reason': str(exc)})
                failed = True
        return not failed
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=(*MODELS, 'all'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--workspace', type=Path, default=Path('/workspace'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--plan', action='store_true', help='CPU-only command preview; no GPU imports or launches')
    p.add_argument('--max-hours', type=float, default=4)
    args = p.parse_args()
    args.root = args.root.resolve()
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        p.error('--max-hours must be finite and positive')
    if args.model == 'all' and not args.smoke:
        p.error('Full runs require one model per invocation')
    if not args.plan:
        if not os.environ.get('TMUX'):
            raise ValueError('Long jobs must run inside tmux')
        if not args.root.is_relative_to(args.workspace.resolve()):
            raise ValueError('Repository and every output must be on the workspace volume')
        for name in ('STAGE1_MATRIX_CELL_JSON', 'STAGE1_FROZEN_PROMPT_POLICY_JSON'):
            if os.environ.get(name):
                raise ValueError(f'Unexpected inherited experiment override: {name}')
        os.environ.update(HF_HOME=str(args.workspace / '.cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        gpu_lock = (args.workspace / '.fullrun_gpu.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.smoke:
            from fullrun_summary import rehearsal_report
            gate = rehearsal_report(args.root)
            if not gate['passed']:
                raise ValueError('Full run requires complete four-model rehearsal and reading-probe gate')
    deadline = time.monotonic() + args.max_hours * 3600
    ok = True
    for model in MODELS if args.model == 'all' else [args.model]:
        try:
            ok = run_model(args, model, deadline) and ok
        except Exception as exc:
            print(f'FAILED {model} preflight: {exc}', flush=True)
            ok = False
        if time.monotonic() >= deadline:
            break
    if not args.plan:
        if args.smoke and args.model == 'all' and time.monotonic() < deadline:
            from fullrun_summary import rehearsal_report, print_report
            report = rehearsal_report(args.root)
            print_report(report)
            ok = ok and report['passed']
        print('Session ended. STOP the Pod; MAX_HOURS does not stop billing.', flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
