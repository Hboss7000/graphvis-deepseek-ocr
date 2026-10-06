#!/usr/bin/env python3
"""Find the largest batch dividing 16 with a real 8-example/2-step dry run.

Each candidate uses a fresh process and the eight longest audited records.
Only CUDA OOM is classified as insufficient capacity; any other failure stops.
This is a paid GPU job, requiring the same approval/guard/tmux rules as training.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=('llava', 'qwen'), default='llava')
    p.add_argument('--token-diagnostics', type=Path)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--lora-lr', type=float, default=2e-5)
    p.add_argument('--projector-lr', type=float, default=2e-5)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh batch-probe output directory')
    args.output_dir.mkdir(parents=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for batch in (16, 8, 4, 2, 1):
        output = args.output_dir / f'batch-{batch}'
        log_path = args.log_dir / f'{args.backbone}-batch-{batch}.log'
        command = [sys.executable, str(Path(__file__).with_name(f'train_{args.backbone}_stage1.py')),
                   '--mode', 'dry', '--data-dir', str(args.data_dir), '--output-dir', str(output),
                   '--per-device-batch-size', str(batch), '--lora-lr', str(args.lora_lr),
                   '--projector-lr', str(args.projector_lr)]
        if args.token_diagnostics:
            command += ['--token-diagnostics', str(args.token_diagnostics)]
        print('DRY CANDIDATE: ' + json.dumps(command), flush=True)
        start = time.perf_counter()
        with log_path.open('x') as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        attempt = {'batch': batch, 'exit_code': result.returncode,
                   'wall_seconds': time.perf_counter() - start, 'log': str(log_path)}
        attempts.append(attempt)
        if result.returncode == 0:
            report = json.loads((output / 'smoke_report.json').read_text())
            if not report.get('passed'):
                raise ValueError('Dry-run verification did not pass')
            report.update({'selected_batch': batch, 'batch_probe_attempts': attempts,
                           'selection': 'largest of 16/8/4/2/1 passing two steps on the eight longest records'})
            (args.output_dir / 'batch_probe_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            print(json.dumps(report, indent=2), flush=True)
            print('BATCH PROBE COMPLETED: review speed/VRAM/cost; STOP the Pod when done.', flush=True)
            return
        text = log_path.read_text()
        if 'CUDA out of memory' not in text:
            print('FAILED: STOP the Pod. Fix and repeat the dry run.', flush=True)
            raise RuntimeError(f'Non-OOM dry failure; inspect {log_path}')
        print(f'CUDA OOM at batch {batch}; capacity trial failed. STOP the Pod if not continuing trials.', flush=True)
    raise RuntimeError('No batch fits; STOP the Pod')


if __name__ == '__main__':
    main()
