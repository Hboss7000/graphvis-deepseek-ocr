#!/usr/bin/env python3
"""Guarded real adapter evaluation rehearsal/full run; unchanged shared scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from llava_adapter import adapter_provenance
from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION
from llava_stage1_training import file_sha

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py'
QA = ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot_llava.py'


def jobs(phase, adapter, data, dry_inputs, output, backbone='llava', scope='all'):
    count, qa_count = (9, 8) if phase == 'dry' else (900, 500)
    stage_input = dry_inputs / 'stage1_dry.jsonl' if phase == 'dry' else data / 'test/stage1_subset100.jsonl'
    qa_input = dry_inputs / 'qa_dry.jsonl' if phase == 'dry' else data / 'test/stage2_obqa_0_500.jsonl'
    stage_runner, qa_runner = STAGE, QA
    if backbone == 'qwen':
        stage_runner = Path(str(STAGE).replace('run_stage1_llava.py', 'run_stage1_qwen.py'))
        qa_runner = Path(str(QA).replace('run_zero_shot_llava.py', 'run_zero_shot_qwen.py'))
    common = ['--adapter', str(adapter), '--image-root', str(data), '--graph-metadata',
              str(data / 'test/graph_metadata_0_500.jsonl'), '--prompt-template', 'llava_v1', '--seed', '13']
    result = []
    if backbone == 'qwen':
        from qwen_stage1_training import DEFAULT_REVISION as qwen_revision
        common = ['--adapter', str(adapter), '--image-root', str(data), '--graph-metadata',
                  str(data / 'test/graph_metadata_0_500.jsonl'), '--revision', qwen_revision,
                  '--min-pixels', '262144', '--max-pixels', '1310720']
    arms = [('stage1', 'none'), ('stage1_prefix', 'gold-template')] if backbone == 'llava' else [('stage1', 'none')]
    for name, prefix in (arms if scope != 'qa' else []):
        command = [sys.executable, str(stage_runner), *common, '--input-jsonl', str(stage_input),
                   '--output-dir', str(output / name), '--expected-count', str(count), '--task-set', 'extended',
                   '--extractor', 'span_extended', '--answer-format', 'none', '--max-new-tokens', '1024',
                   '--assistant-prefix-mode', prefix, '--approve-prompts', '--resume']
        if backbone == 'qwen':
            i = command.index('--assistant-prefix-mode')
            del command[i:i+2]
            command += ['--seed', '13']
        result.append((name, command))
    for condition in (('image', 'text_noref', 'text', 'kg_text') if scope != 'stage1' else ()):
        name = 'qa_' + condition
        command = [sys.executable, str(qa_runner), *common, '--input-jsonl', str(qa_input),
                   '--output-jsonl', str(output / name / 'predictions.jsonl'), '--expected-count', str(qa_count),
                   '--condition', condition, '--max-new-tokens', '64', '--approve-prompt-diff', '--resume']
        result.append((name, command))
    return result


def code_hashes(backbone='llava'):
    stage = STAGE if backbone == 'llava' else Path(str(STAGE).replace('run_stage1_llava.py', 'run_stage1_qwen.py'))
    qa = QA if backbone == 'llava' else Path(str(QA).replace('run_zero_shot_llava.py', 'run_zero_shot_qwen.py'))
    return {str(p.relative_to(ROOT)): file_sha(p) for p in (stage, qa,
        ROOT / 'scripts/llava_stage1_training.py',
        *((ROOT / 'scripts/qwen_stage1_training.py',) if backbone == 'qwen' else ()),
        ROOT / 'scripts/llava_adapter.py', ROOT / 'scripts/llava_stage1_prompt.py',
        ROOT / 'scripts/llava_common.py', ROOT / 'scripts/evaluate_llava_stage1_adapter.py',
        ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/score_stage1.py',
        ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/prompt_common.py')}


def summarize(directory, phase, backbone='llava', scope='all'):
    expected = 9 if phase == 'dry' else 900
    tasks = {}
    total_seconds = 0.
    peak = 0
    artifact_hashes = set()
    names = [name for name, _ in jobs(phase, Path('.'), Path('.'), Path('.'), directory, backbone, scope)]
    for name in names:
        folder = directory / name
        config = json.loads((folder / 'run_config.json').read_text())
        artifact_hashes.add(config['adapter']['sha256'])
        rows = [json.loads(line) for path in folder.glob('predictions*.jsonl')
                for line in path.read_text().splitlines() if line.strip()]
        n = expected if name.startswith('stage1') else (8 if phase == 'dry' else 500)
        keys = [(r['statement_idx'], r.get('task_type', name)) for r in rows]
        if len(rows) != n or len(set(keys)) != n:
            raise ValueError(f'Evaluation coverage differs: {name}')
        mean = sum(r['item_elapsed_seconds'] for r in rows) / n
        loading = json.loads((folder / 'runtime_metrics.json').read_text())['loading_elapsed_seconds']
        full_n = 900 if name.startswith('stage1') else 500
        extrapolated = mean * full_n + loading
        total_seconds += extrapolated
        peak = max(peak, max(r['peak_memory_allocated_bytes'] for r in rows))
        tasks[name] = {'n': n, 'mean_item_seconds': mean, 'loading_seconds': loading,
                       'full_extrapolated_seconds': extrapolated,
                       'ceiling_count': sum(r['hit_token_ceiling'] for r in rows)}
    if len(artifact_hashes) != 1:
        raise ValueError('Evaluation jobs used different adapters')
    return {'passed': True, 'phase': phase, 'backbone': backbone, 'scope': scope, 'adapter_sha256': artifact_hashes.pop(),
            'jobs': tasks, 'full_extrapolated_seconds': total_seconds,
            'estimated_hours': total_seconds / 3600, 'estimated_cost_usd': total_seconds / 3600 * 2.09,
            'guard_hours': total_seconds / 3600 * 1.5, 'peak_vram_bytes': peak,
            'limitation': 'One graph per Stage 1 arm and eight QA examples per condition; response lengths may vary in the full run.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=('llava', 'qwen'), default='llava')
    p.add_argument('--scope', choices=('all', 'stage1', 'qa'), default='all')
    p.add_argument('--phase', choices=('dry', 'full'), required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--dry-inputs', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--dry-report', type=Path)
    p.add_argument('--print-only', action='store_true')
    args = p.parse_args()
    commands = jobs(args.phase, args.adapter, args.data_dir, args.dry_inputs, args.output_dir, args.backbone, args.scope)
    if args.print_only:
        for name, command in commands:
            print(name + ': ' + json.dumps(command))
        return
    if not os.environ.get('TMUX'):
        raise ValueError('Evaluation runs inside tmux')
    manifest = json.loads((args.dry_inputs / 'manifest.json').read_text())
    for name, digest in manifest['source_files_sha256'].items():
        if file_sha(args.data_dir / name) != digest:
            raise ValueError('Evaluation full input changed')
    for name, digest in manifest['dry_files_sha256'].items():
        if file_sha(args.dry_inputs / name) != digest:
            raise ValueError('Evaluation dry input changed')
    model_id, revision = DEFAULT_MODEL_ID, DEFAULT_REVISION
    if args.backbone == 'qwen':
        from qwen_stage1_training import DEFAULT_MODEL_ID as model_id, DEFAULT_REVISION as revision
    artifact = adapter_provenance(args.adapter, model_id, revision)
    if args.phase == 'full':
        if not args.dry_report:
            raise ValueError('Full evaluation requires the same-adapter rehearsal report')
        dry = json.loads(args.dry_report.read_text())
        if (not dry.get('passed') or dry.get('phase') != 'dry'
                or dry['adapter_sha256'] != artifact['sha256'] or dry['code_sha256'] != code_hashes(args.backbone)
                or dry.get('inputs') != manifest or dry.get('backbone', 'llava') != args.backbone
                or dry.get('scope', 'all') != args.scope):
            raise ValueError('Rehearsal adapter/code mismatch; fix then repeat the dry evaluation')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for name, command in commands:
        print('LAUNCH: ' + json.dumps(command), flush=True)
        with (args.log_dir / (args.phase + '_' + name + '.log')).open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f'FAILED {name}: STOP the Pod; inspect its log.', flush=True)
            raise RuntimeError(f'Evaluation failed: {name}')
        print(f'COMPLETED {name}: STOP the Pod if pausing the session.', flush=True)
    report = {**summarize(args.output_dir, args.phase, args.backbone, args.scope), 'code_sha256': code_hashes(args.backbone), 'inputs': manifest}
    (args.output_dir / 'evaluation_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print('EVALUATION COMPLETED: STOP the Pod.', flush=True)


if __name__ == '__main__':
    main()
