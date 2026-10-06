#!/usr/bin/env python3
"""Single-GPU pinned HF+PEFT Stage 1 training, dry run, smoke and exact resume."""
import argparse
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION, configure_processor, prepare_inputs
from llava_stage1_training import (Stage1Collator, attach_lora, file_sha, load_checkpoint,
                                   save_checkpoint, training_text, projector)
from llava_training_logging import TrainingLogger
from stage1_storage import InsufficientCheckpointSpace, retain_checkpoints

ROOT = Path(__file__).resolve().parents[1]


def optional_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def code_hashes(backbone='llava'):
    return {name: file_sha(ROOT / 'scripts' / name) for name in
            ('train_llava_stage1.py', 'llava_stage1_training.py', 'llava_stage1_prompt.py',
             'llava_common.py', 'llava_adapter.py', 'llava_training_logging.py', 'stage1_storage.py',
             *(() if backbone == 'llava' else ('qwen_stage1_training.py', 'train_qwen_stage1.py')))} | ({} if backbone == 'llava' else {
        'run_zero_shot_qwen.py': file_sha(ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot_qwen.py')})


def optimizer_and_scheduler(model, lr, projector_lr, steps):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    lora, projection = [], []
    bridge_ids = {id(p) for p in projector(model).parameters()}
    for name, value in model.named_parameters():
        if value.requires_grad:
            (projection if id(value) in bridge_ids else lora).append(value)
    if not lora or not projection:
        raise ValueError('Expected both LoRA and projector trainable parameters')
    optimizer = torch.optim.AdamW([
        {'params': lora, 'lr': lr, 'name': 'lora'},
        {'params': projection, 'lr': projector_lr, 'name': 'projector'}],
        weight_decay=0., betas=(.9, .999), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps * .03), steps)
    return optimizer, scheduler


def move_batch(batch, device, dtype):
    return {name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for name, value in batch.items()}


def validation(model, records, collator, device, dtype, batch_size):
    import torch
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    totals = {}
    with torch.no_grad():
        # Grouping by task makes per-task loss cheap and exactly token-weighted.
        for task in sorted({r['task_type'] for r in records}):
            subset = [r for r in records if r['task_type'] == task]
            loss_sum, tokens = 0., 0
            for start in range(0, len(subset), batch_size):
                batch = move_batch(collator(subset[start:start + batch_size]), device, dtype)
                count = int((batch['labels'][:, 1:] != -100).sum())
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite validation loss')
                loss_sum += loss.item() * count
                tokens += count
            totals[task] = (loss_sum, tokens)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    metrics = {'validation_loss': sum(v[0] for v in totals.values()) / sum(v[1] for v in totals.values()),
               'validation_seconds': time.perf_counter() - started,
               **{f'validation_{task}_loss': value[0] / value[1] for task, value in totals.items()}}
    model.train(was_training)
    return metrics


def train_steps(model, optimizer, scheduler, records, validation_records, collator,
                output_dir, config, logger, device, dtype, resume=None, stop_after_step=None):
    """Shared real training loop; CPU fixtures inject a tiny HF model into this loop."""
    import torch
    output_dir = Path(output_dir)
    config_path = output_dir / 'run_config.json'
    if config_path.exists():
        if not resume or json.loads(config_path.read_text()) != config:
            raise ValueError('Training run configuration differs or --resume is absent')
    else:
        if resume:
            raise ValueError('Resume output is missing its run configuration')
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    step, cursor = load_checkpoint(model, optimizer, scheduler, resume, config) if resume else (0, 0)
    target = min(config['optimizer_steps'], stop_after_step or config['optimizer_steps'])
    if step > target:
        raise ValueError('Checkpoint step is beyond requested stop step')
    saved_at = time.perf_counter()
    started = saved_at
    checkpoint = Path(resume) if resume else None
    all_metrics = []
    checkpoint_seconds = 0.
    validation_seconds = 0.
    model.train()
    # A crash during validation must retain the preceding optimizer update.
    # Its checkpoint may therefore need that validation replayed on resume.
    if (resume and step > 0 and (step % 100 == 0 or step == config['optimizer_steps'])
            and not logger.has_event('validation', step)):
        replay = validation(model, validation_records, collator, device, dtype,
                            config['per_device_batch_size'])
        logger.log('validation', step, replay)
        validation_seconds += replay['validation_seconds']
    while step < target:
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        n = config['global_batch_size']
        if config['mode'] == 'full':
            n = min(n, len(records) - cursor)
        if n <= 0:
            raise ValueError('Training data exhausted before expected step count')
        selected = [records[(cursor + i) % len(records)] for i in range(n)]
        microbatches = [selected[i:i + config['per_device_batch_size']]
                        for i in range(0, n, config['per_device_batch_size'])]
        losses, tokens = [], 0
        # Standard mean microbatch-loss accumulation, as in the released Trainer recipe.
        for subset in microbatches:
            batch = move_batch(collator(subset), device, dtype)
            tokens += int(batch['attention_mask'].sum())
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite loss at optimizer step {step + 1}')
            losses.append(loss.detach().item())
            (loss / len(microbatches)).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1., error_if_nonfinite=True)
        rates = [group['lr'] for group in optimizer.param_groups]
        optimizer.step()
        scheduler.step()
        cursor += n
        step += 1
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        metrics = {'loss': sum(losses) / len(losses), 'lr': rates[0], 'projector_lr': rates[1],
                   'grad_norm': float(norm), 'tokens_per_second': tokens / elapsed,
                   'examples_per_second': n / elapsed, 'step_seconds': elapsed,
                   'peak_vram_bytes': int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0,
                   'examples': n, 'tokens': tokens}
        logger.log('train', step, metrics)
        all_metrics.append(metrics)
        print(json.dumps({'step': step, **metrics}), flush=True)
        if step % 100 == 0 or step == target or time.perf_counter() - saved_at >= 1200:
            checkpoint = output_dir / f'checkpoint-{step:06d}'
            save_started = time.perf_counter()
            digest = save_checkpoint(model, optimizer, scheduler, checkpoint, config, step, cursor)
            logger.checkpoint(checkpoint, digest)
            retain_checkpoints(output_dir, checkpoint, config, keep=config.get('checkpoint_keep', 2))
            checkpoint_seconds += time.perf_counter() - save_started
            saved_at = time.perf_counter()
        if step % 100 == 0 or step == config['optimizer_steps']:
            result = validation(model, validation_records, collator, device, dtype,
                                config['per_device_batch_size'])
            logger.log('validation', step, result)
            validation_seconds += result['validation_seconds']
            print(json.dumps({'step': step, **result}), flush=True)
    return {'step': step, 'data_cursor': cursor, 'checkpoint': str(checkpoint),
            'elapsed_seconds': time.perf_counter() - started, 'metrics': all_metrics,
            'checkpoint_seconds': checkpoint_seconds, 'validation_seconds': validation_seconds,
            'complete': step == config['optimizer_steps']}


def load_records(data_dir, mode, audit_path=None, revision=DEFAULT_REVISION, template='llava_v1'):
    manifest_path = data_dir / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_sha(data_dir / name) != digest:
            raise ValueError(f'Data manifest mismatch: {name}')
    audit_path = audit_path or data_dir / 'token_diagnostics.json'
    audit = json.loads(audit_path.read_text())
    if (audit['data_manifest_sha256'] != file_sha(manifest_path) or audit['model_max_length'] != 4096
            or audit['revision'] != revision or audit['template'] != template
            or audit['overlength_examples']):
        raise ValueError('Missing/incompatible zero-overlength audit')
    result = []
    for name in ('training', 'validation'):
        subset = manifest['subsets'][name]
        path = data_dir / subset['records']
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if {r['task_type'] for r in records} != set(manifest['tasks']):
            raise ValueError('Unexpected or missing training tasks')
        rng = random.Random(13)
        rng.shuffle(records)
        if mode != 'full':
            if mode == 'dry' and name == 'training':
                lengths = {(r['statement_idx'], r['task_type']): r['total_tokens']
                           for r in audit['subsets']['training']['items']}
                records.sort(key=lambda r: -lengths[(r['statement_idx'], r['task_type'])])
            records = records[:8 if mode == 'dry' else 64]
        result.append(records)
    return *result, manifest, audit


def parse_args(backbone='llava'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--token-diagnostics', type=Path, help='Qwen audit of the reused manifests')
    p.add_argument('--mode', choices=('dry', 'smoke', 'full'), required=True)
    p.add_argument('--per-device-batch-size', type=int, default=1)
    p.add_argument('--lora-lr', type=float, default=2e-5)
    p.add_argument('--projector-lr', type=float, default=2e-5)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after-step', type=int)
    p.add_argument('--smoke-report', type=Path, help='Required successful real-model smoke evidence for full training')
    return p.parse_args()


def main(backbone='llava'):
    args = parse_args(backbone)
    if backbone not in ('llava', 'qwen'):
        raise ValueError('Unknown backbone')
    model_id, revision, template = DEFAULT_MODEL_ID, DEFAULT_REVISION, 'llava_v1'
    collator_class, attach, text = Stage1Collator, attach_lora, training_text
    if backbone == 'qwen':
        import qwen_stage1_training as qwen
        model_id, revision, template = qwen.DEFAULT_MODEL_ID, qwen.DEFAULT_REVISION, qwen.TEMPLATE
        collator_class, attach, text = qwen.Stage1Collator, qwen.attach_lora, qwen.training_text
        if args.token_diagnostics is None or (args.lora_lr, args.projector_lr) != (2e-5, 2e-5):
            raise ValueError('Qwen requires its audited native lengths and the fixed 2e-5 learning rates')
    import torch
    import transformers
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor, Qwen3VLForConditionalGeneration
    model_class = LlavaNextForConditionalGeneration if backbone == 'llava' else Qwen3VLForConditionalGeneration
    audit_path = args.token_diagnostics or args.data_dir / 'token_diagnostics.json'
    from llava_adapter import load_adapter
    workspace = Path(os.environ.get('WORKSPACE', '/workspace')).resolve()
    for path in (args.data_dir, args.output_dir, audit_path, Path(os.environ.get('HF_HOME', ''))):
        if not path.resolve().is_relative_to(workspace):
            raise ValueError('Pod inputs/outputs/cache must be under WORKSPACE')
    if not os.environ.get('TMUX'):
        raise ValueError('Launch long pod jobs inside tmux')
    if torch.__version__ != '2.8.0+cu128' or transformers.__version__ != '5.16.1':
        raise ValueError('Required torch==2.8.0+cu128 and transformers==5.16.1')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one CUDA GPU is required')
    if 'RTX PRO 6000' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
        raise ValueError('Expected the approved RTX PRO 6000 with bf16 support')
    if args.per_device_batch_size not in (1, 2, 4, 8, 16) or min(args.lora_lr, args.projector_lr) <= 0:
        raise ValueError('Microbatch must divide global batch 16; learning rates must be positive')
    records, val_records, manifest, audit = load_records(args.data_dir, args.mode, audit_path, revision, template)
    if args.mode == 'full':
        if args.smoke_report is None:
            raise ValueError('Full training requires a separately reviewed successful smoke report')
        evidence = json.loads(args.smoke_report.read_text())
        required = {'model_id': model_id, 'model_revision': revision,
                    'data_manifest_sha256': file_sha(args.data_dir / 'data_manifest.json'),
                    'token_diagnostics_sha256': file_sha(audit_path),
                    'per_device_batch_size': args.per_device_batch_size,
                    'lora_lr': args.lora_lr, 'projector_lr': args.projector_lr,
                    'mode': 'smoke', 'dtype': 'bf16', 'step': 20, 'optimizer_steps': 20,
                    'training_examples': 64, 'complete': True, 'resume_verified': True,
                    'adapter_reload_identical': True, 'mlflow_readable': True,
                    'gpu_type': torch.cuda.get_device_name(0), 'code_sha256': code_hashes(backbone)}
        if not evidence.get('passed') or any(evidence.get(k) != v for k, v in required.items()):
            raise ValueError('Smoke evidence does not match this full-run recipe')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a fresh output directory or --resume')
    transformers.set_seed(13)
    torch.backends.cuda.matmul.allow_tf32 = True
    if backbone == 'llava':
        processor = LlavaNextProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
        model_config = LlavaNextConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
        processor_settings = configure_processor(processor, model_config)
        image_processing = 'same pinned inference anyres processor'
    else:
        processor, api = qwen.pinned_processor()
        processor_settings = {'min_pixels': qwen.DEFAULT_MIN_PIXELS, 'max_pixels': qwen.DEFAULT_MAX_PIXELS,
                              'processor_pixel_budget_api': api, 'default_system_prompt_injected': False}
        if (audit.get('min_pixels'), audit.get('max_pixels')) != (qwen.DEFAULT_MIN_PIXELS, qwen.DEFAULT_MAX_PIXELS):
            raise ValueError('Qwen audit image budget differs')
        image_processing = 'same pinned inference dynamic-resolution processor'
    collator = collator_class(processor, args.data_dir)
    load_kwargs = {'attn_implementation': 'sdpa'} if backbone == 'qwen' else {}
    base = model_class.from_pretrained(model_id, revision=revision,
        dtype=torch.bfloat16, local_files_only=True, **load_kwargs).to('cuda')
    model, names = attach(base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    steps = {'dry': 2, 'smoke': 20, 'full': math.ceil(len(records) / 16)}[args.mode]
    config = {'model_id': model_id, 'model_revision': revision, 'template': template,
              'mode': args.mode, 'dtype': 'bf16', 'vision_tower_frozen': True,
              'image_processing': image_processing, 'processor_settings': processor_settings,
              'lora_r': 128, 'lora_alpha': 256, 'lora_dropout': .05, 'lora_target_modules': names,
              'projector_trained': True, 'lora_lr': args.lora_lr, 'projector_lr': args.projector_lr,
              'weight_decay': 0., 'warmup_ratio': .03, 'scheduler': 'cosine', 'max_grad_norm': 1.,
              'adam_betas': [.9, .999], 'adam_epsilon': 1e-8, 'epochs': 1, 'tf32': True,
              'optimizer_steps': steps, 'global_batch_size': 16, 'per_device_batch_size': args.per_device_batch_size,
              'gradient_accumulation_steps': 16 // args.per_device_batch_size, 'model_max_length': 4096,
              'gradient_checkpointing': True, 'seed': 13, 'checkpoint_steps': 100,
              'checkpoint_max_seconds': 1200, 'validation_steps': 100, 'training_examples': len(records),
              'checkpoint_keep': 2, 'checkpoint_keep_final': True, 'checkpoint_free_space_multiplier': 2,
              'validation_examples': len(val_records), 'data_manifest_sha256': file_sha(args.data_dir / 'data_manifest.json'),
              'token_diagnostics_sha256': file_sha(audit_path),
              'subset_manifest_hashes': {k: v['sha256'] for k, v in manifest['subsets'].items()},
              'length_distributions': {k: {n: v[n] for n in ('total_tokens', 'answer_eos_tokens', 'exceed_4096')}
                                       for k, v in audit['subsets'].items()},
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'code_sha256': code_hashes(backbone),
              'gpu_type': torch.cuda.get_device_name(0),
              'versions': {name: __import__(name).__version__ for name in ('torch', 'transformers', 'peft', 'accelerate')},
              'mlflow_version': optional_version('mlflow')}
    if backbone == 'qwen':
        config['merger_bridges'] = ['model.visual.merger', *[f'model.visual.deepstack_merger_list.{i}' for i in range(3)]]
        config['merger_parameter_names'] = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
        config['attention_implementation'] = 'sdpa'
    print('RUN CONFIG: ' + json.dumps(config, sort_keys=True), flush=True)
    optimizer, scheduler = optimizer_and_scheduler(model, args.lora_lr, args.projector_lr, steps)
    resume_step = json.loads((args.resume / 'run_config.json').read_text()) if args.resume else None
    if resume_step is not None and resume_step != config:
        raise ValueError('Resume recipe/code/data changed')
    # Validate checkpoint before touching logger state.
    if args.resume:
        from llava_adapter import verify_checkpoint
        verify_checkpoint(args.resume)
        state = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=False)
        resume_step = state['step']
        del state
    logger = TrainingLogger(args.output_dir, config,
        os.environ.get('MLFLOW_TRACKING_URI', f'file:{workspace}/mlruns'), resume_step=resume_step)
    try:
        result = train_steps(model, optimizer, scheduler, records, val_records, collator, args.output_dir,
                             config, logger, torch.device('cuda'), torch.bfloat16,
                             args.resume, args.stop_after_step)
        report = {**config, **{k: v for k, v in result.items() if k != 'metrics'}}
        if result['complete'] and args.mode in ('dry', 'smoke'):
            # Reload in a fresh model to exercise the exact evaluation loader.
            prompt, _ = text(processor, records[0])
            from PIL import Image
            with Image.open(args.data_dir / records[0]['image']) as image:
                prepared = (prepare_inputs(processor, prompt, image.convert('RGB')) if backbone == 'llava' else
                            qwen.prepare_inputs(processor, prompt, image.convert('RGB'),
                                argparse.Namespace(min_pixels=qwen.DEFAULT_MIN_PIXELS, max_pixels=qwen.DEFAULT_MAX_PIXELS)))
                probe = move_batch(prepared,
                                   torch.device('cuda'), torch.bfloat16)
            model.eval()
            model.config.use_cache = True
            model.config.text_config.use_cache = True
            with torch.inference_mode():
                expected = model.generate(**probe, max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True).cpu()
            del model, base, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
            reloaded = model_class.from_pretrained(model_id, revision=revision,
                dtype=torch.bfloat16, local_files_only=True, **load_kwargs).to('cuda')
            reloaded, artifact = load_adapter(reloaded, result['checkpoint'], model_id, revision)
            with torch.inference_mode():
                actual = reloaded.generate(**probe, max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True).cpu()
            report['adapter_reload_identical'] = torch.equal(expected, actual)
            report['adapter'] = artifact
            with (args.output_dir / 'metrics.csv').open() as handle:
                import csv
                measured = [r for r in csv.DictReader(handle) if r['event'] == 'train']
            seconds = sum(float(r['step_seconds']) for r in measured)
            examples = sum(int(r['examples']) for r in measured)
            report['examples_per_second'] = examples / seconds
            report['seconds_per_optimizer_step'] = seconds / len(measured)
            report['peak_vram_bytes'] = max(int(r['peak_vram_bytes']) for r in measured)
            # Validation/save/loading remain separate from train-step extrapolation.
            report['full_training_compute_hours'] = 7200 / report['examples_per_second'] / 3600
            report['full_training_compute_cost_usd'] = report['full_training_compute_hours'] * 2.09
            report['estimate_limitation'] = 'Train-step compute only; add measured validation/checkpoint/loading overhead and 50% guard margin.'
            report['mlflow_readable'] = logger.readable()
            report['resume_verified'] = bool(args.resume and resume_step > 0)
            report['passed'] = (report['adapter_reload_identical'] and report['mlflow_readable']
                                and (report['resume_verified'] or args.mode == 'dry'))
        logger.close('FINISHED' if result['complete'] else 'KILLED')
        (args.output_dir / 'smoke_report.json' if args.mode != 'full' else args.output_dir / 'training_report.json').write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        if result['complete'] and args.mode in ('dry', 'smoke') and not report['passed']:
            raise RuntimeError('Smoke verification failed; inspect report and CSV')
        print(('COMPLETED' if result['complete'] else 'PAUSED') + ': STOP the Pod when finished reviewing.', flush=True)
    except InsufficientCheckpointSpace as error:
        logger.close('KILLED')
        print(str(error), flush=True)
        (args.output_dir / 'disk_pause.json').write_text(json.dumps({'reason': str(error), 'status': 'PAUSED'}) + '\n')
        raise SystemExit(75)
    except BaseException:
        if not logger.handle.closed:
            logger.close('FAILED')
        print('FAILED: STOP the Pod; fix, then repeat the dry run before any full run.', flush=True)
        raise


if __name__ == '__main__':
    main()
