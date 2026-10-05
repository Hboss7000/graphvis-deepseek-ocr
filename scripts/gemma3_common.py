"""Shared stock-Transformers Gemma 3 inference and legibility gate.

Crop controls are checked against the *installed* slow image processor API.
Reference: https://huggingface.co/docs/transformers/model_doc/gemma3
No Qwen dynamic-resolution control has a Gemma equivalent.
"""
from __future__ import annotations

from fullrun_runtime import capture_token_ids

import argparse
import inspect
import json
import re
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE2_SCRIPTS = ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'
STAGE1_SCRIPTS = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
for directory in (STAGE2_SCRIPTS, STAGE1_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from prompt_common import sha256_file

DEFAULT_MODEL_ID = 'google/gemma-3-12b-it'
CROP_KEYS = ('pan_and_scan_min_crop_size', 'pan_and_scan_max_num_crops',
             'pan_and_scan_min_ratio_to_activate')
RESOLUTION_NOTE = (
    'Resolution control is not comparable across backbones. Gemma uses a fixed '
    '896x896 encoder plus optional crops; Qwen min_pixels/max_pixels control '
    'dynamic-resolution tokenisation and have no mapping to these controls. '
    'No shared flag equalises vision-token budgets across Gemma, Qwen and '
    'DeepSeek-OCR-2. Match realised image_soft_tokens post hoc.'
)


def add_common_args(parser):
    parser.add_argument('--attn-impl', choices=('eager', 'sdpa'), default='eager')
    parser.add_argument('--pan-and-scan', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pan-and-scan-min-crop-size', type=int)
    parser.add_argument('--pan-and-scan-max-num-crops', type=int)
    parser.add_argument('--pan-and-scan-min-ratio-to-activate', type=float)
    parser.add_argument('--cache-implementation',
                        choices=('dynamic', 'static'), default='dynamic',
                        help="'static' enables torch.compile; with variable-length "
                             'prompts it recompiles repeatedly and breaks run-to-run '
                             'determinism, so the default is dynamic.')
    parser.add_argument('--batch-size', type=int, choices=(1,), default=1)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--preflight-report', type=Path,
                        help='Passed Stage 1 legibility report, required for full image inference')
    parser.add_argument('--allow-failed-preflight', action='store_true',
                        help='Record a failed readability gate as legibility-confounded (fullrun D6).')


def validate_args(args):
    if not args.revision or not re.fullmatch(r'[0-9a-fA-F]{40}', args.revision):
        raise ValueError('--revision must be an explicit 40-character Hub commit hash')
    if re.search(r'gemma-3-1b(?:-|$)', args.model_id):
        raise ValueError('Gemma 3 1B is text-only and out of scope')
    if args.max_new_tokens <= 0 or args.expected_count <= 0:
        raise ValueError('--max-new-tokens and --expected-count must be positive')
    for key in CROP_KEYS:
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f'{key} must be positive')


def load_processor(AutoProcessor, args):
    # Pin the slow processor so its explicit signature can be verified, rather
    # than accepting **kwargs as proof that a crop option is supported.
    processor = AutoProcessor.from_pretrained(
        args.model_id, revision=args.revision, padding_side='left', use_fast=False,
    )
    ip = processor.image_processor
    try:
        signatures = {name: inspect.signature(getattr(type(ip), name))
                      for name in ('__init__', 'preprocess')}
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f'Installed {type(ip).__name__} does not explicitly expose '
                           'inspectable crop-control signatures') from exc
    supported = set().union(*(set(sig.parameters) for sig in signatures.values()))
    # transformers 5.x declares these via typed **kwargs unpacking
    # (PEP 692), so inspect.signature no longer lists them individually.
    # Fall back to the annotated kwargs TypedDict, which is still an
    # explicit declaration -- not a bare **kwargs catch-all.
    kwargs_typeddicts = []
    for sig in signatures.values():
        for param in sig.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                ann = param.annotation
                # PEP 692: **kwargs: Unpack[SomeTypedDict] -- unwrap it.
                unpacked = typing.get_args(ann)
                if unpacked:
                    ann = unpacked[0]
                names = getattr(ann, '__annotations__', None)
                if names:
                    supported |= set(names)
                    kwargs_typeddicts.append(getattr(ann, '__name__', str(ann)))
    missing = set(CROP_KEYS + ('do_pan_and_scan',)) - supported
    if missing:
        raise RuntimeError(f'Installed {type(ip).__name__} does not explicitly expose '
                           f'{sorted(missing)}; signatures={signatures}')
    kwargs = {}
    for key in CROP_KEYS:
        value = getattr(args, key)
        if value is None:
            value = getattr(ip, key, None)
        if value is None:
            raise RuntimeError(f'Processor has no resolved {key}; specify its CLI control')
        kwargs[key] = value
    if any(value <= 0 for value in kwargs.values()):
        raise ValueError(f'Invalid resolved crop controls: {kwargs}')
    if dict(ip.size) != {'height': 896, 'width': 896}:
        raise RuntimeError(f'Expected a fixed 896x896 encoder; found processor size={ip.size}')
    processor.tokenizer.padding_side = 'left'
    details = {
        'class': type(ip).__name__,
        'use_fast': False,
        'size': dict(ip.size),
        'verified_signatures': {name: str(sig) for name, sig in signatures.items()},
        'verified_kwargs_typeddicts': kwargs_typeddicts,
        'pan_and_scan_kwargs': kwargs,
    }
    print('GEMMA PROCESSOR: ' + json.dumps(details, sort_keys=True), flush=True)
    return processor, details


def load_model(Gemma3ForConditionalGeneration, torch, args):
    from transformers import set_seed
    set_seed(args.seed)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16,
        device_map='auto', attn_implementation=args.attn_impl,
    ).eval()
    # Some checkpoints store sampling defaults; clear them rather than passing
    # temperature/top_p/top_k alongside greedy generation.
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    for name in ('temperature', 'top_p', 'top_k'):
        setattr(model.generation_config, name, None)
    return model, 'dtype', str(next(model.parameters()).dtype)


def build_messages(body, condition, image=None, system_text=None):
    """Consume the exact shared Qwen body; replace only its image placeholder.

    Qwen supplies no explicit system instruction, so neither do the runners.
    The helper supports a structured system role without inventing prompt text.
    """
    content = []
    if condition == 'image':
        marker = '<image>\n'
        if not body.startswith(marker):
            raise ValueError('Shared image prompt lacks the <image> marker')
        body = body[len(marker):]
        content.append({'type': 'image', **({'image': image} if image is not None else {})})
    elif condition not in ('text', 'text_noref', 'kg_text'):
        raise ValueError(f'Unknown condition: {condition}')
    elif image is not None:
        raise ValueError('Text conditions must not receive images')
    content.append({'type': 'text', 'text': body})
    messages = []
    if system_text is not None:
        messages.append({'role': 'system', 'content': [{'type': 'text', 'text': system_text}]})
    messages.append({'role': 'user', 'content': content})
    return messages


def format_prompt(processor, body, condition):
    return processor.apply_chat_template(build_messages(body, condition),
                                         tokenize=False, add_generation_prompt=True)


def prepare_inputs(processor, body, condition, image, args, details):
    if condition == 'image' and image is None:
        raise ValueError('Image inference requires a PIL image')
    kwargs = {}
    if condition == 'image':
        kwargs = {'do_pan_and_scan': args.pan_and_scan, **details['pan_and_scan_kwargs']}
    return processor.apply_chat_template(
        build_messages(body, condition, image), tokenize=True, return_dict=True,
        return_tensors='pt', add_generation_prompt=True, **kwargs,
    )


def image_token_counts(processor, inputs, condition):
    """Return structural image views and expanded soft image tokens separately."""
    input_ids = inputs['input_ids']
    image_views = int((input_ids == int(processor.image_token_id)).sum().item())
    soft_token_id = int(processor.tokenizer.image_token_id)
    image_soft_tokens = int((input_ids == soft_token_id).sum().item())
    expected_image = condition == 'image'
    if ((image_views > 0) != expected_image
            or (image_soft_tokens > 0) != expected_image):
        raise RuntimeError(
            f'Unexpected image token counts views={image_views} '
            f'soft_tokens={image_soft_tokens} for {condition}'
        )
    return image_views, image_soft_tokens


def image_budget_diagnostics(processor, inputs, raw_size):
    result = {
        'source_size': {'width': raw_size[0], 'height': raw_size[1]},
        'resolved_size': dict(processor.image_processor.size),
        **dict(zip(('image_views', 'image_soft_tokens'),
                   image_token_counts(processor, inputs, 'image'))),
        'input_tokens': int(inputs['input_ids'].shape[-1]),
    }
    print('FIRST IMAGE BUDGET: ' + json.dumps(result, sort_keys=True), flush=True)
    return result


def infer_one(model, processor, body, condition, image, args, torch, details):
    inputs = prepare_inputs(processor, body, condition, image, args, details)
    image_views, image_soft_tokens = image_token_counts(processor, inputs, condition)
    inputs = inputs.to(model.device)
    input_len = inputs['input_ids'].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False, num_beams=1,
                             **({} if args.cache_implementation == 'dynamic'
                                else {'cache_implementation': args.cache_implementation}))
    generated = out[0][input_len:]
    capture_token_ids(generated)
    response = processor.decode(generated, skip_special_tokens=True).strip()
    generated_count = int(generated.shape[-1])
    return (response, generated_count, generated_count >= args.max_new_tokens,
            image_views, image_soft_tokens)


def manifest_fields(args, details, resolved_dtype):
    return {
        'revision': args.revision,
        'dtype': resolved_dtype,
        'attn_implementation': args.attn_impl,
        'pan_and_scan': args.pan_and_scan,
        'pan_and_scan_kwargs': details['pan_and_scan_kwargs'],
        'gemma_processor': details,
        'resolution_control_note': RESOLUTION_NOTE,
        'system_text': None,
        'system_prompt_note': 'No explicit system instruction, matching the Qwen runners.',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'cache_implementation': args.cache_implementation,
    }


def gate_identity(args, details):
    return {
        'model_id': args.model_id, 'revision': args.revision,
        'attn_implementation': args.attn_impl,
        'pan_and_scan_kwargs': details['pan_and_scan_kwargs'],
        'graph_metadata_sha256': sha256_file(args.graph_metadata),
        'image_root': str(args.image_root.resolve()),
    }


def require_preflight(args, details):
    if args.preflight_report is None:
        raise ValueError('Full image inference requires --preflight-report from the Stage 1 gate')
    report = json.loads(args.preflight_report.read_text(encoding='utf-8'))
    if report['identity'] != gate_identity(args, details):
        raise ValueError('Preflight model, revision, attention, crops or source graphs do not match')
    setting = 'pan_and_scan' if args.pan_and_scan else 'no_pan_and_scan'
    failed = not report['settings'][setting]['passed']
    if failed and not getattr(args, 'allow_failed_preflight', False):
        raise ValueError(f'Legibility gate failed for {setting}; resolve it before full image inference')
    for item in report['images']:
        if sha256_file(args.image_root / item['image']) != item['sha256']:
            raise ValueError(f'Preflight image changed: {item["image"]}')
    result = {'path': str(args.preflight_report.resolve()),
              'sha256': sha256_file(args.preflight_report), 'setting': setting}
    if getattr(args, 'allow_failed_preflight', False):
        result.update(passed=not failed, legibility_confounded=failed)
    return result
