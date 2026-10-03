#!/usr/bin/env python3
"""Run Gemma 3 on image-only Stage 1 tasks, preserving the Qwen contract."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import copy

import gemma3_common as gemma
from score_stage1 import (
    TASK_TYPES,
    TASK_SETS,
    EXTENDED_TASK_TYPES,
    aggregate_task,
    answer_format_diagnostics,
    read_jsonl,
    require_task,
    score_record,
)

from stage1_common import (
    ANSWER_FORMATS,
    answer_format_provenance,
    output_path,
    prompt_bodies_sha256,
    raw_image_prompt,
    score_completed_run,
    validate_stage1,
    write_run_config,
)


from prompt_common import sha256_file

DEFAULT_MODEL_ID = gemma.DEFAULT_MODEL_ID
MAX_NEW_TOKENS = 1024
MODEL_NAME = "gemma3"
EXTENDED_TASKS = EXTENDED_TASK_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", help="Required immutable 40-character Hub commit")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=3000)
    parser.add_argument("--expected-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="none")
    parser.add_argument(
        "--max-new-tokens", type=int,
        help="Default: 2048 for constrained answers, otherwise 1024",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--approve-prompts",
        action="store_true",
        help="Confirm the printed prompts were reviewed and permit inference",
    )
    parser.add_argument('--task-set', choices=('paper', 'extended'), default='paper')
    parser.add_argument('--extractor', choices=('legacy', 'span', 'span_extended'), default='span')
    parser.add_argument('--preflight', nargs='?', const=50, type=int,
                        help='Run triple_listing on N graphs with BOTH crop settings (default N=50)')
    parser.add_argument('--preflight-min-recall', type=float, default=0.95,
                        help='Required raw macro gold-node recall; operational default 0.95')
    gemma.add_common_args(parser)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.answer_format == "constrained" else MAX_NEW_TOKENS
    return args


def validate_inputs(args):
    return validate_stage1(args.input_jsonl, args.graph_metadata, args.expected_count,
                           TASK_SETS[args.task_set], expected_split=args.expected_split)


def make_result(record, response, generated_tokens, hit_ceiling, image_views,
                image_soft_tokens, metadata, args):
    task = record['task_type']
    format_diagnostics = answer_format_diagnostics(task, response, args.answer_format)
    format_diagnostics['ceiling_hit_before_complete_answer'] = bool(
        args.answer_format == 'constrained'
        and hit_ceiling
        and not format_diagnostics['final_format_compliant']
    )
    result = {
        "statement_idx": int(record['statement_idx']),
        "task_type": task,
        "image": record["image"],
        "prompt": record["prompt"],
        "gold": record["answer"],
        "raw_response": response,
        "generated_token_count": generated_tokens,
        "hit_token_ceiling": hit_ceiling,
        **format_diagnostics,
        **score_record(record, response, metadata, extractor=args.extractor),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_views": image_views,
        "image_soft_tokens": image_soft_tokens,
        "vision_tokens_per_item": image_soft_tokens,
    }
    if isinstance(record.get('gold'), dict):
        result['structured_gold'] = record['gold']
    return result


def run_preflight(args, records, metadata, model, processor, details, torch, run_config):
    from PIL import Image
    answer_format = getattr(args, 'answer_format', 'none')
    if not args.resume and any((args.output_dir / setting / 'predictions_gemma3_triple_listing.jsonl').exists()
           for setting in ('pan_and_scan', 'no_pan_and_scan')):
        raise FileExistsError('Use a fresh preflight output directory for both crop settings')
    report = {'identity': gemma.gate_identity(args, details),
              'minimum_raw_macro_recall': args.preflight_min_recall,
              'threshold_note': 'Operational threshold, not a measured Qwen baseline.',
              'graph_count': len(records), 'settings': {},
              'images': [{'image': r['image'], 'sha256': sha256_file(args.image_root / r['image'])}
                         for r in records]}
    for enabled in (True, False):
        setting = 'pan_and_scan' if enabled else 'no_pan_and_scan'
        trial_args = copy.copy(args)
        trial_args.pan_and_scan = enabled
        trial_dir = args.output_dir / setting
        trial_dir.mkdir(parents=True, exist_ok=True)
        path = output_path(trial_dir, MODEL_NAME, 'triple_listing')
        if path.exists() and not args.resume:
            raise FileExistsError(f'Use --resume or a fresh preflight output directory: {path}')
        config = copy.deepcopy(run_config)
        config.update(gemma.manifest_fields(trial_args, details, run_config['resolved_dtype']))
        config['preflight'] = {'graphs': len(records), 'minimum_raw_macro_recall': args.preflight_min_recall}
        rows = read_jsonl(path) if path.exists() else []
        expected = {(int(r['statement_idx']), r['task_type']) for r in records}
        done = {(int(r['statement_idx']), r['task_type']) for r in rows}
        if len(done) != len(rows) or not done <= expected:
            raise ValueError(f'Duplicate or unexpected preflight predictions: {path}')
        config_path = trial_dir / 'run_config.json'
        if rows and not config_path.exists():
            raise ValueError(f'Cannot resume preflight without config: {trial_dir}')
        with Image.open(args.image_root / records[0]['image']) as opened:
            first_image = opened.convert('RGB')
            inputs = gemma.prepare_inputs(processor, raw_image_prompt(records[0], answer_format), 'image',
                                          first_image, trial_args, details)
            config['image_processing']['first_image_budget'] = gemma.image_budget_diagnostics(
                processor, inputs, first_image.size)
        write_run_config(trial_dir, config)
        with path.open('a' if args.resume else 'w', encoding='utf-8') as handle:
            for record in records:
                key = (int(record['statement_idx']), record['task_type'])
                if key in done:
                    continue
                with Image.open(args.image_root / record['image']) as opened:
                    image = opened.convert('RGB')
                    response, tokens, ceiling, image_views, image_soft_tokens = gemma.infer_one(
                        model, processor, raw_image_prompt(record, answer_format),
                        'image', image, trial_args, torch, details)
                row = make_result(record, response, tokens, ceiling, image_views,
                                  image_soft_tokens,
                                  metadata[int(record['statement_idx'])], trial_args)
                rows.append(row)
                done.add(key)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                print(f'PREFLIGHT {setting} [{len(rows)}/{len(records)}] '
                      f'recall={row["gold_node_component_recall"]["raw"]["recall"]}', flush=True)
        metrics = aggregate_task('triple_listing', rows)
        recall = metrics['gold_node_component_recall']['raw']['macro_recall']
        report['settings'][setting] = {'passed': recall >= args.preflight_min_recall,
                                       'metrics': metrics}
        print(f'PREFLIGHT {setting}: gold-node recall={recall:.6f}; '
              f'passed={report["settings"][setting]["passed"]}', flush=True)
    report_path = args.output_dir / 'preflight_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not report['settings']['pan_and_scan' if args.pan_and_scan else 'no_pan_and_scan']['passed']:
        raise SystemExit('Legibility gate failed; resolve image readability before full image inference')


def main() -> None:
    args = parse_args()
    gemma.validate_args(args)
    if args.preflight is not None and args.preflight <= 0:
        raise ValueError('--preflight must be positive')
    if not 0 <= args.preflight_min_recall <= 1:
        raise ValueError('--preflight-min-recall must be in [0, 1]')
    records, metadata_by_idx, task_counts = validate_inputs(args)
    tasks = TASK_SETS[args.task_set]
    if args.preflight is not None:
        records = [r for r in records if r['task_type'] == 'triple_listing'][:args.preflight]
        if len(records) != args.preflight:
            raise ValueError('Not enough triple_listing graphs for the requested preflight')
        task_counts = {'triple_listing': len(records)}

    # Processor imports and loading are needed to preview the exact chat-template
    # text, but the model and GPU remain untouched until approval.
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    processor, processor_details = gemma.load_processor(AutoProcessor, args)
    formatted_previews = {}
    for record in records:
        if record['task_type'] not in formatted_previews:
            formatted = gemma.format_prompt(
                processor, raw_image_prompt(record, args.answer_format), 'image'
            )
            formatted_previews[record['task_type']] = formatted
            print(f'PROMPT PREVIEW task={record["task_type"]}\n{formatted}', flush=True)
    system_prompt_injected = False
    first = records[0]
    first_image_path = args.image_root / first["image"]
    if not first_image_path.is_file():
        raise FileNotFoundError(f"Missing graph image: {first_image_path}")
    with Image.open(first_image_path) as opened:
        first_image = opened.convert("RGB")
        first_inputs = gemma.prepare_inputs(
            processor, raw_image_prompt(first, args.answer_format), 'image',
            first_image, args, processor_details)
        first_image_budget = gemma.image_budget_diagnostics(processor, first_inputs, first_image.size)
    if args.preview_only:
        print("Preview only: no model was loaded and no predictions were generated.", flush=True)
        return
    if not args.approve_prompts:
        raise SystemExit(
            "Refusing inference until the prompt previews are approved. "
            "Re-run with --approve-prompts."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU is visible. Run through sbatch.")

    preflight_provenance = None
    if args.preflight is None:
        preflight_provenance = gemma.require_preflight(args, processor_details)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = {
        task: output_path(args.output_dir, MODEL_NAME, task) for task in tasks
    }
    if not args.resume:
        existing = [path for path in prediction_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Prediction files already exist; use --resume or a new output directory: "
                + ", ".join(map(str, existing))
            )
    done = set()
    if args.resume:
        for task, path in prediction_paths.items():
            if path.exists():
                for row in read_jsonl(path):
                    key = (int(row['statement_idx']), row['task_type'])
                    if key in done or key[1] != task:
                        raise ValueError(f'Duplicate or misplaced prediction: {key}')
                    done.add(key)
    input_keys = {
        (int(record["statement_idx"]), record["task_type"]) for record in records
    }
    unknown_done = done - input_keys
    if unknown_done:
        raise ValueError(f"Existing predictions are not in the current input: {sorted(unknown_done)[:10]}")

    model, dtype_argument, resolved_dtype = gemma.load_model(
        Gemma3ForConditionalGeneration, torch, args
    )
    run_config = {
        "model_name": MODEL_NAME,
        "extractor": args.extractor,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": "image",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "input_jsonl": {
            "path": str(args.input_jsonl.resolve()),
            "sha256": sha256_file(args.input_jsonl),
            "record_count": len(records),
            "task_counts": task_counts,
        },
        "graph_metadata": {
            "path": str(args.graph_metadata.resolve()),
            "sha256": sha256_file(args.graph_metadata),
            "record_count": len(metadata_by_idx),
        },
        "prompt": "Gemma image content plus record['prompt'] through apply_chat_template",
        "answer_format": answer_format_provenance(args.answer_format, tasks),
        "prompt_bodies_sha256": prompt_bodies_sha256(records, args.answer_format),
        "expected_split": args.expected_split,
        "default_system_prompt_injected": system_prompt_injected,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "effective_max_new_tokens": args.max_new_tokens,
        "image_processing": {
            "processor_pixel_budget_api": "fixed_896_plus_pan_and_scan",
            "first_image_budget": first_image_budget,
            "visual_budget_note": gemma.RESOLUTION_NOTE,
        },
        "dtype_argument": dtype_argument,
        "resolved_dtype": resolved_dtype,
        "attention_implementation": args.attn_impl,
        "token_ceiling_detection": (
            "Exact generated sequence length before decoding; length >= max_new_tokens "
            "is marked as a ceiling hit."
        ),
        "source_image_generation": {
            "command_verified_in_shell_history": (
                "python scripts/generate_graphvis_datasets.py --split test --start 0 "
                "--limit 500 --tasks-per-graph 6 --data-root "
                "data_preprocessed_release --out-dir outputs/graphvis_obqa --seed 13"
            ),
            "expected_renderer_and_pruning_settings": {
                "max_nodes": 18,
                "max_edges": 30,
                "max_degree": 5,
                "engine": "dot",
                "dpi": 200,
                "disconnected_rows": 3,
                "hide_relatedto_labels": False,
                "reveal_correct_answer": False,
            },
            "settings_provenance": (
                "Expected generator defaults; these settings are not independently "
                "encoded in graph_metadata_0_500.jsonl."
            ),
        },
    }
    run_config.update(gemma.manifest_fields(args, processor_details, resolved_dtype))
    run_config['task_set'] = args.task_set
    run_config['scoring_contract'] = 'Shared Stage 1 scorer with paper and extended task sets.'
    run_config['preflight_report'] = preflight_provenance
    if args.preflight is not None:
        run_preflight(args, records, metadata_by_idx, model, processor, processor_details, torch, run_config)
        return
    write_run_config(args.output_dir, run_config)

    generated_count = 0
    with ExitStack() as stack:
        handles = {
            task: stack.enter_context(path.open("a", encoding="utf-8"))
            for task, path in prediction_paths.items()
        }
        for position, record in enumerate(records, start=1):
            idx = int(record["statement_idx"])
            task = record["task_type"]
            require_task(task, tasks)
            key = (idx, task)
            if key in done:
                continue
            image_path = args.image_root / record["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing graph image: {image_path}")
            body = raw_image_prompt(record, args.answer_format)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                response, generated_tokens, hit_ceiling, image_views, image_soft_tokens = gemma.infer_one(
                    model, processor, body, 'image', image, args, torch, processor_details)
            result = make_result(record, response, generated_tokens, hit_ceiling,
                                 image_views, image_soft_tokens,
                                 metadata_by_idx[idx], args)
            handle = handles[task]
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
            generated_count += 1
            print(
                f"[{position}/{len(records)}] task={task} statement_idx={idx} "
                f"correct_raw={result.get('is_correct', 'unscored')} tokens={generated_tokens} "
                f"ceiling={hit_ceiling}",
                flush=True,
            )

    if len(done) != len(records):
        raise RuntimeError(f"Run ended with {len(done)}/{len(records)} completed predictions")
    print(f"Generated {generated_count} new predictions; {len(done)} total complete.", flush=True)
    score_completed_run(args.input_jsonl, args.graph_metadata, args.output_dir, MODEL_NAME, args.task_set, args.extractor)


if __name__ == "__main__":
    main()
