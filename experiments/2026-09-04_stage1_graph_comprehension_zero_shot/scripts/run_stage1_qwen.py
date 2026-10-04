#!/usr/bin/env python3
"""Run Qwen3-VL-8B-Instruct on paper or extended image-only Stage 1 tasks."""

from __future__ import annotations

import sys as _fullrun_sys
from pathlib import Path as _FullrunPath
_fullrun_sys.path.insert(0, str(_FullrunPath(__file__).resolve().parents[3] / "scripts"))
from fullrun_runtime import enrich_config, begin_item, finish_item

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from score_stage1 import (
    TASK_TYPES,
    TASK_SETS,
    answer_format_diagnostics,
    require_task,
    score_record,
)
from stage1_common import (
    ANSWER_FORMATS,
    answer_format_provenance,
    completed_keys,
    output_path,
    print_six_prompt_previews,
    prompt_bodies_sha256,
    raw_image_prompt,
    score_completed_run,
    validate_stage1,
    write_run_config,
)


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
OLD_SCRIPTS_DIR = EXPERIMENTS_DIR / "2026-08-25_zero_shot_obqa_500_multimodal" / "scripts"
sys.path.insert(0, str(OLD_SCRIPTS_DIR))
from prompt_common import sha256_file  # noqa: E402
from run_zero_shot_qwen import (  # noqa: E402
    ATTN_IMPLEMENTATION,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    REQUIRED_TRANSFORMERS_MIN,
    format_qwen_prompt,
    image_budget_diagnostics,
    load_model,
    load_processor,
    prepare_inputs,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_NEW_TOKENS = 1024
MODEL_NAME = "qwen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=3000)
    parser.add_argument("--expected-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--task-set", choices=TASK_SETS, default="paper")
    parser.add_argument("--extractor", choices=("legacy", "span", "span_extended"), default="span")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="none")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--max-new-tokens", type=int,
        help="Default: 2048 for constrained answers, otherwise 1024",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--behavior-only", action="store_true",
        help="Permit a partial task set and skip aggregate scoring (pilot probes only).",
    )
    parser.add_argument(
        "--approve-prompts",
        action="store_true",
        help="Confirm the printed prompts were reviewed and permit inference",
    )
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.answer_format == "constrained" else MAX_NEW_TOKENS
    return args


def infer_one(model, processor, prompt_text: str, image, args, torch):
    inputs = prepare_inputs(processor, prompt_text, image, args).to("cuda")
    vision_tokens = int(
        (inputs["input_ids"] == int(processor.image_token_id)).sum().item()
    )
    input_length = int(inputs["input_ids"].shape[1])
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
        )
    elapsed = perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    generated = output_ids[:, input_length:]
    generated_tokens = int(generated.shape[1])
    response = processor.batch_decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")].rstrip()
    return (response, generated_tokens, generated_tokens >= args.max_new_tokens,
            vision_tokens, elapsed, peak)


def main() -> None:
    args = parse_args()
    task_types = TASK_SETS[args.task_set]
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise SystemExit("Require 0 < --min-pixels <= --max-pixels")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not args.revision:
        raise SystemExit("--revision must pin an immutable model revision")
    records, metadata_by_idx, task_counts = validate_stage1(
        args.input_jsonl, args.graph_metadata, args.expected_count, task_types,
        expected_split=args.expected_split,
    )

    # Processor imports and loading are needed to preview the exact chat-template
    # text, but the model and GPU remain untouched until approval.
    from packaging.version import Version
    from PIL import Image
    import torch
    import transformers
    transformers.set_seed(args.seed)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if Version(transformers.__version__) < Version(REQUIRED_TRANSFORMERS_MIN):
        raise RuntimeError(
            f"Qwen3-VL requires transformers>={REQUIRED_TRANSFORMERS_MIN}; "
            f"found {transformers.__version__}"
        )
    processor, processor_budget_api = load_processor(AutoProcessor, args)
    preview_tasks = (
        tuple(dict.fromkeys(record["task_type"] for record in records))
        if args.behavior_only else task_types
    )
    formatted_previews = print_six_prompt_previews(
        records,
        formatter=lambda body: format_qwen_prompt(processor, body, "image"),
        task_types=preview_tasks,
        answer_format=args.answer_format,
    )
    system_prompt_injected = any(
        "<|im_start|>system" in prompt for prompt in formatted_previews.values()
    )
    print(f"Qwen default system prompt injected: {system_prompt_injected}", flush=True)
    first = records[0]
    first_image_path = args.image_root / first["image"]
    if not first_image_path.is_file():
        raise FileNotFoundError(f"Missing graph image: {first_image_path}")
    with Image.open(first_image_path) as opened:
        first_image = opened.convert("RGB")
        first_inputs = prepare_inputs(
            processor,
            formatted_previews[first["task_type"]],
            first_image,
            args,
        )
        first_image_budget = image_budget_diagnostics(
            processor, first_inputs, first_image.size
        )
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = {
        task: output_path(args.output_dir, MODEL_NAME, task) for task in task_types
    }
    if not args.resume:
        existing = [path for path in prediction_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Prediction files already exist; use --resume or a new output directory: "
                + ", ".join(map(str, existing))
            )
    done = completed_keys(args.output_dir, MODEL_NAME, task_types) if args.resume else set()
    input_keys = {
        (int(record["statement_idx"]), record["task_type"]) for record in records
    }
    unknown_done = done - input_keys
    if unknown_done:
        raise ValueError(f"Existing predictions are not in the current input: {sorted(unknown_done)[:10]}")

    load_started = perf_counter()
    model, dtype_argument, resolved_dtype = load_model(
        Qwen3VLForConditionalGeneration, torch, args
    )
    loading_seconds = perf_counter() - load_started
    run_config = {
        "model_name": MODEL_NAME,
        "extractor": args.extractor,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": "image",
        "seed": args.seed,
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
        "prompt": "Qwen image content plus record['prompt'] through apply_chat_template",
        "answer_format": answer_format_provenance(args.answer_format, task_types),
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
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "processor_pixel_budget_api": processor_budget_api,
            "first_image_budget": first_image_budget,
            "visual_budget_note": (
                "Qwen's pinned dynamic-resolution budget is not equivalent to "
                "DeepSeek-OCR-2 base_size=1024, image_size=768, crop_mode=True"
            ),
        },
        "dtype_argument": dtype_argument,
        "resolved_dtype": resolved_dtype,
        "attention_implementation": ATTN_IMPLEMENTATION,
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
    if args.task_set != "paper":
        run_config["task_set"] = args.task_set
    run_config = enrich_config(run_config)
    write_run_config(args.output_dir, run_config)
    runtime_path = args.output_dir / "runtime_metrics.json"
    if not runtime_path.exists():
        runtime_path.write_text(
            json.dumps({"loading_elapsed_seconds": loading_seconds}, indent=2) + "\n",
            encoding="utf-8",
        )

    generated_count = 0
    with ExitStack() as stack:
        handles = {
            task: stack.enter_context(path.open("a", encoding="utf-8"))
            for task, path in prediction_paths.items()
        }
        for position, record in enumerate(records, start=1):
            idx = int(record["statement_idx"])
            task = record["task_type"]
            require_task(task, task_types)
            key = (idx, task)
            if key in done:
                continue
            begin_item()
            image_path = args.image_root / record["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing graph image: {image_path}")
            prompt_text = format_qwen_prompt(
                processor, raw_image_prompt(record, args.answer_format), "image"
            )
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                inference = infer_one(model, processor, prompt_text, image, args, torch)
                response, generated_tokens, hit_ceiling = inference[:3]
                vision_tokens = inference[3] if len(inference) > 3 else None
                generation_elapsed_seconds = inference[4] if len(inference) > 4 else None
                peak_memory_allocated_bytes = inference[5] if len(inference) > 5 else None
            format_diagnostics = answer_format_diagnostics(
                task, response, args.answer_format
            )
            format_diagnostics["ceiling_hit_before_complete_answer"] = bool(
                args.answer_format == "constrained"
                and hit_ceiling
                and not format_diagnostics["final_format_compliant"]
            )
            result = {
                "statement_idx": idx,
                "task_type": task,
                "image": record["image"],
                "prompt": record["prompt"],
                "gold": record["answer"],
                **({"structured_gold": record["gold"]} if isinstance(record.get("gold"), dict) else {}),
                "raw_response": response,
                "generated_token_count": generated_tokens,
                "hit_token_ceiling": hit_ceiling,
                "vision_tokens_per_item": vision_tokens,
                "generation_elapsed_seconds": generation_elapsed_seconds,
                "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
                **format_diagnostics,
                **score_record(record, response, metadata_by_idx[idx], extractor=args.extractor),
                "model_id": args.model_id,
                "model_revision": args.revision,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle = handles[task]
            finish_item(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
            generated_count += 1
            print(
                f"[{position}/{len(records)}] task={task} statement_idx={idx} "
                f"correct_raw={result['is_correct']} tokens={generated_tokens} "
                f"ceiling={hit_ceiling}",
                flush=True,
            )

    if len(done) != len(records):
        raise RuntimeError(f"Run ended with {len(done)}/{len(records)} completed predictions")
    print(f"Generated {generated_count} new predictions; {len(done)} total complete.", flush=True)
    if args.behavior_only:
        print("Behavior-only probe complete; aggregate scoring intentionally skipped.", flush=True)
    else:
        score_completed_run(
            args.input_jsonl, args.graph_metadata, args.output_dir, MODEL_NAME, args.task_set, args.extractor
        )


if __name__ == "__main__":
    main()
