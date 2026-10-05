#!/usr/bin/env python3
"""Run pinned LLaVA-v1.6-Mistral-7B on Stage 1 graph-comprehension tasks."""

from __future__ import annotations

import sys as _fullrun_sys
from pathlib import Path as _FullrunPath
_fullrun_sys.path.insert(0, str(_FullrunPath(__file__).resolve().parents[3] / "scripts"))
from fullrun_runtime import enrich_config, begin_item, finish_item

import argparse
import json
import re
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from score_stage1 import TASK_SETS, answer_format_diagnostics, require_task, score_record
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

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
STAGE2_SCRIPTS = ROOT / "experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts"
for directory in (SCRIPTS, STAGE2_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from llava_common import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    MODEL_NAME,
    completed_stage1_keys,
    configure_processor,
    format_prompt,
    image_diagnostics,
    image_token_id,
    prepare_inputs,
)
from prompt_common import sha256_file  # noqa: E402


from llava_stage1_prompt import (PROMPT_TEMPLATES, ASSISTANT_PREFIX_MODES,
                                format_stage1_prompt, prefix_for_record, append_assistant_prefix,
                                config_settings, validate_diagnostic_settings, write_compatible_config)


MAX_NEW_TOKENS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--task-set", choices=TASK_SETS, default="extended")
    parser.add_argument("--extractor", choices=("legacy", "span", "span_extended"), default="span")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="none")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt-template", choices=PROMPT_TEMPLATES, default="hf-chat")
    parser.add_argument("--assistant-prefix-mode", choices=ASSISTANT_PREFIX_MODES, default="none")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--approve-prompts", action="store_true")
    parser.add_argument(
        "--behavior-only", action="store_true",
        help="Permit a partial task set and skip aggregate scoring (pilot probes only).",
    )
    return parser.parse_args()


def render_prompt(processor, record, args):
    template = getattr(args, "prompt_template", "hf-chat")
    formatted = format_stage1_prompt(processor, raw_image_prompt(record, args.answer_format), template)
    prefix = prefix_for_record(record, getattr(args, "assistant_prefix_mode", "none"))
    return append_assistant_prefix(formatted, prefix, template)


def make_result(record, response, generated_tokens, hit_ceiling, vision_tokens,
                elapsed_seconds, peak_memory_bytes, metadata, args):
    prefix = prefix_for_record(record, getattr(args, "assistant_prefix_mode", "none"))
    continuation = response
    response = prefix + continuation
    task = str(record["task_type"])
    diagnostics = answer_format_diagnostics(task, response, args.answer_format)
    diagnostics["ceiling_hit_before_complete_answer"] = bool(
        args.answer_format == "constrained"
        and hit_ceiling
        and not diagnostics["final_format_compliant"]
    )
    result = {
        "statement_idx": int(record["statement_idx"]),
        "task_type": task,
        "image": record["image"],
        "prompt": record["prompt"],
        "gold": record["answer"],
        "raw_response": response,
        "generated_token_count": generated_tokens,
        "hit_token_ceiling": hit_ceiling,
        "vision_tokens_per_item": vision_tokens,
        "generation_elapsed_seconds": elapsed_seconds,
        "peak_memory_allocated_bytes": peak_memory_bytes,
        **diagnostics,
        **score_record(record, response, metadata, extractor=args.extractor),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if prefix:
        result["assistant_prefix"] = prefix
        result["continuation_response"] = continuation
    if isinstance(record.get("gold"), dict):
        result["structured_gold"] = record["gold"]
    return result


def infer_one(model, processor, prompt_text, image, args, torch):
    inputs = prepare_inputs(processor, prompt_text, image)
    vision_tokens = int((inputs["input_ids"] == image_token_id(processor)).sum().item())
    input_length = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(model.device)
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
    generated = output_ids[0, input_length:]
    count = int(generated.shape[-1])
    response = processor.decode(generated, skip_special_tokens=True)
    if getattr(args, "assistant_prefix_mode", "none") == "none":
        response = response.strip()
    return response, count, count >= args.max_new_tokens, vision_tokens, elapsed, peak


def main() -> None:
    args = parse_args()
    validate_diagnostic_settings(args)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("--revision must be an explicit 40-character Hub commit")
    if args.expected_count <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--expected-count and --max-new-tokens must be positive")
    task_types = TASK_SETS[args.task_set]
    records, metadata_by_idx, task_counts = validate_stage1(
        args.input_jsonl,
        args.graph_metadata,
        args.expected_count,
        task_types,
        expected_split=args.expected_split,
    )

    from PIL import Image
    import torch
    import transformers
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

    transformers.set_seed(args.seed)
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.revision)
    model_config = LlavaNextConfig.from_pretrained(args.model_id, revision=args.revision)
    processor_expansion = configure_processor(processor, model_config)
    first_by_task = {}
    for record in records:
        first_by_task.setdefault(record["task_type"], record)
    previews = {}
    for task in task_types:
        if task not in first_by_task:
            continue
        previews[task] = render_prompt(processor, first_by_task[task], args)
        print(f"PROMPT PREVIEW task={task}\n{previews[task]}", flush=True)

    first = records[0]
    first_image_path = args.image_root / first["image"]
    if not first_image_path.is_file():
        raise FileNotFoundError(f"Missing graph image: {first_image_path}")
    with Image.open(first_image_path) as opened:
        first_image = opened.convert("RGB")
        first_inputs = prepare_inputs(
            processor,
            render_prompt(processor, first, args),
            first_image,
        )
        first_image_budget = image_diagnostics(processor, first_inputs, first_image.size)
    print("FIRST IMAGE BUDGET: " + json.dumps(first_image_budget, sort_keys=True), flush=True)
    if args.preview_only:
        print("Preview only: no model was loaded and no predictions were generated.", flush=True)
        return
    if not args.approve_prompts:
        raise SystemExit("Refusing inference until prompt previews are approved; use --approve-prompts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = {task: output_path(args.output_dir, MODEL_NAME, task) for task in task_types}
    if not args.resume:
        existing = [path for path in prediction_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Use --resume or a fresh output directory: {existing[0]}")
    done = completed_stage1_keys(args.output_dir, task_types) if args.resume else set()
    input_keys = {(int(row["statement_idx"]), str(row["task_type"])) for row in records}
    if not done <= input_keys:
        raise ValueError(f"Existing predictions are not in current input: {sorted(done - input_keys)[:10]}")

    load_started = perf_counter()
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    loading_seconds = perf_counter() - load_started
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    run_config = {
        "model_name": MODEL_NAME,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": "image",
        "task_set": args.task_set,
        "extractor": args.extractor,
        "seed": args.seed,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "input_jsonl": {"path": str(args.input_jsonl.resolve()), "sha256": sha256_file(args.input_jsonl),
                        "record_count": len(records), "task_counts": task_counts},
        "graph_metadata": {"path": str(args.graph_metadata.resolve()),
                           "sha256": sha256_file(args.graph_metadata),
                           "record_count": len(metadata_by_idx)},
        "prompt": "Shared Stage 1 body through the pinned processor's chat template",
        "answer_format": answer_format_provenance(args.answer_format, task_types),
        "prompt_bodies_sha256": prompt_bodies_sha256(records, args.answer_format),
        "generation": {"do_sample": False, "num_beams": 1,
                       "max_new_tokens": args.max_new_tokens},
        "effective_max_new_tokens": args.max_new_tokens,
        "dtype": "torch.bfloat16",
        "image_processing": {"mode": "processor_default_anyres",
                             "processor_token_expansion": processor_expansion,
                             "first_image_budget": first_image_budget},
        "vision_tokens_per_item": "Count of expanded image token IDs in processor input_ids",
    }
    run_config.update(config_settings(args, run_config["prompt_bodies_sha256"]))
    if args.prompt_template == "llava_v1":
        run_config["prompt"] = "Unchanged shared Stage 1 body through GraphVis conv_llava_v1 (TWO)"
    run_config = enrich_config(run_config)
    write_compatible_config(args.output_dir, run_config, write_run_config)
    runtime_path = args.output_dir / "runtime_metrics.json"
    if not runtime_path.exists():
        runtime_path.write_text(
            json.dumps({"loading_elapsed_seconds": loading_seconds}, indent=2) + "\n",
            encoding="utf-8",
        )

    generated_count = 0
    with ExitStack() as stack:
        handles = {task: stack.enter_context(path.open("a", encoding="utf-8"))
                   for task, path in prediction_paths.items()}
        for position, record in enumerate(records, start=1):
            idx, task = int(record["statement_idx"]), str(record["task_type"])
            require_task(task, task_types)
            if (idx, task) in done:
                continue
            begin_item()
            image_path = args.image_root / record["image"]
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                values = infer_one(
                    model,
                    processor,
                    render_prompt(processor, record, args),
                    image,
                    args,
                    torch,
                )
            result = make_result(record, *values, metadata_by_idx[idx], args)
            finish_item(result)
            handles[task].write(json.dumps(result, ensure_ascii=False) + "\n")
            handles[task].flush()
            done.add((idx, task))
            generated_count += 1
            print(f"[{position}/{len(records)}] task={task} q={idx} tokens={values[1]} ceiling={values[2]}", flush=True)
    if done != input_keys:
        raise RuntimeError(f"Run ended with {len(done)}/{len(input_keys)} completed keys")
    print(f"Generated {generated_count} new predictions; {len(done)} total complete.", flush=True)
    if args.behavior_only:
        print("Behavior-only probe complete; aggregate scoring intentionally skipped.", flush=True)
    else:
        score_completed_run(
            args.input_jsonl, args.graph_metadata, args.output_dir, MODEL_NAME, args.task_set, args.extractor
        )


if __name__ == "__main__":
    main()
