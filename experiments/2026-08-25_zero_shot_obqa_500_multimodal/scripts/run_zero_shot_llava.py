#!/usr/bin/env python3
"""Run pinned LLaVA-v1.6-Mistral-7B on the four shared OBQA conditions."""

from __future__ import annotations

import sys as _fullrun_sys
from pathlib import Path as _FullrunPath
_fullrun_sys.path.insert(0, str(_FullrunPath(__file__).resolve().parents[3] / "scripts"))
from fullrun_runtime import enrich_config, begin_item, finish_item

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from prompt_common import (
    CONDITIONS,
    format_kg_block,
    index_graph_metadata,
    parse_answer,
    prompt_bodies_sha256,
    read_jsonl,
    render_prompt,
    sha256_file,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from llava_common import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    completed_qa_indices,
    configure_processor,
    format_prompt,
    image_diagnostics,
    image_token_id,
    prepare_inputs,
)


MAX_NEW_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, help="Process only the first N records; input identity stays fixed for resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--approve-prompt-diff", action="store_true")
    return parser.parse_args()


def require_fresh_output_progress(output_existed: bool, input_count: int,
                                  generated_count: int) -> None:
    if not output_existed and input_count > 0 and generated_count == 0:
        raise RuntimeError(
            "LLaVA QA generated zero predictions for a non-empty input and a fresh output path; "
            "resume selection skipped every row."
        )


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
    response = processor.decode(generated, skip_special_tokens=True).strip()
    return response, count, count >= args.max_new_tokens, vision_tokens, elapsed, peak


def write_run_config(args, records, metadata_by_idx, transformers_version,
                     torch_version, loading_seconds, first_image_budget,
                     processor_expansion):
    config = {
        "model_name": "llava",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": args.condition,
        "seed": args.seed,
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "input_jsonl": {"path": str(args.input_jsonl.resolve()),
                        "sha256": sha256_file(args.input_jsonl),
                        "record_count": len(records)},
        "graph_metadata": {"path": str(args.graph_metadata.resolve()),
                           "sha256": sha256_file(args.graph_metadata),
                           "record_count": len(metadata_by_idx)},
        "prompt": "Shared OBQA body through the pinned processor's chat template",
        "prompt_bodies_sha256": prompt_bodies_sha256(records, metadata_by_idx, args.condition),
        "generation": {"do_sample": False, "num_beams": 1,
                       "max_new_tokens": args.max_new_tokens},
        "effective_max_new_tokens": args.max_new_tokens,
        "dtype": "torch.bfloat16",
        "image_processing": {"mode": "processor_default_anyres",
                             "processor_token_expansion": processor_expansion,
                             "first_image_budget": first_image_budget},
        "vision_tokens_per_item": "Count of expanded image token IDs in processor input_ids",
    }
    config = enrich_config(config)
    path = args.output_jsonl.parent / "run_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Refusing to overwrite incompatible run configuration: {path}")
        return
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("--revision must be an explicit 40-character Hub commit")
    if args.expected_count <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--expected-count and --max-new-tokens must be positive")
    records = read_jsonl(args.input_jsonl)
    validate_records(records, args.expected_count)
    metadata_by_idx = index_graph_metadata(args.graph_metadata, records)

    from PIL import Image
    import torch
    import transformers
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

    transformers.set_seed(args.seed)
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.revision)
    model_config = LlavaNextConfig.from_pretrained(args.model_id, revision=args.revision)
    processor_expansion = configure_processor(processor, model_config)
    first = min(records, key=lambda row: int(row["statement_idx"]))
    first_idx = int(first["statement_idx"])
    kg_block = format_kg_block(metadata_by_idx[first_idx])
    bodies = {condition: render_prompt(
        first["prompt"], condition,
        kg_block=kg_block if condition == "kg_text" else None,
    ) for condition in CONDITIONS}
    formatted = {condition: format_prompt(processor, body, condition)
                 for condition, body in bodies.items()}
    for condition in CONDITIONS:
        print(f"SHARED BODY: {condition}\n{bodies[condition]}", flush=True)
        print(f"LLAVA-FORMATTED PROMPT: {condition}\n{formatted[condition]}", flush=True)
    if "<image>" not in formatted["image"]:
        raise RuntimeError("LLaVA image prompt lacks its image placeholder")
    if any("<image>" in formatted[name] for name in CONDITIONS if name != "image"):
        raise RuntimeError("LLaVA text-only prompt contains an image placeholder")

    first_image_budget = {}
    if args.condition == "image":
        image_path = args.image_root / first["image"]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            first_inputs = prepare_inputs(processor, formatted["image"], image)
            first_image_budget = image_diagnostics(processor, first_inputs, image.size)
        print("FIRST IMAGE BUDGET: " + json.dumps(first_image_budget, sort_keys=True), flush=True)
    if args.preview_only:
        print("Preview only: no model was loaded and no predictions were generated.", flush=True)
        return
    if not args.approve_prompt_diff:
        raise SystemExit("Refusing inference until prompt previews are approved; use --approve-prompt-diff")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_existed = args.output_jsonl.exists()
    if output_existed and not args.resume:
        raise FileExistsError(f"Use --resume or a fresh output path: {args.output_jsonl}")
    done = completed_qa_indices(args.output_jsonl) if args.resume else set()
    input_indices = {int(row["statement_idx"]) for row in records}
    if not done <= input_indices:
        raise ValueError(f"Existing predictions are not in current input: {sorted(done - input_indices)[:10]}")

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
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(args, records, metadata_by_idx, transformers.__version__,
                     torch.__version__, loading_seconds, first_image_budget,
                     processor_expansion)
    runtime_path = args.output_jsonl.parent / "runtime_metrics.json"
    if not runtime_path.exists():
        runtime_path.write_text(
            json.dumps({"loading_elapsed_seconds": loading_seconds}, indent=2) + "\n",
            encoding="utf-8",
        )

    tiers = Counter()
    if args.resume and args.output_jsonl.exists():
        tiers.update(row.get("parse_tier", "MISSING") for row in read_jsonl(args.output_jsonl))
    generated_count = 0
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for position, record in enumerate(records[:args.limit], start=1):
            statement_idx = int(record["statement_idx"])
            if statement_idx in done:
                continue
            begin_item()
            per_item_kg = (format_kg_block(metadata_by_idx[statement_idx])
                           if args.condition == "kg_text" else None)
            body = render_prompt(record["prompt"], args.condition, kg_block=per_item_kg)
            prompt_text = format_prompt(processor, body, args.condition)
            image = None
            if args.condition == "image":
                with Image.open(args.image_root / record["image"]) as opened:
                    image = opened.convert("RGB")
            response, tokens, ceiling, vision_tokens, elapsed, peak = infer_one(
                model, processor, prompt_text, image, args, torch
            )
            parsed, tier = parse_answer(response, n_choices=4)
            predicted = None if parsed == "FAILED" else parsed
            tiers[tier] += 1
            result = {
                "statement_idx": statement_idx,
                "image": record["image"],
                "gold_option": record["answer"],
                "predicted_option": predicted,
                "raw_response": response,
                "parse_tier": tier,
                "is_correct": predicted == record["answer"],
                "generated_token_count": tokens,
                "hit_token_ceiling": ceiling,
                "vision_tokens_per_item": vision_tokens,
                "generation_elapsed_seconds": elapsed,
                "peak_memory_allocated_bytes": peak,
                "model_id": args.model_id,
                "model_revision": args.revision,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            finish_item(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            done.add(statement_idx)
            generated_count += 1
            print(f"[{position}/{len(records)}] condition={args.condition} q={statement_idx} "
                  f"predicted={predicted or 'FAILED'} tokens={tokens} ceiling={ceiling}", flush=True)
    require_fresh_output_progress(output_existed, len(records[:args.limit]), generated_count)
    if not {int(r["statement_idx"]) for r in records[:args.limit]} <= done:
        raise RuntimeError(f"Run ended with {len(done)}/{len(input_indices)} completed keys")
    print("PARSE TIER DISTRIBUTION: " + json.dumps(dict(sorted(tiers.items()))), flush=True)


if __name__ == "__main__":
    main()
