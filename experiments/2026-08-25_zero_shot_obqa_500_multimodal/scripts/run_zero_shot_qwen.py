#!/usr/bin/env python3
"""Run Qwen3-VL zero-shot OBQA evaluation with shared prompt bodies."""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prompt_common import (
    CONDITIONS,
    IMAGE_REFERENCE_SENTENCE,
    KG_TEXT_REFERENCE_SENTENCE,
    assert_no_answer_letter_annotation,
    completed_indices,
    format_kg_block,
    index_graph_metadata,
    parse_answer,
    prompt_bodies_sha256,
    read_jsonl,
    render_prompt,
    sha256_file,
    validate_records,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REQUIRED_TRANSFORMERS_MIN = "4.57.0"
QWEN_MAX_NEW_TOKENS = 64
QWEN_IMAGE_FACTOR = 32
DEFAULT_MIN_PIXELS = 256 * QWEN_IMAGE_FACTOR * QWEN_IMAGE_FACTOR
DEFAULT_MAX_PIXELS = 1280 * QWEN_IMAGE_FACTOR * QWEN_IMAGE_FACTOR
ATTN_IMPLEMENTATION = "sdpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        help="Immutable Hugging Face revision/commit; required for inference",
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-new-tokens", type=int, default=QWEN_MAX_NEW_TOKENS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--approve-prompt-diff", action="store_true")
    return parser.parse_args()


def qwen_vision_block(processor) -> str:
    return (
        str(getattr(processor, "vision_start_token", "<|vision_start|>"))
        + str(getattr(processor, "image_token", "<|image_pad|>"))
        + str(getattr(processor, "vision_end_token", "<|vision_end|>"))
    )


def format_qwen_prompt(processor, body: str, condition: str) -> str:
    if condition == "image":
        marker = "<image>\n"
        if not body.startswith(marker):
            raise RuntimeError("Shared image prompt body lacks the expected <image> marker")
        text_body = body[len(marker) :]
        content = [{"type": "image"}, {"type": "text", "text": text_body}]
    else:
        content = [{"type": "text", "text": body}]
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def assert_body_invariants(bodies: dict[str, str], kg_block: str) -> None:
    marker = "<image>\n"
    if bodies["image"].count(marker) != 1 or bodies["image"].replace(
        marker, "", 1
    ) != bodies["text"]:
        raise RuntimeError("Shared image and text bodies differ by more than <image>\\n")
    if bodies["text"].count(IMAGE_REFERENCE_SENTENCE) != 1 or bodies[
        "text"
    ].replace(IMAGE_REFERENCE_SENTENCE, "", 1) != bodies["text_noref"]:
        raise RuntimeError(
            "Shared text and text_noref bodies differ by more than "
            "IMAGE_REFERENCE_SENTENCE"
        )
    kg_prefix = f"{KG_TEXT_REFERENCE_SENTENCE}\n{kg_block}\n\n"
    if bodies["kg_text"].count(kg_prefix) != 1 or bodies["kg_text"].replace(
        kg_prefix, "", 1
    ) != bodies["text_noref"]:
        raise RuntimeError(
            "Shared kg_text and text_noref bodies differ by more than the KG prefix"
        )
    assert_no_answer_letter_annotation(bodies["kg_text"])


def assert_qwen_formatted_invariants(
    processor,
    formatted: dict[str, str],
) -> None:
    vision_block = qwen_vision_block(processor)
    if formatted["image"].count(vision_block) != 1:
        raise RuntimeError("Qwen image prompt must contain exactly one vision-token block")
    if any(formatted[name].count(vision_block) for name in CONDITIONS if name != "image"):
        raise RuntimeError("A Qwen text-only prompt unexpectedly contains vision tokens")

    # The image body retains IMAGE_REFERENCE_SENTENCE, so removing only the
    # vision segment must yield `text`; removing the reference then yields
    # `text_noref`. Requiring image-minus-vision == text_noref would discard a
    # real shared-body difference and is therefore not a valid invariant.
    candidates = (
        formatted["image"].replace(vision_block + "\n", "", 1),
        formatted["image"].replace(vision_block, "", 1),
    )
    if formatted["text"] not in candidates:
        raise RuntimeError(
            "Removing Qwen's vision-token segment does not yield its text prompt"
        )
    image_without_vision = formatted["text"]
    if image_without_vision.count(IMAGE_REFERENCE_SENTENCE) != 1 or image_without_vision.replace(
        IMAGE_REFERENCE_SENTENCE, "", 1
    ) != formatted["text_noref"]:
        raise RuntimeError(
            "Qwen image and text_noref prompts differ by more than vision tokens "
            "and IMAGE_REFERENCE_SENTENCE"
        )


def print_prompt_preview(
    statement_idx: int,
    bodies: dict[str, str],
    formatted: dict[str, str],
) -> bool:
    print(f"QWEN PROMPT PREVIEW: statement_idx={statement_idx}", flush=True)
    for condition in CONDITIONS:
        print("=" * 80, flush=True)
        print(f"SHARED BODY: {condition}", flush=True)
        print("-" * 80, flush=True)
        print(bodies[condition], flush=True)
    for condition in CONDITIONS:
        print("=" * 80, flush=True)
        print(f"QWEN-FORMATTED PROMPT: {condition}", flush=True)
        print("-" * 80, flush=True)
        print(formatted[condition], flush=True)

    for target in ("image", "kg_text"):
        print("=" * 80, flush=True)
        print(f"UNIFIED DIFF (Qwen text_noref -> {target})", flush=True)
        print("-" * 80, flush=True)
        print(
            "".join(
                difflib.unified_diff(
                    formatted["text_noref"].splitlines(keepends=True),
                    formatted[target].splitlines(keepends=True),
                    fromfile="text_noref",
                    tofile=target,
                )
            ),
            end="",
            flush=True,
        )
    print("=" * 80, flush=True)

    system_injected = any("<|im_start|>system" in value for value in formatted.values())
    print(f"Qwen default system prompt injected: {system_injected}", flush=True)
    return system_injected


def _size_value(size, key: str):
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def processor_pixel_bounds(processor) -> tuple[int | None, int | None]:
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", {})
    minimum = _size_value(size, "shortest_edge")
    maximum = _size_value(size, "longest_edge")
    if minimum is None:
        minimum = getattr(image_processor, "min_pixels", None)
    if maximum is None:
        maximum = getattr(image_processor, "max_pixels", None)
    return minimum, maximum


def load_processor(AutoProcessor, args: argparse.Namespace):
    load_kwargs = {
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision
    processor = AutoProcessor.from_pretrained(args.model_id, **load_kwargs)
    budget_api = "min_pixels/max_pixels"
    if processor_pixel_bounds(processor) != (args.min_pixels, args.max_pixels):
        print(
            "Processor did not retain min_pixels/max_pixels; retrying with "
            "size.shortest_edge/longest_edge",
            flush=True,
        )
        load_kwargs.pop("min_pixels")
        load_kwargs.pop("max_pixels")
        load_kwargs["size"] = {
            "shortest_edge": args.min_pixels,
            "longest_edge": args.max_pixels,
        }
        processor = AutoProcessor.from_pretrained(args.model_id, **load_kwargs)
        budget_api = "size.shortest_edge/longest_edge"
    resolved = processor_pixel_bounds(processor)
    if resolved != (args.min_pixels, args.max_pixels):
        raise RuntimeError(
            "Unable to pin Qwen image pixel bounds: "
            f"requested={(args.min_pixels, args.max_pixels)}, resolved={resolved}"
        )
    print(
        f"Qwen pixel budget: min_pixels={resolved[0]} max_pixels={resolved[1]} "
        f"via {budget_api}",
        flush=True,
    )
    return processor, budget_api


def prepare_inputs(processor, prompt_text: str, image, args: argparse.Namespace):
    kwargs = {
        "text": [prompt_text],
        "images": [image] if image is not None else None,
        "return_tensors": "pt",
    }
    if image is not None:
        # Transformers 4.57 declares these as Qwen3VL image kwargs. Passing
        # them per call also guards against processor-load kwargs being ignored.
        kwargs["min_pixels"] = args.min_pixels
        kwargs["max_pixels"] = args.max_pixels
    inputs = processor(**kwargs)
    inputs.pop("token_type_ids", None)
    return inputs


def image_budget_diagnostics(processor, inputs, raw_size: tuple[int, int]) -> dict:
    grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
    patch_size = int(processor.image_processor.patch_size)
    resolved_size = {
        "width": grid[2] * patch_size,
        "height": grid[1] * patch_size,
    }
    image_token_id = int(processor.image_token_id)
    image_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
    diagnostics = {
        "source_size": {"width": raw_size[0], "height": raw_size[1]},
        "resolved_size": resolved_size,
        "image_grid_thw": grid,
        "image_tokens": image_tokens,
        "input_tokens": int(inputs["input_ids"].shape[1]),
    }
    print("FIRST IMAGE BUDGET: " + json.dumps(diagnostics, sort_keys=True), flush=True)
    print(
        "Visual-budget divergence: Qwen uses the pinned dynamic pixel range; "
        "DeepSeek-OCR-2 uses base_size=1024, image_size=768, crop_mode=True. "
        "These budgets are not equivalent.",
        flush=True,
    )
    return diagnostics


def load_model(Qwen3VLForConditionalGeneration, torch, args: argparse.Namespace):
    load_kwargs = {"attn_implementation": ATTN_IMPLEMENTATION}
    if args.revision:
        load_kwargs["revision"] = args.revision
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            **load_kwargs,
        )
        dtype_argument = "dtype"
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        print(
            f"Model loader rejected dtype ({exc}); falling back to torch_dtype",
            flush=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )
        dtype_argument = "torch_dtype"
    model = model.eval().cuda()
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    resolved_dtype = str(next(model.parameters()).dtype)
    print(
        f"Loaded Qwen with {dtype_argument}=torch.bfloat16; resolved dtype={resolved_dtype}",
        flush=True,
    )
    return model, dtype_argument, resolved_dtype


def infer_one(model, processor, prompt_text: str, image, args, torch) -> str:
    inputs = prepare_inputs(processor, prompt_text, image, args).to("cuda")
    input_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
        )
    generated = output_ids[:, input_length:]
    response = processor.batch_decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")]
    return response.strip()


def write_run_config(
    args: argparse.Namespace,
    records: list[dict],
    metadata_by_idx: dict[int, dict],
    transformers_version: str,
    processor_budget_api: str,
    dtype_argument: str,
    resolved_dtype: str,
    system_prompt_injected: bool,
    first_image_budget: dict,
) -> None:
    config = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": args.condition,
        "prompt_bodies_sha256": prompt_bodies_sha256(
            records, metadata_by_idx, args.condition
        ),
        "input_jsonl": {
            "path": str(args.input_jsonl.resolve()),
            "sha256": sha256_file(args.input_jsonl),
        },
        "graph_metadata": {
            "path": str(args.graph_metadata.resolve()),
            "sha256": sha256_file(args.graph_metadata),
        },
        "triple_sort_order": ["source_name", "relation", "target_name"],
        "relation_text_source": (
            "RELATION_TEXT copied verbatim from scripts/generate_graphvis_datasets.py"
        ),
        "source_image_generation_flags": {
            "split": "test",
            "start": 0,
            "limit": 500,
            "tasks_per_graph": 6,
            "seed": 13,
            "max_nodes": 18,
            "max_edges": 30,
            "max_degree": 5,
            "engine": "dot",
            "dpi": 200,
            "disconnected_rows": 3,
            "hide_relatedto_labels": False,
            "reveal_correct_answer": False,
        },
        "source_image_generation_flags_provenance": (
            "Expected settings from the experiment README and shell history; "
            "the metadata file does not encode or independently verify them"
        ),
        "kg_text_graph_asymmetry": (
            "visible_nodes with connected=false are drawn in the image condition "
            "but omitted from kg_text because they produce no triples"
        ),
        "transformers_version": transformers_version,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "processor_pixel_budget_api": processor_budget_api,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "dtype_argument": dtype_argument,
        "resolved_dtype": resolved_dtype,
        "default_system_prompt_injected": system_prompt_injected,
        "first_image_budget": first_image_budget,
        "visual_budget_note": (
            "Qwen's pinned dynamic-resolution budget is not equivalent to "
            "DeepSeek-OCR-2 base_size=1024, image_size=768, crop_mode=True"
        ),
        "decoding_budget_note": (
            "Qwen max_new_tokens is 64 by default; DeepSeek-OCR-2 uses its "
            "remote-code value 8192. The budgets differ by necessity."
        ),
    }
    config_path = args.output_jsonl.parent / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                f"Refusing to overwrite incompatible run configuration: {config_path}"
            )
        return
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise SystemExit("Require 0 < --min-pixels <= --max-pixels")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not args.preview_only and not args.revision:
        raise SystemExit("--revision is required for Qwen inference")

    records = read_jsonl(args.input_jsonl)
    validate_records(records, args.expected_count)
    metadata_by_idx = index_graph_metadata(args.graph_metadata, records)

    from packaging.version import Version
    from PIL import Image
    import torch
    import transformers

    observed_version = transformers.__version__
    print(f"Observed transformers version: {observed_version}", flush=True)
    if Version(observed_version) < Version(REQUIRED_TRANSFORMERS_MIN):
        raise RuntimeError(
            f"Qwen3-VL requires transformers>={REQUIRED_TRANSFORMERS_MIN}; "
            f"found {observed_version}"
        )
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor, processor_budget_api = load_processor(AutoProcessor, args)
    first = min(records, key=lambda row: int(row["statement_idx"]))
    first_idx = int(first["statement_idx"])
    first_kg_block = format_kg_block(metadata_by_idx[first_idx])
    bodies = {
        condition: render_prompt(
            first["prompt"],
            condition,
            kg_block=first_kg_block if condition == "kg_text" else None,
        )
        for condition in CONDITIONS
    }
    assert_body_invariants(bodies, first_kg_block)
    formatted = {
        condition: format_qwen_prompt(processor, bodies[condition], condition)
        for condition in CONDITIONS
    }
    assert_qwen_formatted_invariants(processor, formatted)
    system_prompt_injected = print_prompt_preview(first_idx, bodies, formatted)

    first_image_path = args.image_root / first["image"]
    if not first_image_path.is_file():
        raise FileNotFoundError(f"Missing graph image: {first_image_path}")
    with Image.open(first_image_path) as opened_image:
        first_image = opened_image.convert("RGB")
        first_inputs = prepare_inputs(
            processor, formatted["image"], first_image, args
        )
        first_image_budget = image_budget_diagnostics(
            processor, first_inputs, first_image.size
        )

    if args.preview_only:
        print("Preview only: no model was loaded and no predictions were generated.", flush=True)
        return
    if not args.approve_prompt_diff:
        raise SystemExit(
            "Refusing to run inference until the prompt diff is approved. "
            "Re-run with --approve-prompt-diff after reviewing the preview."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU is visible. Run this through sbatch.")

    model, dtype_argument, resolved_dtype = load_model(
        Qwen3VLForConditionalGeneration, torch, args
    )
    done = completed_indices(args.output_jsonl) if args.resume else set()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(
        args,
        records,
        metadata_by_idx,
        observed_version,
        processor_budget_api,
        dtype_argument,
        resolved_dtype,
        system_prompt_injected,
        first_image_budget,
    )

    parse_tier_counts = Counter()
    if args.resume and args.output_jsonl.exists():
        parse_tier_counts.update(
            row.get("parse_tier", "MISSING") for row in read_jsonl(args.output_jsonl)
        )

    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for number, record in enumerate(records, start=1):
            statement_idx = int(record["statement_idx"])
            if statement_idx in done:
                continue

            kg_block = (
                format_kg_block(metadata_by_idx[statement_idx])
                if args.condition == "kg_text"
                else None
            )
            body = render_prompt(record["prompt"], args.condition, kg_block=kg_block)
            prompt_text = format_qwen_prompt(processor, body, args.condition)

            image = None
            if args.condition == "image":
                image_path = args.image_root / record["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing graph image: {image_path}")
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB")
            response = infer_one(model, processor, prompt_text, image, args, torch)
            parsed, parse_tier = parse_answer(response, n_choices=4)
            parse_tier_counts[parse_tier] += 1
            predicted = None if parsed == "FAILED" else parsed
            result = {
                "statement_idx": statement_idx,
                "image": record["image"],
                "gold_option": record["answer"],
                "predicted_option": predicted,
                "raw_response": response,
                "parse_tier": parse_tier,
                "is_correct": predicted == record["answer"],
                "model_id": args.model_id,
                "model_revision": args.revision,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(
                f"[{number}/{len(records)}] condition={args.condition} "
                f"q={statement_idx} gold={record['answer']} "
                f"predicted={predicted or 'FAILED'}",
                flush=True,
            )

    print(
        f"PARSE TIER DISTRIBUTION condition={args.condition}: "
        + json.dumps(dict(sorted(parse_tier_counts.items()))),
        flush=True,
    )


if __name__ == "__main__":
    main()
