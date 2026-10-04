#!/usr/bin/env python3
"""Run Gemma 3 zero-shot OBQA evaluation with the unchanged Qwen prompt/scoring contract."""

from __future__ import annotations

import sys as _fullrun_sys
from pathlib import Path as _FullrunPath
_fullrun_sys.path.insert(0, str(_FullrunPath(__file__).resolve().parents[1] / "scripts"))
from fullrun_runtime import enrich_config, begin_item, finish_item

import argparse
import difflib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gemma3_common as gemma

from run_zero_shot_qwen import assert_body_invariants
from prompt_common import (
    CONDITIONS,
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


DEFAULT_MODEL_ID = gemma.DEFAULT_MODEL_ID
MAX_NEW_TOKENS = 64


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
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, help="Process only the first N records; input identity stays fixed for resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--approve-prompt-diff", action="store_true")
    gemma.add_common_args(parser)
    return parser.parse_args()


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
    processor_details: dict,
    preflight_provenance: dict | None,
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
        "effective_max_new_tokens": args.max_new_tokens,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
        "transformers_version": transformers_version,
        "processor_pixel_budget_api": processor_budget_api,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_impl,
        "dtype_argument": dtype_argument,
        "resolved_dtype": resolved_dtype,
        "default_system_prompt_injected": system_prompt_injected,
        "first_image_budget": first_image_budget,
        "visual_budget_note": gemma.RESOLUTION_NOTE,
        "decoding_budget_note": (
            "Phase-1 and planned QA runs use the common 64-token ceiling for every model."
        ),
    }
    config.update(gemma.manifest_fields(args, processor_details, resolved_dtype))
    config['preflight_report'] = preflight_provenance
    config = enrich_config(config)
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
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    gemma.validate_args(args)

    records = read_jsonl(args.input_jsonl)
    validate_records(records, args.expected_count)
    metadata_by_idx = index_graph_metadata(args.graph_metadata, records)

    from PIL import Image
    import torch
    import transformers
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    observed_version = transformers.__version__
    processor, processor_details = gemma.load_processor(AutoProcessor, args)
    processor_budget_api = 'fixed_896_plus_pan_and_scan'
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
        condition: gemma.format_prompt(processor, bodies[condition], condition)
        for condition in CONDITIONS
    }
    system_prompt_injected = False
    for condition in CONDITIONS:
        print(f'SHARED BODY: {condition}\n{bodies[condition]}', flush=True)
        print(f'GEMMA-FORMATTED PROMPT: {condition}\n{formatted[condition]}', flush=True)
    for target in ('image', 'kg_text'):
        print(f'UNIFIED DIFF (Gemma text_noref -> {target})', flush=True)
        print(''.join(difflib.unified_diff(
            formatted['text_noref'].splitlines(keepends=True),
            formatted[target].splitlines(keepends=True),
            fromfile='text_noref', tofile=target)), flush=True)
    first_image_budget = {}
    # Text inference must neither load nor tokenise an image.
    if args.condition == 'image':
        first_image_path = args.image_root / first['image']
        with Image.open(first_image_path) as opened_image:
            first_image = opened_image.convert('RGB')
            first_inputs = gemma.prepare_inputs(
                processor, bodies['image'], 'image', first_image, args, processor_details)
            first_image_budget = gemma.image_budget_diagnostics(processor, first_inputs, first_image.size)

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

    preflight_provenance = (gemma.require_preflight(args, processor_details)
                            if args.condition == 'image' else None)
    model, dtype_argument, resolved_dtype = gemma.load_model(
        Gemma3ForConditionalGeneration, torch, args
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
        processor_details,
        preflight_provenance,
    )

    parse_tier_counts = Counter()
    if args.resume and args.output_jsonl.exists():
        parse_tier_counts.update(
            row.get("parse_tier", "MISSING") for row in read_jsonl(args.output_jsonl)
        )

    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for number, record in enumerate(records[:args.limit], start=1):
            statement_idx = int(record["statement_idx"])
            if statement_idx in done:
                continue
            begin_item()

            kg_block = (
                format_kg_block(metadata_by_idx[statement_idx])
                if args.condition == "kg_text"
                else None
            )
            body = render_prompt(record["prompt"], args.condition, kg_block=kg_block)

            image = None
            if args.condition == "image":
                image_path = args.image_root / record["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing graph image: {image_path}")
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB")
            response, generated_tokens, hit_ceiling, image_views, image_soft_tokens = gemma.infer_one(
                model, processor, body, args.condition, image, args, torch, processor_details)
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
                "generated_token_count": generated_tokens,
                "hit_token_ceiling": hit_ceiling,
                "vision_tokens_per_item": image_soft_tokens,
                "model_id": args.model_id,
                "model_revision": args.revision,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "image_views": image_views,
                "image_soft_tokens": image_soft_tokens,
            }
            finish_item(result)
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
