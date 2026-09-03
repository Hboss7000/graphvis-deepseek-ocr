#!/usr/bin/env python3
"""Run DeepSeek-OCR-2 zero-shot OBQA evaluation with image or text graphs.

The image condition uses DeepSeek-OCR-2's validated remote-code ``infer`` call.
The text conditions use the same model, tokenizer, remote-code prompt formatter,
token encoder, and decoding configuration, but calls ``generate`` without any
image placeholder or image tensors.

The script always prints the three baseline prompts for the first record, plus the
KG-text prompt when metadata is supplied. A full run requires
--approve-prompt-diff; use --preview-only for the approval smoke test.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prompt_common import (
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

REQUIRED_TRANSFORMERS_VERSION = "4.46.3"
BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True

# These are the values hard-coded by DeepSeek-OCR-2 infer(..., eval_mode=True).
MAX_NEW_TOKENS = 8192
NO_REPEAT_NGRAM_SIZE = 35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision/commit")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--graph-metadata",
        type=Path,
        help=(
            "graph_metadata_*.jsonl providing the pruned subgraph; "
            "required for --condition kg_text"
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory relative to which each input record's `image` path is resolved",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--infer-output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("image", "text", "text_noref", "kg_text"),
        required=True,
    )
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Print the first image/text/text_noref prompts, plus kg_text when "
            "metadata is supplied, and exit without inference"
        ),
    )
    parser.add_argument(
        "--approve-prompt-diff",
        action="store_true",
        help="Confirm that the printed prompt diff was reviewed and permit inference",
    )
    return parser.parse_args()


def format_plain_prompt(remote_module, raw_prompt: str) -> str:
    """Replicate the single plain-formatting step inside model.infer()."""
    conversation = [
        {"role": "<|User|>", "content": raw_prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    return remote_module.format_messages(
        conversations=conversation,
        sft_format="plain",
        system_prompt="",
    )


def print_prompt_preview(
    image_raw_prompt: str,
    image_plain_prompt: str,
    text_plain_prompt: str,
    text_noref_plain_prompt: str,
    statement_idx: int,
    kg_text_plain_prompt: str | None = None,
    kg_block: str | None = None,
) -> None:
    marker = "<image>\n"
    if (
        image_plain_prompt.count(marker) != 1
        or image_plain_prompt.replace(marker, "", 1) != text_plain_prompt
    ):
        raise RuntimeError(
            "Rendered prompts differ by more than the single expected <image> placeholder"
        )
    if (
        text_plain_prompt.count(IMAGE_REFERENCE_SENTENCE) != 1
        or text_plain_prompt.replace(IMAGE_REFERENCE_SENTENCE, "", 1)
        != text_noref_plain_prompt
    ):
        raise RuntimeError(
            "text and text_noref prompts differ by more than IMAGE_REFERENCE_SENTENCE"
        )
    if (kg_text_plain_prompt is None) != (kg_block is None):
        raise RuntimeError("KG-text preview requires both its prompt and triple block")
    if kg_text_plain_prompt is not None and kg_block is not None:
        kg_prefix = f"{KG_TEXT_REFERENCE_SENTENCE}\n{kg_block}\n\n"
        if (
            kg_text_plain_prompt.count(kg_prefix) != 1
            or kg_text_plain_prompt.replace(kg_prefix, "", 1)
            != text_noref_plain_prompt
        ):
            raise RuntimeError(
                "kg_text and text_noref prompts differ by more than "
                "KG_TEXT_REFERENCE_SENTENCE plus the triple block"
            )
        assert_no_answer_letter_annotation(kg_text_plain_prompt)
    print(f"PROMPT PREVIEW: statement_idx={statement_idx}", flush=True)
    print("=" * 80, flush=True)
    print("IMAGE CONDITION (raw body passed to model.infer)", flush=True)
    print("-" * 80, flush=True)
    print(image_raw_prompt, flush=True)
    print("=" * 80, flush=True)
    print("IMAGE CONDITION (effective plain-formatted model input)", flush=True)
    print("-" * 80, flush=True)
    print(image_plain_prompt, flush=True)
    print("=" * 80, flush=True)
    print("TEXT CONDITION (effective plain-formatted model input)", flush=True)
    print("-" * 80, flush=True)
    print(text_plain_prompt, flush=True)
    print("=" * 80, flush=True)
    print("TEXT_NOREF CONDITION (effective plain-formatted model input)", flush=True)
    print("-" * 80, flush=True)
    print(text_noref_plain_prompt, flush=True)
    print("=" * 80, flush=True)
    if kg_text_plain_prompt is not None:
        print("KG_TEXT CONDITION (effective plain-formatted model input)", flush=True)
        print("-" * 80, flush=True)
        print(kg_text_plain_prompt, flush=True)
        print("=" * 80, flush=True)
    print("UNIFIED DIFF (effective text input -> effective image input)", flush=True)
    print("-" * 80, flush=True)
    print(
        "".join(
            difflib.unified_diff(
                text_plain_prompt.splitlines(keepends=True),
                image_plain_prompt.splitlines(keepends=True),
                fromfile="text",
                tofile="image",
            )
        ),
        end="",
        flush=True,
    )
    print("=" * 80, flush=True)
    if kg_text_plain_prompt is not None:
        print(
            "UNIFIED DIFF (effective text_noref input -> effective kg_text input)",
            flush=True,
        )
        print("-" * 80, flush=True)
        print(
            "".join(
                difflib.unified_diff(
                    text_noref_plain_prompt.splitlines(keepends=True),
                    kg_text_plain_prompt.splitlines(keepends=True),
                    fromfile="text_noref",
                    tofile="kg_text",
                )
            ),
            end="",
            flush=True,
        )
        print("=" * 80, flush=True)
    print("UNIFIED DIFF (effective text input -> effective text_noref input)", flush=True)
    print("-" * 80, flush=True)
    print(
        "".join(
            difflib.unified_diff(
                text_plain_prompt.splitlines(keepends=True),
                text_noref_plain_prompt.splitlines(keepends=True),
                fromfile="text",
                tofile="text_noref",
            )
        ),
        end="",
        flush=True,
    )
    print("=" * 80, flush=True)


def infer_text_only(model, tokenizer, remote_module, raw_prompt: str, torch) -> str:
    """Run the text LM with zero compatibility tensors that bypass vision."""
    rendered_prompt = format_plain_prompt(remote_module, raw_prompt)
    token_ids = remote_module.text_encode(
        tokenizer,
        rendered_prompt,
        bos=True,
        eos=False,
    )
    input_ids = torch.LongTensor(token_ids).unsqueeze(0).cuda()
    images_ori = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    images_crop = torch.zeros((1, 3, BASE_SIZE, BASE_SIZE))
    images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[(images_crop.cuda(), images_ori.cuda())],
                images_seq_mask=torch.zeros_like(input_ids, dtype=torch.bool).cuda(),
                images_spatial_crop=images_spatial_crop,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                use_cache=True,
            )

    response = tokenizer.decode(output_ids[0, input_ids.shape[1] :])
    stop_str = "<｜end▁of▁sentence｜>"
    if response.endswith(stop_str):
        response = response[: -len(stop_str)]
    return response.strip()


def write_kg_text_run_config(args: argparse.Namespace) -> None:
    """Write an idempotent provenance record beside KG-text predictions."""
    if args.graph_metadata is None:
        raise RuntimeError("Cannot write kg_text run_config without graph metadata")
    config_records = read_jsonl(args.input_jsonl)
    config_metadata = index_graph_metadata(args.graph_metadata, config_records)
    config = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": args.condition,
        "prompt_bodies_sha256": prompt_bodies_sha256(
            config_records,
            config_metadata,
            args.condition,
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
            "RELATION_TEXT copied verbatim from "
            "scripts/generate_graphvis_datasets.py"
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
    if args.condition == "kg_text" and args.graph_metadata is None:
        raise SystemExit("--graph-metadata is required for --condition kg_text")

    records = read_jsonl(args.input_jsonl)
    validate_records(records, args.expected_count)
    metadata_by_idx = (
        index_graph_metadata(args.graph_metadata, records)
        if args.graph_metadata is not None
        else None
    )

    # Heavy imports intentionally remain inside main so --help works on login nodes.
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This experiment requires transformers=={REQUIRED_TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU is visible. Run this through sbatch.")

    load_kwargs = {
        "trust_remote_code": True,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **load_kwargs)
    model_load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "_attn_implementation": "eager",
        "use_safetensors": True,
    }
    if args.revision:
        model_load_kwargs["revision"] = args.revision
    model = AutoModel.from_pretrained(args.model_id, **model_load_kwargs)
    model = model.eval().cuda().to(torch.bfloat16)
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1

    remote_module = importlib.import_module(model.__class__.__module__)
    first = records[0]
    first_image_raw_prompt = render_prompt(first["prompt"], condition="image")
    first_text_raw_prompt = render_prompt(first["prompt"], condition="text")
    first_text_noref_raw_prompt = render_prompt(first["prompt"], condition="text_noref")
    first_image_plain_prompt = format_plain_prompt(remote_module, first_image_raw_prompt)
    first_text_plain_prompt = format_plain_prompt(remote_module, first_text_raw_prompt)
    first_text_noref_plain_prompt = format_plain_prompt(
        remote_module, first_text_noref_raw_prompt
    )
    first_kg_block = None
    first_kg_text_plain_prompt = None
    if metadata_by_idx is not None:
        first_statement_idx = int(first["statement_idx"])
        first_kg_block = format_kg_block(metadata_by_idx[first_statement_idx])
        first_kg_text_raw_prompt = render_prompt(
            first["prompt"], condition="kg_text", kg_block=first_kg_block
        )
        first_kg_text_plain_prompt = format_plain_prompt(
            remote_module, first_kg_text_raw_prompt
        )
    print_prompt_preview(
        first_image_raw_prompt,
        first_image_plain_prompt,
        first_text_plain_prompt,
        first_text_noref_plain_prompt,
        int(first["statement_idx"]),
        kg_text_plain_prompt=first_kg_text_plain_prompt,
        kg_block=first_kg_block,
    )

    if args.preview_only:
        print("Preview only: no predictions were generated.", flush=True)
        return
    if not args.approve_prompt_diff:
        raise SystemExit(
            "Refusing to run inference until the prompt diff is approved. "
            "Re-run with --approve-prompt-diff after reviewing the preview."
        )

    done = completed_indices(args.output_jsonl) if args.resume else set()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.infer_output_dir.mkdir(parents=True, exist_ok=True)
    if args.condition == "kg_text":
        write_kg_text_run_config(args)

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

            if args.condition == "kg_text":
                if metadata_by_idx is None:
                    raise RuntimeError("kg_text metadata was not loaded")
                kg_block = format_kg_block(metadata_by_idx[statement_idx])
                raw_prompt = render_prompt(
                    record["prompt"], condition="kg_text", kg_block=kg_block
                )
            else:
                raw_prompt = render_prompt(record["prompt"], condition=args.condition)
            if args.condition == "image":
                image_path = args.image_root / record["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing graph image: {image_path}")
                response = model.infer(
                    tokenizer,
                    prompt=raw_prompt,
                    image_file=str(image_path),
                    output_path=str(args.infer_output_dir),
                    save_results=False,
                    eval_mode=True,
                    base_size=BASE_SIZE,
                    image_size=IMAGE_SIZE,
                    crop_mode=CROP_MODE,
                )
            else:
                response = infer_text_only(model, tokenizer, remote_module, raw_prompt, torch)

            response = "" if response is None else str(response).strip()
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
