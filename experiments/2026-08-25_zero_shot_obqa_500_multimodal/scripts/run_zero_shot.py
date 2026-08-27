#!/usr/bin/env python3
"""Run DeepSeek-OCR-2 zero-shot OBQA evaluation with or without graph images.

The image condition uses DeepSeek-OCR-2's validated remote-code ``infer`` call.
The text condition uses the same model, tokenizer, remote-code prompt formatter,
token encoder, and decoding configuration, but calls ``generate`` without any
image placeholder or image tensors.

The script always prints both rendered prompts for the first record. A full run
requires --approve-prompt-diff; use --preview-only for the approval smoke test.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


OPTIONS = ["A", "B", "C", "D", "E"]
REQUIRED_TRANSFORMERS_VERSION = "4.44.2"
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
        "--image-root",
        type=Path,
        required=True,
        help="Directory relative to which each input record's `image` path is resolved",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--infer-output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("image", "text"), required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print the first image/text prompt pair and exit without inference",
    )
    parser.add_argument(
        "--approve-prompt-diff",
        action="store_true",
        help="Confirm that the printed prompt diff was reviewed and permit inference",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(row["statement_idx"])
        for row in read_jsonl(path)
        if "statement_idx" in row
    }


def parse_answer(pred_text: str, n_choices: int) -> str:
    """Verbatim parser from eval_obqa_llava.py / LLaVA ScienceQA evaluation."""
    opts = OPTIONS[:n_choices]
    if pred_text in opts:
        return pred_text
    if len(pred_text) >= 3 and pred_text[0] in opts and pred_text[1:3] == ". ":
        return pred_text[0]
    res = re.compile(r"The answer is ([A-Z]).").findall(pred_text)
    if len(res) == 1:
        return res[0]
    return "FAILED"


def validate_records(records: list[dict], expected_count: int) -> None:
    required = {"statement_idx", "image", "prompt", "answer"}
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} input rows, found {len(records)}")
    indices = []
    for line_number, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Input row {line_number} lacks fields: {sorted(missing)}")
        indices.append(int(record["statement_idx"]))
    if len(set(indices)) != expected_count:
        raise ValueError("Input does not contain the expected number of unique statement_idx values")


def render_prompt(remote_module, prompt_body: str, with_image: bool) -> str:
    """Use DeepSeek's own SFT formatter; only the image marker may differ."""
    content = f"<image>\n{prompt_body}" if with_image else prompt_body
    conversation = [
        {"role": "<|User|>", "content": content},
        {"role": "<|Assistant|>", "content": ""},
    ]
    return remote_module.format_messages(
        conversations=conversation,
        sft_format="deepseek",
        system_prompt="",
    )


def print_prompt_preview(image_prompt: str, text_prompt: str, statement_idx: int) -> None:
    marker = "<image>\n"
    if image_prompt.count(marker) != 1 or image_prompt.replace(marker, "", 1) != text_prompt:
        raise RuntimeError(
            "Rendered prompts differ by more than the single expected <image> placeholder"
        )
    print(f"PROMPT PREVIEW: statement_idx={statement_idx}", flush=True)
    print("=" * 80, flush=True)
    print("IMAGE CONDITION (fully rendered)", flush=True)
    print("-" * 80, flush=True)
    print(image_prompt, flush=True)
    print("=" * 80, flush=True)
    print("TEXT CONDITION (fully rendered)", flush=True)
    print("-" * 80, flush=True)
    print(text_prompt, flush=True)
    print("=" * 80, flush=True)
    print("UNIFIED DIFF (text -> image)", flush=True)
    print("-" * 80, flush=True)
    print(
        "".join(
            difflib.unified_diff(
                text_prompt.splitlines(keepends=True),
                image_prompt.splitlines(keepends=True),
                fromfile="text",
                tofile="image",
            )
        ),
        end="",
        flush=True,
    )
    print("=" * 80, flush=True)


def infer_text_only(model, tokenizer, remote_module, rendered_prompt: str, torch) -> str:
    """Run the plain DeepSeek-V2 LM path with no image placeholder or tensors."""
    token_ids = remote_module.text_encode(
        tokenizer,
        rendered_prompt,
        bos=True,
        eos=False,
    )
    input_ids = torch.LongTensor(token_ids).unsqueeze(0).cuda()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                images_seq_mask=None,
                images_spatial_crop=None,
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


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    validate_records(records, args.expected_count)

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
    first_image_prompt = render_prompt(remote_module, first["prompt"], with_image=True)
    first_text_prompt = render_prompt(remote_module, first["prompt"], with_image=False)
    print_prompt_preview(first_image_prompt, first_text_prompt, int(first["statement_idx"]))

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

    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for number, record in enumerate(records, start=1):
            statement_idx = int(record["statement_idx"])
            if statement_idx in done:
                continue

            rendered_prompt = render_prompt(
                remote_module,
                record["prompt"],
                with_image=args.condition == "image",
            )
            if args.condition == "image":
                image_path = args.image_root / record["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing graph image: {image_path}")
                response = model.infer(
                    tokenizer,
                    prompt=rendered_prompt,
                    image_file=str(image_path),
                    output_path=str(args.infer_output_dir),
                    save_results=False,
                    eval_mode=True,
                    base_size=BASE_SIZE,
                    image_size=IMAGE_SIZE,
                    crop_mode=CROP_MODE,
                )
            else:
                response = infer_text_only(model, tokenizer, remote_module, rendered_prompt, torch)

            response = "" if response is None else str(response).strip()
            parsed = parse_answer(response, n_choices=4)
            predicted = None if parsed == "FAILED" else parsed
            result = {
                "statement_idx": statement_idx,
                "image": record["image"],
                "gold_option": record["answer"],
                "predicted_option": predicted,
                "raw_response": response,
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


if __name__ == "__main__":
    main()
