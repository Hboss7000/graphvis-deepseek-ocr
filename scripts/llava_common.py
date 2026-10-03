#!/usr/bin/env python3
"""Shared LLaVA-NeXT prompt, resume, and image-budget helpers."""

from __future__ import annotations

import json
from pathlib import Path


MODEL_NAME = "llava"
DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DEFAULT_REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"


def configure_processor(processor, model_config) -> dict:
    """Supply model-derived token-expansion fields absent from older processor JSON."""
    resolved = {
        "patch_size": int(model_config.vision_config.patch_size),
        "vision_feature_select_strategy": str(model_config.vision_feature_select_strategy),
        "num_additional_image_tokens": 1,
    }
    for name, value in resolved.items():
        setattr(processor, name, value)
    return resolved


def build_messages(body: str, condition: str) -> list[dict]:
    if condition == "image":
        marker = "<image>\n"
        if not body.startswith(marker) or body.count(marker) != 1:
            raise ValueError("Shared image body must start with exactly one <image> marker")
        content = body
    else:
        if "<image>" in body:
            raise ValueError("Text-only LLaVA prompt unexpectedly contains <image>")
        content = body
    return [{"role": "user", "content": content}]


def format_prompt(processor, body: str, condition: str) -> str:
    template_owner = getattr(processor, "tokenizer", processor)
    return template_owner.apply_chat_template(
        build_messages(body, condition), tokenize=False, add_generation_prompt=True
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_stage1_keys(output_dir: Path, task_types) -> set[tuple[int, str]]:
    completed: set[tuple[int, str]] = set()
    for task in task_types:
        path = output_dir / f"predictions_{MODEL_NAME}_{task}.jsonl"
        for row in read_jsonl(path):
            key = (int(row["statement_idx"]), str(row["task_type"]))
            if key in completed:
                raise ValueError(f"Duplicate existing prediction key: {key}")
            if key[1] != task:
                raise ValueError(f"Prediction stored in wrong task file: {key} in {path}")
            completed.add(key)
    return completed


def completed_qa_indices(path: Path) -> set[int]:
    rows = read_jsonl(path)
    indices = [int(row["statement_idx"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate existing QA prediction key in {path}")
    return set(indices)


def select_best_resolution(original_size, possible_resolutions):
    """Mirror Transformers' anyres choice and return the selected grid pinpoint."""
    original_height, original_width = original_size
    best_fit = None
    max_effective = -1
    min_wasted = float("inf")
    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted = width * height - effective
        if effective > max_effective or (effective == max_effective and wasted < min_wasted):
            best_fit = [int(height), int(width)]
            max_effective = effective
            min_wasted = wasted
    return best_fit


def image_diagnostics(processor, inputs, source_size) -> dict:
    pinpoints = [list(map(int, pair)) for pair in processor.image_processor.image_grid_pinpoints]
    image_sizes = inputs.get("image_sizes")
    encoded_size = (
        [int(value) for value in image_sizes[0].tolist()]
        if image_sizes is not None else None
    )
    realised_tokens = int((inputs["input_ids"] == image_token_id(processor)).sum().item())
    pixel_values = inputs.get("pixel_values")
    return {
        "source_size": {"width": int(source_size[0]), "height": int(source_size[1])},
        "processor_image_sizes": encoded_size,
        "configured_grid_pinpoints": pinpoints,
        "selected_grid_pinpoint": dict(zip(
            ("height", "width"),
            select_best_resolution((source_size[1], source_size[0]), pinpoints),
        )),
        "pixel_value_tiles": int(pixel_values.shape[1]) if pixel_values is not None and pixel_values.ndim == 5 else (
            int(pixel_values.shape[0]) if pixel_values is not None else None
        ),
        "realised_image_tokens": realised_tokens,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
    }


def prepare_inputs(processor, prompt_text: str, image):
    kwargs = {"text": prompt_text, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    inputs = processor(**kwargs)
    inputs.pop("token_type_ids", None)
    return inputs


def image_token_id(processor) -> int:
    value = getattr(processor, "image_token_id", None)
    if value is None:
        value = processor.tokenizer.convert_tokens_to_ids("<image>")
    return int(value)
