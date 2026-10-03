"""Shared validation, preview, output, and scoring helpers for Stage 1 runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from score_stage1 import (
    TASK_TYPES,
    TASK_SETS,
    NEW_TASK_TYPES,
    require_task,
    structured_gold,
    read_jsonl,
    score_files,
    write_failure_cases,
    write_json,
)


TASK_ORDER = {task: index for index, task in enumerate(TASK_TYPES)}
EXPECTED_SPLIT = "test"
EXPECTED_SOURCE = "graphvis_stage1_clean_union_of_four"
ANSWER_FORMATS = ("none", "constrained")
ANSWER_FORMAT_SEPARATOR = "\n\n"
LEGACY_PLACEHOLDER_ANSWER_SUFFIXES = {
    "node_number": (
        "You may reason before answering. Finish with exactly one final line and no text after it:\n"
        "Answer: <integer>"
    ),
    "edge_number": (
        "You may reason before answering. Finish with exactly one final line and no text after it:\n"
        "Answer: <integer>"
    ),
    "node_degree": (
        "You may reason before answering. Finish with exactly one final line and no text after it:\n"
        "Answer: <integer>"
    ),
    "highest_node_degree": (
        "You may reason before answering. Finish with exactly one final line and no text after it:\n"
        "Answer: <node>, <degree>"
    ),
    "node_description": (
        "You may reason before answering. Finish with exactly one final line and no text after it, "
        "listing every node exactly once:\n"
        "Answer: <node1>, <node2>, ..."
    ),
    "triple_listing": (
        "You may reason before answering. Finish with an answer block and no text after it. "
        "Put Answer: on its own line, then write exactly one triple per line:\n"
        "Answer:\n"
        "(head, relation, tail)\n"
        "..."
    ),
}
CONSTRAINED_ANSWER_SUFFIXES = {
    "node_number": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the integer. Do not write anything after that line."
    ),
    "edge_number": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the integer. Do not write anything after that line."
    ),
    "node_degree": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the integer. Do not write anything after that line."
    ),
    "highest_node_degree": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the node, a comma, and the integer degree. "
        "Do not write anything after that line."
    ),
    "node_description": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the comma-separated node names, listing every "
        "node exactly once. Do not write anything after that line."
    ),
    "triple_listing": (
        "You may reason before answering. Finish with a final answer block and no text "
        "after it. Put 'Answer:' on its own line, then one triple per line in the form "
        "(head, relation, tail)."
    ),
}

DEEPSEEK_PROMPT_VARIANTS = (
    "standard", "old-placeholder", "completion", "describe"
)
DEEPSEEK_COMPLETION_PREFIXES = {
    "node_number": "The total number of nodes in the graph is",
    "edge_number": "The total number of edges in the graph is",
    "node_description": "The image depicts the following nodes:",
    "triple_listing": "The triples in the graph are listed as:",
    "highest_node_degree": "One node with the highest degree is",
}


def validate_stage1(
    input_path: Path,
    metadata_path: Path,
    expected_count: int | None,
    task_types=TASK_TYPES,
    expected_split: str = EXPECTED_SPLIT,
) -> tuple[list[dict], dict[int, dict], dict[str, int]]:
    records = read_jsonl(input_path)
    task_order = {task: index for index, task in enumerate(task_types)}
    required = {"image", "split", "statement_idx", "task_type", "prompt", "answer", "source"}
    seen = set()
    counts = Counter()
    images_by_idx: dict[int, str] = {}
    for line_number, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Stage 1 row {line_number} lacks fields: {sorted(missing)}")
        task = str(record["task_type"])
        require_task(task, task_types)
        if task in NEW_TASK_TYPES:
            structured_gold(record)
        if record["split"] != expected_split:
            raise ValueError(
                f"Unexpected split at row {line_number}: {record['split']!r}; "
                f"expected {expected_split!r}"
            )
        if record["source"] != EXPECTED_SOURCE:
            raise ValueError(
                f"Unexpected source at row {line_number}: {record['source']!r}"
            )
        idx = int(record["statement_idx"])
        key = (idx, task)
        if key in seen:
            raise ValueError(f"Duplicate Stage 1 key: {key}")
        seen.add(key)
        counts[task] += 1
        image = str(record["image"])
        if idx in images_by_idx and images_by_idx[idx] != image:
            raise ValueError(f"Stage 1 image mismatch within statement_idx={idx}")
        images_by_idx[idx] = image

    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} Stage 1 rows, found {len(records)}")
    metadata_by_idx = {}
    for meta in read_jsonl(metadata_path):
        idx = int(meta["statement_idx"])
        if idx in metadata_by_idx:
            raise ValueError(f"Duplicate metadata statement_idx={idx}")
        metadata_by_idx[idx] = meta
    missing_metadata = sorted(set(images_by_idx) - metadata_by_idx.keys())
    if missing_metadata:
        raise ValueError(f"Metadata is missing statement_idx values: {missing_metadata[:10]}")
    for idx, image in images_by_idx.items():
        if str(metadata_by_idx[idx]["image"]) != image:
            raise ValueError(f"Stage 1/metadata image mismatch at statement_idx={idx}")

    ordered_counts = {task: counts[task] for task in task_types}
    print("STAGE 1 TASK COUNTS: " + json.dumps(ordered_counts), flush=True)
    records.sort(key=lambda row: (int(row["statement_idx"]), task_order[row["task_type"]]))
    return records, metadata_by_idx, ordered_counts


def answer_format_suffix(task: str, answer_format: str) -> str:
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"Unknown answer format: {answer_format!r}")
    if answer_format == "none":
        return ""
    try:
        return CONSTRAINED_ANSWER_SUFFIXES[task]
    except KeyError as error:
        raise ValueError(
            f"No constrained answer format is defined for task {task!r}; "
            "use --answer-format none"
        ) from error


def answer_format_provenance(answer_format: str, task_types=TASK_TYPES) -> dict:
    suffixes = {
        task: answer_format_suffix(task, answer_format)
        for task in task_types
    }
    return {
        "mode": answer_format,
        "application": "Inference-time only; source record prompts are not modified.",
        "separator": ANSWER_FORMAT_SEPARATOR if answer_format == "constrained" else "",
        "suffixes": suffixes,
    }


def raw_image_prompt(record: dict, answer_format: str = "none") -> str:
    original = str(record["prompt"])
    suffix = answer_format_suffix(str(record["task_type"]), answer_format)
    body = original if not suffix else original + ANSWER_FORMAT_SEPARATOR + suffix
    return f"<image>\n{body}"


def deepseek_image_prompt(
    record: dict, answer_format: str = "none", prompt_variant: str = "standard"
) -> str:
    """Build one DeepSeek ablation prompt without changing source records."""
    if prompt_variant not in DEEPSEEK_PROMPT_VARIANTS:
        raise ValueError(f"Unknown DeepSeek prompt variant: {prompt_variant!r}")
    if prompt_variant == "standard":
        return raw_image_prompt(record, answer_format)
    if prompt_variant == "old-placeholder":
        suffix = LEGACY_PLACEHOLDER_ANSWER_SUFFIXES[str(record["task_type"])]
        return f"<image>\n{record['prompt']}{ANSWER_FORMAT_SEPARATOR}{suffix}"
    if prompt_variant == "describe":
        return "<image>\nDescribe this image in detail."
    task = str(record["task_type"])
    if task == "node_degree":
        match = re.search(
            r'node(?:\s+labeled)?\s+["“]([^"”]+)["”]', str(record["prompt"]),
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Cannot identify node-degree entity: {record['prompt']!r}")
        prefix = f'The degree of the node "{match.group(1)}" is'
    else:
        prefix = DEEPSEEK_COMPLETION_PREFIXES[task]
    return f"<image>\n{prefix}"


def prompts_sha256(records: list[dict], builder) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(builder(record), ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def prompt_bodies_sha256(records: list[dict], answer_format: str = "none") -> str:
    digest = hashlib.sha256()
    for record in records:
        line = json.dumps(raw_image_prompt(record, answer_format), ensure_ascii=False) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def print_six_prompt_previews(
    records: list[dict],
    formatter=None,
    task_types=TASK_TYPES,
    answer_format: str = "none",
    prompt_builder=None,
) -> dict[str, str]:
    first_by_task = {}
    for record in records:
        first_by_task.setdefault(record["task_type"], record)
    missing = [task for task in task_types if task not in first_by_task and task not in NEW_TASK_TYPES]
    if missing:
        raise ValueError(f"Cannot preview absent task types: {missing}")
    previews = {}
    for task in task_types:
        if task not in first_by_task:
            continue
        record = first_by_task[task]
        body = prompt_builder(record) if prompt_builder is not None else raw_image_prompt(record, answer_format)
        if body.count("<image>\n") != 1 or not body.startswith("<image>\n"):
            raise RuntimeError(f"Invalid Stage 1 image prompt for task={task}")
        preview = formatter(body) if formatter is not None else body
        previews[task] = preview
        print("=" * 80, flush=True)
        print(
            f"PROMPT PREVIEW task={task} statement_idx={record['statement_idx']}",
            flush=True,
        )
        print("-" * 80, flush=True)
        print(preview, flush=True)
    print("=" * 80, flush=True)
    return previews


def output_path(output_dir: Path, model_name: str, task: str) -> Path:
    return output_dir / f"predictions_{model_name}_{task}.jsonl"


def completed_keys(output_dir: Path, model_name: str, task_types=TASK_TYPES) -> set[tuple[int, str]]:
    completed = set()
    for task in task_types:
        path = output_path(output_dir, model_name, task)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = (int(row["statement_idx"]), str(row["task_type"]))
            if key in completed:
                raise ValueError(f"Duplicate existing prediction key: {key}")
            if key[1] != task:
                raise ValueError(f"Prediction stored in wrong task file: {key} in {path}")
            completed.add(key)
    return completed


def write_run_config(output_dir: Path, config: dict) -> None:
    # Only matrix launches set this variable; standalone evaluator behavior stays intact.
    cell_json = os.environ.get('STAGE1_MATRIX_CELL_JSON')
    if cell_json:
        scripts = Path(__file__).resolve().parents[3] / 'scripts'
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from build_pruning_matrix import run_config_fields, validate_run_config, sha_object
        cell = json.loads(cell_json)
        for key in ('input_jsonl', 'graph_metadata'):
            if Path(config[key]['path']).resolve() != Path(cell[key]).resolve():
                raise ValueError(f'Matrix evaluator used a different {key} path')
        config = {**config, **run_config_fields(cell)}
        config['graph_metadata'] = {**config['graph_metadata'],
            'path_sha256': sha_object(config['graph_metadata']['path'])}
        config['source_image_generation'] = {
            'generation': cell['generation'], 'pruning': cell['pruning_flags'],
            'source_sha256': cell['source_sha256'],
            'settings_provenance': 'Validated pruning matrix manifest and per-graph metadata.'}
        validate_run_config(cell, config, preflight=os.environ.get('STAGE1_MATRIX_PHASE') == 'preflight')
    path = output_dir / "run_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Refusing to overwrite incompatible run config: {path}")
        return
    write_json(path, config)


def score_completed_run(
    input_jsonl: Path,
    graph_metadata: Path,
    output_dir: Path,
    model_name: str,
    task_set: str = "paper",
    extractor: str = "span",
) -> None:
    task_types = TASK_SETS[task_set]
    scoring_args = argparse.Namespace(
        input_jsonl=input_jsonl,
        graph_metadata=graph_metadata,
        predictions_dir=output_dir,
        model_name=model_name,
        task_set=task_set,
        extractor=extractor,
    )
    metrics, rows_by_task = score_files(scoring_args)
    metrics_path = output_dir / f"metrics_{model_name}.json"
    failure_path = output_dir / f"failure_cases_{model_name}.md"
    write_json(metrics_path, metrics)
    write_failure_cases(failure_path, model_name, rows_by_task, task_types)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {failure_path}", flush=True)
    ceilings = {
        task: metrics["tasks"][task]["hit_token_ceiling_fraction"]
        for task in task_types
    }
    print("TOKEN CEILING FRACTIONS: " + json.dumps(ceilings, sort_keys=True), flush=True)
