#!/usr/bin/env python3
"""CPU-only deterministic Stage 1 model stub for session-script dry runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCORER_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
)
sys.path.insert(0, str(SCORER_DIR))

from score_stage1 import TASK_SETS, answer_format_diagnostics, score_record  # noqa: E402
from stage1_common import score_completed_run, validate_stage1  # noqa: E402


MODEL_NAMES = {
    "deepseek": "deepseek",
    "qwen": "qwen",
    "gemma": "gemma3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--task-set", choices=TASK_SETS, default="paper")
    parser.add_argument("--answer-format", choices=("none", "constrained"), required=True)
    parser.add_argument("--prompt-variant", default="standard")
    parser.add_argument("--behavior-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_for(record: dict, metadata: dict, answer_format: str) -> str:
    answer = str(record["answer"]).strip()
    if answer_format != "constrained":
        return answer
    gold = score_record(record, answer, metadata)
    task = str(record["task_type"])
    if task in {"node_number", "edge_number", "node_degree"}:
        payload = str(gold["gold_integer"])
    elif task == "highest_node_degree":
        payload = f'{gold["valid_highest_names"][0]}, {gold["gold_degree"]}'
    elif task == "node_description":
        payload = ", ".join(gold["parsed_gold_nodes"])
    elif task == "triple_listing":
        payload = "\n".join(f"({', '.join(triple)})" for triple in gold["parsed_gold_triples"])
        return "Answer:\n" + payload
    else:
        raise ValueError(f"No constrained stub response for task {task!r}")
    return "Answer: " + payload


def main() -> None:
    args = parse_args()
    tasks = TASK_SETS[args.task_set]
    records, metadata_by_idx, task_counts = validate_stage1(
        args.input_jsonl,
        args.graph_metadata,
        args.expected_count,
        tasks,
        expected_split=args.expected_split,
    )
    missing = [str(args.image_root / row["image"]) for row in records
               if not (args.image_root / row["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rendered graph image: {missing[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dry_run": True,
        "inference": "deterministic CPU stub; no model loaded",
        "model_name": MODEL_NAMES[args.model],
        "model_id": args.model_id,
        "model_revision": args.revision,
        "input_jsonl": {
            "path": str(args.input_jsonl.resolve()),
            "sha256": sha256(args.input_jsonl),
            "record_count": len(records),
            "task_counts": task_counts,
        },
        "graph_metadata": {
            "path": str(args.graph_metadata.resolve()),
            "sha256": sha256(args.graph_metadata),
            "record_count": len(metadata_by_idx),
        },
        "expected_split": args.expected_split,
        "task_set": args.task_set,
        "answer_format": args.answer_format,
        "prompt_variant": args.prompt_variant,
        "behavior_only": args.behavior_only,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.preflight:
        report = {
            "dry_run": True,
            "passed": True,
            "records_checked": len(records),
            "recall": 1.0,
            "note": "CPU stub preflight; rendered-image existence was checked.",
        }
        (args.output_dir / "preflight_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"STUB PREFLIGHT COMPLETE: checked {len(records)} rendered inputs", flush=True)
        return

    model_name = MODEL_NAMES[args.model]
    handles = {
        task: (args.output_dir / f"predictions_{model_name}_{task}.jsonl").open(
            "w", encoding="utf-8"
        )
        for task in tasks
    }
    try:
        for position, record in enumerate(records, start=1):
            idx = int(record["statement_idx"])
            task = str(record["task_type"])
            response = response_for(record, metadata_by_idx[idx], args.answer_format)
            diagnostics = answer_format_diagnostics(task, response, args.answer_format)
            result = {
                "statement_idx": idx,
                "task_type": task,
                "image": record["image"],
                "prompt": record["prompt"],
                "gold": record["answer"],
                **({"structured_gold": record["gold"]}
                   if isinstance(record.get("gold"), dict) else {}),
                "raw_response": response,
                "model_infer_returned_none": False,
                "model_infer_returned_empty_string": False,
                "normalized_response_empty": False,
                "generation_elapsed_seconds": 0.0,
                "generated_token_count_approx": len(response.split()),
                "generated_token_count": len(response.split()),
                "hit_token_ceiling": False,
                "ceiling_hit_before_complete_answer": False,
                **diagnostics,
                **score_record(record, response, metadata_by_idx[idx]),
                "model_id": args.model_id,
                "model_revision": args.revision,
                "dry_run": True,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            handles[task].write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"STUB [{position}/{len(records)}] model={args.model} "
                f"task={task} statement_idx={idx} correct_raw={result['is_correct']}",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    print(f"STUB INFERENCE COMPLETE: {len(records)} predictions", flush=True)
    if args.behavior_only:
        print("Behavior-only ablation complete; aggregate task scoring intentionally skipped.")
    else:
        score_completed_run(
            args.input_jsonl, args.graph_metadata, args.output_dir, model_name, args.task_set
        )


if __name__ == "__main__":
    main()
