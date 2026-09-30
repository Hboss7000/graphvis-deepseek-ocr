#!/usr/bin/env python3
"""Summarize Stage 1 format, extraction, accuracy, ceiling, and OCR diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

SCORER_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
)
sys.path.insert(0, str(SCORER_DIR))
import score_stage1 as scorer


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(label: str, predictions_dir: Path, answer_format: str) -> list[dict]:
    rows = []
    paths = sorted(predictions_dir.glob("predictions_*_*.jsonl"))
    if not paths:
        paths = sorted(predictions_dir.glob("scored_*_*.jsonl"))
    for path in paths:
        records = read_jsonl(path)
        if not records:
            continue
        task = str(records[0]["task_type"])
        model = str(records[0].get("model_name") or records[0].get("model_id") or "unknown")
        counts: defaultdict[str, int] = defaultdict(int)
        for record in records:
            diagnostics = scorer.answer_format_diagnostics(
                task, str(record.get("raw_response", "")), answer_format
            )
            counts["marker"] += int(diagnostics["final_answer_marker_found"])
            counts["marker_success"] += int(diagnostics["answer_marker_extraction_success"])
            counts["compliant"] += int(bool(diagnostics["final_format_compliant"]))
            counts["correct"] += int(bool(record.get("is_correct")))
            ceiling = bool(record.get("hit_token_ceiling"))
            counts["ceiling"] += int(ceiling)
            if ceiling:
                counts["ceiling_" + scorer.classify_ceiling_response(
                    task, str(record.get("raw_response", ""))
                )] += 1
            counts["ceiling_cutoff"] += int(
                answer_format == "constrained"
                and ceiling
                and not bool(diagnostics["final_format_compliant"])
            )
            counts["ocr"] += int(diagnostics["ocr_style_dump"])
        total = len(records)
        ratio = lambda key: counts[key] / total
        rows.append(
            {
                "run": label,
                "model": model,
                "task": task,
                "records": total,
                "final_format_compliant": counts["compliant"],
                "final_format_compliance_rate": (
                    ratio("compliant") if answer_format == "constrained" else None
                ),
                "answer_marker_found": counts["marker"],
                "marker_extraction_success": counts["marker_success"],
                "marker_extraction_success_rate": ratio("marker_success"),
                "strict_correct": counts["correct"],
                "strict_metric": ratio("correct"),
                "token_ceiling_hits": counts["ceiling"],
                "token_ceiling_hit_rate": ratio("ceiling"),
                "ceiling_hits_cutting_off_answer": counts["ceiling_cutoff"],
                "ceiling_progressing_reasoning": counts["ceiling_progressing_reasoning"],
                "ceiling_repetition_loop": counts["ceiling_repetition_loop"],
                "ceiling_answer_block_too_long": counts["ceiling_answer_block_too_long"],
                "ocr_style_dumps": counts["ocr"],
                "ocr_style_dump_rate": ratio("ocr"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No prediction JSONL files found in {predictions_dir}")
    return rows


def markdown(rows: list[dict], answer_format: str) -> str:
    lines = [
        "# Stage 1 answer-format report",
        "",
        f"Answer-format condition: `{answer_format}`.",
        "OCR/HTML is counted only; it is not specially parsed.",
        "",
        "| Run | Model | Task | N | Format | Marker extraction | Strict | Ceiling | Ceiling classes (progress/loop/answer) | Ceiling cut off answer | OCR/HTML |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    fmt = lambda count, rate: f"{count}/{rate:.1%}"
    for row in rows:
        compliance = (
            "n/a"
            if row["final_format_compliance_rate"] is None
            else fmt(row["final_format_compliant"], row["final_format_compliance_rate"])
        )
        lines.append(
            f"| {row['run']} | {row['model']} | {row['task']} | {row['records']} | "
            f"{compliance} | {fmt(row['marker_extraction_success'], row['marker_extraction_success_rate'])} | "
            f"{fmt(row['strict_correct'], row['strict_metric'])} | "
            f"{fmt(row['token_ceiling_hits'], row['token_ceiling_hit_rate'])} | "
            f"{row['ceiling_progressing_reasoning']}/{row['ceiling_repetition_loop']}/"
            f"{row['ceiling_answer_block_too_long']} | "
            f"{row['ceiling_hits_cutting_off_answer']} | "
            f"{fmt(row['ocr_style_dumps'], row['ocr_style_dump_rate'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Repeat for every model/cap output directory.",
    )
    parser.add_argument("--answer-format", choices=("none", "constrained"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for value in args.run:
        label, separator, directory = value.partition("=")
        if not separator or not label or not directory:
            parser.error(f"Invalid --run {value!r}; expected LABEL=DIR")
        rows.extend(summarize(label, Path(directory), args.answer_format))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"answer_format": args.answer_format, "rows": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown(rows, args.answer_format), encoding="utf-8")
    print(markdown(rows, args.answer_format), end="")


if __name__ == "__main__":
    main()
