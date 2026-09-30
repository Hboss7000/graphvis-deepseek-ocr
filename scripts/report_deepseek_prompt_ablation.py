#!/usr/bin/env python3
"""Report DeepSeek prompt-ablation behavior from prediction JSONL files."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCORER_DIR = Path(__file__).resolve().parents[1] / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
sys.path.insert(0, str(SCORER_DIR))
import score_stage1 as scorer

SCOREABLE = {"original", "natural", "completion"}


def load(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("predictions_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not rows:
        raise FileNotFoundError(f"No predictions in {directory}")
    return rows


def summarize_rows(name: str, task: str, rows: list[dict]) -> dict:
    counts = [int(row.get("generated_token_count_approx", 0)) for row in rows]
    elapsed = [float(row.get("generation_elapsed_seconds", 0.0)) for row in rows]
    nonempty = sum(bool(str(row.get("raw_response", "")).strip()) for row in rows)
    returned_none = sum(bool(row.get("model_infer_returned_none")) for row in rows)
    loops = sum(scorer.has_repetition_loop(str(row.get("raw_response", ""))) for row in rows)
    return {
        "variant": name, "task": task, "records": len(rows), "nonempty": nonempty,
        "nonempty_rate": nonempty / len(rows), "infer_returned_none": returned_none,
        "empty_string_returns": sum(bool(row.get("model_infer_returned_empty_string")) for row in rows),
        "token_count": {"min": min(counts), "median": statistics.median(counts),
                        "mean": statistics.fmean(counts), "max": max(counts)},
        "elapsed_seconds": {"min": min(elapsed), "median": statistics.median(elapsed),
                            "mean": statistics.fmean(elapsed), "max": max(elapsed)},
        "repetition_loops": loops, "loop_rate": loops / len(rows),
        "strict_correct": sum(bool(row.get("is_correct")) for row in rows) if name in SCOREABLE else None,
        "strict_rate": (sum(bool(row.get("is_correct")) for row in rows) / len(rows)
                        if name in SCOREABLE else None),
    }


def summarize(name: str, directory: Path) -> list[dict]:
    rows = load(directory)
    tasks = sorted({str(row["task_type"]) for row in rows})
    return [summarize_rows(name, "all", rows)] + [
        summarize_rows(name, task, [row for row in rows if row["task_type"] == task])
        for task in tasks
    ]


def markdown(rows: list[dict]) -> str:
    lines = ["# DeepSeek-OCR-2 prompt ablation", "",
             "Variants `original`, `natural`, and `completion` are scoreable; the old-placeholder and describe controls are behavioral only.", "",
             "| Variant | Task | N | Nonempty | None / empty string | Tokens min/median/mean/max | Seconds min/median/mean/max | Loops | Strict |",
             "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in rows:
        token = row["token_count"]
        elapsed = row["elapsed_seconds"]
        strict = "n/a" if row["strict_rate"] is None else f'{row["strict_correct"]}/{row["strict_rate"]:.1%}'
        lines.append(
            f'| {row["variant"]} | {row["task"]} | {row["records"]} | {row["nonempty"]}/{row["nonempty_rate"]:.1%} | '
            f'{row["infer_returned_none"]}/{row["empty_string_returns"]} | '
            f'{token["min"]}/{token["median"]:g}/{token["mean"]:.1f}/{token["max"]} | '
            f'{elapsed["min"]:.1f}/{elapsed["median"]:.1f}/{elapsed["mean"]:.1f}/{elapsed["max"]:.1f} | '
            f'{row["repetition_loops"]}/{row["loop_rate"]:.1%} | {strict} |'
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="VARIANT=DIR")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for spec in args.run:
        name, sep, directory = spec.partition("=")
        if not sep:
            parser.error(f"invalid --run: {spec!r}")
        rows.extend(summarize(name, Path(directory)))
    payload = {"rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = markdown(rows)
    args.output_markdown.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
