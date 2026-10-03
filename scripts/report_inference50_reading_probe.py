#!/usr/bin/env python3
"""Report raw macro gold-node recall for inference50 reading-probe arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
sys.path.insert(0, str(STAGE1))
from score_stage1 import aggregate_task, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="LABEL=predictions_node_description.jsonl")
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "metric": "set_metrics.raw.macro_recall",
        "interpretation": "Raw per-item gold-node recall for node_description.",
        "arms": {},
    }
    expected_indices = None
    for spec in args.arm:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label:
            raise ValueError(f"Invalid --arm {spec!r}; expected LABEL=PATH")
        path = Path(raw_path)
        rows = read_jsonl(path)
        indices = [int(row["statement_idx"]) for row in rows]
        if len(rows) != args.expected_count or len(indices) != len(set(indices)):
            raise ValueError(f"{label}: expected {args.expected_count} unique rows")
        if any(row["task_type"] != "node_description" for row in rows):
            raise ValueError(f"{label}: non-node_description row found")
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise ValueError(f"{label}: statement order differs from the first arm")
        metrics = aggregate_task("node_description", rows)
        recall = metrics["set_metrics"]["raw"]["macro_recall"]
        report["arms"][label] = {
            "predictions": str(path.resolve()),
            "n": len(rows),
            "raw_macro_gold_node_recall": recall,
        }
    report["statement_indices"] = expected_indices
    payload = json.dumps(report, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite different reading-probe report: {args.output}")
        print(json.dumps(report, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
