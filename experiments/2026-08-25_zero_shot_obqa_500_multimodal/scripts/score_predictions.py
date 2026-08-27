#!/usr/bin/env python3
"""Validate and score exact option-letter predictions from run_zero_shot.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


VALID_OPTIONS = {"A", "B", "C", "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--metrics-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    indices = [row.get("statement_idx") for row in rows]
    duplicates = [index for index, count in Counter(indices).items() if count > 1]
    invalid = [row.get("statement_idx") for row in rows if row.get("predicted_option") not in VALID_OPTIONS]
    correct = sum(row.get("is_correct") is True for row in rows)
    metrics = {
        "prediction_rows": len(rows),
        "unique_statement_indices": len(set(indices)),
        "expected_count": args.expected_count,
        "duplicate_statement_indices": duplicates,
        "invalid_prediction_indices": invalid,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "complete_and_valid": (
            len(rows) == args.expected_count
            and len(set(indices)) == args.expected_count
            and not duplicates
            and not invalid
        ),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not metrics["complete_and_valid"]:
        raise SystemExit("Result is incomplete or contains invalid predictions; do not report its accuracy yet.")


if __name__ == "__main__":
    main()
