#!/usr/bin/env python3
"""Compare a new Qwen Stage 1 run with stored COMA predictions item by item."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(directory: Path) -> dict[tuple[int, str], dict]:
    result = {}
    paths = sorted(directory.glob("predictions_qwen_*.jsonl"))
    if not paths:
        paths = sorted(directory.glob("scored_qwen_*.jsonl"))
    for path in paths:
        for row in read_jsonl(path):
            key = (int(row["statement_idx"]), str(row["task_type"]))
            if key in result:
                raise ValueError(f"Duplicate prediction key {key} in {directory}")
            result[key] = row
    if not result:
        raise FileNotFoundError(f"No Qwen prediction/scored files in {directory}")
    return result


def compare(label: str, reference_dir: Path, candidate_dir: Path) -> dict:
    reference = keyed(reference_dir)
    candidate = keyed(candidate_dir)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())[:10]
        extra = sorted(candidate.keys() - reference.keys())[:10]
        raise ValueError(f"Prediction keys differ for {label}: missing={missing}; extra={extra}")
    by_task: dict[str, dict[str, int]] = {}
    raw_matches = correctness_matches = 0
    mismatches = []
    for key in sorted(reference):
        old, new = reference[key], candidate[key]
        raw_equal = old.get("raw_response") == new.get("raw_response")
        correct_equal = bool(old.get("is_correct")) == bool(new.get("is_correct"))
        raw_matches += int(raw_equal)
        correctness_matches += int(correct_equal)
        task = key[1]
        task_counts = by_task.setdefault(
            task, {"records": 0, "raw_response_exact_matches": 0, "scored_correctness_matches": 0}
        )
        task_counts["records"] += 1
        task_counts["raw_response_exact_matches"] += int(raw_equal)
        task_counts["scored_correctness_matches"] += int(correct_equal)
        if not raw_equal or not correct_equal:
            mismatches.append(
                {
                    "statement_idx": key[0],
                    "task": task,
                    "raw_response_exact_match": raw_equal,
                    "scored_correctness_match": correct_equal,
                    "reference_is_correct": bool(old.get("is_correct")),
                    "candidate_is_correct": bool(new.get("is_correct")),
                }
            )
    total = len(reference)
    return {
        "label": label,
        "records": total,
        "raw_response_exact_matches": raw_matches,
        "raw_response_exact_match_rate": raw_matches / total,
        "scored_correctness_matches": correctness_matches,
        "scored_correctness_agreement_rate": correctness_matches / total,
        "by_task": by_task,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", action="append", required=True, metavar="LABEL=REFERENCE=CANDIDATE"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for value in args.comparison:
        parts = value.split("=", 2)
        if len(parts) != 3 or not all(parts):
            parser.error(f"Invalid comparison {value!r}")
        reports.append(compare(parts[0], Path(parts[1]), Path(parts[2])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"comparisons": reports}, indent=2, sort_keys=True) + "\n")
    for report in reports:
        print(
            f"{report['label']}: raw exact {report['raw_response_exact_matches']}/{report['records']} "
            f"({report['raw_response_exact_match_rate']:.2%}); scored correctness "
            f"{report['scored_correctness_matches']}/{report['records']} "
            f"({report['scored_correctness_agreement_rate']:.2%})"
        )


if __name__ == "__main__":
    main()
