#!/usr/bin/env python3
"""Unbiased pairwise comparison of answer-node degree, correct vs distractor.

Companion to analyze_answer_connectivity.py, which reports a max-over-distractors
verdict. That verdict compares one correct option against the maximum over three
distractors, so it is biased against the correct answer even under a null where
degree carries no information. This script avoids that asymmetry.

Statistic: over all (correct, distractor) option pairs within a question, the
fraction where the correct option's answer-node degree exceeds the distractor's.
Ties count as one half. Chance is exactly 0.5.

Also reports the degree histogram, so the effect of the max_degree pruning cap on
the available dynamic range is visible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--exclusive-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def option_scores(meta: dict, exclusive_only: bool) -> dict[str, int]:
    degree: dict[int, int] = defaultdict(int)
    for edge in meta["edges"]:
        degree[int(edge["source_cid"])] += 1
        degree[int(edge["target_cid"])] += 1

    best = {choice["label"]: 0 for choice in meta["choices"]}
    for node in meta["visible_nodes"]:
        in_choices = list(node.get("in_choices") or [])
        if not in_choices:
            continue
        if exclusive_only and len(in_choices) != 1:
            continue
        node_degree = degree[int(node["cid"])]
        for letter in in_choices:
            if letter in best:
                best[letter] = max(best[letter], node_degree)
    return best


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.metadata)

    wins = ties = losses = 0
    degree_hist: Counter[int] = Counter()
    correct_zero = 0

    for meta in records:
        gold = meta["answerKey"]
        best = option_scores(meta, args.exclusive_only)
        gold_score = best[gold]
        if gold_score == 0:
            correct_zero += 1
        for letter, score in best.items():
            degree_hist[score] += 1
            if letter == gold:
                continue
            if gold_score > score:
                wins += 1
            elif gold_score == score:
                ties += 1
            else:
                losses += 1

    pairs = wins + ties + losses
    win_rate = (wins + 0.5 * ties) / pairs if pairs else None

    metrics = {
        "metadata_path": str(args.metadata.resolve()),
        "questions": len(records),
        "exclusive_only": args.exclusive_only,
        "pairs_compared": pairs,
        "correct_higher": wins,
        "tied": ties,
        "correct_lower": losses,
        "pairwise_win_rate": win_rate,
        "chance_level": 0.5,
        "tie_fraction": ties / pairs if pairs else None,
        "questions_correct_answer_degree_zero": correct_zero,
        "fraction_correct_answer_degree_zero": correct_zero / len(records) if records else None,
        "option_degree_histogram": dict(sorted(degree_hist.items())),
    }
    print(json.dumps(metrics, indent=2))

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.metrics_out}")


if __name__ == "__main__":
    main()
