#!/usr/bin/env python3
"""Structural analysis of answer-node connectivity in the pruned OBQA subgraphs.

Asks whether the retrieved and pruned subgraph distinguishes the correct answer
at all. Reads graph_metadata_*.jsonl only. No GPU, no model, no dependencies
beyond the standard library.

Definitions used throughout:

  answer node   a visible node whose `in_choices` contains at least one option
                letter. A node may belong to several options at once.
  degree        number of metadata edges incident on the node, counted on the
                pruned graph that was actually rendered.
  exclusive     an answer node belonging to exactly one option letter. Nodes
                shared between options carry no discriminative signal, so the
                "exclusive" variants of each statistic are the informative ones.

Per question, the correct option's score is the maximum degree over its answer
nodes (0 if it has none). Same for each distractor. The question is then
classified by comparing those scores.
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
    parser.add_argument(
        "--exclusive-only",
        action="store_true",
        help="Count only answer nodes belonging to a single option letter",
    )
    parser.add_argument(
        "--list-examples",
        type=int,
        default=0,
        help="Print N questions where a distractor outranks the correct answer",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def option_scores(meta: dict, exclusive_only: bool) -> tuple[dict[str, int], dict[str, int]]:
    """Return (max degree per option letter, answer-node count per option letter)."""
    degree: dict[int, int] = defaultdict(int)
    for edge in meta["edges"]:
        degree[int(edge["source_cid"])] += 1
        degree[int(edge["target_cid"])] += 1

    letters = [choice["label"] for choice in meta["choices"]]
    best = {letter: 0 for letter in letters}
    counts = {letter: 0 for letter in letters}

    for node in meta["visible_nodes"]:
        in_choices = list(node.get("in_choices") or [])
        if not in_choices:
            continue
        if exclusive_only and len(in_choices) != 1:
            continue
        node_degree = degree[int(node["cid"])]
        for letter in in_choices:
            if letter not in best:
                continue
            counts[letter] += 1
            best[letter] = max(best[letter], node_degree)

    return best, counts


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.metadata)
    if not records:
        raise SystemExit("Metadata file is empty")

    total = len(records)
    correct_degrees: list[int] = []
    distractor_degrees: list[int] = []
    verdicts: Counter[str] = Counter()
    no_answer_nodes = 0
    correct_absent = 0
    correct_isolated = 0
    losers: list[tuple[int, str, int, int]] = []

    for meta in records:
        gold = meta["answerKey"]
        best, counts = option_scores(meta, args.exclusive_only)
        if gold not in best:
            raise SystemExit(
                f"statement_idx={meta['statement_idx']}: answerKey {gold!r} "
                f"not among choice labels {sorted(best)}"
            )

        if all(count == 0 for count in counts.values()):
            no_answer_nodes += 1
        if counts[gold] == 0:
            correct_absent += 1
        elif best[gold] == 0:
            correct_isolated += 1

        gold_score = best[gold]
        rival_scores = [score for letter, score in best.items() if letter != gold]
        top_rival = max(rival_scores) if rival_scores else 0

        correct_degrees.append(gold_score)
        distractor_degrees.extend(rival_scores)

        if gold_score > top_rival:
            verdicts["correct_strictly_highest"] += 1
        elif gold_score == top_rival:
            verdicts["tied_with_distractor"] += 1
        else:
            verdicts["distractor_strictly_higher"] += 1
            losers.append((int(meta["statement_idx"]), gold, gold_score, top_rival))

    def mean(values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None

    discriminative = verdicts["correct_strictly_highest"]
    metrics = {
        "metadata_path": str(args.metadata.resolve()),
        "questions": total,
        "exclusive_only": args.exclusive_only,
        "questions_with_no_answer_nodes": no_answer_nodes,
        "correct_answer_has_no_node": correct_absent,
        "correct_answer_node_isolated": correct_isolated,
        "verdicts": dict(verdicts),
        "verdict_fractions": {k: v / total for k, v in verdicts.items()},
        "degree_discriminates_correct_answer": discriminative / total,
        "mean_degree_correct_answer": mean(correct_degrees),
        "mean_degree_distractors": mean(distractor_degrees),
    }

    print(json.dumps(metrics, indent=2))

    if args.list_examples and losers:
        print(f"\nQuestions where a distractor outranks the correct answer "
              f"(showing {min(args.list_examples, len(losers))} of {len(losers)}):")
        for statement_idx, gold, gold_score, rival_score in losers[: args.list_examples]:
            print(f"  statement_idx={statement_idx:>4}  gold={gold}  "
                  f"gold_degree={gold_score}  best_distractor_degree={rival_score}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.metrics_out}")


if __name__ == "__main__":
    main()
