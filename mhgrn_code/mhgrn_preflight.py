#!/usr/bin/env python3
"""Validate inputs and alignment for a small OBQA MHGRN retrieval comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from retrieval_compare_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_TRANSE_ROOT,
    assert_required_paths,
    install_networkx_gpickle_shim,
    load_jsonl,
    parse_indices,
    peak_rss_mib,
    required_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "train"), default="test")
    parser.add_argument("--indices", default="60,0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--transe-root", type=Path, default=DEFAULT_TRANSE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = parse_indices(args.indices)
    paths = required_paths(args.split, args.data_root.resolve(), args.transe_root.resolve())
    assert_required_paths(paths)

    with paths["concept_vocab"].open("r", encoding="utf-8") as handle:
        vocab_size = sum(1 for _ in handle)
    entity_shape = np.load(paths["transe_entities"], mmap_mode="r").shape
    relation_shape = np.load(paths["transe_relations"], mmap_mode="r").shape
    print(f"concept vocabulary rows: {vocab_size}")
    print(f"TransE entity shape: {entity_shape}")
    print(f"TransE relation shape: {relation_shape}")
    if not entity_shape or entity_shape[0] != vocab_size:
        raise ValueError(
            f"Concept/TransE misalignment: concept.txt has {vocab_size} rows, "
            f"entity embeddings have {entity_shape[0] if entity_shape else 0}"
        )

    statements = load_jsonl(paths["statements"])
    grounded = load_jsonl(paths["grounded"])
    print(f"statement rows: {len(statements)}")
    print(f"grounded rows: {len(grounded)}")
    if len(grounded) != 4 * len(statements):
        raise ValueError("Grounded row count must equal four times the statement row count")
    if any(index >= len(statements) for index in indices):
        raise IndexError(f"Selected index is outside 0..{len(statements) - 1}: {indices}")

    for question_idx in indices[: min(3, len(indices))]:
        statement = statements[question_idx]
        print(f"alignment question_idx={question_idx}: {statement['question']['stem']}")
        for choice_idx, choice in enumerate(statement["question"]["choices"]):
            row = grounded[4 * question_idx + choice_idx]
            print(
                "  " + json.dumps({
                    "choice": choice,
                    "grounded_sent": row.get("sent"),
                    "grounded_ans": row.get("ans"),
                }, ensure_ascii=False)
            )

    install_networkx_gpickle_shim()
    graph = nx.read_gpickle(paths["cpnet_graph"])
    print(f"ConceptNet nodes: {graph.number_of_nodes()}")
    print(f"ConceptNet edges: {graph.number_of_edges()}")
    print(f"peak RSS after ConceptNet load: {peak_rss_mib():.1f} MiB")
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
