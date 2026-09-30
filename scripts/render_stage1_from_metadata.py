#!/usr/bin/env python3
"""Re-render Stage 1 PNGs from stored graph metadata without re-pruning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_graphvis_datasets import render_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", default="dot")
    parser.add_argument("--node-fontsize", type=int, default=18)
    parser.add_argument("--edge-fontsize", type=int, default=14)
    parser.add_argument("--nodesep", type=float, default=0.5)
    parser.add_argument("--ranksep", type=float, default=0.7)
    parser.add_argument("--rankdir", choices=("LR", "TB"), default="LR")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--disconnected-rows", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.graph_metadata.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        target = args.out_dir / str(row["image"])
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        nodes = {
            int(node["cid"]): {
                "name": str(node["name"]),
                "in_question": bool(node["in_question"]),
                "in_choices": set(node["in_choices"]),
            }
            for node in row["visible_nodes"]
        }
        edges = [
            (int(edge["source_cid"]), str(edge["relation"]), int(edge["target_cid"]))
            for edge in row["edges"]
        ]
        disconnected = [int(cid) for cid in row.get("disconnected_answer_cids", [])]
        graph = {
            "connected_nodes": sorted(
                int(node["cid"]) for node in row["visible_nodes"] if node["connected"]
            ),
            "visible_nodes": sorted(nodes),
            "edges": edges,
            "disconnected_answers": disconnected,
        }
        render_graph(
            target.with_suffix(""), nodes, graph, str(row["answerKey"]), args.engine, False,
            False, args.dpi, args.disconnected_rows, args.node_fontsize, args.edge_fontsize,
            args.nodesep, args.ranksep, None, None, args.rankdir,
        )
        print(target, flush=True)


if __name__ == "__main__":
    main()
