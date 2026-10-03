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
from score_stage1 import normalize_component, parse_node_items  # noqa: E402

# Kept local so this read-only report works in inference venvs without Graphviz.
RELATION_TEXT = {
    'antonym': 'antonym', 'atlocation': 'at location', 'capableof': 'capable of',
    'causes': 'causes', 'createdby': 'created by', 'isa': 'is a', 'desires': 'desires',
    'hassubevent': 'has subevent', 'partof': 'part of', 'hascontext': 'has context',
    'hasproperty': 'has property', 'madeof': 'made of', 'notcapableof': 'not capable of',
    'notdesires': 'not desires', 'receivesaction': 'receives action',
    'relatedto': 'related to', 'usedfor': 'used for',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="LABEL=predictions_node_description.jsonl")
    parser.add_argument("--graph-metadata", type=Path, required=True,
                        help="First-20 metadata used to identify visible relation labels.")
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "metric": "set_metrics.raw.macro_recall",
        "interpretation": "Raw per-item gold-node recall for node_description.",
        "arms": {},
    }
    metadata = {int(row["statement_idx"]): row
                for row in read_jsonl(args.graph_metadata)}
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
        raw_metrics = metrics["set_metrics"]["raw"]
        raw_predicted = []
        distinct_predicted = 0
        raw_intrusions = 0
        distinct_intrusions = 0
        for row in rows:
            meta = metadata[int(row["statement_idx"])]
            relation_labels = {
                normalize_component(RELATION_TEXT.get(str(edge["relation"]), str(edge["relation"])), "basic")
                for edge in meta.get("edges", [])
            }
            items = parse_node_items(row.get("raw_response", ""))
            raw_predicted.extend((item, relation_labels) for item in items)
            raw_intrusions += sum(normalize_component(item, "basic") in relation_labels
                                  for item in items)
            unique_items = {normalize_component(item, "basic") for item in items}
            distinct_predicted += len(unique_items)
            distinct_intrusions += sum(item in relation_labels for item in unique_items)
        recall = raw_metrics["macro_recall"]
        report["arms"][label] = {
            "predictions": str(path.resolve()),
            "n": len(rows),
            "raw_macro_gold_node_recall": recall,
            "raw_macro_precision": raw_metrics["macro_precision"],
            "raw_macro_f1": raw_metrics["macro_f1"],
            "edge_label_intrusion": {
                "raw_predicted_items": {
                    "predicted_item_count": len(raw_predicted),
                    "matching_predicted_item_count": raw_intrusions,
                    "rate": raw_intrusions / len(raw_predicted) if raw_predicted else 0.0,
                },
                "distinct_predicted_items_per_graph": {
                    "predicted_item_count": distinct_predicted,
                    "matching_predicted_item_count": distinct_intrusions,
                    "rate": distinct_intrusions / distinct_predicted if distinct_predicted else 0.0,
                    "definition": "Deduplicate normalized predicted items within each graph, then pool graph-level counts.",
                },
                "definition": "An intrusion exactly matches a rendered relation label of that item's graph after basic normalization.",
            },
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
