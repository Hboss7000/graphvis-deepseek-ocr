#!/usr/bin/env python3
"""Build the fixed first-20 node-description reading probe without rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite reading-probe artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")

    records = sorted(
        (row for row in read_jsonl(args.stage1_jsonl)
         if row["task_type"] == "node_description"),
        key=lambda row: int(row["statement_idx"]),
    )[:args.n]
    if len(records) != args.n:
        raise ValueError(f"Expected {args.n} node_description records, found {len(records)}")
    indices = [int(row["statement_idx"]) for row in records]
    metadata = {int(row["statement_idx"]): row for row in read_jsonl(args.graph_metadata)}
    selected_metadata = [metadata[index] for index in indices]
    stage1_out = args.output_dir / "stage1_node_description_first20.jsonl"
    metadata_out = args.output_dir / "graph_metadata_first20.jsonl"
    write_new(stage1_out, records)
    write_new(metadata_out, selected_metadata)
    manifest = {
        "task": "node_description",
        "n": args.n,
        "statement_indices": indices,
        "source_stage1": {"path": str(args.stage1_jsonl.resolve()),
                          "sha256": sha256_file(args.stage1_jsonl)},
        "source_graph_metadata": {"path": str(args.graph_metadata.resolve()),
                                  "sha256": sha256_file(args.graph_metadata)},
        "stage1_jsonl": str(stage1_out.resolve()),
        "graph_metadata": str(metadata_out.resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite reading-probe artifact: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
