#!/usr/bin/env python3
"""Shared, project-local helpers for the small MHGRN retrieval comparison."""

from __future__ import annotations

import json
import pickle
import re
import resource
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MHGRN_ROOT = HERE / "MHGRN"
GRAPHVIS_ROOT = HERE / "GraphVis"
DEFAULT_DATA_ROOT = REPO_ROOT / "data_preprocessed_release"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "retrieval_compare"
DEFAULT_TRANSE_ROOT = REPO_ROOT / "data" / "transe"


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("--indices must be a comma-separated list of integers") from error
    if not indices:
        raise ValueError("--indices must select at least one question")
    if any(index < 0 for index in indices):
        raise ValueError("question indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError("question indices must not contain duplicates")
    return indices


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_new(path: Path, rows: Iterable[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def required_paths(split: str, data_root: Path, transe_root: Path) -> dict[str, Path]:
    return {
        "mhgrn": MHGRN_ROOT,
        "graphvis": GRAPHVIS_ROOT,
        "concept_vocab": data_root / "cpnet" / "concept.txt",
        "cpnet_graph": data_root / "cpnet" / "conceptnet.en.pruned.graph",
        "grounded": data_root / "obqa" / "grounded" / f"{split}.grounded.jsonl",
        "statements": data_root / "obqa" / "statement" / f"{split}.statement.jsonl",
        "qagnn_graph": data_root / "obqa" / "graph" / f"{split}.graph.adj.pk",
        "transe_entities": transe_root / "glove.transe.sgd.ent.npy",
        "transe_relations": transe_root / "glove.transe.sgd.rel.npy",
    }


def assert_required_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        grounded_missing = any(line.startswith("grounded:") for line in missing)
        extra = (
            " Grounding must not be regenerated here because MHGRN's grounding path "
            "requires spaCy 2.x."
            if grounded_missing else ""
        )
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing) + extra)


def mhgrn_default_threshold() -> float:
    preprocess = (MHGRN_ROOT / "preprocess.py").read_text(encoding="utf-8")
    match = re.search(
        r"add_argument\(['\"]--path_prune_threshold['\"].*?default\s*=\s*([0-9.]+)",
        preprocess,
    )
    if not match:
        raise RuntimeError("Could not read --path_prune_threshold default from MHGRN/preprocess.py")
    return float(match.group(1))


def peak_rss_mib(include_children: bool = False) -> float:
    # Linux reports ru_maxrss in KiB; this project runs on Linux locally/Runpod.
    rss_values = [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]
    if include_children:
        rss_values.append(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return max(rss_values) / 1024.0


def install_networkx_gpickle_shim() -> None:
    """Keep unmodified MHGRN usable if invoked from an environment with NetworkX 3."""
    import networkx as nx

    if not hasattr(nx, "read_gpickle"):
        def read_gpickle(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)
        nx.read_gpickle = read_gpickle  # type: ignore[attr-defined]
    if not hasattr(nx, "write_gpickle"):
        def write_gpickle(graph, path):
            with open(path, "wb") as handle:
                pickle.dump(graph, handle)
        nx.write_gpickle = write_gpickle  # type: ignore[attr-defined]


def filter_paths_by_max_nodes(rows: list[dict] | list[list[dict]], max_nodes: int) -> list:
    """Keep MHGRN raw paths up to ``max_nodes`` while preserving row structure."""
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")
    filtered = []
    for qa_pairs in rows:
        new_pairs = []
        for pair in qa_pairs:
            pair = dict(pair)
            paths = pair.get("pf_res")
            if paths is not None:
                # MHGRN enumerates shortest paths first, so this is exactly a
                # two-hop cutoff on its already ordered raw candidate paths.
                pair["pf_res"] = [
                    item for item in paths if 2 <= len(item["path"]) <= max_nodes
                ]
            new_pairs.append(pair)
        filtered.append(new_pairs)
    return filtered


def filter_paths_2hop(rows: list[dict] | list[list[dict]]) -> list:
    """Keep MHGRN raw paths with 2--3 nodes (at most two graph hops)."""
    return filter_paths_by_max_nodes(rows, 3)


def count_paths(qa_pairs: list[dict]) -> int:
    return sum(len(pair["pf_res"] or []) for pair in qa_pairs)
