#!/usr/bin/env python3
"""Run the released MHGRN path retrieval on explicitly selected OBQA questions."""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

from retrieval_compare_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRANSE_ROOT,
    MHGRN_ROOT,
    assert_required_paths,
    count_paths,
    filter_paths_2hop,
    install_networkx_gpickle_shim,
    load_jsonl,
    mhgrn_default_threshold,
    parse_indices,
    peak_rss_mib,
    required_paths,
    write_jsonl_new,
    write_text_new,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def install_score_triples_json_float_shim(paths_module) -> None:
    """Make MHGRN's NumPy scalar score JSON-safe without changing its value."""
    original = paths_module.score_triples
    if getattr(original, "_mhgrn_json_float_shim", False):
        return

    @functools.wraps(original)
    def score_triples_as_python_float(*args, **kwargs):
        return float(original(*args, **kwargs))

    score_triples_as_python_float._mhgrn_json_float_shim = True
    paths_module.score_triples = score_triples_as_python_float


def score_precision_diagnostic(
    paths_module, raw_path: Path, scores_path: Path, entity_path: Path,
    relation_path: Path, threshold: float,
) -> dict[str, int]:
    """Compare released float32 keep/drop decisions with float64 rescoring."""
    raw_rows = load_jsonl(raw_path)
    float32_rows = load_jsonl(scores_path)
    if len(raw_rows) != len(float32_rows):
        raise ValueError("Raw path and score files have different row counts")

    original_concepts = paths_module.concept_embs
    original_relations = paths_module.relation_embs
    paths_module.concept_embs = np.load(entity_path).astype(np.float64, copy=False)
    paths_module.relation_embs = np.load(relation_path).astype(np.float64, copy=False)
    decisions_changed = 0
    within_1e4 = 0
    paths_compared = 0
    try:
        for qa_pairs, qa_scores in zip(raw_rows, float32_rows):
            if len(qa_pairs) != len(qa_scores):
                raise ValueError("Raw QA-pair and score counts differ")
            for pair, scores in zip(qa_pairs, qa_scores):
                raw_paths = pair["pf_res"]
                if raw_paths is None:
                    if scores is not None:
                        raise ValueError("Expected null scores for null raw paths")
                    continue
                if len(raw_paths) != len(scores):
                    raise ValueError("Raw path and path-score counts differ")
                for path, score32 in zip(raw_paths, scores):
                    # score_triples mutates relation-id lists while adding
                    # inverse aliases, so pass fresh lists from the raw JSON.
                    score64 = paths_module.score_triples(
                        concept_id=list(path["path"]),
                        relation_id=[list(relations) for relations in path["rel"]],
                    )
                    score32 = float(score32)
                    decisions_changed += (score32 >= threshold) != (score64 >= threshold)
                    within_1e4 += abs(score32 - threshold) <= 1e-4
                    paths_compared += 1
    finally:
        paths_module.concept_embs = original_concepts
        paths_module.relation_embs = original_relations
        gc.collect()

    return {
        "paths_compared": paths_compared,
        "keep_drop_differences_float32_vs_float64": decisions_changed,
        "paths_within_1e-4_of_threshold_float32": within_1e4,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "train"), required=True)
    parser.add_argument("--indices", required=True, help="Comma-separated question indices")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--transe-root", type=Path, default=DEFAULT_TRANSE_ROOT)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_subset(
    grounded_rows: list[dict], split: str, indices: list[int], grounded_path: Path,
    index_path: Path,
) -> None:
    subset = []
    mapping = []
    n_questions = len(grounded_rows) // 4
    for subset_position, question_idx in enumerate(indices):
        if question_idx >= n_questions:
            raise IndexError(f"Question index {question_idx} is outside 0..{n_questions - 1}")
        for choice_idx in range(4):
            subset.append(grounded_rows[4 * question_idx + choice_idx])
            mapping.append({
                "subset_position": subset_position,
                "split": split,
                "question_idx": question_idx,
                "choice_idx": choice_idx,
                "source_grounded_line": 4 * question_idx + choice_idx,
            })
    write_jsonl_new(grounded_path, subset)
    write_text_new(index_path, json.dumps(mapping, indent=2) + "\n")


def write_2hop_raw(source: Path, target: Path) -> None:
    rows = load_jsonl(source)
    write_jsonl_new(target, filter_paths_2hop(rows))


def report_counts(indices: list[int], variant: str, raw_path: Path, pruned_path: Path, graph_path: Path) -> None:
    raw_rows = load_jsonl(raw_path)
    pruned_rows = load_jsonl(pruned_path)
    graph_rows = load_jsonl(graph_path)
    for position, question_idx in enumerate(indices):
        start = 4 * position
        raw_count = sum(count_paths(row) for row in raw_rows[start:start + 4])
        kept_count = sum(count_paths(row) for row in pruned_rows[start:start + 4])
        graph_sizes = [
            {"choice_idx": choice_idx, "nodes": len(graph["nodes"]), "links": len(graph["links"])}
            for choice_idx, graph in enumerate(graph_rows[start:start + 4])
        ]
        print(
            f"question_idx={question_idx} variant={variant} raw_paths={raw_count} "
            f"pruned_paths={kept_count} graph_lines={json.dumps(graph_sizes)}"
        )


def main() -> None:
    # MHGRN initializes module-global ConceptNet/vocabulary state in the parent
    # and its Pool workers inherit that state. Python 3.14 defaults to
    # forkserver on POSIX, so force fork to preserve MHGRN's released behavior.
    multiprocessing.set_start_method("fork", force=True)
    args = parse_args()
    if args.num_processes < 1:
        raise ValueError("--num-processes must be at least 1")
    indices = parse_indices(args.indices)
    paths = required_paths(args.split, args.data_root.resolve(), args.transe_root.resolve())
    assert_required_paths(paths)
    threshold = mhgrn_default_threshold() if args.threshold is None else args.threshold

    split_root = args.out_dir.resolve() / args.split
    log_path = split_root / "mhgrn_run.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {log_path}")
    split_root.mkdir(parents=True, exist_ok=True)

    subset_grounded = split_root / "subset.grounded.jsonl"
    subset_index = split_root / "subset_index.json"
    write_subset(load_jsonl(paths["grounded"]), args.split, indices, subset_grounded, subset_index)

    install_networkx_gpickle_shim()
    sys.path.insert(0, str(MHGRN_ROOT))
    # These are the released implementations; no retrieval/scoring logic is duplicated here.
    from utils.graph import generate_graph
    from utils import paths as mhgrn_paths
    install_score_triples_json_float_shim(mhgrn_paths)
    find_paths = mhgrn_paths.find_paths
    prune_paths = mhgrn_paths.prune_paths
    score_paths = mhgrn_paths.score_paths

    with log_path.open("x", encoding="utf-8") as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)), \
            contextlib.redirect_stderr(Tee(sys.stderr, log)):
        print(f"split={args.split} indices={indices}")
        print(f"MHGRN path_prune_threshold={threshold}")
        print(f"num_processes={args.num_processes}")

        default_root = split_root / "mhgrn_default"
        hop2_root = split_root / "mhgrn_2hop"
        default_root.mkdir()
        hop2_root.mkdir()
        default_raw = default_root / "paths.raw.jsonl"

        started = time.monotonic()
        find_paths(
            str(subset_grounded), str(paths["concept_vocab"]), str(paths["cpnet_graph"]),
            str(default_raw), args.num_processes, args.seed,
        )
        find_seconds = time.monotonic() - started
        # MHGRN uses a multiprocessing Pool even for one worker, so include
        # completed child-process high-water marks in the reported peak.
        find_peak = peak_rss_mib(include_children=True)
        print(f"find_paths runtime_seconds={find_seconds:.3f}")
        print(f"find_paths peak_rss_mib={find_peak:.1f}")

        write_2hop_raw(default_raw, hop2_root / "paths.raw.jsonl")
        precision_diagnostic = None
        for name, root in (("mhgrn_default", default_root), ("mhgrn_2hop", hop2_root)):
            raw = root / "paths.raw.jsonl"
            scores = root / "paths.scores.jsonl"
            pruned = root / "paths.pruned.jsonl"
            graph = root / "graph.jsonl"
            score_paths(
                str(raw), str(paths["transe_entities"]), str(paths["transe_relations"]),
                str(paths["concept_vocab"]), str(scores), args.num_processes,
                method="triple_cls",
            )
            if name == "mhgrn_default":
                precision_diagnostic = score_precision_diagnostic(
                    mhgrn_paths, raw, scores, paths["transe_entities"],
                    paths["transe_relations"], threshold,
                )
                print(
                    "precision diagnostic: "
                    f"keep_drop_differences_float32_vs_float64="
                    f"{precision_diagnostic['keep_drop_differences_float32_vs_float64']}"
                )
                print(
                    "precision diagnostic: "
                    f"paths_within_1e-4_of_threshold_float32="
                    f"{precision_diagnostic['paths_within_1e-4_of_threshold_float32']}"
                )
            prune_paths(str(raw), str(scores), str(pruned), threshold)
            generate_graph(
                str(subset_grounded), str(pruned), str(paths["concept_vocab"]),
                str(paths["cpnet_graph"]), str(graph),
            )
            report_counts(indices, name, raw, pruned, graph)

        metadata = {
            "split": args.split,
            "indices": indices,
            "threshold": threshold,
            "num_processes": args.num_processes,
            "find_paths_runtime_seconds": find_seconds,
            "find_paths_peak_rss_mib": find_peak,
            "score_precision_diagnostic": precision_diagnostic,
        }
        write_text_new(split_root / "run_metadata.json", json.dumps(metadata, indent=2) + "\n")
        print(f"monitor with: tail -f {log_path}")
        print("MHGRN SUBSET COMPLETE")


if __name__ == "__main__":
    main()
