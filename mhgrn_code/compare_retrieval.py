#!/usr/bin/env python3
"""Convert, render, and measure QA-GNN and MHGRN retrieval for an OBQA subset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import networkx as nx
from PIL import Image, ImageDraw

from retrieval_compare_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    load_jsonl,
    parse_indices,
    required_paths,
    write_text_new,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_graphvis_datasets import (  # noqa: E402
    RELATIONS,
    merge_choice_graphs,
    prune_graph,
    render_graph,
)


cpnet = None


def get_edge(src_concept, tgt_concept):
    # Copied verbatim from GraphVis/generate_data.py lines 111--120.
    global cpnet
    rel_list = cpnet[src_concept][tgt_concept]  # list of dicts
    seen = set()
    res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
    # convert everyhing larger than 16 to 15 and make sure no duplicates
    res = [r for r in res if r <= 16]
    if len(res) > 1 and 15 in res:
        res.remove(15)
    return res


def graphvis_faithful_triples(graphs: list[dict], id2concept: list[str], graph) -> dict:
    """Apply GraphVis's node/link loop while recording, not repairing, its quirks."""
    global cpnet
    cpnet = graph
    nodes = set()
    node_cids = set()
    triples = set()
    counts = {
        "links_total": 0,
        "links_dropped_direction": 0,
        "links_dropped_missing": 0,
        "nodes_cid_minus1": 0,
    }

    for item in graphs:
        node2cid = {}
        for node in item["nodes"]:
            cid = int(node["cid"])
            node2cid[node["id"]] = cid
            node_cids.add(cid)
            if cid == -1:
                counts["nodes_cid_minus1"] += 1
            # This deliberately preserves GraphVis's Python -1 indexing quirk.
            nodes.add(id2concept[cid].replace("_", " "))

        for link in item["links"]:
            counts["links_total"] += 1
            src = node2cid[link["source"]]
            tgt = node2cid[link["target"]]
            try:
                edges = get_edge(src, tgt)
            except KeyError:
                counts["links_dropped_missing"] += 1
                continue
            if not edges:
                forward_raw = [attrs["rel"] for attrs in graph[src][tgt].values()]
                reverse_forward = []
                try:
                    reverse_forward = [
                        attrs["rel"] for attrs in graph[tgt][src].values()
                        if attrs["rel"] <= 16
                    ]
                except KeyError:
                    pass
                if forward_raw and all(rel > 16 for rel in forward_raw) and reverse_forward:
                    counts["links_dropped_direction"] += 1
                else:
                    counts["links_dropped_missing"] += 1
                continue
            rel_id = int(edges[0])
            triples.add((src, rel_id, tgt))

    serial_triples = [
        {
            "source_cid": src,
            "source": id2concept[src].replace("_", " "),
            "relation_id": rel_id,
            "relation": RELATIONS[rel_id].replace("_", " "),
            "target_cid": tgt,
            "target": id2concept[tgt].replace("_", " "),
        }
        for src, rel_id, tgt in sorted(triples)
    ]
    return {
        "nodes": sorted(nodes),
        "node_cids": sorted(node_cids),
        "triples": serial_triples,
        **counts,
        "graphvis_would_skip": not triples,
    }


def merge_mhgrn_choice_graphs(
    graphs: list[dict], grounded_rows: list[dict], choices: list[dict],
    id2concept: list[str], graphvis_result: dict,
) -> tuple[dict, set, str | None]:
    """Return the same node/edge value shapes as merge_choice_graphs."""
    if len(graphs) != 4 or len(grounded_rows) != 4 or len(choices) != 4:
        raise ValueError("MHGRN OBQA merge requires exactly four answer-choice rows")
    if len(RELATIONS) != 17:
        raise AssertionError("GraphVis/MHGRN relation indexing requires exactly 17 relations")
    concept2id = {name: cid for cid, name in enumerate(id2concept)}
    merged_nodes = {}
    qcs = set().union(*(set(row.get("qc", [])) for row in grounded_rows))

    for item in graphs:
        for node in item["nodes"]:
            cid = int(node["cid"])
            if cid == -1:
                continue
            merged_nodes.setdefault(cid, {
                "name": id2concept[cid],
                "in_question": id2concept[cid] in qcs,
                "in_choices": set(),
            })

    for choice, grounded in zip(choices, grounded_rows):
        for name in grounded.get("ac", []):
            cid = concept2id.get(name)
            if cid in merged_nodes:
                merged_nodes[cid]["in_choices"].add(choice["label"])

    merged_edges = {
        (int(item["source_cid"]), RELATIONS[int(item["relation_id"])], int(item["target_cid"]))
        for item in graphvis_result["triples"]
        if int(item["source_cid"]) != -1 and int(item["target_cid"]) != -1
    }
    # answerKey is returned separately by the caller, matching merge_choice_graphs's third slot.
    return merged_nodes, merged_edges, None


def degree_map(edges: set | list) -> dict[int, int]:
    degree = defaultdict(int)
    for src, _rel, tgt in edges:
        degree[src] += 1
        degree[tgt] += 1
    return degree


def relation_histogram(edges: set | list) -> dict[str, int]:
    counts = Counter(rel for _src, rel, _tgt in edges)
    return {rel: counts[rel] for rel in RELATIONS if counts[rel]}


def graph_metrics(merged_nodes: dict, edges: set | list, answer_key: str, node_ids=None) -> dict:
    node_ids = set(merged_nodes) if node_ids is None else set(node_ids)
    edges = [edge for edge in edges if edge[0] in node_ids and edge[2] in node_ids]
    degree = degree_map(edges)
    gold_ids = sorted(
        cid for cid in node_ids if answer_key in merged_nodes[cid]["in_choices"]
    )
    choice_ids = [cid for cid in node_ids if merged_nodes[cid]["in_choices"]]
    question_ids = [cid for cid in node_ids if merged_nodes[cid]["in_question"]]
    return {
        "n_nodes": len(node_ids),
        "n_edges": len(edges),
        "gold_concepts": [merged_nodes[cid]["name"].replace("_", " ") for cid in gold_ids],
        "gold_present": bool(gold_ids),
        "gold_degree": fmean(degree[cid] for cid in gold_ids) if gold_ids else 0.0,
        "choice_concepts_deg0": sum(degree[cid] == 0 for cid in choice_ids),
        "question_concepts_deg0": sum(degree[cid] == 0 for cid in question_ids),
        "relation_histogram": relation_histogram(edges),
    }


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def graph_sets(merged_nodes: dict, edges, node_ids=None) -> tuple[set, set]:
    ids = set(merged_nodes) if node_ids is None else set(node_ids)
    triples = {(src, rel, tgt) for src, rel, tgt in edges if src in ids and tgt in ids}
    return ids, triples


def make_montage(paths: list[tuple[str, Path]], target: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in paths]
    label_height = 42
    canvas = Image.new(
        "RGB", (sum(image.width for _label, image in images),
                max(image.height for _label, image in images) + label_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in images:
        draw.text((x + 10, 12), label, fill="black")
        canvas.paste(image, (x, label_height))
        x += image.width
    canvas.save(target)
    for _label, image in images:
        image.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "train"), required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-nodes", type=int, default=18, help="Node cap; 0 disables it.")
    parser.add_argument("--max-edges", type=int, default=60, help="Edge cap; 0 disables it.")
    parser.add_argument("--max-degree", type=int, default=0, help="Degree cap; 0 disables it.")
    parser.add_argument("--bridge-rule", choices=("qa-bridge", "core-neighbor", "any", "none"), default="qa-bridge")
    parser.add_argument("--no-lifelines", dest="lifelines", action="store_false")
    parser.add_argument("--engine", default="dot")
    parser.add_argument("--node-fontsize", type=int, default=18)
    parser.add_argument("--edge-fontsize", type=int, default=14)
    parser.add_argument("--nodesep", type=float, default=0.5)
    parser.add_argument("--ranksep", type=float, default=0.7)
    parser.add_argument("--graph-size")
    parser.add_argument("--graph-ratio")
    parser.add_argument("--rankdir", choices=("LR", "TB"), default="LR")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--disconnected-rows", type=int, default=3)
    parser.add_argument("--hide-relatedto-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.max_nodes, args.max_edges, args.max_degree) < 0:
        raise ValueError("Require max_nodes >= 0, max_edges >= 0 and max_degree >= 0")
    indices = parse_indices(args.indices)
    data_root = args.data_root.resolve()
    split_root = args.out_dir.resolve() / args.split
    paths = required_paths(args.split, data_root, REPO_ROOT / "data" / "transe")
    needed = [paths["concept_vocab"], paths["cpnet_graph"], paths["grounded"],
              paths["statements"], paths["qagnn_graph"], split_root / "subset_index.json"]
    for variant in ("mhgrn_default", "mhgrn_2hop"):
        needed.append(split_root / variant / "graph.jsonl")
    missing = [path for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n  " + "\n  ".join(map(str, missing)))

    output_targets = [split_root / "comparison.jsonl", split_root / "summary.md"]
    for question_idx in indices:
        output_targets.extend(
            split_root / "images" / f"q{question_idx:05d}_{method}.png"
            for method in ("qagnn", "mhgrn_default", "mhgrn_2hop", "compare")
        )
        output_targets.extend(
            split_root / variant / "graphvis_triples" / f"q{question_idx:05d}.json"
            for variant in ("mhgrn_default", "mhgrn_2hop")
        )
    collisions = [path for path in output_targets if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite outputs:\n  " + "\n  ".join(map(str, collisions)))

    id2concept = paths["concept_vocab"].read_text(encoding="utf-8").splitlines()
    statements = load_jsonl(paths["statements"])
    grounded_all = load_jsonl(paths["grounded"])
    with paths["qagnn_graph"].open("rb") as handle:
        qagnn_entries = pickle.load(handle)
    with paths["cpnet_graph"].open("rb") as handle:
        conceptnet = pickle.load(handle)
    variant_graphs = {
        variant: load_jsonl(split_root / variant / "graph.jsonl")
        for variant in ("mhgrn_default", "mhgrn_2hop")
    }
    subset_map = json.loads((split_root / "subset_index.json").read_text(encoding="utf-8"))
    mapped_indices = [subset_map[pos * 4]["question_idx"] for pos in range(len(indices))]
    if mapped_indices != indices:
        raise ValueError(f"--indices do not match subset_index.json: {indices} != {mapped_indices}")

    image_dir = split_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset_position, question_idx in enumerate(indices):
        statement = statements[question_idx]
        choices = statement["question"]["choices"]
        answer_key = statement["answerKey"]
        grounded = grounded_all[4 * question_idx:4 * question_idx + 4]
        methods = {}

        q_nodes, q_edges, q_answer = merge_choice_graphs(
            question_idx, qagnn_entries, statement, id2concept,
        )
        methods["qagnn"] = (q_nodes, q_edges, q_answer, {})

        for variant in ("mhgrn_default", "mhgrn_2hop"):
            graphs = variant_graphs[variant][4 * subset_position:4 * subset_position + 4]
            faithful = graphvis_faithful_triples(graphs, id2concept, conceptnet)
            payload = {
                "split": args.split,
                "question_idx": question_idx,
                "question": statement["question"]["stem"],
                "choices": choices,
                "answerKey": answer_key,
                **faithful,
            }
            triple_path = split_root / variant / "graphvis_triples" / f"q{question_idx:05d}.json"
            write_text_new(triple_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            m_nodes, m_edges, _ = merge_mhgrn_choice_graphs(
                graphs, grounded, choices, id2concept, faithful,
            )
            quirks = {key: faithful[key] for key in (
                "links_total", "links_dropped_direction", "links_dropped_missing",
                "nodes_cid_minus1", "graphvis_would_skip",
            )}
            methods[variant] = (m_nodes, m_edges, answer_key, quirks)

        evaluated = {}
        montage_inputs = []
        for method, (merged_nodes, merged_edges, correct, quirks) in methods.items():
            pruned = prune_graph(
                merged_nodes, merged_edges, args.max_nodes, args.max_edges, args.max_degree,
                bridge_rule=args.bridge_rule, lifelines=args.lifelines,
            )
            image_stem = image_dir / f"q{question_idx:05d}_{method}"
            image_path = render_graph(
                image_stem, merged_nodes, pruned, correct, args.engine,
                args.hide_relatedto_labels, reveal_correct_answer=False, dpi=args.dpi,
                disconnected_rows=args.disconnected_rows, node_fontsize=args.node_fontsize,
                edge_fontsize=args.edge_fontsize, nodesep=args.nodesep, ranksep=args.ranksep,
                graph_size=args.graph_size, graph_ratio=args.graph_ratio, rankdir=args.rankdir,
            )
            montage_inputs.append((method, image_path))
            before = graph_metrics(merged_nodes, merged_edges, answer_key)
            after = graph_metrics(merged_nodes, pruned["edges"], answer_key, pruned["visible_nodes"])
            before_sets = graph_sets(merged_nodes, merged_edges)
            after_sets = graph_sets(merged_nodes, pruned["edges"], pruned["visible_nodes"])
            evaluated[method] = (before_sets, after_sets)
            rows.append({
                "split": args.split,
                "question_idx": question_idx,
                "method": method,
                "before_pruning": before,
                "after_pruning": after,
                **quirks,
            })

        q_before, q_after = evaluated["qagnn"]
        for row in rows[-3:]:
            if row["method"] == "qagnn":
                row["node_jaccard_vs_qagnn"] = None
                row["triple_jaccard_vs_qagnn"] = None
            else:
                before_sets, after_sets = evaluated[row["method"]]
                row["node_jaccard_vs_qagnn"] = {
                    "before_pruning": jaccard(before_sets[0], q_before[0]),
                    "after_pruning": jaccard(after_sets[0], q_after[0]),
                }
                row["triple_jaccard_vs_qagnn"] = {
                    "before_pruning": jaccard(before_sets[1], q_before[1]),
                    "after_pruning": jaccard(after_sets[1], q_after[1]),
                }
        make_montage(montage_inputs, image_dir / f"q{question_idx:05d}_compare.png")

    comparison_path = split_root / "comparison.jsonl"
    write_text_new(
        comparison_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )

    summary_columns = [
        ("nodes_pre", lambda row: row["before_pruning"]["n_nodes"]),
        ("edges_pre", lambda row: row["before_pruning"]["n_edges"]),
        ("nodes_post", lambda row: row["after_pruning"]["n_nodes"]),
        ("edges_post", lambda row: row["after_pruning"]["n_edges"]),
        ("gold_present_pre", lambda row: int(row["before_pruning"]["gold_present"])),
        ("gold_present_post", lambda row: int(row["after_pruning"]["gold_present"])),
        ("gold_degree_pre", lambda row: row["before_pruning"]["gold_degree"]),
        ("gold_degree_post", lambda row: row["after_pruning"]["gold_degree"]),
        ("choice_deg0_pre", lambda row: row["before_pruning"]["choice_concepts_deg0"]),
        ("choice_deg0_post", lambda row: row["after_pruning"]["choice_concepts_deg0"]),
        ("question_deg0_pre", lambda row: row["before_pruning"]["question_concepts_deg0"]),
        ("question_deg0_post", lambda row: row["after_pruning"]["question_concepts_deg0"]),
        ("node_jaccard_pre", lambda row: 1.0 if row["node_jaccard_vs_qagnn"] is None else row["node_jaccard_vs_qagnn"]["before_pruning"]),
        ("node_jaccard_post", lambda row: 1.0 if row["node_jaccard_vs_qagnn"] is None else row["node_jaccard_vs_qagnn"]["after_pruning"]),
        ("triple_jaccard_pre", lambda row: 1.0 if row["triple_jaccard_vs_qagnn"] is None else row["triple_jaccard_vs_qagnn"]["before_pruning"]),
        ("triple_jaccard_post", lambda row: 1.0 if row["triple_jaccard_vs_qagnn"] is None else row["triple_jaccard_vs_qagnn"]["after_pruning"]),
        ("links_total", lambda row: row.get("links_total", 0)),
        ("links_drop_direction", lambda row: row.get("links_dropped_direction", 0)),
        ("links_drop_missing", lambda row: row.get("links_dropped_missing", 0)),
        ("cid_minus1", lambda row: row.get("nodes_cid_minus1", 0)),
        ("graphvis_skip", lambda row: int(row.get("graphvis_would_skip", False))),
    ]
    header = ["method", *[name for name, _getter in summary_columns]]
    table = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    for method in ("qagnn", "mhgrn_default", "mhgrn_2hop"):
        selected = [row for row in rows if row["method"] == method]
        values = [f"{fmean(getter(row) for row in selected):.4f}" for _name, getter in summary_columns]
        table.append("| " + " | ".join([method, *values]) + " |")
    summary = "\n".join(table) + "\n"
    write_text_new(split_root / "summary.md", summary)
    print(summary, end="")
    print(f"comparison: {comparison_path}")
    print(f"montages: {image_dir}/q*_compare.png")


if __name__ == "__main__":
    main()
