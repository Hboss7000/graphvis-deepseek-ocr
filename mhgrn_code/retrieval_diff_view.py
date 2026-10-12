#!/usr/bin/env python3
"""Render diagnostic visual diffs for QA-GNN and MHGRN OBQA retrieval."""

from __future__ import annotations

import argparse
import html
import json
import math
import pickle
import shutil
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import graphviz
from PIL import Image, ImageDraw

from retrieval_compare_common import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, load_jsonl


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_graphvis_datasets import (  # noqa: E402
    RELATIONS,
    RELATION_TEXT,
    merge_choice_graphs,
    node_style,
    prune_graph,
    render_graph,  # imported deliberately: clean images use this implementation
)
from compare_retrieval import (  # noqa: E402
    graph_metrics,
    graphvis_faithful_triples,  # imported for provenance; saved faithful triples are read-only inputs
    merge_mhgrn_choice_graphs,
)


BASE_METHODS = ("qagnn", "mhgrn_default", "mhgrn_2hop")
LOSSLESS_METHODS = ("mhgrn_default_lossless", "mhgrn_2hop_lossless")
METHOD_NAMES = {
    "qagnn": "QA-GNN",
    "mhgrn_default": "GraphVis-faithful (MHGRN default + released get_edge)",
    "mhgrn_2hop": "GraphVis-faithful (MHGRN 2-hop + released get_edge)",
}
MIN_FREE_BYTES = 2 * 1024 ** 3
METHOD_COLORS = {
    "qagnn": "#d62728",
    "mhgrn_default": "#2ca02c",
    "mhgrn_2hop": "#ff7f0e",
    "mhgrn_default_lossless": "#9467bd",
    "mhgrn_2hop_lossless": "#17becf",
}


def parse_pair(value: str) -> tuple[str, str]:
    methods = tuple(part.strip() for part in value.split(","))
    if len(methods) != 2 or any(method not in BASE_METHODS for method in methods):
        raise argparse.ArgumentTypeError(
            "--pair requires two comma-separated names chosen from: " + ", ".join(BASE_METHODS)
        )
    if methods[0] == methods[1]:
        raise argparse.ArgumentTypeError("--pair methods must be distinct")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "train"), default="test")
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--with-lossless", action="store_true")
    parser.add_argument("--pair", type=parse_pair, metavar="A,B")
    parser.add_argument("--max-nodes", type=int, default=18)
    parser.add_argument("--max-edges", type=int, default=30)
    parser.add_argument("--max-degree", type=int, default=5)
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
    return parser.parse_args()


def prepare_output_dir(path: Path, minimum_free: int = MIN_FREE_BYTES) -> None:
    probe = path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < minimum_free:
        raise OSError(
            f"Refusing diagnostic render: only {free / 1024 ** 3:.2f} GiB free; "
            f"at least {minimum_free / 1024 ** 3:.2f} GiB is required"
        )
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic folder: {path}")
    path.mkdir(parents=True)
    (path / "images").mkdir()
    (path / "montages").mkdir()


def diagnostic_image_path(diff_root: Path, question_idx: int, method: str) -> Path:
    return diff_root / "images" / f"q{question_idx:05d}_{method}_DIAGNOSTIC.png"


def membership_classes(items_by_method: dict[str, set]) -> dict[str, set]:
    """Group items by the exact set of methods containing each item."""
    methods = tuple(items_by_method)
    holders = defaultdict(set)
    for method, items in items_by_method.items():
        for item in items:
            holders[item].add(method)
    classes = defaultdict(set)
    for item, present in holders.items():
        ordered = tuple(method for method in methods if method in present)
        if len(ordered) == len(methods):
            name = "shared_by_all"
        elif len(ordered) == 1:
            name = f"unique_to__{ordered[0]}"
        else:
            name = "shared_by__" + "__".join(ordered)
        classes[name].add(item)
    return dict(classes)


def pair_membership_classes(items_by_method: dict[str, set]) -> dict[str, set]:
    if len(items_by_method) != 2:
        raise ValueError("Pair membership requires exactly two methods")
    first, second = items_by_method
    first_items = items_by_method[first]
    second_items = items_by_method[second]
    return {
        "in_both": first_items & second_items,
        f"only_in__{first}": first_items - second_items,
        f"only_in__{second}": second_items - first_items,
    }


def membership_lookup(classes: dict[str, set]) -> dict[object, str]:
    return {item: class_name for class_name, items in classes.items() for item in items}


def assert_post_pruning_consistency(
    question_idx: int, method: str, actual: dict, reference_row: dict,
) -> None:
    reference = reference_row["after_pruning"]
    checks = {
        "n_nodes": actual["n_nodes"],
        "n_edges": actual["n_edges"],
        "gold_degree": actual["gold_degree"],
        "choice_concepts_deg0": actual["choice_concepts_deg0"],
    }
    mismatches = {}
    for key, value in checks.items():
        expected = reference[key]
        equal = math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12) \
            if isinstance(value, float) or isinstance(expected, float) else value == expected
        if not equal:
            mismatches[key] = {"recomputed": value, "comparison_jsonl": expected}
    if mismatches:
        raise ValueError(
            f"Post-pruning consistency mismatch q{question_idx} {method}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def lossless_mhgrn_edges(graphs: list[dict], conceptnet) -> tuple[set[tuple[int, int, int]], int]:
    """Return all distinct forward ConceptNet relations for each MHGRN link."""
    edges = set()
    links_total = 0
    for graph in graphs:
        node2cid = {node["id"]: int(node["cid"]) for node in graph["nodes"]}
        for link in graph["links"]:
            links_total += 1
            u = node2cid[link["source"]]
            v = node2cid[link["target"]]
            for source, target in ((u, v), (v, u)):
                try:
                    relation_records = conceptnet[source][target].values()
                except KeyError:
                    continue
                for record in relation_records:
                    relation_id = int(record["rel"])
                    if 0 <= relation_id < len(RELATIONS):
                        edges.add((source, relation_id, target))
    return edges, links_total


def lossless_graphvis_result(graphs: list[dict], id2concept: list[str], conceptnet) -> dict:
    edges, links_total = lossless_mhgrn_edges(graphs, conceptnet)
    triples = [
        {
            "source_cid": source,
            "source": id2concept[source].replace("_", " "),
            "relation_id": relation_id,
            "relation": RELATIONS[relation_id].replace("_", " "),
            "target_cid": target,
            "target": id2concept[target].replace("_", " "),
        }
        for source, relation_id, target in sorted(edges)
        if source != -1 and target != -1
    ]
    return {"triples": triples, "links_total": links_total, "lossless_edges": len(triples)}


def edge_degree(edges) -> dict[int, int]:
    degree = defaultdict(int)
    for source, _relation, target in edges:
        degree[source] += 1
        degree[target] += 1
    return degree


def edge_text(edge: tuple[int, str, int], id2concept: list[str]) -> str:
    source, relation, target = edge
    return (
        f"({id2concept[source].replace('_', ' ')}, "
        f"{RELATION_TEXT.get(relation, relation)}, "
        f"{id2concept[target].replace('_', ' ')})"
    )


def gold_answer_edges(
    merged_nodes: dict, edges: set[tuple[int, str, int]], answer_key: str,
    id2concept: list[str],
) -> list[str]:
    gold_ids = {
        cid for cid, node in merged_nodes.items() if answer_key in node["in_choices"]
    }
    return sorted(
        edge_text(edge, id2concept) for edge in edges
        if edge[0] in gold_ids or edge[2] in gold_ids
    )


def class_style(class_name: str, method: str, n_methods: int) -> tuple[str, str, str]:
    if class_name == "shared_by_all":
        return "#777777", "1.5", "solid"
    if class_name == f"unique_to__{method}":
        return METHOD_COLORS[method], "4.0", "solid"
    members = class_name.split("__")[1:] if class_name.startswith("shared_by__") else []
    if len(members) == 2:
        return "#1f77b4", "3.0", "dashed"
    # Only possible in the optional five-method view.
    return "#8c6bb1", "2.5", "dashed"


def pair_node_style(class_name: str) -> tuple[str, str]:
    return ("#777777", "1.5") if class_name == "in_both" else ("#d62728", "4.0")


def pair_edge_style(class_name: str, pair: tuple[str, str]) -> tuple[str, str, str]:
    if class_name == "in_both":
        return "#777777", "1.5", "solid"
    if class_name == f"only_in__{pair[0]}":
        return "#ff7f0e", "4.0", "solid"
    if class_name == f"only_in__{pair[1]}":
        return "#1f77b4", "4.0", "solid"
    raise ValueError(f"Unknown pair edge membership: {class_name}")


def render_diagnostic(
    target: Path, question_idx: int, method: str, merged_nodes: dict, pruned: dict,
    correct_label: str, node_classes: dict[str, set], edge_classes: dict[str, set],
    metrics: dict, args: argparse.Namespace, n_methods: int,
    pair: tuple[str, str] | None = None,
) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic image: {target}")
    node_membership = membership_lookup(node_classes)
    edge_membership = membership_lookup(edge_classes)
    dot = graphviz.Digraph(format="png", engine=args.engine)
    attrs = {
        "overlap": "false", "splines": "true", "dpi": str(args.dpi),
        "bgcolor": "white", "pad": "0.3", "nodesep": str(args.nodesep),
        "ranksep": str(args.ranksep),
        "label": (
            f"q{question_idx} {method} | nodes {metrics['n_nodes']}, "
            f"edges {metrics['n_edges']} | gold degree {metrics['gold_degree']:.2f} | "
            "DIAGNOSTIC: reveals answer"
        ),
        "labelloc": "t", "fontsize": "18", "fontname": "Helvetica-Bold",
    }
    if args.engine == "dot":
        attrs["rankdir"] = args.rankdir
    if args.graph_size is not None:
        attrs["size"] = args.graph_size
    if args.graph_ratio is not None:
        attrs["ratio"] = args.graph_ratio
    dot.attr("graph", **attrs)
    dot.attr(
        "node", shape="box", style="rounded,filled", fontname="Helvetica-Bold",
        fontsize=str(args.node_fontsize), margin="0.2,0.1",
    )
    dot.attr("edge", fontname="Helvetica", fontsize=str(args.edge_fontsize), arrowsize="0.8")

    for cid in pruned["visible_nodes"]:
        label, fill, _pipeline_penwidth = node_style(
            cid, merged_nodes, correct_label, reveal_correct_answer=True,
        )
        if pair:
            color, penwidth = pair_node_style(node_membership[cid])
            border_style = "solid"
        else:
            color, penwidth, border_style = class_style(node_membership[cid], method, n_methods)
        style = "rounded,filled" + (",dashed" if border_style == "dashed" else "")
        dot.node(str(cid), label=label, fillcolor=fill, color=color, penwidth=penwidth, style=style)

    for edge in pruned["edges"]:
        source, relation, target_cid = edge
        if pair:
            color, penwidth, edge_style = pair_edge_style(edge_membership[edge], pair)
        else:
            color, penwidth, edge_style = class_style(edge_membership[edge], method, n_methods)
        dot.edge(
            str(source), str(target_cid), label=RELATION_TEXT.get(relation, relation),
            color=color, fontcolor=color, penwidth=penwidth, style=edge_style,
        )

    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="membership", fontsize="11", color="gray70")
        if pair:
            legend.node("legend_both", "gray = in both", fillcolor="white", color="#777777", penwidth="1.5")
            legend.node("legend_node_only", "red border = node only in this method", fillcolor="white", color="#d62728", penwidth="4")
            legend.node("legend_edge_a", f"orange edge = only {METHOD_NAMES[pair[0]]}", shape="plaintext", fontcolor="#ff7f0e")
            legend.node("legend_edge_b", f"blue edge = only {METHOD_NAMES[pair[1]]}", shape="plaintext", fontcolor="#1f77b4")
        else:
            legend.node("legend_all", "shared by all", fillcolor="white", color="#777777", penwidth="1.5")
            legend.node("legend_pair", "shared with one other", fillcolor="white", color="#1f77b4", penwidth="3", style="rounded,filled,dashed")
            legend.node("legend_unique", "unique here", fillcolor="white", color=METHOD_COLORS[method], penwidth="4")

    rendered = Path(dot.render(str(target.with_suffix("")), cleanup=True))
    if rendered != target or not target.exists():
        raise RuntimeError(f"Graphviz did not produce expected output: {target}")


def save_montage_exclusive(
    clean_paths: list[Path], diagnostic_paths: list[Path], target: Path,
    statement: dict, column_titles: list[str] | None = None,
) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite montage: {target}")
    clean = [Image.open(path).convert("RGB") for path in clean_paths]
    diagnostic = [Image.open(path).convert("RGB") for path in diagnostic_paths]
    try:
        widths = [max(top.width, bottom.width) for top, bottom in zip(clean, diagnostic)]
        top_height = max(image.height for image in clean)
        bottom_height = max(image.height for image in diagnostic)
        header_lines = [statement["question"]["stem"]]
        header_lines.extend(
            f"{choice['label']}. {choice['text']}" for choice in statement["question"]["choices"]
        )
        header_lines.append("DIAGNOSTIC: gold answer marked, not for training or evaluation")
        header = "\n".join(textwrap.fill(line, width=140) for line in header_lines)
        header_height = 20 * (header.count("\n") + 1) + 20
        title_height = 34 if column_titles else 0
        canvas = Image.new(
            "RGB", (sum(widths), header_height + title_height + top_height + bottom_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text((12, 10), header, fill="black", spacing=4)
        x = 0
        for column, (width, top, bottom) in enumerate(zip(widths, clean, diagnostic)):
            if column_titles:
                draw.text((x + 10, header_height + 8), column_titles[column], fill="black")
            top_y = header_height + title_height
            canvas.paste(top, (x + (width - top.width) // 2, top_y))
            canvas.paste(bottom, (x + (width - bottom.width) // 2, top_y + top_height))
            x += width
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            canvas.save(handle, format="PNG")
    finally:
        for image in clean + diagnostic:
            image.close()


def exclusive_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def html_page(
    question_rows: list[dict], summary_rows: list[dict], methods: tuple[str, ...],
    pair: tuple[str, str] | None = None,
) -> str:
    style = """
body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4;color:#222}
.banner{padding:1rem;background:#ffe7a3;border:2px solid #c77d00;font-weight:700}
table{border-collapse:collapse;margin:1rem 0;width:100%}th,td{border:1px solid #bbb;padding:.35rem;text-align:right}th:first-child,td:first-child{text-align:left}
section{margin:3rem 0;border-top:3px solid #555;padding-top:1rem}img{max-width:100%;height:auto}.gold{background:#b7efb0;font-weight:700}code{white-space:normal}
"""
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Retrieval diff</title>",
             f"<style>{style}</style></head><body>",
             "<div class='banner'>DIAGNOSTIC ONLY: these images reveal the gold answer and must not be used for training or evaluation.</div>",
             "<h1>QA-GNN vs MHGRN retrieval after pruning</h1>",
             "<table><thead><tr><th>method</th><th>mean nodes</th><th>mean edges</th><th>mean gold degree</th><th>mean choice deg-0</th><th>node share shared all</th><th>edge share shared all</th></tr></thead><tbody>"]
    for row in summary_rows:
        parts.append(
            "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in (
                "method", "nodes_post", "edges_post", "gold_degree_post",
                "choice_deg0_post", "node_share_all", "edge_share_all",
            )) + "</tr>"
        )
    parts.append("</tbody></table>")
    for row in question_rows:
        parts.append(f"<section><h2>q{row['question_idx']}: {html.escape(row['question'])}</h2>")
        parts.append(f"<p>Sort key (largest remaining bridge budget): {row['sort_key_bridge_budget']}</p><ol type='A'>")
        for choice in row["choices"]:
            css = " class='gold'" if choice["label"] == row["answerKey"] else ""
            parts.append(f"<li{css}>{html.escape(choice['text'])}</li>")
        parts.append("</ol><table><thead><tr><th>method</th><th>nodes</th><th>edges</th><th>gold degree</th><th>choice deg-0</th><th>core</th><th>bridge budget</th></tr></thead><tbody>")
        for method in methods:
            parts.append(
                f"<tr><td>{html.escape(method)}</td>"
                f"<td>{row['nodes_post'][method]}</td><td>{row['edges_post'][method]}</td>"
                f"<td>{row['gold_degree_post'][method]:.4f}</td>"
                f"<td>{row['choice_deg0_post'][method]}</td><td>{row['core_size'][method]}</td>"
                f"<td>{row['remaining_bridge_budget'][method]}</td></tr>"
            )
        parts.append("</tbody></table>")
        if pair:
            parts.append("<h3>Pairwise edge differences</h3>")
            for method in pair:
                edges = row["edge_lists_by_class"].get(f"only_in__{method}", [])
                parts.append(
                    f"<p><strong>Edges only in {html.escape(METHOD_NAMES[method])}:</strong> "
                    f"{html.escape('; '.join(edges) or 'none')}</p>"
                )
            parts.append("<h3>Gold-answer edges</h3>")
            for method in pair:
                parts.append(
                    f"<p><strong>{html.escape(METHOD_NAMES[method])}</strong> "
                    f"(gold degree {row['gold_degree_post'][method]:.4f}): "
                    f"{html.escape('; '.join(row['gold_answer_edges'][method]) or 'none')}</p>"
                )
        else:
            parts.append("<h3>Unique nodes</h3>")
            for method in methods:
                labels = row["node_labels_by_class"].get(f"unique_to__{method}", [])
                parts.append(f"<p><strong>{html.escape(method)}:</strong> {html.escape(', '.join(labels) or 'none')}</p>")
        parts.append(f"<img src='montages/q{row['question_idx']:05d}_diff.png' alt='retrieval diff montage'>")
        parts.append("</section>")
    parts.append("</body></html>\n")
    return "".join(parts)


def main() -> None:
    args = parse_args()
    if min(args.max_nodes, args.max_edges, args.max_degree) < 0:
        raise ValueError("Pruning limits must be non-negative")
    if args.pair and args.with_lossless:
        raise ValueError("--pair cannot be combined with --with-lossless")
    split_root = args.retrieval_dir.resolve() / args.split
    diff_root = (
        split_root / f"diff_view_pair_{args.pair[0]}_vs_{args.pair[1]}"
        if args.pair else split_root / "diff_view"
    )
    prepare_output_dir(diff_root)

    data_root = args.data_root.resolve()
    concept_path = data_root / "cpnet" / "concept.txt"
    cpnet_path = data_root / "cpnet" / "conceptnet.en.pruned.graph"
    statement_path = data_root / "obqa" / "statement" / f"{args.split}.statement.jsonl"
    grounded_path = data_root / "obqa" / "grounded" / f"{args.split}.grounded.jsonl"
    qagnn_path = data_root / "obqa" / "graph" / f"{args.split}.graph.adj.pk"
    required = [
        concept_path, statement_path, grounded_path,
        split_root / "comparison.jsonl", split_root / "qa_node_counts.jsonl",
        split_root / "subset_index.json",
    ]
    methods = args.pair or (BASE_METHODS + (LOSSLESS_METHODS if args.with_lossless else ()))
    if "qagnn" in methods:
        required.append(qagnn_path)
    needed_variants = tuple(
        variant for variant in ("mhgrn_default", "mhgrn_2hop")
        if variant in methods or f"{variant}_lossless" in methods
    )
    for variant in needed_variants:
        required.append(split_root / variant / "graph.jsonl")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(map(str, missing)))

    subset_map = json.loads((split_root / "subset_index.json").read_text(encoding="utf-8"))
    indices = [subset_map[offset]["question_idx"] for offset in range(0, len(subset_map), 4)]
    id2concept = concept_path.read_text(encoding="utf-8").splitlines()
    statements = load_jsonl(statement_path)
    grounded_all = load_jsonl(grounded_path)
    qagnn_entries = None
    if "qagnn" in methods:
        with qagnn_path.open("rb") as handle:
            qagnn_entries = pickle.load(handle)
    variant_graphs = {
        variant: load_jsonl(split_root / variant / "graph.jsonl")
        for variant in needed_variants
    }
    references = {
        (row["question_idx"], row["method"]): row
        for row in load_jsonl(split_root / "comparison.jsonl")
    }
    qa_counts = {
        (row["question_idx"], row["method"]): row
        for row in load_jsonl(split_root / "qa_node_counts.jsonl")
    }
    conceptnet = None
    if args.with_lossless and not args.pair:
        with cpnet_path.open("rb") as handle:
            conceptnet = pickle.load(handle)

    question_rows = []
    consistency_matches = 0
    summary_accumulator = defaultdict(list)
    for subset_position, question_idx in enumerate(indices):
        statement = statements[question_idx]
        choices = statement["question"]["choices"]
        answer_key = statement["answerKey"]
        grounded = grounded_all[4 * question_idx:4 * question_idx + 4]
        assembled = {}

        if "qagnn" in methods:
            q_nodes, q_edges, q_answer = merge_choice_graphs(
                question_idx, qagnn_entries, statement, id2concept,
            )
            assembled["qagnn"] = (q_nodes, q_edges, q_answer)
        for variant in needed_variants:
            graphs = variant_graphs[variant][4 * subset_position:4 * subset_position + 4]
            faithful_path = split_root / variant / "graphvis_triples" / f"q{question_idx:05d}.json"
            faithful = json.loads(faithful_path.read_text(encoding="utf-8"))
            nodes, edges, _ = merge_mhgrn_choice_graphs(
                graphs, grounded, choices, id2concept, faithful,
            )
            assembled[variant] = (nodes, edges, answer_key)
            if args.with_lossless and not args.pair:
                lossless = lossless_graphvis_result(graphs, id2concept, conceptnet)
                lossless_name = f"{variant}_lossless"
                ll_nodes, ll_edges, _ = merge_mhgrn_choice_graphs(
                    graphs, grounded, choices, id2concept, lossless,
                )
                assembled[lossless_name] = (ll_nodes, ll_edges, answer_key)
                print(
                    f"q{question_idx} {variant}: links={lossless['links_total']} "
                    f"graphvis_edges={len(faithful['triples'])} "
                    f"lossless_edges={lossless['lossless_edges']}"
                )

        evaluated = {}
        node_sets = {}
        edge_sets = {}
        for method in methods:
            merged_nodes, merged_edges, correct = assembled[method]
            pruned = prune_graph(
                merged_nodes, merged_edges, args.max_nodes, args.max_edges, args.max_degree,
                bridge_rule=args.bridge_rule, lifelines=args.lifelines,
            )
            metrics = graph_metrics(
                merged_nodes, pruned["edges"], answer_key, pruned["visible_nodes"],
            )
            if method in BASE_METHODS:
                assert_post_pruning_consistency(
                    question_idx, method, metrics, references[(question_idx, method)],
                )
                consistency_matches += 1
            evaluated[method] = {
                "merged_nodes": merged_nodes, "merged_edges": merged_edges,
                "correct": correct, "pruned": pruned, "metrics": metrics,
            }
            node_sets[method] = set(pruned["visible_nodes"])
            edge_sets[method] = set(map(tuple, pruned["edges"]))

        if args.pair:
            node_classes = pair_membership_classes(node_sets)
            edge_classes = pair_membership_classes(edge_sets)
            all_class = "in_both"
        else:
            node_classes = membership_classes(node_sets)
            edge_classes = membership_classes(edge_sets)
            all_class = "shared_by_all"
        all_nodes = node_classes.get(all_class, set())
        all_edges = edge_classes.get(all_class, set())
        diagnostic_paths = []
        clean_paths = []
        for method in methods:
            item = evaluated[method]
            target = diagnostic_image_path(diff_root, question_idx, method)
            render_diagnostic(
                target, question_idx, method, item["merged_nodes"], item["pruned"],
                item["correct"], node_classes, edge_classes, item["metrics"], args,
                len(methods), pair=args.pair,
            )
            diagnostic_paths.append(target)
            if method in BASE_METHODS:
                clean_paths.append(split_root / "images" / f"q{question_idx:05d}_{method}.png")
            else:
                # Lossless has no prior clean image; render one with the canonical renderer.
                clean_target = diff_root / "images" / f"q{question_idx:05d}_{method}_clean"
                rendered = render_graph(
                    clean_target, item["merged_nodes"], item["pruned"], item["correct"],
                    args.engine, False, reveal_correct_answer=False, dpi=args.dpi,
                    disconnected_rows=args.disconnected_rows, node_fontsize=args.node_fontsize,
                    edge_fontsize=args.edge_fontsize, nodesep=args.nodesep, ranksep=args.ranksep,
                    graph_size=args.graph_size, graph_ratio=args.graph_ratio, rankdir=args.rankdir,
                )
                clean_paths.append(rendered)
        save_montage_exclusive(
            clean_paths, diagnostic_paths,
            diff_root / "montages" / f"q{question_idx:05d}_diff.png", statement,
            column_titles=[METHOD_NAMES[method] for method in methods] if args.pair else None,
        )

        core_size = {}
        core_exceeds = {}
        bridge_budget = {}
        for method in methods:
            count_method = method.removesuffix("_lossless")
            count = qa_counts[(question_idx, count_method)]
            core_size[method] = count["core"]
            core_exceeds[method] = count["core_exceeds_cap"]
            bridge_budget[method] = max(0, args.max_nodes - count["core"])
        row = {
            "split": args.split,
            "question_idx": question_idx,
            "question": statement["question"]["stem"],
            "choices": choices,
            "answerKey": answer_key,
            "methods": list(methods),
            "node_labels_by_class": {
                name: sorted(id2concept[cid].replace("_", " ") for cid in cids)
                for name, cids in sorted(node_classes.items())
            },
            "edge_counts_by_class": {
                name: len(edges) for name, edges in sorted(edge_classes.items())
            },
            "nodes_post": {method: evaluated[method]["metrics"]["n_nodes"] for method in methods},
            "edges_post": {method: evaluated[method]["metrics"]["n_edges"] for method in methods},
            "gold_degree_post": {method: evaluated[method]["metrics"]["gold_degree"] for method in methods},
            "choice_deg0_post": {method: evaluated[method]["metrics"]["choice_concepts_deg0"] for method in methods},
            "core_size": core_size,
            "core_exceeds_max_nodes": core_exceeds,
            "remaining_bridge_budget": bridge_budget,
            "sort_key_bridge_budget": max(bridge_budget.values()),
        }
        if args.pair:
            row["edge_lists_by_class"] = {
                name: sorted(edge_text(edge, id2concept) for edge in edges)
                for name, edges in sorted(edge_classes.items())
            }
            row["gold_answer_edges"] = {
                method: gold_answer_edges(
                    evaluated[method]["merged_nodes"], edge_sets[method], answer_key, id2concept,
                )
                for method in methods
            }
        question_rows.append(row)
        for method in methods:
            metrics = evaluated[method]["metrics"]
            summary_accumulator[method].append({
                "nodes": metrics["n_nodes"], "edges": metrics["n_edges"],
                "gold_degree": metrics["gold_degree"],
                "choice_deg0": metrics["choice_concepts_deg0"],
                "node_share_all": len(all_nodes) / len(node_sets[method]) if node_sets[method] else 1.0,
                "edge_share_all": len(all_edges) / len(edge_sets[method]) if edge_sets[method] else 1.0,
                "edges_pre": len(assembled[method][1]),
            })

    expected_matches = len(indices) * len(methods if args.pair else BASE_METHODS)
    if consistency_matches != expected_matches:
        raise RuntimeError(f"Expected {expected_matches} consistency matches, got {consistency_matches}")
    print(f"consistency checks matched: {consistency_matches}/{expected_matches}")
    question_rows.sort(key=lambda row: (-row["sort_key_bridge_budget"], row["question_idx"]))
    summary_rows = []
    print("method\tnodes_post\tedges_post\tgold_degree_post\tchoice_deg0_post\tnode_share_all\tedge_share_all")
    for method in methods:
        values = summary_accumulator[method]
        row = {
            "method": method,
            "nodes_post": f"{fmean(item['nodes'] for item in values):.4f}",
            "edges_post": f"{fmean(item['edges'] for item in values):.4f}",
            "gold_degree_post": f"{fmean(item['gold_degree'] for item in values):.4f}",
            "choice_deg0_post": f"{fmean(item['choice_deg0'] for item in values):.4f}",
            "node_share_all": f"{fmean(item['node_share_all'] for item in values):.4f}",
            "edge_share_all": f"{fmean(item['edge_share_all'] for item in values):.4f}",
        }
        summary_rows.append(row)
        print("\t".join(str(row[key]) for key in row))
        if args.with_lossless and method in LOSSLESS_METHODS:
            print(
                f"lossless {method}: mean_edges_pre="
                f"{fmean(item['edges_pre'] for item in values):.4f} "
                f"mean_edges_post={row['edges_post']} mean_gold_degree_post={row['gold_degree_post']}"
            )

    exclusive_jsonl(diff_root / "diff.jsonl", question_rows)
    with (diff_root / "index.html").open("x", encoding="utf-8", errors="strict") as handle:
        handle.write(html_page(question_rows, summary_rows, methods, pair=args.pair))
    print(f"index: {diff_root / 'index.html'}")


if __name__ == "__main__":
    main()
