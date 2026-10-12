#!/usr/bin/env python3
"""Compare question/answer node counts of QA-GNN vs MHGRN retrieval, BEFORE pruning.

For every question in an MHGRN subset run (<retrieval-dir>/<split>/subset_index.json),
the four per-choice graphs are merged into one question-level graph, as in the
rendering pipeline, and the following are counted per retrieval method:

  nodes_total          all nodes in the merged graph
  q_nodes              nodes flagged as question concepts
  a_nodes              nodes flagged as answer concepts (any choice)
  qa_overlap           nodes flagged as both
  core                 |Q union A|, which is the set prune_graph tries to keep first
  a_per_choice         answer nodes per choice label
  gold_a_nodes         answer nodes of the correct choice
  grounded_q_missing   grounded question concepts not flagged as Q in the graph
  grounded_a_missing   grounded answer concepts not flagged as A in the graph
  gold_grounded_missing  grounded concepts of the correct choice not flagged as that choice

How Q/A flags are obtained (they differ by necessity):
  qagnn  QA-GNN's own qmask/amask, merged with merge_choice_graphs() from
         scripts/generate_graphvis_datasets.py, i.e. exactly what the current
         image pipeline feeds into prune_graph.
  mhgrn  graph.jsonl stores only concept ids, so a node is Q (A) if it is in that
         statement's grounded qc (ac) list. Only nodes present in the retrieved
         graph count. MHGRN's -1 placeholder nodes are excluded.

Grounded concepts come from <data-root>/obqa/grounded/<split>.grounded.jsonl for
both methods, so "missing" counts are comparable.

Nothing is pruned here. Read-only on all inputs; refuses to overwrite its output.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("mhgrn_default", "mhgrn_2hop")
NUMERIC = ("nodes_total", "q_nodes", "a_nodes", "qa_overlap", "core", "gold_a_nodes",
           "grounded_q_missing", "grounded_a_missing", "gold_grounded_missing")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=("train", "dev", "test"))
    p.add_argument("--data-root", type=Path, default=ROOT / "data_preprocessed_release")
    p.add_argument("--retrieval-dir", type=Path, default=ROOT / "outputs" / "retrieval_compare")
    p.add_argument("--max-nodes", type=int, default=18,
                   help="Only used to flag questions whose core exceeds the node cap.")
    p.add_argument("--out", type=Path, default=None,
                   help="JSONL output (default: <retrieval-dir>/<split>/qa_node_counts.jsonl).")
    return p.parse_args()


def to_ids(concepts: list[str], concept2id: dict[str, int], where: str) -> set[int]:
    missing = [c for c in concepts if c not in concept2id]
    if missing:
        raise KeyError(f"{where}: grounded concepts not in concept.txt: {missing[:5]}")
    return {concept2id[c] for c in concepts}


def merge_mhgrn(graph_rows: list[dict], grounded_rows: list[dict], labels: list[str],
                concept2id: dict[str, int], where: str) -> dict[int, dict]:
    """Union of the 4 MHGRN statement graphs, with Q/A flags from the grounded file."""
    merged: dict[int, dict] = {}
    for label, graph, grounded in zip(labels, graph_rows, grounded_rows):
        qc = to_ids(grounded["qc"], concept2id, where)
        ac = to_ids(grounded["ac"], concept2id, where)
        for node in graph["nodes"]:
            cid = int(node["cid"])
            if cid < 0:          # MHGRN placeholder when a statement has no qc/ac
                continue
            entry = merged.setdefault(cid, {"in_question": False, "in_choices": set()})
            if cid in qc:
                entry["in_question"] = True
            if cid in ac:
                entry["in_choices"].add(label)
    return merged


def count(merged: dict[int, dict], labels: list[str], gold: str,
          grounded_q: set[int], grounded_a_by_label: dict[str, set[int]], max_nodes: int) -> dict:
    q = {cid for cid, n in merged.items() if n["in_question"]}
    a = {cid for cid, n in merged.items() if n["in_choices"]}
    flagged_by_label = {lab: {cid for cid, n in merged.items() if lab in n["in_choices"]} for lab in labels}
    grounded_a = set().union(*grounded_a_by_label.values())
    core = len(q | a)
    return {
        "nodes_total": len(merged),
        "q_nodes": len(q),
        "a_nodes": len(a),
        "qa_overlap": len(q & a),
        "core": core,
        "core_exceeds_cap": core > max_nodes,
        "a_per_choice": {lab: len(flagged_by_label[lab]) for lab in labels},
        "gold_a_nodes": len(flagged_by_label[gold]),
        "grounded_q_missing": len(grounded_q - q),
        "grounded_a_missing": len(grounded_a - a),
        "gold_grounded_missing": len(grounded_a_by_label[gold] - flagged_by_label[gold]),
    }


def main() -> None:
    args = parse_args()
    split_dir = args.retrieval_dir / args.split
    out_path = args.out or split_dir / "qa_node_counts.jsonl"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}; pass --out to choose another path.")

    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_graphvis_datasets import merge_choice_graphs  # the pipeline's own merge

    cpnet = args.data_root / "cpnet" / "concept.txt"
    with cpnet.open(encoding="utf-8") as f:
        id2concept = [line.strip() for line in f]
    concept2id = {c: i for i, c in enumerate(id2concept)}
    statements = load_jsonl(args.data_root / "obqa" / "statement" / f"{args.split}.statement.jsonl")
    grounded = load_jsonl(args.data_root / "obqa" / "grounded" / f"{args.split}.grounded.jsonl")
    with (args.data_root / "obqa" / "graph" / f"{args.split}.graph.adj.pk").open("rb") as f:
        qagnn_entries = pickle.load(f)
    if not (len(grounded) == len(qagnn_entries) == 4 * len(statements)):
        raise ValueError(f"Row counts disagree: statements={len(statements)} grounded={len(grounded)} "
                         f"adj={len(qagnn_entries)} (expected 4 rows per statement)")

    mapping = json.loads((split_dir / "subset_index.json").read_text(encoding="utf-8"))
    mhgrn_graphs = {v: load_jsonl(split_dir / v / "graph.jsonl") for v in VARIANTS}
    for v, rows in mhgrn_graphs.items():
        if len(rows) != len(mapping):
            raise ValueError(f"{v}/graph.jsonl has {len(rows)} rows, subset_index.json has {len(mapping)}")

    # Group subset rows by question, checking the 4 choices are consecutive and in order.
    rows_by_q: dict[int, list[int]] = defaultdict(list)
    order: list[int] = []
    for row, m in enumerate(mapping):
        q_idx = int(m["question_idx"])
        if q_idx not in rows_by_q:
            order.append(q_idx)
        rows_by_q[q_idx].append(row)
        if m["choice_idx"] != len(rows_by_q[q_idx]) - 1:
            raise ValueError(f"subset row {row}: choices of question {q_idx} not in order 0..3")

    records = []
    for q_idx in order:
        st = statements[q_idx]
        labels = [c["label"] for c in st["question"]["choices"]]
        gold = st["answerKey"]
        g_rows = grounded[4 * q_idx: 4 * q_idx + 4]
        where = f"question {q_idx}"
        grounded_q = set().union(*(to_ids(r["qc"], concept2id, where) for r in g_rows))
        grounded_a_by_label = {lab: to_ids(r["ac"], concept2id, where) for lab, r in zip(labels, g_rows)}
        base = {
            "split": args.split, "question_idx": q_idx, "answerKey": gold,
            "grounded_q": len(grounded_q),
            "grounded_a": len(set().union(*grounded_a_by_label.values())),
        }

        merged = merge_choice_graphs(q_idx, qagnn_entries, st, id2concept)[0]
        records.append({**base, "method": "qagnn",
                        **count(merged, labels, gold, grounded_q, grounded_a_by_label, args.max_nodes)})
        for v in VARIANTS:
            rows = [mhgrn_graphs[v][r] for r in rows_by_q[q_idx]]
            merged = merge_mhgrn(rows, g_rows, labels, concept2id, where)
            records.append({**base, "method": v,
                            **count(merged, labels, gold, grounded_q, grounded_a_by_label, args.max_nodes)})

    with out_path.open("x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Per-question table
    head = (f"{'q':>6} {'method':<14} {'gQ':>3} {'gA':>3} | {'nodes':>6} {'Q':>3} {'A':>3} {'Q&A':>3} "
            f"{'core':>4} {'>cap':>4} {'goldA':>5} {'A/choice':<16} {'missQ':>5} {'missA':>5} {'missGold':>8}")
    print(head)
    print("-" * len(head))
    for r in records:
        per = " ".join(f"{k}:{n}" for k, n in r["a_per_choice"].items())
        print(f"{r['question_idx']:>6} {r['method']:<14} {r['grounded_q']:>3} {r['grounded_a']:>3} | "
              f"{r['nodes_total']:>6} {r['q_nodes']:>3} {r['a_nodes']:>3} {r['qa_overlap']:>3} "
              f"{r['core']:>4} {'yes' if r['core_exceeds_cap'] else '':>4} {r['gold_a_nodes']:>5} "
              f"{per:<16} {r['grounded_q_missing']:>5} {r['grounded_a_missing']:>5} "
              f"{r['gold_grounded_missing']:>8}")
        if r["method"] == VARIANTS[-1]:
            print()

    # Summary per method
    print(f"Summary over {len(order)} questions (mean; core>cap and missing counts as question totals)")
    print(f"{'method':<14} " + " ".join(f"{k:>12}" for k in NUMERIC) + f" {'core>cap':>9} {'any_missQ':>9} "
          f"{'any_missA':>9} {'missGold':>9}")
    for method in ("qagnn", *VARIANTS):
        rs = [r for r in records if r["method"] == method]
        means = " ".join(f"{statistics.mean(r[k] for r in rs):>12.2f}" for k in NUMERIC)
        print(f"{method:<14} {means} {sum(r['core_exceeds_cap'] for r in rs):>9} "
              f"{sum(r['grounded_q_missing'] > 0 for r in rs):>9} "
              f"{sum(r['grounded_a_missing'] > 0 for r in rs):>9} "
              f"{sum(r['gold_grounded_missing'] > 0 for r in rs):>9}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
