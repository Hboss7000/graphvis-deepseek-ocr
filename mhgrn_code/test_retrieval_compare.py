#!/usr/bin/env python3
"""Synthetic regression tests for the isolated MHGRN retrieval comparison."""

from __future__ import annotations

import json
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

nx = pytest.importorskip("networkx", reason="isolated comparison dependencies are in venv_mhgrn")
import numpy as np
from scipy.sparse import coo_matrix


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import compare_retrieval as compare
import run_mhgrn_subset as subset_runner
from generate_graphvis_datasets import merge_choice_graphs
from retrieval_compare_common import filter_paths_2hop
from run_mhgrn_subset import write_subset


def test_main_setup_forces_fork_before_argument_parsing(monkeypatch):
    class StopAfterSetup(Exception):
        pass

    def stop_at_parse_args():
        raise StopAfterSetup

    monkeypatch.setattr(subset_runner, "parse_args", stop_at_parse_args)
    with pytest.raises(StopAfterSetup):
        subset_runner.main()
    assert multiprocessing.get_start_method() == "fork"


def test_score_triples_wrapper_returns_python_float_without_changing_value():
    original_value = np.float32(0.12345679)
    paths_module = SimpleNamespace(score_triples=lambda *args, **kwargs: original_value)

    subset_runner.install_score_triples_json_float_shim(paths_module)
    result = paths_module.score_triples(concept_id=[1, 2], relation_id=[[0]])

    assert type(result) is float
    assert result == float(original_value)


def test_merge_mhgrn_choice_graphs_matches_qagnn_value_types():
    concepts = np.array([0, 1])
    entries = []
    for _choice_idx in range(4):
        adj = coo_matrix(([1], ([0], [1])), shape=(17 * 2, 2))
        entries.append({
            "concepts": concepts,
            "qmask": np.array([True, False]),
            "amask": np.array([False, True]),
            "adj": adj,
        })
    statement = {
        "answerKey": "A",
        "question": {"choices": [{"label": label} for label in "ABCD"]},
    }
    q_nodes, q_edges, _answer = merge_choice_graphs(0, entries, statement, ["question", "answer"])

    graphs = [{
        "nodes": [{"id": 0, "cid": 0}, {"id": 1, "cid": 1}],
        "links": [{"source": 0, "target": 1}],
    } for _ in range(4)]
    grounded = [{"qc": ["question"], "ac": ["answer"]} for _ in range(4)]
    faithful = {"triples": [{"source_cid": 0, "relation_id": 0, "target_cid": 1}]}
    m_nodes, m_edges, _answer = compare.merge_mhgrn_choice_graphs(
        graphs, grounded, statement["question"]["choices"], ["question", "answer"], faithful,
    )

    assert type(m_nodes) is type(q_nodes) is dict
    assert type(m_edges) is type(q_edges) is set
    assert set(m_nodes) == set(q_nodes)
    for cid in m_nodes:
        assert set(m_nodes[cid]) == set(q_nodes[cid])
        for key in m_nodes[cid]:
            assert type(m_nodes[cid][key]) is type(q_nodes[cid][key])
    assert all(type(edge) is tuple for edge in m_edges)


def test_copied_get_edge_matches_graphvis_filter_and_order():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, rel=20)
    graph.add_edge(1, 0, rel=2)
    graph.add_edge(1, 2, rel=15)
    graph.add_edge(1, 2, rel=7)
    graph.add_edge(1, 2, rel=7)
    graph.add_edge(1, 2, rel=4)
    graph.add_edge(2, 3, rel=15)
    graph.add_edge(3, 4, rel=0)
    graph.add_edge(3, 4, rel=17)
    compare.cpnet = graph

    assert compare.get_edge(0, 1) == []
    assert compare.get_edge(1, 2) == [7, 4]
    assert compare.get_edge(2, 3) == [15]
    assert compare.get_edge(3, 4) == [0]


def test_two_hop_filter_keeps_two_and_three_node_paths_only():
    rows = [[{
        "qc": 0,
        "ac": 5,
        "pf_res": [
            {"path": [0, 1], "rel": [[0]]},
            {"path": [0, 1, 2], "rel": [[0], [1]]},
            {"path": [0, 1, 2, 3], "rel": [[0], [1], [2]]},
            {"path": [0, 1, 2, 3, 4], "rel": [[0], [1], [2], [3]]},
        ],
    }]]
    filtered = filter_paths_2hop(rows)
    assert [len(item["path"]) for item in filtered[0][0]["pf_res"]] == [2, 3]
    assert len(rows[0][0]["pf_res"]) == 4


def test_subset_writer_maps_question_to_four_grounded_lines(tmp_path):
    grounded = [{"source_line": index} for index in range(12)]
    grounded_path = tmp_path / "subset.grounded.jsonl"
    index_path = tmp_path / "subset_index.json"
    write_subset(grounded, "dev", [2, 0], grounded_path, index_path)

    selected = [json.loads(line) for line in grounded_path.read_text().splitlines()]
    mapping = json.loads(index_path.read_text())
    assert [row["source_line"] for row in selected] == [8, 9, 10, 11, 0, 1, 2, 3]
    assert [(row["question_idx"], row["choice_idx"]) for row in mapping] == [
        (2, 0), (2, 1), (2, 2), (2, 3),
        (0, 0), (0, 1), (0, 2), (0, 3),
    ]
