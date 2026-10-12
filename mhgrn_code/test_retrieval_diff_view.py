#!/usr/bin/env python3
"""Tests for the standalone retrieval diagnostic view."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

nx = pytest.importorskip("networkx", reason="isolated comparison dependencies are in venv_mhgrn")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import retrieval_diff_view as diff


def test_membership_classification_three_methods():
    classes = diff.membership_classes({
        "a": {1, 2, 3, 5},
        "b": {2, 3, 4, 6},
        "c": {3, 5, 6, 7},
    })
    assert classes == {
        "unique_to__a": {1},
        "shared_by__a__b": {2},
        "shared_by_all": {3},
        "unique_to__b": {4},
        "shared_by__a__c": {5},
        "shared_by__b__c": {6},
        "unique_to__c": {7},
    }


def test_pair_membership_has_only_both_and_method_only_classes():
    classes = diff.pair_membership_classes({
        "qagnn": {1, 2, 3},
        "mhgrn_2hop": {2, 3, 4},
    })
    assert classes == {
        "in_both": {2, 3},
        "only_in__qagnn": {1},
        "only_in__mhgrn_2hop": {4},
    }


@pytest.mark.parametrize("value", [
    "qagnn",
    "qagnn,mhgrn_2hop, mhgrn_default",
    "qagnn,unknown",
    "qagnn,qagnn",
])
def test_pair_argument_rejects_invalid_values(value):
    with pytest.raises(diff.argparse.ArgumentTypeError):
        diff.parse_pair(value)


def test_pair_argument_accepts_valid_ordered_pair():
    assert diff.parse_pair("qagnn,mhgrn_2hop") == ("qagnn", "mhgrn_2hop")


def test_lossless_extraction_keeps_forward_direction_all_relations_and_relatedto():
    cpnet = nx.MultiDiGraph()
    cpnet.add_edge(0, 1, rel=18)  # inverse id: ignored
    cpnet.add_edge(1, 0, rel=1)   # forward edge represented in reverse direction
    cpnet.add_edge(0, 1, rel=4)
    cpnet.add_edge(0, 1, rel=15)  # relatedto must remain
    cpnet.add_edge(0, 1, rel=4)   # duplicate relation
    graphs = [{
        "nodes": [{"id": "u", "cid": 0}, {"id": "v", "cid": 1}],
        "links": [{"source": "u", "target": "v"}],
    }]
    edges, links = diff.lossless_mhgrn_edges(graphs, cpnet)
    assert links == 1
    assert edges == {(1, 1, 0), (0, 4, 1), (0, 15, 1)}


def test_consistency_check_raises_on_mismatch():
    actual = {
        "n_nodes": 4, "n_edges": 3, "gold_degree": 1.0,
        "choice_concepts_deg0": 0,
    }
    reference = {"after_pruning": {
        "n_nodes": 5, "n_edges": 3, "gold_degree": 1.0,
        "choice_concepts_deg0": 0,
    }}
    with pytest.raises(ValueError, match=r"q9 qagnn.*n_nodes"):
        diff.assert_post_pruning_consistency(9, "qagnn", actual, reference)


def test_prepare_output_refuses_existing_folder(tmp_path):
    target = tmp_path / "diff_view"
    target.mkdir()
    with pytest.raises(FileExistsError):
        diff.prepare_output_dir(target, minimum_free=0)


def test_prepare_output_refuses_low_disk_before_writing(tmp_path, monkeypatch):
    target = tmp_path / "diff_view"
    monkeypatch.setattr(diff.shutil, "disk_usage", lambda _path: SimpleNamespace(free=99))
    with pytest.raises(OSError, match="at least"):
        diff.prepare_output_dir(target, minimum_free=100)
    assert not target.exists()


def test_diagnostic_filename_is_inside_diff_view_and_marked(tmp_path):
    diff_root = tmp_path / "diff_view"
    target = diff.diagnostic_image_path(diff_root, 60, "qagnn")
    assert target.parent == diff_root / "images"
    assert target.name == "q00060_qagnn_DIAGNOSTIC.png"
    assert not list(tmp_path.rglob("*"))
