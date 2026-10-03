from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_graphvis_datasets as generator
import select_subset


def test_subset_selection_is_deterministic(tmp_path):
    source = tmp_path / "test.statement.jsonl"
    source.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(500)))
    first = select_subset.build_subset(source, seed=13, n=50)
    second = select_subset.build_subset(source, seed=13, n=50)
    assert first == second
    assert first["indices"] == sorted(first["indices"])
    assert len(first["indices"]) == len(set(first["indices"])) == 50
    assert first["source_n"] == 500
    assert first["sha256_of_statement_file"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_indices_manifest_validation_and_suffix(tmp_path):
    source = tmp_path / "test.statement.jsonl"
    source.write_text("{}\n" * 20)
    manifest = tmp_path / "obqa_test_subset20_seed13.json"
    manifest.write_text(json.dumps({
        "split": "test",
        "seed": 13,
        "n": 20,
        "source_n": 20,
        "indices": list(reversed(range(20))),
        "sha256_of_statement_file": hashlib.sha256(source.read_bytes()).hexdigest(),
    }))
    assert generator.load_indices_file(manifest, "test", 20, source) == list(range(20))
    assert generator.indices_suffix(manifest, "test") == "subset20_seed13"


@pytest.mark.skipif(
    not (ROOT / "data_preprocessed_release/obqa/statement/test.statement.jsonl").is_file(),
    reason="QA-GNN OBQA release is not present",
)
def test_indices_file_matches_contiguous_slice_byte_for_byte(tmp_path):
    statement = ROOT / "data_preprocessed_release/obqa/statement/test.statement.jsonl"
    manifest = tmp_path / "obqa_test_identity20.json"
    manifest.write_text(json.dumps({
        "split": "test",
        "seed": 13,
        "n": 20,
        "source_n": 500,
        "indices": list(range(20)),
        "sha256_of_statement_file": hashlib.sha256(statement.read_bytes()).hexdigest(),
    }))
    sliced = tmp_path / "sliced"
    indexed = tmp_path / "indexed"
    common = [
        sys.executable,
        str(ROOT / "scripts/generate_graphvis_datasets.py"),
        "--split", "test",
        "--stage1-task-set", "extended",
        "--stage1-balance", "per-task",
        "--seed", "13",
        "--max-nodes", "18",
        "--max-edges", "60",
        "--max-degree", "0",
        "--data-root", str(ROOT / "data_preprocessed_release"),
    ]
    subprocess.run(common + ["--start", "0", "--limit", "20", "--out-dir", str(sliced)], check=True)
    subprocess.run(common + ["--indices-file", str(manifest), "--out-dir", str(indexed)], check=True)

    sliced_root = sliced / "test"
    indexed_root = indexed / "test"
    for prefix in ("stage1_graph_comprehension", "stage2_obqa", "graph_metadata"):
        assert (sliced_root / f"{prefix}_0_20.jsonl").read_bytes() == (
            indexed_root / f"{prefix}_identity20.jsonl"
        ).read_bytes()
    for index in range(20):
        for relative in (f"images/q{index:05d}_clean.png", f"graphs/q{index:05d}.jsonl"):
            assert (sliced_root / relative).read_bytes() == (indexed_root / relative).read_bytes()
