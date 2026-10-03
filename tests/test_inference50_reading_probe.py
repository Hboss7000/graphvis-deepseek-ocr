from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
sys.path.insert(0, str(STAGE1))
from score_stage1 import score_record


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_probe_selects_first_twenty_node_descriptions(tmp_path):
    records = [
        {"statement_idx": index, "task_type": "node_description", "image": "x.png",
         "prompt": "nodes?", "answer": "The image depicts the following nodes: alpha."}
        for index in reversed(range(25))
    ] + [{"statement_idx": 0, "task_type": "node_number"}]
    metadata = [{"statement_idx": index} for index in range(25)]
    source, meta, output = tmp_path / "stage1.jsonl", tmp_path / "meta.jsonl", tmp_path / "probe"
    dump(source, records)
    dump(meta, metadata)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/build_inference50_reading_probe.py"),
        "--stage1-jsonl", str(source), "--graph-metadata", str(meta),
        "--output-dir", str(output),
    ], check=True, capture_output=True)
    selected = [json.loads(line) for line in
                (output / "stage1_node_description_first20.jsonl").read_text().splitlines()]
    assert [row["statement_idx"] for row in selected] == list(range(20))


def test_probe_report_uses_raw_macro_node_recall_and_paired_indices(tmp_path):
    record = {
        "task_type": "node_description",
        "answer": "The image depicts the following nodes: alpha, beta.",
    }
    arms = {}
    for label, response in (("perfect", record["answer"]), ("half", "alpha"), ("repeat", record["answer"])):
        rows = []
        for index in (3, 7):
            row = {"statement_idx": index, "task_type": "node_description",
                   "hit_token_ceiling": False}
            row.update(score_record(record, response, {}))
            rows.append(row)
        path = tmp_path / f"{label}.jsonl"
        dump(path, rows)
        arms[label] = path
    output = tmp_path / "report.json"
    command = [sys.executable, str(ROOT / "scripts/report_inference50_reading_probe.py")]
    for label, path in arms.items():
        command += ["--arm", f"{label}={path}"]
    command += ["--expected-count", "2", "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["arms"]["perfect"]["raw_macro_gold_node_recall"] == 1.0
    assert report["arms"]["half"]["raw_macro_gold_node_recall"] == 0.5
    assert report["statement_indices"] == [3, 7]
