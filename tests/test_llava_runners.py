from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
QA_SCRIPTS = ROOT / "experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts"
STAGE1_SCRIPTS = ROOT / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
for directory in (SCRIPTS, QA_SCRIPTS, STAGE1_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import llava_common
import prompt_common
import stage1_common
import run_stage1_llava
import run_stage1_qwen
import run_zero_shot_llava
import run_zero_shot_qwen


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "[INST] " + messages[0]["content"] + " [/INST]"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_llava_uses_shared_stage1_prompt_body_hash():
    records = [
        {"statement_idx": 4, "task_type": "node_number", "prompt": "How many nodes?"},
        {"statement_idx": 9, "task_type": "edge_number", "prompt": "How many edges?"},
    ]
    # Both Qwen and LLaVA runners call this exact model-independent contract.
    expected_qwen_hash = run_stage1_qwen.prompt_bodies_sha256(records, "none")
    llava_hash = run_stage1_llava.prompt_bodies_sha256(records, "none")
    assert llava_hash == expected_qwen_hash


def test_llava_uses_shared_qa_prompt_body_hash():
    records = [{"statement_idx": 4, "prompt": prompt_common.IMAGE_REFERENCE_SENTENCE + "Question"}]
    metadata = {4: {
        "statement_idx": 4,
        "visible_nodes": [
            {"cid": 1, "name": "alpha", "connected": True},
            {"cid": 2, "name": "beta", "connected": True},
        ],
        "edges": [{"source_cid": 1, "relation": "isa", "target_cid": 2}],
    }}
    for condition in prompt_common.CONDITIONS:
        expected_qwen_hash = run_zero_shot_qwen.prompt_bodies_sha256(records, metadata, condition)
        llava_hash = run_zero_shot_llava.prompt_bodies_sha256(records, metadata, condition)
        assert llava_hash == expected_qwen_hash


def test_llava_image_placeholder_only_in_image_condition():
    processor = FakeProcessor()
    image = llava_common.format_prompt(processor, "<image>\nQuestion", "image")
    assert image.count("<image>") == 1
    for condition in ("text", "text_noref", "kg_text"):
        formatted = llava_common.format_prompt(processor, "Question", condition)
        assert "<image>" not in formatted


def test_llava_resume_skips_completed_keys_and_rejects_duplicates(tmp_path):
    task_types = ("node_number", "edge_number")
    node_path = tmp_path / "predictions_llava_node_number.jsonl"
    write_jsonl(node_path, [{"statement_idx": 3, "task_type": "node_number"}])
    done = llava_common.completed_stage1_keys(tmp_path, task_types)
    all_keys = {(3, "node_number"), (3, "edge_number")}
    assert all_keys - done == {(3, "edge_number")}
    write_jsonl(node_path, [
        {"statement_idx": 3, "task_type": "node_number"},
        {"statement_idx": 3, "task_type": "node_number"},
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        llava_common.completed_stage1_keys(tmp_path, task_types)


def test_llava_qa_resume_skips_completed_indices_and_rejects_duplicates(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [{"statement_idx": 3}])
    done = llava_common.completed_qa_indices(path)
    assert {3, 7} - done == {7}
    write_jsonl(path, [{"statement_idx": 3}, {"statement_idx": 3}])
    with pytest.raises(ValueError, match="Duplicate"):
        llava_common.completed_qa_indices(path)
