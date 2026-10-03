from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("condition", prompt_common.CONDITIONS)
def test_llava_qa_fresh_output_generates_all_eight_rows(tmp_path, monkeypatch, condition):
    import types

    records = []
    metadata = []
    for index in range(8):
        prompt = prompt_common.IMAGE_REFERENCE_SENTENCE + "Question? Choices A-D."
        records.append({"statement_idx": index, "image": f"q{index}.png",
                        "prompt": prompt, "answer": "A"})
        metadata.append({"statement_idx": index, "visible_nodes": [
            {"cid": 1, "name": "alpha", "connected": True},
            {"cid": 2, "name": "beta", "connected": True}],
            "edges": [{"source_cid": 1, "target_cid": 2, "relation": "isa"}]})
    input_path, metadata_path = tmp_path / "input.jsonl", tmp_path / "metadata.jsonl"
    write_jsonl(input_path, records)
    write_jsonl(metadata_path, metadata)
    output_path = tmp_path / "fresh" / "predictions.jsonl"

    class FakeImage:
        size = (100, 80)

        def convert(self, mode):
            assert mode == "RGB"
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeProcessor:
        tokenizer = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            instance = cls()
            instance.tokenizer = instance
            return instance

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[0]["content"]

    class FakeModel:
        generation_config = SimpleNamespace()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "test"
    fake_transformers.set_seed = lambda seed: None
    fake_transformers.LlavaNextConfig = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=14), vision_feature_select_strategy="default"))
    fake_transformers.LlavaNextForConditionalGeneration = FakeModel
    fake_transformers.LlavaNextProcessor = FakeProcessor
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                  bfloat16="bf16", __version__="test-torch")
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = SimpleNamespace(open=lambda path: FakeImage())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setattr(run_zero_shot_llava, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(run_zero_shot_llava, "image_diagnostics", lambda *args: {"stub": True})
    monkeypatch.setattr(run_zero_shot_llava, "infer_one",
                        lambda *args: ("A", 1, False, 0, 0.01, 123))
    monkeypatch.setattr(sys, "argv", ["run_zero_shot_llava.py", "--revision", "a" * 40,
        "--input-jsonl", str(input_path), "--graph-metadata", str(metadata_path),
        "--image-root", str(tmp_path), "--output-jsonl", str(output_path),
        "--condition", condition, "--expected-count", "8", "--max-new-tokens", "64",
        "--approve-prompt-diff", "--resume"])
    run_zero_shot_llava.main()
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    config = json.loads((output_path.parent / "run_config.json").read_text())
    assert len(rows) == 8
    assert config["effective_max_new_tokens"] == config["generation"]["max_new_tokens"] == 64


def test_llava_qa_fresh_path_cannot_silently_skip_every_input(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="generated zero predictions"):
        run_zero_shot_llava.require_fresh_output_progress(False, 8, 0)
    run_zero_shot_llava.require_fresh_output_progress(False, 8, 8)
    run_zero_shot_llava.require_fresh_output_progress(True, 8, 0)
