"""Tiny random Qwen CPU checks: no training, pretrained weights, CUDA or pod access."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLForConditionalGeneration
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qwen_stage1_training_cpu_checks import setup  # reuse the native pinned processor/tiny config fixture
from qwen_stage1_training import attach_lora, DEFAULT_MODEL_ID, DEFAULT_REVISION
from llava_stage1_training import projector, file_sha
from llava_adapter import load_adapter
from run_synth_mcq import load_sanity, infer_one
import run_zero_shot_qwen as qa
from audit_synth_mcq_images import qwen_image_measurement


def test_real_cpu_processor_cap_height_uses_vertical_grid_scale(setup):
    processor, _ = setup
    image = Image.new('RGB', (96, 160), 'white')
    args = SimpleNamespace(min_pixels=256, max_pixels=4096)
    row = qwen_image_measurement(processor, image, '<image>\nQuestion: Which object?', args, 36, 28)
    grid = row['image_grid_thw']
    assert row['width'] == 96 and row['height'] == 160
    assert row['resized_width'] == grid[2] * processor.image_processor.patch_size
    assert row['resized_height'] == grid[1] * processor.image_processor.patch_size
    assert row['realised_image_tokens'] == grid[0] * grid[1] * grid[2] // processor.image_processor.merge_size ** 2
    assert row['estimated_label_cap_px_qwen'] == pytest.approx(36 * row['resized_height'] / 160)
    assert row['estimated_node_cap_px_qwen'] == row['estimated_label_cap_px_qwen']
    assert row['estimated_edge_cap_px_qwen'] == pytest.approx(28 * row['resized_height'] / 160)
    assert row['resized_width'] * row['resized_height'] <= args.max_pixels


def test_pristine_and_complete_checkpoint_sanity(setup, tmp_path):
    _, config = setup
    base = Qwen3VLForConditionalGeneration(copy.deepcopy(config))
    assert load_sanity(base, 'base')['lora_tensors'] == 0
    tuned, _ = attach_lora(base, r=2, alpha=4, dropout=0)
    with torch.no_grad():
        for n, p in tuned.named_parameters():
            if 'lora_B' in n: p.fill_(.03)
        for p in projector(tuned).parameters(): p.add_(.01)
    tuned.save_pretrained(tmp_path / 'adapter', safe_serialization=True)
    torch.save(projector(tuned).state_dict(), tmp_path / 'projector.pt')
    (tmp_path / 'run_config.json').write_text(json.dumps(dict(model_id=DEFAULT_MODEL_ID, model_revision=DEFAULT_REVISION)))
    manifest = {str(p.relative_to(tmp_path)): file_sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path / 'checkpoint_manifest.json').write_text(json.dumps(manifest))
    fresh = Qwen3VLForConditionalGeneration(copy.deepcopy(config))
    assert load_sanity(fresh, 'base')['pristine_base']
    loaded, _ = load_adapter(fresh, tmp_path, DEFAULT_MODEL_ID, DEFAULT_REVISION)
    sanity = load_sanity(loaded, 'tuned', tmp_path)
    assert sanity['lora_tensors'] > 0 and sanity['merger_tensors'] > 0
    assert sanity['merger_groups'] == 4
    for name, value in projector(loaded).state_dict().items():
        torch.testing.assert_close(value, projector(tuned).state_dict()[name], rtol=0, atol=0)
    expected = {n: p for n, p in tuned.named_parameters() if 'lora_' in n}
    for name, value in loaded.named_parameters():
        if 'lora_' in name: torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(AssertionError, match='PEFT/LoRA'): load_sanity(loaded, 'base')


@pytest.mark.parametrize('condition', ['image', 'text', 'none'])
def test_actual_first_generation_logits_native_template_and_parser_cpu(setup, tmp_path, monkeypatch, condition):
    processor, config = setup
    model = Qwen3VLForConditionalGeneration(copy.deepcopy(config)).eval()
    image_path = tmp_path / 'graph.png'
    Image.new('RGB', (16, 16), 'white').save(image_path)
    core = "Question: Which object?\nChoices:\nA. a\nB. b\nC. c\nD. d\nAnswer with the correct option's letter."
    body = '<image>\n' + core if condition == 'image' else '(a, is a, b)\n\n' + core if condition == 'text' else core
    actual_prepare = qa.prepare_inputs
    class CpuInputs(dict):
        def to(self, device):
            assert device == 'cuda'
            return self
    monkeypatch.setattr(qa, 'prepare_inputs', lambda *args: CpuInputs(actual_prepare(*args)))
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    args = SimpleNamespace(min_pixels=256, max_pixels=1024, max_new_tokens=2)
    result = infer_one(model, processor, body, condition, image_path, args, torch)
    assert sum(result['first_letter_probs'].values()) == pytest.approx(1, abs=1e-6)
    logits = torch.tensor(list(result['first_letter_logits'].values()))
    torch.testing.assert_close(torch.tensor(list(result['first_letter_probs'].values())), logits.softmax(-1))
    assert result['generated_token_count'] <= 2
    assert bool(result['vision_tokens']) == (condition == 'image')
    assert result['parser_protocol'] == 'explicit_v1'
    if condition == 'image':
        grid = result['image_budget']['grid_thw']
        assert result['vision_tokens'] == grid[0] * grid[1] * grid[2] // processor.image_processor.merge_size ** 2
        assert result['image_budget']['realised_vision_tokens'] == result['vision_tokens']
    assert result['parse_ok'] == (result['predicted_letter'] is not None)
    assert result['peak_memory_allocated_bytes'] == 0
