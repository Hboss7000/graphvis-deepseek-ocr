"""Native tiny random LLaVA CPU inference; no pretrained weights or CUDA calls."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from llava_stage1_training_cpu_checks import setup
import synth_mcq_llava_backend as backend
from synth_mcq_llava_common import letter_variants, VERIFIED_LETTER_IDS
from llava_stage1_training import projector
from llava_stage1_prompt import format_llava_prompt


def test_native_pinned_tokenizer_ids(setup):
    processor, _ = setup
    assert letter_variants(processor.tokenizer, verify_pin=True) == VERIFIED_LETTER_IDS
    from synth_mcq_llava_common import ANSWER_PREFIX
    prompt = format_llava_prompt(processor, 'Question: Which name?', 'text', 'llava_v1')
    original = processor.tokenizer.encode(prompt)
    forced = processor.tokenizer.encode(prompt + ' ' + ANSWER_PREFIX)
    assert forced == original + processor.tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    from synth_mcq_llava_common import token_letter
    for reply in ('C.', 'B.'):
        tokens = processor.tokenizer.encode(reply, add_special_tokens=False)
        assert token_letter(tokens[0], VERIFIED_LETTER_IDS) == reply[0]


def test_pristine_sanity_and_projector_digest_cpu(setup):
    _, config = setup
    model = LlavaNextForConditionalGeneration(copy.deepcopy(config))
    before = backend.load_sanity(model, 'base')
    assert before['lora_tensors'] == before['peft_modules'] == 0
    with torch.no_grad():
        for p in projector(model).parameters(): p.add_(.1)
    assert backend.projector_digest(model) != before['projector_sha256']


@pytest.mark.parametrize('condition', ['image', 'text', 'none'])
def test_native_generation_logits_variants_prompt_and_tokens_cpu(setup, tmp_path, monkeypatch, condition):
    processor, config = setup
    model = LlavaNextForConditionalGeneration(copy.deepcopy(config)).eval()
    path = tmp_path / 'graph.png'; Image.new('RGB', (32, 16), 'white').save(path)
    body = 'Question: Which object?\nChoices:\nA. a\nB. b\nC. c\nD. d\nAnswer with the correct option\'s letter.'
    if condition == 'image': body = '<image>\n' + body
    if condition == 'text': body = '(a, is a, b)\n\n' + body
    args = SimpleNamespace(prompt_template='llava_v1', letter_ids=letter_variants(processor.tokenizer, verify_pin=True))
    assert backend.format_qa_prompt(processor, body, condition, args) == format_llava_prompt(processor, body, condition, 'llava_v1')
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    result = backend.infer_one(model, processor, body, condition, path, args, torch)
    assert model.device.type == 'cpu' and result['generated_token_count'] <= 64
    assert result['parser_protocol'] == 'explicit_v1'
    assert bool(result['vision_tokens']) == (condition == 'image')
    assert result['first_letter_token_ids'] == VERIFIED_LETTER_IDS
    for letter in 'ABCD':
        assert result['first_letter_logits'][letter] == max(result['first_letter_variant_logits'][letter])
        assert result['prefix_letter_logits'][letter] == max(result['prefix_letter_variant_logits'][letter])
    assert result['prefix_probe'] == 'The correct answer is'
    assert result['prefix_token_ids'] == processor.tokenizer.encode(result['prefix_probe'], add_special_tokens=False)
    assert result['prefix_probe_elapsed_seconds'] > 0
    assert result['first_generated_letter'] == next((l for l, ids in VERIFIED_LETTER_IDS.items()
        if result['generated_token_ids'][0] in ids), None)
    expected = torch.tensor(list(result['first_letter_logits'].values())).softmax(0)
    torch.testing.assert_close(torch.tensor(list(result['first_letter_probs'].values())), expected)
    assert result['peak_memory_allocated_bytes'] == 0


def test_prefix_probe_fake_tokenizer_both_variants_and_multimodal_inputs():
    from test_synth_mcq_llava_backend import FakeTokenizer
    from synth_mcq_llava_common import ANSWER_PREFIX
    tokenizer = FakeTokenizer()
    variants = letter_variants(tokenizer, verify_pin=True)
    inputs = dict(input_ids=torch.tensor([[11, 22]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
                  pixel_values=torch.tensor([3.]), image_sizes=torch.tensor([[32, 16]]))
    def model(**kwargs):
        assert kwargs['input_ids'].tolist() == [[11, 22] + tokenizer.encode(ANSWER_PREFIX)]
        assert kwargs['attention_mask'].tolist() == [[1] * 6]
        assert kwargs['pixel_values'] is inputs['pixel_values']
        assert kwargs['image_sizes'] is inputs['image_sizes']
        assert kwargs['use_cache'] is False and not torch.is_grad_enabled()
        assert kwargs['logits_to_keep'] == 1
        logits = torch.zeros(1, 6, 32001)
        for index, letter in enumerate('ABCD'):
            logits[0, -1, variants[letter][index % 2]] = index + 1
        # Earlier positions have contradictory logits: only final position counts.
        logits[0, 0, variants['A'][0]] = 100
        return SimpleNamespace(logits=logits)
    before = inputs['input_ids'].clone()
    result = backend.forced_choice_probe(model, inputs, tokenizer, variants, torch)
    assert result['prefix_letter_logits'] == dict(A=1., B=2., C=3., D=4.)
    torch.testing.assert_close(inputs['input_ids'], before)
