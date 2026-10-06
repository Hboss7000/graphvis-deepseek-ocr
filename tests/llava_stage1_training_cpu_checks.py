"""Explicit HF/PEFT CPU checks with random tiny weights and the pinned tokenizer.

Missing dependencies are a collection error, never a successful skipped suite.
No 7B weights are loaded. Float32 CPU tests do not establish CUDA/bf16 behavior.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import (CLIPVisionConfig, LlavaNextConfig, LlavaNextForConditionalGeneration,
                          LlavaNextImageProcessor, LlavaNextProcessor, MistralConfig)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION, configure_processor, image_token_id
from llava_stage1_training import (Stage1Collator, attach_lora, load_checkpoint, projector,
                                   save_checkpoint, training_text)


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1)
    original = LlavaNextProcessor.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION,
                                                  local_files_only=True)
    original.tokenizer.pad_token = original.tokenizer.eos_token  # Exercise pad==EOS.
    image_processor = LlavaNextImageProcessor(size={'shortest_edge': 16},
        crop_size={'height': 16, 'width': 16}, image_grid_pinpoints=[[16, 16], [16, 32], [32, 16]])
    processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=original.tokenizer)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=4, image_size=16, patch_size=4)
    text = MistralConfig(vocab_size=len(original.tokenizer), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=4096, attention_dropout=0.,
                        bos_token_id=original.tokenizer.bos_token_id,
                        eos_token_id=original.tokenizer.eos_token_id,
                        pad_token_id=original.tokenizer.pad_token_id)
    config = LlavaNextConfig(vision_config=vision.to_dict(), text_config=text.to_dict(),
        image_token_index=image_token_id(original), image_grid_pinpoints=image_processor.image_grid_pinpoints,
        vision_feature_select_strategy='default', vision_feature_layer=-2)
    configure_processor(processor, config)
    return processor, config


@pytest.fixture
def examples(tmp_path):
    Image.new('RGB', (16, 16), '#ADD8E6').save(tmp_path / 'a.png')
    Image.new('RGB', (32, 16), '#ADD8E6').save(tmp_path / 'b.png')
    return tmp_path, [
        {'image': 'a.png', 'task_type': 'node_number', 'prompt': 'How many nodes are there in the graph?',
         'answer': 'There are 3 nodes in the graph.'},
        {'image': 'b.png', 'task_type': 'triple_listing', 'prompt': 'List all the triples in the graph.',
         'answer': 'The triples in the graph are listed as: (a, is a, b), (b, part of, c).'}]


def new_model(config):
    torch.manual_seed(13)
    model = LlavaNextForConditionalGeneration(copy.deepcopy(config))
    model, names = attach_lora(model)  # Fixed r=128, alpha=256, dropout=.05.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    assert len(names) == 7  # Mistral q/k/v/o and gate/up/down; no head/vision/projector.
    assert all('.language_model.' in name for name in names)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    return model, optimizer, scheduler


def step(model, optimizer, scheduler, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients.values())
    assert any(g.abs().sum() > 0 for name, g in gradients.items() if 'lora_B' in name)
    assert any(g.abs().sum() > 0 for name, g in gradients.items() if 'multi_modal_projector' in name)
    assert all(p.grad is None for name, p in model.named_parameters() if 'vision_tower' in name)
    optimizer.step()
    scheduler.step()
    return output.loss.detach().clone()


def test_collator_answer_mask_image_expansion_eos_and_no_truncation(setup, examples):
    processor, _ = setup
    root, records = examples
    batch = Stage1Collator(processor, root)(records)
    for i, row in enumerate(records):
        prefix, full = training_text(processor, row)
        n = int(batch['attention_mask'][i].sum())
        labels = batch['labels'][i]
        supervised = labels[labels != -100]
        decoded = processor.tokenizer.decode(supervised)
        assert decoded.strip() == row['answer'] + processor.tokenizer.eos_token
        assert supervised[-1] == processor.tokenizer.eos_token_id
        assert (supervised == processor.tokenizer.eos_token_id).sum() == 1
        image_mask = batch['input_ids'][i] == image_token_id(processor)
        assert image_mask.sum() > 1 and (labels[image_mask] == -100).all()
        assert (labels[n:] == -100).all()
        first = int(torch.where(labels != -100)[0][0])
        assert (labels[:first] == -100).all()
        assert first == len(processor.tokenizer(prefix)['input_ids']) + int(image_mask.sum()) - 1
        assert full.endswith(' ASSISTANT: ' + row['answer'] + '</s>')
    with pytest.raises(ValueError, match='exceeds'):
        Stage1Collator(processor, root, model_max_length=32)(records)
    with pytest.raises(ValueError, match='injected'):
        Stage1Collator(processor, root)([{**records[0], 'answer': 'Oops</s>'}])


def test_finite_real_vision_forward_backward_and_exact_resume(setup, examples, tmp_path):
    processor, config = setup
    root, records = examples
    batch = Stage1Collator(processor, root)(records)
    model, optimizer, scheduler = new_model(config)
    step(model, optimizer, scheduler, batch)
    run_config = {'test': 'tiny random weights', 'revision': DEFAULT_REVISION,
                  'template': 'llava_v1', 'lora_r': 128, 'lora_alpha': 256, 'dropout': .05}
    checkpoint = tmp_path / 'checkpoint-1'
    digest = save_checkpoint(model, optimizer, scheduler, checkpoint, run_config, 1, 2)
    assert len(digest) == 64 and (checkpoint / 'adapter/adapter_model.safetensors').exists()
    expected_loss = step(model, optimizer, scheduler, batch)
    expected = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    resumed, opt2, sched2 = new_model(config)
    assert load_checkpoint(resumed, opt2, sched2, checkpoint, run_config) == (1, 2)
    actual_loss = step(resumed, opt2, sched2, batch)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for name, p in resumed.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, expected[name], rtol=0, atol=0)
    assert opt2.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    with pytest.raises(ValueError, match='configuration differs'):
        load_checkpoint(resumed, opt2, sched2, checkpoint, {**run_config, 'dropout': 0})
    with (checkpoint / 'projector.pt').open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_checkpoint(resumed, opt2, sched2, checkpoint, run_config)
