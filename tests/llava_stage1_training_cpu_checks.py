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
def examples(tmp_path, monkeypatch):
    monkeypatch.setenv('WORKSPACE', str(tmp_path))
    monkeypatch.setenv('VOLUME_CAP_GB', '200')
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


def test_real_training_loop_resume_adapter_inference_and_file_store(setup, examples, tmp_path, monkeypatch):
    import csv
    import socket
    from train_llava_stage1 import train_steps, optimizer_and_scheduler
    from llava_training_logging import TrainingLogger
    from llava_adapter import load_adapter

    def deny_network(*args, **kwargs):
        raise AssertionError('Training/MLflow must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    processor, config = setup
    root, records = examples
    collator = Stage1Collator(processor, root)
    contract = {'model_id': DEFAULT_MODEL_ID, 'model_revision': DEFAULT_REVISION,
                'mode': 'full', 'optimizer_steps': 2, 'global_batch_size': 2,
                'per_device_batch_size': 1, 'test': 'random tiny model, float32 CPU'}
    store = (tmp_path / 'mlruns').as_uri()
    reference, _, _ = new_model(config)
    optimizer, scheduler = optimizer_and_scheduler(reference, 2e-5, 2e-5, 2)
    reference_dir = tmp_path / 'reference'
    log = TrainingLogger(reference_dir, contract, store)
    train_steps(reference, optimizer, scheduler, records * 2, records, collator,
                reference_dir, contract, log, torch.device('cpu'), torch.float32)
    assert log.readable()
    log.close()
    expected = {name: p.detach().clone() for name, p in reference.named_parameters() if p.requires_grad}

    model, _, _ = new_model(config)
    optimizer, scheduler = optimizer_and_scheduler(model, 2e-5, 2e-5, 2)
    directory = tmp_path / 'resumed'
    log = TrainingLogger(directory, contract, store)
    first = train_steps(model, optimizer, scheduler, records * 2, records, collator,
                        directory, contract, log, torch.device('cpu'), torch.float32, stop_after_step=1)
    run_id = log.run_id
    log.close('KILLED')
    checkpoint = Path(first['checkpoint'])
    resumed, _, _ = new_model(config)
    optimizer, scheduler = optimizer_and_scheduler(resumed, 2e-5, 2e-5, 2)
    log = TrainingLogger(directory, contract, store, resume_step=1)
    final = train_steps(resumed, optimizer, scheduler, records * 2, records, collator,
                        directory, contract, log, torch.device('cpu'), torch.float32, resume=checkpoint)
    assert final['complete'] and log.run_id == run_id and log.readable()
    log.close()
    for name, value in resumed.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with (directory / 'metrics.csv').open() as handle:
        logged = list(csv.DictReader(handle))
    assert [int(r['step']) for r in logged if r['event'] == 'train'] == [1, 2]
    assert any(r['validation_loss'] for r in logged)

    # Frozen random backbone must match exactly; the adapter loader supplies only trained weights.
    torch.manual_seed(13)
    bare = LlavaNextForConditionalGeneration(copy.deepcopy(config))
    inference, provenance = load_adapter(bare, final['checkpoint'], DEFAULT_MODEL_ID, DEFAULT_REVISION)
    batch = collator(records)
    resumed.eval()
    with torch.no_grad():
        expected_logits = resumed(**batch).logits
        actual_logits = inference(**batch).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    from llava_common import prepare_inputs
    prompt, _ = training_text(processor, records[0])
    with Image.open(root / records[0]['image']) as image:
        probe = prepare_inputs(processor, prompt, image.convert('RGB'))
    with torch.inference_mode():
        expected_ids = resumed.generate(**probe, max_new_tokens=2, do_sample=False, num_beams=1, use_cache=True)
        actual_ids = inference.generate(**probe, max_new_tokens=2, do_sample=False, num_beams=1, use_cache=True)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    assert len(provenance['sha256']) == 64
    with pytest.raises(ValueError, match='revision differs'):
        load_adapter(bare, final['checkpoint'], DEFAULT_MODEL_ID, '0' * 40)

    from plot_training import read_csv, read_mlflow, plot
    csv_series = read_csv(directory / 'metrics.csv')
    assert read_mlflow(tmp_path / 'mlruns', run_id) == csv_series
    plot(csv_series, tmp_path / 'plots')
    for name in ('loss.png', 'lr.png'):
        with Image.open(tmp_path / 'plots' / name) as figure:
            assert figure.size == (1920, 1080)


def test_mlflow_failure_keeps_csv_and_training_independent(tmp_path, monkeypatch):
    import csv
    from llava_training_logging import TrainingLogger
    logger = TrainingLogger(tmp_path / 'out', {'lr': 2e-5}, (tmp_path / 'mlruns').as_uri())
    def broken(*args, **kwargs):
        raise RuntimeError('Deliberate MLflow failure')
    monkeypatch.setattr(logger.client, 'log_metric', broken)
    with pytest.warns(UserWarning, match='CSV continues'):
        logger.log('train', 1, {'loss': 1.25})
    logger.close()
    with logger.csv_path.open() as handle:
        assert list(csv.DictReader(handle))[0]['loss'] == '1.25'


def test_validation_failure_keeps_update_and_replays_on_resume(setup, examples, tmp_path, monkeypatch):
    import train_llava_stage1 as trainer
    from llava_training_logging import TrainingLogger
    processor, config = setup
    root, records = examples
    collator = Stage1Collator(processor, root)
    contract = {'mode': 'full', 'optimizer_steps': 2, 'global_batch_size': 2,
                'per_device_batch_size': 1}
    model, _, _ = new_model(config)
    optimizer, scheduler = trainer.optimizer_and_scheduler(model, 2e-5, 2e-5, 2)
    output = tmp_path / 'training'
    logger = TrainingLogger(output, contract)
    original_validation = trainer.validation
    def fail_validation(*args, **kwargs):
        raise RuntimeError('Simulated validation crash')
    monkeypatch.setattr(trainer, 'validation', fail_validation)
    with pytest.raises(RuntimeError, match='validation crash'):
        trainer.train_steps(model, optimizer, scheduler, records * 2, records, collator,
                            output, contract, logger, torch.device('cpu'), torch.float32)
    logger.close('FAILED')
    checkpoint = output / 'checkpoint-000002'
    assert (checkpoint / 'checkpoint_manifest.json').exists()
    monkeypatch.setattr(trainer, 'validation', original_validation)
    resumed, _, _ = new_model(config)
    optimizer, scheduler = trainer.optimizer_and_scheduler(resumed, 2e-5, 2e-5, 2)
    logger = TrainingLogger(output, contract, resume_step=2)
    result = trainer.train_steps(resumed, optimizer, scheduler, records * 2, records, collator,
                                 output, contract, logger, torch.device('cpu'), torch.float32,
                                 resume=checkpoint)
    assert result['complete'] and logger.has_event('validation', 2)
    logger.close()


def test_quota_pause_preserves_last_checkpoint_before_creating_new_save(setup, examples, tmp_path, monkeypatch):
    from stage1_storage import InsufficientCheckpointSpace
    from llava_stage1_training import tree_sha
    processor, config = setup
    model, optimizer, scheduler = new_model(config)
    first = tmp_path / 'checkpoint-000001'
    contract = {'test': 'quota pause preserves resume'}
    digest = save_checkpoint(model, optimizer, scheduler, first, contract, 1, 2)
    monkeypatch.setenv('VOLUME_CAP_GB', '0.000000001')  # One-byte cap, below fixture usage.
    with pytest.raises(InsufficientCheckpointSpace, match='cap_minus_usage='):
        save_checkpoint(model, optimizer, scheduler, tmp_path / 'checkpoint-000002', contract, 2, 4)
    assert tree_sha(first) == digest
    assert not (tmp_path / 'checkpoint-000002').exists()
    assert not (tmp_path / 'checkpoint-000002.incomplete').exists()
