"""Real Qwen vision/DeepStack CPU checks, pinned tokenizer, random tiny weights."""
import copy
from pathlib import Path
import sys
import pytest
import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers import Qwen2VLImageProcessor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qwen_stage1_training import (pinned_processor, Stage1Collator, training_text, attach_lora,
                                 DEFAULT_MODEL_ID, DEFAULT_REVISION, prepare_inputs)
from llava_stage1_training import projector, save_checkpoint, load_checkpoint
from train_llava_stage1 import optimizer_and_scheduler, train_steps

@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    original, _ = pinned_processor(root / 'outputs/qwen_stage1_assets' / DEFAULT_REVISION)
    original.tokenizer.pad_token = original.tokenizer.eos_token
    image_processor = Qwen2VLImageProcessor(patch_size=4, temporal_patch_size=2, merge_size=2,
        size={'shortest_edge': 256, 'longest_edge': 1024})
    processor = Qwen3VLProcessor(image_processor=image_processor, tokenizer=original.tokenizer,
        video_processor=original.video_processor, chat_template=original.chat_template)
    vision = Qwen3VLVisionConfig(depth=3, hidden_size=16, intermediate_size=32, num_heads=4,
        patch_size=4, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
        num_position_embeddings=16, deepstack_visual_indexes=[0,1,2])
    text = Qwen3VLTextConfig(vocab_size=len(original.tokenizer), hidden_size=32,
        intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=4096,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'mrope_section':[1,1,2], 'mrope_interleaved':True},
        bos_token_id=151643, eos_token_id=151645, pad_token_id=151645)
    config = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
        vision_end_token_id=151653)
    return processor, config

@pytest.fixture
def examples(tmp_path):
    Image.new('RGB',(16,16),'#ADD8E6').save(tmp_path/'a.png')
    Image.new('RGB',(32,16),'#ADD8E6').save(tmp_path/'b.png')
    return tmp_path, [
        {'image':'a.png','prompt':'How many nodes are there in the graph?', 'answer':'There are 3 nodes in the graph.', 'task_type':'node_number'},
        {'image':'b.png','prompt':'List all the triples in the graph.', 'answer':'The triples are: (a, is a, b), (b, part of, c).','task_type':'triple_listing'}]

def new_model(config):
    torch.manual_seed(13)
    model, names = attach_lora(Qwen3VLForConditionalGeneration(copy.deepcopy(config)))
    assert len(names) == 28 and all('.language_model.' in n for n in names)
    assert len(projector(model)) == 4
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer, scheduler = optimizer_and_scheduler(model, 2e-5, 2e-5, 2)
    return model, optimizer, scheduler

def collator(processor, root):
    return Stage1Collator(processor,root,min_pixels=256,max_pixels=1024)

def test_native_mask_padding_image_tokens_and_length_guard(setup, examples):
    processor,_=setup
    root,rows=examples
    batch=collator(processor,root)(rows)
    for i,row in enumerate(rows):
        prefix,full=training_text(processor,row)
        assert '<|im_start|>system' not in prefix
        assert full == prefix + row['answer'] + '<|im_end|>'
        labels=batch['labels'][i]
        supervised=labels[labels != -100]
        assert processor.tokenizer.decode(supervised) == row['answer']+'<|im_end|>'
        assert supervised[-1] == processor.tokenizer.eos_token_id
        assert (supervised == processor.tokenizer.eos_token_id).sum() == 1
        assert (labels[batch['input_ids'][i] == processor.image_token_id] == -100).all()
        assert (labels[batch['attention_mask'][i] == 0] == -100).all()
    assert 'mm_token_type_ids' in batch
    with pytest.raises(ValueError,match='exceeds'):
        Stage1Collator(processor,root,model_max_length=8,min_pixels=256,max_pixels=1024)(rows)
    with pytest.raises(ValueError,match='injected'):
        training_text(processor,{**rows[0],'answer':'<|im_end|>'})

def test_all_four_bridges_backward_and_exact_checkpoint_resume(setup,examples,tmp_path):
    processor,config=setup
    root,rows=examples
    batch=collator(processor,root)(rows)
    model,opt,sched=new_model(config)
    def step(m,o,s):
        m.train();o.zero_grad(set_to_none=True)
        loss=m(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        assert any(p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'lora_B' in n)
        for bridge in projector(m).values():
            assert any(p.grad.abs().sum()>0 for p in bridge.parameters())
        bridge_ids={id(p) for p in projector(m).parameters()}
        assert all(p.grad is None for n,p in m.named_parameters() if '.visual.' in n and id(p) not in bridge_ids)
        o.step();s.step();return loss.detach().clone()
    step(model,opt,sched)
    contract={'model_id':DEFAULT_MODEL_ID,'model_revision':DEFAULT_REVISION,'test':'tiny random CPU'}
    save_checkpoint(model,opt,sched,tmp_path/'checkpoint',contract,1,2)
    expected_loss=step(model,opt,sched)
    expected={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    resumed,opt2,sched2=new_model(config)
    assert load_checkpoint(resumed,opt2,sched2,tmp_path/'checkpoint',contract)==(1,2)
    torch.testing.assert_close(step(resumed,opt2,sched2),expected_loss,rtol=0,atol=0)
    for n,p in resumed.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p,expected[n],rtol=0,atol=0)

def test_shared_loop_resume_logger_and_adapter_reload(setup,examples,tmp_path,monkeypatch):
    import socket
    import csv
    from types import SimpleNamespace
    from llava_training_logging import TrainingLogger
    from llava_adapter import load_adapter
    def deny(*args,**kwargs):raise AssertionError('No network in CPU training/inference')
    monkeypatch.setattr(socket.socket,'connect',deny)
    processor,config=setup
    root,rows=examples
    collate=collator(processor,root)
    contract={'model_id':DEFAULT_MODEL_ID,'model_revision':DEFAULT_REVISION,'mode':'full',
        'optimizer_steps':2,'global_batch_size':2,'per_device_batch_size':1,'checkpoint_keep':2}
    store=(tmp_path/'mlruns').as_uri()
    reference,opt,sched=new_model(config)
    log=TrainingLogger(tmp_path/'reference',contract,store)
    train_steps(reference,opt,sched,rows*2,rows,collate,tmp_path/'reference',contract,log,
                torch.device('cpu'),torch.float32)
    log.close()
    expected={n:p.detach().clone() for n,p in reference.named_parameters() if p.requires_grad}
    model,opt,sched=new_model(config)
    directory=tmp_path/'resumed'
    log=TrainingLogger(directory,contract,store)
    first=train_steps(model,opt,sched,rows*2,rows,collate,directory,contract,log,
                      torch.device('cpu'),torch.float32,stop_after_step=1)
    run_id=log.run_id;log.close('KILLED')
    resumed,opt,sched=new_model(config)
    log=TrainingLogger(directory,contract,store,resume_step=1)
    final=train_steps(resumed,opt,sched,rows*2,rows,collate,directory,contract,log,
                      torch.device('cpu'),torch.float32,resume=first['checkpoint'])
    assert final['complete'] and log.readable() and log.run_id==run_id
    log.close()
    for n,p in resumed.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
    with (directory/'metrics.csv').open() as f:
        logged=list(csv.DictReader(f))
    assert [int(r['step']) for r in logged if r['event']=='train']==[1,2]
    torch.manual_seed(13)
    loaded,provenance=load_adapter(Qwen3VLForConditionalGeneration(copy.deepcopy(config)),
                                 final['checkpoint'],DEFAULT_MODEL_ID,DEFAULT_REVISION)
    resumed.eval()
    batch=collate(rows)
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch).logits,resumed(**batch).logits,rtol=0,atol=0)
    prefix,_=training_text(processor,rows[0])
    probe=prepare_inputs(processor,prefix,Image.open(root/rows[0]['image']).convert('RGB'),
                         SimpleNamespace(min_pixels=256,max_pixels=1024))
    with torch.no_grad():
        torch.testing.assert_close(loaded.generate(**probe,max_new_tokens=2,do_sample=False,use_cache=True),
            resumed.generate(**probe,max_new_tokens=2,do_sample=False,use_cache=True),rtol=0,atol=0)
    assert len(provenance['sha256'])==64
