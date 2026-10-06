"""Qwen paid-run command contracts and unchanged native evaluation defaults."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
for path in [ROOT/'scripts',ROOT/'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts',ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts']:
    sys.path.insert(0,str(path))
from evaluate_llava_stage1_adapter import jobs,summarize
from compare_llava_stage1_training import compare_stage,compare_qa,verify_config
from llava_stage1_training import file_sha
from qwen_stage1_training import DEFAULT_MODEL_ID,DEFAULT_REVISION,TEMPLATE
import run_zero_shot_qwen as qa
import run_stage1_qwen as stage

@pytest.mark.parametrize('phase,count,qa_count',[('dry',9,8),('full',900,500)])
def test_qwen_stage_first_qa_second_pixel_pins_native_generation(phase,count,qa_count):
    commands=jobs(phase,Path('/adapter'),Path('/data'),Path('/dry'),Path('/out'),'qwen')
    assert [n for n,_ in commands]==['stage1','qa_image','qa_text_noref','qa_text','qa_kg_text']
    assert [n for n,_ in jobs(phase,Path('/a'),Path('/d'),Path('/s'),Path('/o'),'qwen','stage1')]==['stage1']
    assert len(jobs(phase,Path('/a'),Path('/d'),Path('/s'),Path('/o'),'qwen','qa'))==4
    for name,cmd in commands:
        val=lambda flag:cmd[cmd.index(flag)+1]
        assert val('--revision')==DEFAULT_REVISION
        assert val('--min-pixels')=='262144' and val('--max-pixels')=='1310720'
        assert '--prompt-template' not in cmd and '--assistant-prefix-mode' not in cmd
        assert val('--expected-count')==str(count if name=='stage1' else qa_count)
        assert val('--max-new-tokens')==('1024' if name=='stage1' else '64')
        assert 'qwen.py' in cmd[1]
        if name=='stage1':assert val('--extractor')=='span_extended'


def test_qwen_baseline_comparison_uses_real_frozen_nine_tasks_and_four_conditions(tmp_path):
    data=ROOT/'outputs/fullrun_2026-10-04_B/test'
    baseline=ROOT/'outputs/fullrun_2026-10-04_B_results/qwen'
    result=compare_stage({'main':baseline/'stage1','tuned':tmp_path/'missing'},
                         data/'stage1_subset100.jsonl',data/'graph_metadata_0_500.jsonl','qwen')
    assert result['main']['status']=='complete' and len(result['main']['metrics']['tasks'])==9
    assert result['tuned']['status']=='missing'
    result=compare_qa(baseline,data/'stage2_obqa_0_500.jsonl',False,'qwen')
    assert result['status']=='complete' and len(result['metrics'])==4
    assert all(v['n']==500 for v in result['metrics'].values())
    # Deliberately cutting QA must not treat a stage-only output as incomplete QA.
    (tmp_path/'stage1').mkdir()
    assert compare_qa(tmp_path,data/'stage2_obqa_0_500.jsonl',True,'qwen')['status']=='missing'


def test_qwen_comparison_rejects_system_budget_revision_and_artifact_drift(tmp_path):
    source=tmp_path/'source';source.write_text('{}\n')
    config={'model_id':DEFAULT_MODEL_ID,'model_revision':DEFAULT_REVISION,'seed':13,
        'input_jsonl':{'sha256':file_sha(source)},'prompt_template':TEMPLATE,
        'adapter':{'sha256':'a'*64},'effective_max_new_tokens':64,
        'generation':{'do_sample':False,'num_beams':1},'min_pixels':262144,
        'max_pixels':1310720,'default_system_prompt_injected':False}
    verify_config(config,source,TEMPLATE,'none',True,64,'qwen')
    for key,value in [('model_revision','b'*40),('min_pixels',256),('default_system_prompt_injected',True),('adapter',None),('prompt_template','llava_v1')]:
        with pytest.raises(ValueError):verify_config({**config,key:value},source,TEMPLATE,'none',True,64,'qwen')


def test_qwen_adapter_flag_is_opt_in_and_pinned_load_stays_default(monkeypatch,tmp_path):
    common=['--input-jsonl','input','--graph-metadata','meta','--image-root','images']
    monkeypatch.setattr(sys,'argv',['stage',*common,'--output-dir','out'])
    assert stage.parse_args().adapter is None
    monkeypatch.setattr(sys,'argv',['qa',*common,'--output-jsonl','out/p.jsonl','--condition','image'])
    assert qa.parse_args().adapter is None
    monkeypatch.setattr(sys,'argv',['qa',*common,'--output-jsonl','out/p.jsonl','--condition','image','--adapter',str(tmp_path)])
    assert qa.parse_args().adapter==tmp_path
    seen=[]
    class Processor:
        @classmethod
        def from_pretrained(cls,*a,**k):
            seen.append(k)
            return SimpleNamespace(image_processor=SimpleNamespace(min_pixels=262144,max_pixels=1310720))
    args=SimpleNamespace(model_id=DEFAULT_MODEL_ID,revision=DEFAULT_REVISION,min_pixels=262144,max_pixels=1310720)
    qa.load_processor(Processor,args)
    assert seen[-1]=={'min_pixels':262144,'max_pixels':1310720,'revision':DEFAULT_REVISION}
    args.local_files_only=True;qa.load_processor(Processor,args)
    assert seen[-1]['local_files_only'] is True


def test_stage_only_measured_eval_and_adapter_consistency(tmp_path):
    folder=tmp_path/'stage1';folder.mkdir()
    (folder/'run_config.json').write_text(json.dumps({'adapter':{'sha256':'a'*64}}))
    (folder/'runtime_metrics.json').write_text(json.dumps({'loading_elapsed_seconds':3}))
    rows=[{'statement_idx':1,'task_type':str(i),'item_elapsed_seconds':2,'peak_memory_allocated_bytes':99,'hit_token_ceiling':False} for i in range(9)]
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=summarize(tmp_path,'dry','qwen','stage1')
    assert result['scope']=='stage1' and result['full_extrapolated_seconds']==1803
    assert list(result['jobs'])==['stage1']


def test_stage_and_qa_measured_costs_are_separate_and_same_adapter():
    from llava_stage1_execution_plan import plan
    probe={'passed':True,'dtype':'bf16','gpu_type':'RTX PRO 6000','seconds_per_optimizer_step':10,
           'batch_probe_attempts':[{'wall_seconds':20}]}
    stage_report={'passed':True,'phase':'dry','scope':'stage1','backbone':'qwen',
                  'adapter_sha256':'a'*64,'inputs':{},'full_extrapolated_seconds':100}
    qa_report={**stage_report,'scope':'qa','full_extrapolated_seconds':200}
    result=plan(probe,evaluation_dry=stage_report,qa_evaluation_dry=qa_report)
    assert result['jobs']['eval_stage1']['guard_hours']==100*1.5/3600
    assert result['jobs']['eval_qa']['guard_hours']==200*1.5/3600
    with pytest.raises(ValueError,match='different adapters'):
        plan(probe,evaluation_dry=stage_report,qa_evaluation_dry={**qa_report,'adapter_sha256':'b'*64})
