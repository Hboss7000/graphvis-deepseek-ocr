from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fullrun_common as common
import fullrun_runtime as runtime
import fullrun_session as session
import fullrun_gate as gate
import fullrun_rescue as rescue
import fullrun_consolidate as consolidate
from prompt_common import parse_answer
from test_fullrun_controls import spec,row,write_output

FETCH=ROOT/'outputs/fullrun_s0b_fetch_jcjYPY6T/fullrun_2026-10-04_B_s0b'

@pytest.mark.parametrize('text,letter',[
 ('Option A is wrong… so the answer is D','D'),
 ('Option A is wrong. Option B is also wrong.','FAILED'),
 ('correct answer: C','C'),('The ANSWER is (B).','B'),('Reasoning ends here: **D**','D'),
 ('**A** is wrong, so the answer is D','D'),
 ('The answer is A. Later, the answer is D.','FAILED'),
 ('A. apple','A'),('A. \n\nHere is a long explanation mentioning D.','FAILED'),
 ('First A, then B, possibly D','FAILED'),('D.','D'),
 ('The answer is a planet','FAILED'),('**A** and **D** are both mentioned','FAILED'),
])
def test_explicit_long_answer_parser(text,letter):assert parse_answer(text,4)[0]==letter

@pytest.mark.skipif(not FETCH.exists(),reason='Fetched 0b required')
@pytest.mark.parametrize('model,condition,idx',[('gemma','image',8),('gemma','kg_text',8),('qwen','image',8),('qwen','kg_text',7),('qwen','kg_text',8)])
def test_all_five_original_failed_responses(model,condition,idx):
    rows=common.read_rows(FETCH/model/('qa_'+condition)/'predictions.jsonl')
    record=next(r for r in rows if r['statement_idx']==idx)
    assert record['parse_tier']=='FAILED' and record['hit_token_ceiling']
    assert parse_answer(record['raw_response'],4)==('FAILED','FAILED')


def test_ceilings_always_descriptive_parse_only_tripwire(tmp_path):
    s=spec(n=5);path=write_output(tmp_path,s)
    rows=common.read_rows(path)
    for r in rows:r.update(hit_token_ceiling=True,raw_response='No explicit answer',parse_tier='FAILED',predicted_option=None)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=common.verify_job(tmp_path,s)
    assert report['ceilings']==5 and report['FAILED']==5
    assert common.tripwire(rows,'qa',3.5) is None
    assert 'parse' in common.tripwire(rows*4,'qa',3.5)


def test_resume_evidence_survives_sanity_failure(tmp_path,monkeypatch):
    s=spec(n=5);path=write_output(tmp_path,s);rows=common.read_rows(path)
    common.frozen_json(tmp_path/'resume_first3.json',{'rows':rows[:3]})
    # This helper is independent of quality checks, but verifies real unchanged rows/coverage.
    monkeypatch.setattr(session,'verify_job',lambda *a,**k:(_ for _ in ()).throw(ValueError('sanity')))
    session.write_resume_proof(tmp_path,s)
    assert (tmp_path/'resume_verified.json').exists()
    gate.proof(tmp_path) if hasattr(gate,'proof') else None
    rows[0]['raw_response']='changed';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='modified'):session.write_resume_proof(tmp_path,s)

@pytest.mark.skipif(not FETCH.exists(),reason='Fetched 0b required')
def test_gemma_recovered_proof_and_gate():
    gate.proof(FETCH/'gemma/qa_image')
    report=gate.model_gate(ROOT,'gemma')
    assert report['passed'] and report['resume_proof']=='PASSED'
    assert report['preflight_nonblocking_D6']

@pytest.mark.skipif(not FETCH.exists(),reason='Fetched 0b required')
def test_llava_gate_does_not_read_gemma_or_other_model_jobs(monkeypatch):
    seen=[];verify=gate.verify_historical
    def check(root,directory,current):
        seen.append(current['model']);assert current['model']=='llava'
        return verify(root,directory,current)
    monkeypatch.setattr(gate,'verify_historical',check)
    monkeypatch.setattr(gate,'auxiliary_specs',lambda *args:(_ for _ in ()).throw(AssertionError('Gemma must not be accessed')))
    result=gate.model_gate(ROOT,'llava')
    assert result['passed'] and set(seen)=={'llava'}
    assert result['probe']['gate']['difference']==pytest.approx(-.015,abs=.001)


def test_prefix_audit_covers_every_ceiling_even_if_already_parsed():
    originals=[dict(statement_idx=i,model_id='m',model_revision='r',gold_option='A',image='i',raw_response='answer is A',hit_token_ceiling=True,generated_token_ids=list(range(64))) for i in range(3)]
    uncapped=dict(originals[0],statement_idx=9,hit_token_ceiling=False);originals.append(uncapped)
    extended=[dict(r,generated_token_ids=list(range(64))+[100],hit_token_ceiling=False) for r in originals[:3]]
    extended[1]['generated_token_ids'][63]=999
    report=rescue.prefix_audit(originals,extended)
    assert report['selected']==report['checked']==3
    assert report['matched']==2 and report['mismatches']==1
    assert [r['statement_idx'] for r in report['items']]==[0,1,2]
    with pytest.raises(ValueError,match='coverage'):rescue.prefix_audit(originals,extended[:2])
    extended[1]['generated_token_ids']=None
    assert rescue.prefix_audit(originals,extended)['items'][1]['matched'] is False


def test_capture_saves_exact_ids_without_retokenization(tmp_path,monkeypatch):
    p=tmp_path/'contract';p.write_text(json.dumps({'stage':'qa'}));monkeypatch.setenv('FULLRUN_CONTRACT',str(p))
    class Tensor:
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return [[10,11,12]]
    runtime.capture_token_ids(Tensor())
    assert runtime._generated_ids==[10,11,12]


def test_selection_preserves_full_config_source_and_all_capped_rows(tmp_path):
    p=tmp_path/'selected.json';p.write_text('[1,3]')
    records=[dict(statement_idx=i,prompt='p') for i in range(5)]
    args=SimpleNamespace(only_indices_json=p)
    assert runtime.selected_records(records,args)==[records[1],records[3]]
    assert len(records)==5
    p.write_text('[1,1]')
    with pytest.raises(ValueError,match='unique'):runtime.selected_records(records,args)


def test_metrics_unparsed_wrong_and_paired_exact_test():
    a=consolidate.parsed_rows([dict(statement_idx=i,raw_response=text,gold_option='D',hit_token_ceiling=bool(i)) for i,text in enumerate(['D','Option A is wrong… so the answer is D','A or D'])])
    metrics=consolidate.metrics(list(a.values()))
    assert metrics['accuracy']==pytest.approx(2/3) and metrics['accuracy_among_parsed']==1
    assert metrics['parse_failure_rate']==pytest.approx(1/3)
    b=copy.deepcopy(a)
    for r in b.values():r['is_correct']=False
    comparison=consolidate.paired(a,b)
    assert comparison['image_only_correct']==2 and comparison['text_only_correct']==0
    assert comparison['mcnemar_exact_two_sided_p']==.5


def test_strata_ties_and_reachability():
    metadata={i:dict(answerKey='A',visible_nodes=[dict(cid=1,in_question=True,in_choices=[]),dict(cid=2,in_question=False,in_choices=['A'])],edges=[] if i else [dict(source_cid=1,target_cid=2)]) for i in range(3)}
    strata,definitions=consolidate.graph_strata(metadata)
    assert strata[0]['reachable'] and not strata[1]['reachable']
    assert all(r['node_tercile']=='low' for r in strata.values())
    assert definitions['cutpoints']==[2,2]


def test_exact_token_validation_is_required_for_full_qa():
    s=spec();s['mode']='full';r=row(s)
    with pytest.raises(ValueError,match='exact generated'):common.validate_rows([r],s)
    r['generated_token_ids']=[1,2];common.validate_rows([r],s)
    r['generated_token_ids']=[1]*65
    with pytest.raises(ValueError,match='exact generated'):common.validate_rows([r],s)


def test_rescue_command_keeps_full_input_prompt_and_only_selects_capped(tmp_path,monkeypatch):
    s=spec();s.update(input_jsonl='all500.jsonl',graph_metadata='meta',input_files={},code_files={})
    s['mode']='full';s['data_provenance']={}
    strict=tmp_path/'strict';strict.mkdir()
    rows=[dict(row(s,i),gold_option='A',image='i',hit_token_ceiling=(i in (0,2)),generated_token_ids=list(range(64))) for i in range(5)]
    monkeypatch.setattr(rescue,'validate_frozen_files',lambda *a:None)
    monkeypatch.setattr(rescue,'verify_job',lambda *a:None)
    monkeypatch.setattr(rescue,'prediction_rows',lambda *a:rows)
    (strict/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    common.frozen_json(strict/'contract.json',s)
    common.frozen_json(strict/'run_config.json',{'fullrun':s,'seed':13,'gpu_type':'NVIDIA RTX PRO 6000 Blackwell Server Edition','generation':{'do_sample':False,'num_beams':1,'max_new_tokens':64}})
    rescue_spec,originals=rescue.rescue_spec(tmp_path,'llava',s,strict,tmp_path/'rescue')
    assert rescue_spec['effective_max_new_tokens']==512
    assert rescue_spec['input_jsonl']=='all500.jsonl'
    assert rescue_spec['prompt_bodies_sha256']==s['prompt_bodies_sha256']
    assert rescue_spec['expected_keys']==[[0,'obqa_answer'],[2,'obqa_answer']]
    assert json.loads((tmp_path/'rescue/selected_indices.json').read_text())==[0,2]


def test_strict_extended_merge_does_not_select_only_failed(monkeypatch,tmp_path):
    # All 500 paired IDs, with one already-correct capped row replaced by rescue.
    base=tmp_path/'outputs'/(common.DATA_NAME+'_results')/'llava'
    for condition in common.CONDITIONS:
        p=base/('qa_'+condition)/'predictions.jsonl';p.parent.mkdir(parents=True)
        originals=[dict(statement_idx=i,raw_response='A',gold_option='A',hit_token_ceiling=(i==0)) for i in range(500)]
        p.write_text(''.join(json.dumps(r)+'\n' for r in originals))
        p=base/'rescue_512'/condition/'predictions.jsonl';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(originals[0],raw_response='correct answer: D',hit_token_ceiling=False))+'\n')
    metadata=[dict(statement_idx=i,answerKey='A',visible_nodes=[dict(cid=1,in_question=True,in_choices=['A'])],edges=[]) for i in range(500)]
    p=tmp_path/'outputs'/common.DATA_NAME/'test/graph_metadata_0_500.jsonl';p.parent.mkdir(parents=True)
    p.write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    monkeypatch.setattr(consolidate,'verify_model',lambda *a:{'passed':True})
    monkeypatch.setattr(consolidate,'rescue_state',lambda root,model:{'complete':model=='llava','audit':{},'reason':None})
    report,versions=consolidate.consolidate_model(tmp_path,'llava')
    assert report['primary']=='strict_64'
    assert report['extended_available'] is True
    assert report['versions']['strict_64']['conditions']['image']['accuracy']==1
    assert report['versions']['extended_512']['conditions']['image']['accuracy']==499/500
    assert versions['extended_512']['image'][0]['predicted_option']=='D'
    monkeypatch.setattr(consolidate,'rescue_state',lambda *a:{'complete':False,'audit':None,'reason':'not run'})
    absent,_=consolidate.consolidate_model(tmp_path,'llava')
    assert absent['primary']=='strict_64'
    assert absent['versions']['extended_512']['conditions']['image']['accuracy'] is None
    assert absent['versions']['strict_64']['conditions']['image']['accuracy']==1


def test_rescue_orchestration_all_conditions_resumes_and_audits_every_item(tmp_path,monkeypatch):
    records=tmp_path/'all.jsonl';records.write_text(''.join(json.dumps(dict(statement_idx=i))+'\n' for i in range(5)))
    meta=tmp_path/'meta.jsonl';meta.write_text('{}\n')
    base=tmp_path/'outputs'/(common.DATA_NAME+'_results')/'llava'
    specs=[]
    for condition in common.CONDITIONS:
        s=spec();s.update(mode='full',label='qa_'+condition,condition=condition,image_root='.',input_jsonl='all.jsonl',graph_metadata='meta.jsonl',
                          input_files={'all.jsonl':common.sha(records),'meta.jsonl':common.sha(meta)},code_files={},data_provenance={})
        directory=base/s['label'];directory.mkdir(parents=True)
        values=[dict(row(s,i),gold_option='A',image='i',generated_token_ids=list(range(64)),generated_token_count=64,hit_token_ceiling=(i in (1,3))) for i in range(5)]
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
        cfg={**{k:s[k] for k in ('model_id','model_revision','effective_max_new_tokens','prompt_bodies_sha256')},
             'generation':dict(do_sample=False,num_beams=1,max_new_tokens=64),'parser_protocol':'explicit_v1',
             'seed':13,'gpu_type':'NVIDIA RTX PRO 6000 Blackwell Server Edition',
             'torch_version':s['versions']['torch'],'transformers_version':s['versions']['transformers'],'fullrun':s,
             'input_jsonl':{'sha256':common.sha(records)},'graph_metadata':{'sha256':common.sha(meta)}}
        common.frozen_json(directory/'run_config.json',cfg)
        old={**s,'code_files':{'deleted_old_code.py':'old-hash'}}
        cfg['fullrun']=old
        # Source commit has a different code manifest; its code need not exist now.
        (directory/'run_config.json').write_text(json.dumps(cfg))
        common.frozen_json(directory/'contract.json',old);specs.append(s)
    monkeypatch.setattr(rescue,'specs',lambda *args:[{}]+specs)
    monkeypatch.setattr(rescue,'hardware_preflight',lambda *args:None)
    calls=[]
    def fake_watch(cmd,directory,spec,log,deadline):
        calls.append(cmd)
        assert cmd[cmd.index('--max-new-tokens')+1]=='512'
        assert cmd[cmd.index('--expected-count')+1]=='5'
        assert '--only-indices-json' in cmd
        source=common.read_rows(base/('qa_'+spec['condition'])/'predictions.jsonl')
        rows=[dict(r,raw_response='correct answer: A',parse_tier='explicit_answer',generated_token_ids=list(range(64))+[123],generated_token_count=65,hit_token_ceiling=False) for r in source if r['hit_token_ceiling']]
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cfg=json.loads((base/('qa_'+spec['condition'])/'run_config.json').read_text());cfg['fullrun']=spec
        cfg['generation']['max_new_tokens']=512;cfg['effective_max_new_tokens']=512
        common.frozen_json(directory/'run_config.json',cfg)
        return {'state':'COMPLETED','reason':None}
    monkeypatch.setattr(rescue,'watch_job',fake_watch)
    monkeypatch.setattr(consolidate,'print_spot_check',lambda *args:None)
    args=SimpleNamespace(root=tmp_path,workspace=tmp_path)
    assert rescue.run_rescue(args,'llava',time.monotonic()+10)
    report=json.loads((base/'rescue_512/prefix_report.json').read_text())
    assert len(calls)==4
    saved=json.loads((base/'rescue_512/image/contract.json').read_text())
    assert saved['code_files']=={}
    assert saved['rescue']['strict_code_files']=={'deleted_old_code.py':'old-hash'}
    assert all(r['selected']==r['checked']==r['matched']==2 for r in report.values())
    for s in specs:s['code_files']={'another_new_commit.py':'new-hash'}
    assert rescue.run_rescue(args,'llava',time.monotonic()+10)
    assert len(calls)==4,'Completed rescues must not generate again, even on a newer commit'
    attempt=json.loads((base/'rescue_512/code_attempts/attempt_0002.json').read_text())
    assert attempt['code_files']=={'another_new_commit.py':'new-hash'}


def test_llava_full_launcher_checks_only_llava_and_never_rescues(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(sys,'argv',['fullrun_session.py','llava','--root',str(tmp_path),'--workspace',str(tmp_path)])
    monkeypatch.setenv('TMUX','local-test')
    def model_gate(root,model,*args):
        assert model=='llava';calls.append('gate:'+model);return {'passed':True,'errors':[]}
    monkeypatch.setattr(gate,'model_gate',model_gate)
    monkeypatch.setattr(session,'run_model',lambda args,model,deadline:calls.append('full:'+model) or True)
    monkeypatch.setattr(rescue,'run_rescue',lambda args,model,deadline:calls.append('rescue:'+model) or True)
    with pytest.raises(SystemExit) as exc:session.main()
    assert exc.value.code==0
    assert calls==['gate:llava','full:llava']


def test_gemma_startup_smoke_then_own_gate_then_full_without_rescue(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(sys,'argv',['fullrun_session.py','gemma','--root',str(tmp_path),'--workspace',str(tmp_path)])
    monkeypatch.setenv('TMUX','local-test')
    monkeypatch.setattr(gate,'evidence_roots',lambda *args:(tmp_path/'smoke',tmp_path/'recovery'))
    monkeypatch.setattr(session,'specs',lambda root,model,smoke:[{'stage':'stage1'}]+[{'stage':'qa','condition':c} for c in common.CONDITIONS])
    def run(args,model,deadline):
        assert model=='gemma'
        if args.smoke:
            assert len(args.recovery_jobs['gemma'])==4
            assert args.mode_override=='gemma_start_smoke'
            calls.append('startup-smoke')
        else:calls.append('full')
        return True
    def model_gate(root,model,*args):
        assert model=='gemma' and args[-1]==tmp_path/'outputs'/(common.DATA_NAME+'_gemma_start_smoke')
        calls.append('own-gate');return {'passed':True,'errors':[]}
    monkeypatch.setattr(session,'run_model',run);monkeypatch.setattr(gate,'model_gate',model_gate)
    monkeypatch.setattr(rescue,'run_rescue',lambda *args:calls.append('rescue') or True)
    with pytest.raises(SystemExit) as exc:session.main()
    assert exc.value.code==0
    assert calls==['startup-smoke','own-gate','full']


def test_saved_contract_allows_new_code_but_rejects_changed_inputs(tmp_path):
    source=tmp_path/'input';source.write_text('unchanged')
    old=spec();old.update(input_files={'input':common.sha(source)},code_files={'old.py':'old-hash'})
    directory=tmp_path/'strict';directory.mkdir()
    common.frozen_json(directory/'contract.json',old)
    common.frozen_json(directory/'run_config.json',{'fullrun':old})
    current={**old,'code_files':{'new.py':'new-hash'}}
    assert common.saved_spec(tmp_path,directory,current)==old
    with pytest.raises(ValueError,match='experimental settings'):
        common.saved_spec(tmp_path,directory,{**current,'model_revision':'another-revision'})
    source.write_text('changed')
    with pytest.raises(ValueError,match='Missing/changed input'):
        common.saved_spec(tmp_path,directory,current)


def test_headline_requires_all_four_rescued_even_for_one_model(monkeypatch,tmp_path):
    finished={'llava','qwen','deepseek'}
    monkeypatch.setattr(consolidate,'rescue_state',lambda root,m:{'complete':m in finished})
    assert consolidate.headline_rule(tmp_path)[0]=='strict_64'
    finished.add('gemma')
    assert consolidate.headline_rule(tmp_path)[0]=='extended_512'


def test_absent_optional_rescue_is_explicitly_unavailable(tmp_path):
    state=consolidate.rescue_state(tmp_path,'llava')
    assert state['complete'] is False and state['audit'] is None
    assert consolidate.headline_rule(tmp_path)[0]=='strict_64'
    assert not list(tmp_path.iterdir()),'Checking absent rescue must not create output directories'


def test_planner_uses_actual_counts_and_measured_speed_not_512(tmp_path):
    from fullrun_rescue_plan import plan
    for condition in common.CONDITIONS:
        directory=tmp_path/('qa_'+condition);directory.mkdir()
        rows=[dict(statement_idx=i,raw_response='x'*250,generated_token_ids=list(range(64 if i<3 else 16)),
                   item_elapsed_seconds=2,generation_elapsed_seconds=1,hit_token_ceiling=i<3) for i in range(4)]
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=plan('llava',tmp_path)
    assert len(report['capped_examples'])==10
    assert all(len(r['response_first_200'])==200 for r in report['capped_examples'])
    c=report['conditions']['image']
    assert c['capped_answers']==3
    assert c['measured_tokens_per_second']==52
    assert c['estimated_rescue_hours']==pytest.approx(192/52/3600)
    assert c['estimated_rescue_cost_usd']==pytest.approx(c['estimated_rescue_hours']*2.09)
    assert c['estimate_kind']=='observed token workload lower bound'
    estimate=plan('llava',tmp_path,128)
    assert estimate['conditions']['image']['estimated_rescue_hours']==pytest.approx(384/52/3600)


def test_full_qa_runtime_records_seed_gpu_and_checks_rescue_execution(monkeypatch):
    s=spec();s.update(mode='full',input_jsonl='input',graph_metadata='meta',input_files={'input':'a','meta':'b'},data_provenance={})
    cfg={k:s[k] for k in ('model_id','model_revision','effective_max_new_tokens','prompt_bodies_sha256')}
    cfg.update(generation=dict(do_sample=False,num_beams=1,max_new_tokens=64),
               torch_version=s['versions']['torch'],transformers_version=s['versions']['transformers'],
               input_jsonl={'sha256':'a'},graph_metadata={'sha256':'b'})
    seeds=[]
    torch=SimpleNamespace(manual_seed=seeds.append,cuda=SimpleNamespace(get_device_name=lambda i:'NVIDIA RTX PRO 6000 Blackwell Server Edition'))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setattr(runtime,'contract',lambda:s)
    enriched=runtime.enrich_config(cfg)
    assert enriched['seed']==13 and seeds==[13]
    assert enriched['gpu_type']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
    s['rescue']={'strict_execution':{'seed':13,'gpu_type':enriched['gpu_type'],'generation':cfg['generation']}}
    assert runtime.enrich_config(cfg)['seed']==13
    with pytest.raises(ValueError,match='Rescue seed/GPU/decoding'):
        runtime.enrich_config({**cfg,'seed':42})
    torch.cuda.get_device_name=lambda i:'NVIDIA H100'
    with pytest.raises(ValueError,match='RTX PRO 6000'):
        runtime.enrich_config(cfg)


def test_strict_only_group_runs_s2_in_order_and_stops_on_failure(tmp_path):
    import subprocess
    scripts=tmp_path/'scripts';scripts.mkdir()
    path=scripts/'fullrun_manual.sh'
    path.write_text('printf "%s %s\\n" "$1" "$MAX_HOURS" >> calls\nif [[ "$1" == qwen && -f fail ]]; then exit 1; fi\n')
    launcher=common.ROOT/'scripts/fullrun_strict_session.sh'
    success=subprocess.run(['bash',str(launcher),'S2'],cwd=tmp_path,text=True,capture_output=True)
    assert success.returncode==0
    lines=(tmp_path/'calls').read_text().splitlines()
    assert [line.split()[0] for line in lines]==['qwen','deepseek']
    assert all(0<float(line.split()[1])<=3.36 for line in lines)
    (tmp_path/'calls').unlink();(tmp_path/'fail').touch()
    failed=subprocess.run(['bash',str(launcher),'S2'],cwd=tmp_path,text=True,capture_output=True)
    assert failed.returncode!=0
    assert (tmp_path/'calls').read_text().split()[0]=='qwen'
    assert 'deepseek' not in (tmp_path/'calls').read_text()
    assert 'STOP the Pod' in failed.stdout
