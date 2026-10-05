"""CPU-only regression coverage for the one-job Gemma strict resume exception."""
import copy
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fullrun_common as common
import fullrun_override as override
import fullrun_session as session
import verify_fullrun as verifier
from test_fullrun_controls import config, row, spec, STUB


def fixture_job(directory, n=21):
    s = spec('gemma', n=n)
    s.update(label='qa_kg_text', condition='kg_text', mode='full', code_files={'old.py': 'old'})
    cfg = config(s)
    cfg['parser_protocol'] = 'explicit_v1'
    common.frozen_json(directory / 'contract.json', s)
    common.frozen_json(directory / 'run_config.json', cfg)
    rows = []
    for i in range(20):
        r = row(s, i)
        r.update(generated_token_ids=[1]*64, gold_option='A')
        if i < 15:
            r.update(raw_response='Let me explain the reasoning first', generated_token_count=64, hit_token_ceiling=True)
            letter, r['parse_tier'] = common.parse_answer(r['raw_response'], 4)
            r['predicted_option'] = None if letter == 'FAILED' else letter
        rows.append(r)
    (directory / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return s, {**s, 'code_files': {'new.py': 'new'}}


def test_explicit_override_preserves_20_rows_and_records_code(tmp_path):
    old, current = fixture_job(tmp_path)
    before = (tmp_path / 'predictions.jsonl').read_bytes()
    with pytest.raises(ValueError, match='TRIPWIRE'):
        session.check_resume(tmp_path, old)
    saved, attempt = override.prepare_override(tmp_path, tmp_path, current)
    assert saved == old
    assert json.loads(attempt.read_text())['code_files'] == current['code_files']
    assert json.loads((tmp_path / 'run_config.json').read_text())['tripwire_overridden'] is True
    session.check_resume(tmp_path, saved)
    # Exercise the real watchdog and append/resume subprocess, keeping exactly the first 20.
    stub = STUB.replace("generated_token_count=2,", "generated_token_ids=[1,2],gold_option='A',generated_token_count=2,")
    status = session.watch_job([sys.executable, '-c', stub, '21'], tmp_path, saved,
                               tmp_path / 'job.log', time.monotonic()+10, poll_seconds=.01, code_attempt=attempt)
    assert status['state'] == 'COMPLETED' and status['tripwire_overridden'] is True
    assert (tmp_path / 'predictions.jsonl').read_bytes().startswith(before)
    result = common.verify_job(tmp_path, saved)
    assert result['rows'] == 21 and result['FAILED'] == 15
    assert result['accuracy'] == 6/21 and result['accuracy_among_parsed'] == 1
    session.record_status(tmp_path / 'status.json', status)
    assert json.loads((tmp_path / 'status.json').read_text())['tripwire_overridden'] is True
    assert override.prepare_override(tmp_path, tmp_path, current)[0] == saved


@pytest.mark.parametrize('key,value', [('effective_max_new_tokens',512), ('prompt_bodies_sha256','changed'), ('data_provenance',{'changed':True})])
def test_override_rejects_experimental_drift(tmp_path,key,value):
    old,current=fixture_job(tmp_path)
    current[key]=value
    with pytest.raises(ValueError):override.prepare_override(tmp_path,tmp_path,current)
    assert not (tmp_path/'tripwire_override.json').exists()


def test_override_rejects_changed_inputs_and_original_predictions(tmp_path):
    old,current=fixture_job(tmp_path)
    data=tmp_path/'input';data.write_text('original')
    old['input_files']={'input':common.sha(data)}
    current['input_files']=old['input_files']
    (tmp_path/'contract.json').write_text(json.dumps(old))
    cfg=config(old);cfg['parser_protocol']='explicit_v1'
    (tmp_path/'run_config.json').write_text(json.dumps(cfg))
    data.write_text('changed')
    with pytest.raises(ValueError,match='changed input'):override.prepare_override(tmp_path,tmp_path,current)
    data.write_text('original')
    saved,_=override.prepare_override(tmp_path,tmp_path,current)
    predictions=tmp_path/'predictions.jsonl'
    predictions.write_text(predictions.read_text().replace('Let me explain','I will explain',1))
    with pytest.raises(ValueError,match='original predictions'):common.verify_job(tmp_path,saved,complete=False)


def test_other_tripwires_and_validation_still_apply(tmp_path):
    old,current=fixture_job(tmp_path)
    saved,attempt=override.prepare_override(tmp_path,tmp_path,current)
    rows=common.prediction_rows(tmp_path,saved)
    for r in rows:r['item_elapsed_seconds']=100
    assert '3x' in common.tripwire(rows,'qa',4.2,parse_override=True)
    for r in rows:r.update(raw_response='',generated_token_count=0)
    assert '80%' in common.tripwire(rows,'stage1',4.2,parse_override=True)
    for mode,expected in [('stall','TRIPWIRE')]:
        status=session.watch_job([sys.executable,'-c',STUB,mode],tmp_path,saved,tmp_path/'log',time.monotonic()+10,poll_seconds=.01,stall_seconds=.05,code_attempt=attempt)
        assert status['state']==expected
    status=session.watch_job([sys.executable,'-c',STUB,'stall'],tmp_path,saved,tmp_path/'log',time.monotonic()+.05,poll_seconds=.01,code_attempt=attempt)
    assert status['state']=='PAUSED'
    with pytest.raises(ValueError,match='Missing predictions'):common.verify_job(tmp_path,saved)
    with (tmp_path/'predictions.jsonl').open('a') as f:f.write(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='Duplicate'):common.verify_job(tmp_path,saved)


def test_job_selection_never_launches_smoke_preflight_or_other_jobs(tmp_path,monkeypatch):
    base=tmp_path/'outputs'/(common.DATA_NAME+'_results')/'gemma'
    old,current=fixture_job(base/'qa_kg_text')
    common.frozen_json(base/'session.json',{'model':'gemma','mode':'results','data_name':common.DATA_NAME})
    monkeypatch.setattr(session,'specs',lambda *a:[spec('gemma'),current])
    monkeypatch.setattr(session,'auxiliary_specs',lambda *a:[{**spec('gemma'),'label':'preflight'}])
    monkeypatch.setattr(session,'validate_inputs',lambda *a:None)
    monkeypatch.setattr(session,'hardware_preflight',lambda *a:None)
    launched=[]
    def command(root,workspace,s,directory,*a):
        launched.append(s['label'])
        stub=STUB.replace('generated_token_count=2,',"generated_token_ids=[1,2],gold_option='A',generated_token_count=2,")
        return [sys.executable,'-c',stub,'21']
    monkeypatch.setattr(session,'command',command)
    args=SimpleNamespace(root=tmp_path,workspace=tmp_path,smoke=False,plan=False,job='qa_kg_text',override_qa_parse_tripwire=True)
    assert session.run_model(args,'gemma',time.monotonic()+10)
    assert launched==['qa_kg_text']
    monkeypatch.setattr(verifier,'specs',lambda *a:[current])
    report=verifier.verify_model(tmp_path,'gemma',job='qa_kg_text')
    assert report['passed'] and report['jobs'][0]['tripwire_overridden']


def test_fetched_real_20_row_contract_accepts_only_code_changes(tmp_path):
    source=common.ROOT/'outputs/fullrun_S3_fetch_Tr48gbix/gemma/qa_kg_text'
    if not source.exists():pytest.skip('Local fetched evidence unavailable')
    shutil.copytree(source,tmp_path/'job')
    current=common.specs(common.ROOT,'gemma')[-1]
    saved,_=override.prepare_override(common.ROOT,tmp_path/'job',current)
    result=common.verify_job(tmp_path/'job',saved,complete=False)
    assert (result['rows'],result['FAILED'],result['ceilings'])==(20,15,15)
    assert result['tripwire_overridden']


def test_documented_blocks_independent_with_mock_transports(tmp_path):
    import os
    import re
    import subprocess
    docs=common.ROOT/'scripts/FULLRUN_GEMMA_KG_RESUME.md'
    blocks=re.findall(r'```bash\n(.*?)\n```',docs.read_text(),re.S)
    assert len(blocks)==5
    binaries=tmp_path/'bin';binaries.mkdir()
    calls=tmp_path/'calls.jsonl'
    prefix='import json,os,sys,subprocess\na=sys.argv[1:]\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps([os.path.basename(sys.argv[0]),a])+"\\n")\n'
    for name,body in {
        'git':'print("abc123" if a[0]=="rev-parse" else "abc123\\trefs/heads/inference50-core-keep")',
        'ssh':'subprocess.run(["bash","-n","-c",a[-1]],check=True)',
        'rsync':'pass',
    }.items():
        path=binaries/name;path.write_text('#!'+sys.executable+'\n'+prefix+body+'\n');path.chmod(0o755)
    for block in blocks:
        result=subprocess.run(['bash','-c',block],input='127.0.0.1\n22\n',text=True,capture_output=True,cwd=tmp_path,
                              env={**os.environ,'PATH':str(binaries)+os.pathsep+os.environ['PATH'],'CALLS':str(calls)},timeout=10)
        assert result.returncode==0,result.stderr
    commands=[args[-1] for name,args in map(json.loads,calls.read_text().splitlines()) if name=='ssh']
    launch=next(c for c in commands if 'tmux new-session' in c)
    assert 'gemma --job qa_kg_text --override-qa-parse-tripwire' in launch
    assert 'MAX_HOURS=0.275' in launch
    # Validate the inner tmux shell after SSH's quoting, not merely the outer script.
    import shlex
    inner=shlex.split(launch)[-1]
    assert subprocess.run(['bash','-n','-c',inner]).returncode==0


def test_real_config_writer_enrichment_preserves_settings_and_marks_override(tmp_path,monkeypatch):
    import fullrun_runtime as runtime
    source=common.ROOT/'outputs/fullrun_S3_fetch_Tr48gbix/gemma/qa_kg_text'
    if not source.exists():pytest.skip('Local fetched evidence unavailable')
    shutil.copytree(source,tmp_path/'job')
    old,attempt=override.prepare_override(common.ROOT,tmp_path/'job',common.specs(common.ROOT,'gemma')[-1])
    monkeypatch.setenv('FULLRUN_CONTRACT',str(tmp_path/'job/contract.json'))
    cfg=json.loads((tmp_path/'job/run_config.before_parse_override.json').read_text())
    cuda=SimpleNamespace(get_device_name=lambda i:cfg['gpu_type'],synchronize=lambda:None,
                         max_memory_allocated=lambda:1234)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(manual_seed=lambda s:None,cuda=cuda))
    enriched=runtime.enrich_config(cfg)
    assert enriched=={**cfg,'tripwire_overridden':True}
    monkeypatch.setenv('FULLRUN_CODE_ATTEMPT',str(attempt))
    monkeypatch.setattr(runtime,'_started',time.perf_counter()-1)
    monkeypatch.setattr(runtime,'_generated_ids',[1,2])
    result={}
    runtime.finish_item(result)
    assert result['generated_token_ids']==[1,2]
    assert result['code_attempt']=={'path':'code_attempts/'+attempt.name,'sha256':common.sha(attempt)}


def test_parse_override_cannot_apply_to_other_job(tmp_path):
    old,current=fixture_job(tmp_path)
    for key,value in [('model','qwen'),('label','qa_text'),('mode','smoke')]:
        with pytest.raises(ValueError,match='restricted'):
            override.prepare_override(tmp_path,tmp_path,{**current,key:value})
    cfg=json.loads((tmp_path/'run_config.json').read_text());cfg['tripwire_overridden']=True
    (tmp_path/'run_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='audit'):
        common.verify_job(tmp_path,old,complete=False)


def test_consolidation_flags_override_and_keeps_unparsed_wrong(tmp_path):
    import fullrun_consolidate as consolidate
    base=tmp_path/'outputs'/(common.DATA_NAME+'_results')/'gemma'
    for condition in common.CONDITIONS:
        directory=base/('qa_'+condition);directory.mkdir(parents=True)
        rows=[{'statement_idx':i,'raw_response':text,'gold_option':'A','hit_token_ceiling':bool(i)}
              for i,text in enumerate(['A','Let me explain the reasoning first'])]
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        common.frozen_json(directory/'run_config.json',{'tripwire_overridden':condition=='kg_text'})
    meta=tmp_path/'outputs'/common.DATA_NAME/'test/graph_metadata_0_500.jsonl'
    meta.parent.mkdir(parents=True)
    meta.write_text(''.join(json.dumps({'statement_idx':i,'answerKey':'A',
        'visible_nodes':[{'cid':1,'in_question':True,'in_choices':['A']}],'edges':[]})+'\n' for i in range(2)))
    report,versions=consolidate.consolidate_model(tmp_path,'gemma',expected_count=2)
    assert report['tripwire_overridden_jobs']==['qa_kg_text'] and report['tripwire_overridden']
    assert report['primary']=='strict_64'
    for condition in common.CONDITIONS:
        metrics=report['versions']['strict_64']['conditions'][condition]
        assert metrics['tripwire_overridden']==(condition=='kg_text')
        assert metrics['accuracy']==.5 and metrics['accuracy_among_parsed']==1
    assert not report['versions']['extended_512']['available']


def test_targeted_main_reuses_startup_evidence_without_launching_smoke(tmp_path,monkeypatch):
    import fullrun_gate as gate
    monkeypatch.setenv('TMUX','local-test')
    monkeypatch.setattr(sys,'argv',['fullrun_session.py','gemma','--root',str(tmp_path),'--workspace',str(tmp_path),
                                    '--job','qa_kg_text','--override-qa-parse-tripwire'])
    calls=[]
    def model_gate(root,model,*args):
        assert args[-1]==tmp_path/'outputs'/(common.DATA_NAME+'_gemma_start_smoke')
        calls.append('gate');return {'passed':True,'errors':[]}
    def run(args,model,deadline):
        assert not args.smoke and args.job=='qa_kg_text' and args.override_qa_parse_tripwire
        calls.append('resume');return True
    monkeypatch.setattr(gate,'model_gate',model_gate)
    monkeypatch.setattr(session,'run_model',run)
    with pytest.raises(SystemExit) as exc:session.main()
    assert exc.value.code==0 and calls==['gate','resume']
