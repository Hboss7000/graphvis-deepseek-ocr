from pathlib import Path
from types import SimpleNamespace
import copy
import importlib
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fullrun_common as common
import fullrun_runtime as runtime
import fullrun_session as session
import fullrun_s0b as recovery
from fullrun_snapshot_imports import snapshot_imports
from report_fullrun_probes import paired_report

FETCH=ROOT/'outputs/fullrun_fetch_20261004_183521.qtJrSg/fullrun_2026-10-04_B_smoke'
CORPUS=(sorted((ROOT/'outputs/phase1_dryrun').rglob('run_config.json'))+sorted(FETCH.rglob('run_config.json'))
        +sorted((ROOT/'outputs/fullrun_s0b_fetch_jcjYPY6T/fullrun_2026-10-04_B_s0b').rglob('run_config.json')))

@pytest.mark.parametrize('path',CORPUS,ids=lambda p:str(p.relative_to(ROOT)))
def test_every_real_config_enriches(path,tmp_path,monkeypatch):
    original=json.loads(path.read_text());generation=runtime.generation_settings(original)
    s=copy.deepcopy(original.get('fullrun',{}))
    for key in ('model_id','model_revision','prompt_bodies_sha256'):
        s[key]=original[key]
    s['effective_max_new_tokens']=original.get('effective_max_new_tokens',generation['max_new_tokens'])
    s['versions']={name:original.get(name+'_version','test-recorded-runtime') for name in ('torch','transformers')}
    for name,version in s['versions'].items():monkeypatch.setitem(sys.modules,name,SimpleNamespace(__version__=version))
    s.update(input_jsonl='records',graph_metadata='meta',input_files={'records':original['input_jsonl']['sha256'],'meta':original['graph_metadata']['sha256']},data_provenance={'test':'historical schema; missing package versions explicitly stubbed'})
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(s));monkeypatch.setenv('FULLRUN_CONTRACT',str(contract))
    result=runtime.enrich_config(original)
    assert result['generation']==generation
    assert result['effective_max_new_tokens']==generation['max_new_tokens']
    assert json.loads(path.read_text())==original
    assert result['generation']['do_sample'] is False and result['generation']['num_beams']==1

@pytest.mark.parametrize('model',common.MODELS)
def test_real_qa_writer_with_stub_args(model,tmp_path,monkeypatch):
    s=common.specs(ROOT,model,True)[1]
    name='eval_gemma3_stage2' if model=='gemma' else 'run_zero_shot'+('' if model=='deepseek' else '_'+model)
    runner=importlib.import_module(name)
    cmd=session.command(ROOT,Path('/workspace'),s,tmp_path,limit=5)
    monkeypatch.setattr(sys,'argv',[cmd[2],*cmd[3:]])
    args=runner.parse_args()
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(__version__=s['versions']['torch']))
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(s));monkeypatch.setenv('FULLRUN_CONTRACT',str(contract))
    records=common.read_rows(ROOT/s['input_jsonl']);meta={r['statement_idx']:r for r in common.read_rows(ROOT/s['graph_metadata'])}
    if model=='deepseek':runner.write_run_config(args,s['versions']['transformers'],s['versions']['torch'],64)
    elif model=='llava':runner.write_run_config(args,records,meta,s['versions']['transformers'],s['versions']['torch'],0,{}, {})
    else:
        positional=[args,records,meta,s['versions']['transformers'],'stub','dtype','bfloat16',False,{}]
        if model=='gemma':positional += [{'pan_and_scan_kwargs':{}},None]
        runner.write_run_config(*positional)
    cfg=json.loads((tmp_path/'run_config.json').read_text())
    assert cfg['generation']['max_new_tokens']==64
    assert cfg['fullrun']==s
    # Identical second call is idempotent and cannot overwrite changed settings.
    if model=='deepseek':runner.write_run_config(args,s['versions']['transformers'],s['versions']['torch'],64)
    assert common.verify_job(tmp_path,s,complete=False)['rows']==0

@pytest.mark.parametrize('model',common.MODELS)
def test_each_stage1_writer_stub_config(model,tmp_path,monkeypatch):
    runner=importlib.import_module('eval_gemma3_stage1' if model=='gemma' else 'run_stage1_'+model)
    s=common.specs(ROOT,model,True)[0]
    c={k:s[k] for k in ('model_id','model_revision','effective_max_new_tokens','prompt_bodies_sha256')}
    c.update(generation={'do_sample':False,'num_beams':1,'max_new_tokens':1024},torch_version=s['versions']['torch'],transformers_version=s['versions']['transformers'])
    for key in ('input_jsonl','graph_metadata'):c[key]={'path':str(ROOT/s[key]),'sha256':s['input_files'][s[key]]}
    p=tmp_path/'contract.json';p.write_text(json.dumps(s));monkeypatch.setenv('FULLRUN_CONTRACT',str(p))
    runner.write_run_config(tmp_path,runner.enrich_config(c))
    assert json.loads((tmp_path/'run_config.json').read_text())['fullrun']==s


def test_imports_come_from_snapshot_not_requirements(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'auto_map':{'AutoModel':'modeling_x.X'}}))
    (tmp_path/'modeling_x.py').write_text('import torch\nimport os\nfrom .helper import X\ntry:\n import flash_attn\nexcept ImportError:\n pass\ndef lazy():\n import other_optional\n')
    (tmp_path/'helper.py').write_text('from addict import Dict\n')
    report=snapshot_imports(tmp_path)
    assert report['required']==['addict','torch']
    assert report['conditional_or_deferred']==['flash_attn','other_optional']
    assert 'easydict' not in report['required']
    (tmp_path/'helper.py').write_text('from easydict import EasyDict\n')
    assert snapshot_imports(tmp_path)['required']==['easydict','torch']

@pytest.mark.skipif(not FETCH.exists(),reason='Fetched smoke evidence required')
def test_probe_ceilings_are_descriptive_and_exact_paired_gate():
    arms={a:FETCH/'llava'/('probe_'+a)/'predictions_llava_node_description.jsonl' for a in ('uniform','legacy')}
    report=paired_report(ROOT/'outputs'/common.DATA_NAME/'inputs/reading_probe.jsonl',ROOT/'outputs'/common.DATA_NAME/'test/graph_metadata_0_500.jsonl',arms,20)
    assert report['gate']['difference']==pytest.approx(-.07381387122127156)
    assert not report['gate']['passed']
    assert report['pairs']['uniform_minus_legacy']['basic']['better']==5
    assert report['pairs']['uniform_minus_legacy']['basic']['worse']==11
    assert report['pairs']['uniform_minus_legacy']['basic']['equal']==4
    for arm,n in [('uniform',14),('legacy',10)]:
        d=FETCH/'llava'/('probe_'+arm);s=json.loads((d/'contract.json').read_text())
        assert common.verify_job(d,s)['ceilings']==n
    assert len(report['per_graph'])==20
    assert report["arms"]["uniform"]["basic_recall"] == pytest.approx(.5331258, abs=1e-6)

@pytest.mark.skipif(not FETCH.exists(),reason='Fetched smoke evidence required')
def test_targeted_recovery_reuses_completed_and_rejects_experiment_changes():
    entries=recovery.reused(ROOT,FETCH)
    assert len(entries)==8
    jobs=recovery.recovery_jobs(ROOT)
    assert {m:len(j) for m,j in jobs.items()}=={'llava':3,'qwen':4,'gemma':4,'deepseek':5}
    assert all(s['record_count']==50 for s in jobs['llava'])
    d=FETCH/'qwen/stage1';current=common.specs(ROOT,'qwen',True)[0]
    assert recovery.historical_spec(d,current)['code_files'] != current['code_files']
    current['effective_max_new_tokens']=8192
    with pytest.raises(ValueError,match='experimental contract'):recovery.historical_spec(d,current)


def test_probe_tripwire_only_content_bypass():
    rows=[dict(raw_response='',generated_token_count=1,item_elapsed_seconds=1) for _ in range(20)]
    assert common.tripwire(rows,'stage1',3.5,reading_probe=True) is None
    assert common.tripwire(rows,'stage1',3.5,reading_probe=False)
    for row in rows:row['item_elapsed_seconds']=20
    assert common.tripwire(rows,'stage1',3.5,reading_probe=True)

@pytest.mark.parametrize('drop,accepted',[(.05,True),(.0501,False),(.0,True)])
def test_only_paired_basic_recall_decides(drop,accepted,tmp_path,monkeypatch):
    import report_fullrun_probes as probes
    records=tmp_path/'records';meta=tmp_path/'meta'
    records.write_text(json.dumps(dict(statement_idx=1,task_type='node_description',prompt='p',answer='a'))+'\n')
    meta.write_text(json.dumps(dict(statement_idx=1))+'\n')
    paths={}
    for arm in ('uniform','legacy','white'):
        p=tmp_path/arm;p.write_text(json.dumps(dict(statement_idx=1,task_type='node_description',raw_response=arm,prompt='p',gold='a',hit_token_ceiling=True))+'\n');paths[arm]=p
    def score(record,response,metadata,extractor):
        recall=1-drop if response=='uniform' else 1
        return {'set_metrics':{tier:{'recall':recall if tier=='basic' else 0,'precision':0,'f1':0} for tier in ('basic','raw')}}
    monkeypatch.setattr(probes,'score_record',score)
    report=probes.paired_report(records,meta,paths,1)
    assert report['gate']['passed'] is accepted
    assert report['arms']['uniform']['ceilings']==1
    assert report['arms']['white']['basic_recall']==1


def test_white_changes_only_node_fill_in_dot(tmp_path,monkeypatch):
    import generate_graphvis_datasets as gen
    nodes={1:dict(name='q',in_question=True,in_choices=set()),2:dict(name='a',in_question=False,in_choices={'A'}),3:dict(name='b',in_question=False,in_choices=set())}
    graph=dict(connected_nodes=[1,2],disconnected_answers=[3],edges=[(1,'isa',2)])
    sources=[]
    monkeypatch.setattr(gen.graphviz.Digraph,'render',lambda self,*a,**kw:sources.append(self.source))
    for fill in (None,'#FFFFFF'):
        gen.render_graph(tmp_path/'x',nodes,graph,'A','dot',False,node_style_mode='uniform',node_fill=fill)
    assert sources[0].replace('#ADD8E6','#FFFFFF')==sources[1]
    assert 'rounded,filled,solid' in sources[1]
    assert gen.node_fill_hex('ffffff')=='#FFFFFF'
    for bad in ('white','#FFF','#FFFFFG','FFFFFF;'):
        with pytest.raises(Exception):gen.node_fill_hex(bad)

@pytest.mark.skipif(not (ROOT/'outputs/fullrun_2026-10-04_B_uniform_white50').exists(),reason='Rendered white50 required')
def test_all_50_white_geometry_records_and_frozen_inputs():
    from prepare_fullrun_s0b import prepare
    prepare(ROOT)
    for s in recovery.probe_specs(ROOT):
        common.validate_frozen_files(ROOT,s)
        session.validate_inputs(ROOT,s)


def test_snapshot_audit_excludes_demos_and_keeps_conditional_local_dependencies_optional(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'auto_map':{'AutoModel':'modeling_x.X'}}))
    (tmp_path/'modeling_x.py').write_text('import torch\nif use_flash:\n from .flash_helper import X\n')
    (tmp_path/'flash_helper.py').write_text('import flash_attn\n')
    (tmp_path/'demo.py').write_text('import gradio\nimport easydict\n')
    report=snapshot_imports(tmp_path)
    assert report['required']==['torch']
    assert report['conditional_or_deferred']==['flash_attn']
    assert 'demo.py' not in report['source_sha256']


def test_fullrun_requires_recomputed_50_gate_when_recovery_exists(tmp_path,monkeypatch):
    import fullrun_summary
    marker=tmp_path/'outputs'/ (common.DATA_NAME+'_s0b')/'source.json';marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'source_smoke_root':'outputs/old_smoke'}))
    called=[]
    def failed_gate(root,source):
        called.append((root,source));return {'passed':False,'errors':['blue drop exceeds .05']}
    monkeypatch.setattr(recovery,'gate',failed_gate)
    assert fullrun_summary.rehearsal_report(tmp_path)['passed'] is False
    assert called==[(tmp_path,tmp_path/'outputs/old_smoke')]
