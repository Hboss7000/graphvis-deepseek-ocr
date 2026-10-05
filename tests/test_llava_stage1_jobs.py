"""Prepare diagnostic subsets and exercise guarded jobs without model/GPU imports."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import llava_stage1_jobs as jobs
import llava_stage1_summary as summary
from fullrun_common import read_rows, frozen_json, sha
from llava_stage1_prompt import prefix_for_record
from test_llava_stage1_diagnostics import runner


@pytest.fixture
def prepared(tmp_path):
    source=ROOT/'outputs'/jobs.DATA_NAME
    measured=ROOT/'outputs'/(jobs.DATA_NAME+'_results')/'llava/stage1'
    if not source.exists() or not measured.exists():pytest.skip('Main-run source/evidence required')
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs'/jobs.DATA_NAME).symlink_to(source,target_is_directory=True)
    evidence=tmp_path/'outputs'/(jobs.DATA_NAME+'_results')/'llava'
    evidence.mkdir(parents=True)
    (evidence/'stage1').symlink_to(measured,target_is_directory=True)
    manifest=jobs.prepare(tmp_path)
    return tmp_path,manifest


def write_arm(root,manifest,phase,arm):
    s=manifest['phases'][phase];directory=jobs.output_dir(root,phase,arm)
    directory.mkdir(parents=True,exist_ok=True)
    template,prefix=jobs.ARMS[arm]
    cfg={'model_id':manifest['model_id'],'model_revision':manifest['model_revision'],'seed':13,
         'effective_max_new_tokens':1024,'generation':manifest['generation'],
         'prompt_template':template,'assistant_prefix_mode':prefix,'diagnostic_arm':arm,
         'prompt_bodies_sha256':s['prompt_bodies_sha256'],'user_turn_text_sha256':s['prompt_bodies_sha256'],
         'extractor':'span_extended','image_processing':{'mode':'processor_default_anyres'},
         'torch_version':jobs.VERSIONS['torch'],'transformers_version':jobs.VERSIONS['transformers'],
         'input_jsonl':{'sha256':manifest['files'][s['input_jsonl']]},
         'graph_metadata':{'sha256':manifest['files'][manifest['graph_metadata']]}}
    frozen_json(directory/'run_config.json',cfg)
    frozen_json(directory/'runtime_metrics.json',{'loading_elapsed_seconds':30})
    metadata={r['statement_idx']:r for r in read_rows(root/manifest['graph_metadata'])}
    records=read_rows(root/s['input_jsonl'])
    rows=[]
    args=SimpleNamespace(answer_format='none',assistant_prefix_mode=prefix,extractor='span_extended',
                         model_id=manifest['model_id'],revision=manifest['model_revision'])
    for record in records:
        response=record['answer']
        pre=prefix_for_record(record,prefix)
        if pre:
            if record['task_type']=='node_degree':
                name=record['prompt'].split('"')[1]
                canonical=f'The degree of the node "{name}" is'
                response=response[len(canonical):]
            else:response=response[len(pre):]
        row=runner.make_result(record,response,1,False,10,1.,1024,metadata[record['statement_idx']],args)
        row['item_elapsed_seconds']=2.
        rows.append(row)
    for task in s['tasks']:
        (directory/f'predictions_llava_{task}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['task_type']==task))
    return rows


def test_pilot_selection_is_same_20_graphs_four_numeric_tasks_for_every_arm(prepared):
    root,manifest=prepared
    pilot=manifest['phases']['pilot'];dry=manifest['phases']['pilot-dry']
    assert (pilot['record_count'],pilot['graphs'])==(80,20)
    assert (dry['record_count'],dry['graphs'])==(8,2)
    assert set(pilot['tasks'])==set(jobs.NUMERIC_TASKS)
    assert set(dry['indices'])<=set(pilot['indices'])
    assert jobs.load_manifest(root)==manifest
    for arm,(template,prefix) in jobs.ARMS.items():
        cmd=jobs.command(root,Path('/workspace'),manifest,'pilot',arm)
        assert cmd[cmd.index('--prompt-template')+1]==template
        assert cmd[cmd.index('--assistant-prefix-mode')+1]==prefix
        assert cmd[cmd.index('--max-new-tokens')+1]=='1024'
        assert cmd[cmd.index('--seed')+1]=='13'
        assert str(jobs.output_dir(root,'pilot',arm)) in cmd
        assert '--behavior-only' in cmd
    speed=manifest['speed_evidence'];estimate=jobs.estimate(manifest,'pilot',jobs.ARMS)
    expected=(20*sum(speed['seconds_per_task'][t] for t in jobs.NUMERIC_TASKS)+speed['loading_seconds'])*3
    assert estimate['expected_seconds']==pytest.approx(expected)
    assert estimate['guard_seconds']>=expected*1.5


def test_pilot_verification_short_primed_answers_and_summary(prepared):
    root,manifest=prepared
    for arm in jobs.ARMS:
        write_arm(root,manifest,'pilot',arm)
        assert len(jobs.verify_arm(root,manifest,'pilot',arm))==80
    report=summary.summary(root)
    for arm,data in report['arms'].items():
        for task,metrics in data['tasks'].items():
            assert metrics['n']==20
            assert metrics['answer_rate']==metrics['exact_accuracy']==metrics['lenient_containment']==1
            assert metrics['mean_absolute_error']==metrics['ceiling_rate']==0
            assert len(metrics['raw_answers'])==3
    directory=jobs.output_dir(root,'pilot','P1')
    cfg=json.loads((directory/'run_config.json').read_text());cfg['assistant_prefix_mode']='none'
    (directory/'run_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='config drift'):jobs.verify_arm(root,manifest,'pilot','P1')


def test_pilot_cannot_start_without_real_dry_evidence(prepared,monkeypatch):
    root,manifest=prepared
    monkeypatch.setenv('TMUX','test')
    monkeypatch.setattr(jobs,'hardware_preflight',lambda *a:pytest.fail('GPU preflight before dry proof'))
    args=SimpleNamespace(root=root,workspace=root,phase='pilot',arm='P1',plan=False,max_hours=None)
    with pytest.raises(ValueError,match='coverage'):jobs.run(args)


def test_numeric_summary_counts_unparsed_as_wrong_and_mae_only_parsed():
    records=[dict(statement_idx=i,task_type='node_number',answer='There are 18 nodes in the graph.') for i in range(4)]
    rows=[dict(statement_idx=i,task_type='node_number',raw_response=text,hit_token_ceiling=i==3)
          for i,text in enumerate(['18','17','No count given','First 18, final answer: 19'])]
    report=summary.summarize_task(records,rows,{i:{} for i in range(4)},'node_number')
    assert report['answer_rate']==.75 and report['exact_accuracy']==.25
    assert report['lenient_containment']==.5 and report['mean_absolute_error']==pytest.approx(2/3)
    assert report['mean_absolute_error_n']==3 and report['ceiling_rate']==.25


def test_watchdog_real_subprocess_success_stall_and_deadline(tmp_path):
    code='import pathlib,sys,time; time.sleep(float(sys.argv[1])); pathlib.Path(sys.argv[2]).write_text("{}\\n")'
    path=tmp_path/'predictions_llava_node_number.jsonl'
    status=jobs.execute([sys.executable,'-c',code,'0',str(path)],tmp_path,tmp_path/'log',time.monotonic()+10,poll_seconds=.01)
    assert status['state']=='COMPLETED'
    status=jobs.execute([sys.executable,'-c',code,'30',str(path)],tmp_path,tmp_path/'log',time.monotonic()+10,stall_seconds=.05,poll_seconds=.01)
    assert status['state']=='TRIPWIRE'
    status=jobs.execute([sys.executable,'-c',code,'30',str(path)],tmp_path,tmp_path/'log',time.monotonic()+.05,poll_seconds=.01)
    assert status['state']=='PAUSED'


def test_dry_run_projection_uses_actual_arm_timings(prepared):
    root,manifest=prepared
    write_arm(root,manifest,'pilot-dry','P1')
    projected=jobs.dry_projection(root,manifest,'pilot','P1')
    assert projected['expected_seconds']==80*2+30
    assert projected['guard_seconds']==285
    assert projected['peak_vram_bytes']==1024
