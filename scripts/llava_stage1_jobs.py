#!/usr/bin/env python3
"""CPU preparation and guarded, resumable LLaVA Stage 1 diagnostic pilots."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from fullrun_common import ROOT, DATA_NAME, MODELS, sha, read_rows, frozen_json, validate_frozen_files
from fullrun_session import hardware_preflight, stop_process, record_status
from llava_stage1_prompt import diagnostic_arm
from stage1_common import prompt_bodies_sha256, raw_image_prompt
from score_stage1 import TASK_SETS

NAME = 'llava_stage1_diagnostics_2026-10-05'
RUNNER = 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py'
NUMERIC_TASKS = ('node_number', 'edge_number', 'node_degree', 'highest_node_degree')
ARMS = {'P1': ('hf-chat','gold-template'), 'P2': ('llava_v1','none'), 'P3': ('llava_v1','gold-template')}
VERSIONS = {'torch': '2.8.0+cu128', 'transformers': '5.16.1'}
PHASES = {'pilot-dry': (2, NUMERIC_TASKS), 'pilot': (20, NUMERIC_TASKS)}


def row_keys(rows):
    keys = [(int(r['statement_idx']), str(r['task_type'])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate diagnostic keys')
    return keys


def user_turn_hashes(rows):
    return {f"{r['statement_idx']}:{r['task_type']}": hashlib.sha256(raw_image_prompt(r,'none').encode()).hexdigest()
            for r in rows}


def measured_speeds(directory, source):
    cfg = json.loads((directory/'run_config.json').read_text())
    if (cfg['model_id'], cfg['model_revision'], cfg['seed'], cfg['effective_max_new_tokens']) != (*MODELS['llava'],13,1024):
        raise ValueError('Speed evidence differs from the pinned main run')
    if cfg['prompt_bodies_sha256'] != prompt_bodies_sha256(source,'none'):
        raise ValueError('Main-run user-turn hash differs from the 900 diagnostic source rows')
    rows=[];files={}
    for task in TASK_SETS['extended']:
        path=directory/f'predictions_llava_{task}.jsonl'
        rows.extend(read_rows(path));files[str(path)]=sha(path)
    if set(row_keys(rows)) != set(row_keys(source)):
        raise ValueError('Main speed evidence must cover the same 900 records')
    for row in rows:
        if row.get('prompt') != next(r['prompt'] for r in source if
                (r['statement_idx'],r['task_type']) == (row['statement_idx'],row['task_type'])):
            raise ValueError('Main prediction question changed')
        if row.get('model_revision') != MODELS['llava'][1]:
            raise ValueError('Main speed evidence revision differs')
        value=row.get('item_elapsed_seconds')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Invalid main measured item time')
    loading=json.loads((directory/'runtime_metrics.json').read_text())['loading_elapsed_seconds']
    return {'seconds_per_task': {t:sum(r['item_elapsed_seconds'] for r in rows if r['task_type']==t)/100
                                for t in TASK_SETS['extended']},
            'loading_seconds': loading, 'peak_vram_bytes': max(r['peak_memory_allocated_bytes'] for r in rows),
            'source_files_sha256': files, 'source_run_config_sha256': sha(directory/'run_config.json'),
            'note': 'Measured main hf-chat/none run; diagnostic response lengths may differ.'}


def prepare(root, measurements_dir=None):
    source_path=root/'outputs'/DATA_NAME/'test/stage1_subset100.jsonl'
    source=read_rows(source_path)
    expected={(i,t) for i in {r['statement_idx'] for r in source} for t in TASK_SETS['extended']}
    if len(source)!=900 or len(expected)!=900 or set(row_keys(source))!=expected:
        raise ValueError('Expected exactly 100 graphs x all nine tasks')
    base=root/'outputs'/NAME
    indices=sorted({r['statement_idx'] for r in source})
    measurements_dir=measurements_dir or root/'outputs'/(DATA_NAME+'_results')/'llava/stage1'
    speeds=measured_speeds(measurements_dir,source)
    metadata='outputs/'+DATA_NAME+'/test/graph_metadata_0_500.jsonl'
    image_root='outputs/'+DATA_NAME
    files={str(source_path.relative_to(root)):sha(source_path), metadata:sha(root/metadata),
           'outputs/'+DATA_NAME+'/inputs_manifest.json':sha(root/'outputs'/DATA_NAME/'inputs_manifest.json')}
    for row in source:
        path=image_root+'/'+row['image'];files[path]=sha(root/path)
    phases={}
    for phase,(graphs,tasks) in PHASES.items():
        wanted=set(indices[:graphs])
        selected=[r for r in source if r['statement_idx'] in wanted and r['task_type'] in tasks]
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
        path=base/'inputs'/(phase+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_text()!=data:raise ValueError('Diagnostic input collision')
        if not path.exists():path.write_text(data)
        name=str(path.relative_to(root));files[name]=sha(path)
        phases[phase]={'input_jsonl':name,'record_count':len(selected),'graphs':graphs,'tasks':list(tasks),
                       'indices':sorted(wanted),'expected_keys':[list(k) for k in row_keys(selected)],
                       'prompt_bodies_sha256':prompt_bodies_sha256(selected,'none'),
                       'user_turn_hashes':user_turn_hashes(selected)}
    manifest={'schema':1,'model_id':MODELS['llava'][0],'model_revision':MODELS['llava'][1],
              'versions':VERSIONS,'seed':13,'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':1024},
              'image_root':image_root,'graph_metadata':metadata,'files':files,'phases':phases,
              'arms':{a:dict(prompt_template=t,assistant_prefix_mode=p) for a,(t,p) in ARMS.items()},
              'main_user_turn_text_sha256':prompt_bodies_sha256(source,'none'),
              'main_user_turn_hashes':user_turn_hashes(source),'speed_evidence':speeds,
              'selection':'Lowest 20 statement indices of the frozen 100-graph main subset; dry run uses the first two.',
              'image_settings':'Same pinned LLaVA-NeXT default anyres processor and images as the main run; no rendering.'}
    frozen_json(base/'manifest.json',manifest)
    return manifest


def load_manifest(root):
    manifest=json.loads((root/'outputs'/NAME/'manifest.json').read_text())
    if (manifest['model_id'],manifest['model_revision'],manifest['seed'],manifest['generation'],manifest['arms']) != (
            *MODELS['llava'],13,{'do_sample':False,'num_beams':1,'max_new_tokens':1024},
            {a:dict(prompt_template=t,assistant_prefix_mode=p) for a,(t,p) in ARMS.items()}):
        raise ValueError('Diagnostic manifest experimental drift')
    if manifest['versions'] != VERSIONS:raise ValueError('Diagnostic manifest version drift')
    validate_frozen_files(root,{'input_files':manifest['files']})
    for phase,spec in manifest['phases'].items():
        rows=read_rows(root/spec['input_jsonl'])
        if len(rows)!=spec['record_count'] or set(row_keys(rows))!={tuple(k) for k in spec['expected_keys']}:
            raise ValueError('Diagnostic input coverage drift')
        hashes=user_turn_hashes(rows)
        if hashes!=spec['user_turn_hashes'] or any(manifest['main_user_turn_hashes'].get(k)!=v for k,v in hashes.items()):
            raise ValueError('User-turn text differs from main run')
        if prompt_bodies_sha256(rows,'none')!=spec['prompt_bodies_sha256']:
            raise ValueError('Diagnostic user-turn hash drift')
    return manifest


def output_dir(root,phase,arm):
    return root/'outputs'/NAME/phase/arm


def command(root,workspace,manifest,phase,arm):
    spec=manifest['phases'][phase]
    template,prefix=ARMS[arm]
    cmd=[str(workspace/'venvs/venv_qwen/bin/python'),'-u',str(root/RUNNER),
         '--model-id',manifest['model_id'],'--revision',manifest['model_revision'],
         '--input-jsonl',str(root/spec['input_jsonl']),'--graph-metadata',str(root/manifest['graph_metadata']),
         '--image-root',str(root/manifest['image_root']),'--output-dir',str(output_dir(root,phase,arm)),
         '--expected-count',str(spec['record_count']),'--expected-split','test','--task-set','extended',
         '--extractor','span_extended','--answer-format','none','--seed','13','--max-new-tokens','1024',
         '--prompt-template',template,'--assistant-prefix-mode',prefix,'--approve-prompts','--resume']
    cmd+=['--behavior-only']
    return cmd


def predictions(directory,tasks):
    unknown=set(directory.glob('predictions*.jsonl'))-{directory/f'predictions_llava_{t}.jsonl' for t in tasks}
    if unknown:raise ValueError('Unexpected diagnostic task files')
    return [r for t in tasks for r in read_rows(directory/f'predictions_llava_{t}.jsonl')]


def verify_arm(root,manifest,phase,arm,complete=True):
    spec=manifest['phases'][phase];directory=output_dir(root,phase,arm)
    rows=predictions(directory,spec['tasks']);keys=set(row_keys(rows));expected={tuple(k) for k in spec['expected_keys']}
    if not keys<=expected or (complete and keys!=expected):raise ValueError('Diagnostic prediction coverage differs')
    cfg=json.loads((directory/'run_config.json').read_text())
    template,prefix=ARMS[arm]
    for key,value in {'model_id':manifest['model_id'],'model_revision':manifest['model_revision'],
                     'seed':13,'effective_max_new_tokens':1024,'generation':manifest['generation'],
                     'prompt_template':template,'assistant_prefix_mode':prefix,'diagnostic_arm':arm,
                     'prompt_bodies_sha256':spec['prompt_bodies_sha256'],
                     'user_turn_text_sha256':spec['prompt_bodies_sha256'],'extractor':'span_extended'}.items():
        if cfg.get(key)!=value:raise ValueError('Diagnostic config drift: '+key)
    for name,version in VERSIONS.items():
        if cfg.get(name+'_version')!=version:raise ValueError('Diagnostic package version drift')
    for key,path in [('input_jsonl',spec['input_jsonl']),('graph_metadata',manifest['graph_metadata'])]:
        if cfg[key]['sha256']!=manifest['files'][path]:raise ValueError('Diagnostic config input drift')
    if cfg['image_processing']['mode']!='processor_default_anyres':raise ValueError('Diagnostic image settings changed')
    inputs={(r['statement_idx'],r['task_type']):r for r in read_rows(root/spec['input_jsonl'])}
    from llava_stage1_prompt import prefix_for_record
    for row in rows:
        original=inputs[(row['statement_idx'],row['task_type'])]
        for key in ('prompt','image'):
            if row.get(key)!=original[key]:raise ValueError('Diagnostic prediction identity drift: '+key)
        if row.get('gold')!=original['answer']:raise ValueError('Diagnostic gold drift')
        if (row.get('model_id'),row.get('model_revision'))!=MODELS['llava']:raise ValueError('Diagnostic row model drift')
        expected_prefix=prefix_for_record(original,prefix)
        if expected_prefix and (row.get('assistant_prefix')!=expected_prefix or
                row['raw_response']!=expected_prefix+row.get('continuation_response','')):
            raise ValueError('Stored prefix/continuation differs')
        for field in ('item_elapsed_seconds','peak_memory_allocated_bytes'):
            value=row.get(field)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Missing/invalid diagnostic measurement: '+field)
        if type(row.get('hit_token_ceiling')) is not bool or type(row.get('generated_token_count')) is not int:
            raise ValueError('Missing diagnostic token measurements')
    return rows


def estimate(manifest,phase,arms):
    spec=manifest['phases'][phase];evidence=manifest['speed_evidence']
    one=spec['graphs']*sum(evidence['seconds_per_task'][t] for t in spec['tasks'])+evidence['loading_seconds']
    return {'phase':phase,'arms':list(arms),'expected_seconds':one*len(arms),
            'guard_seconds':math.ceil(one*len(arms)*1.5),'basis':evidence['note']}



def dry_projection(root, manifest, target, arm):
    phase = target + '-dry'
    rows = verify_arm(root, manifest, phase, arm)
    spec = manifest['phases'][target]
    per_task = {t: sum(r['item_elapsed_seconds'] for r in rows if r['task_type'] == t)
                    / sum(r['task_type'] == t for r in rows) for t in spec['tasks']}
    loading = json.loads((output_dir(root, phase, arm) / 'runtime_metrics.json').read_text())['loading_elapsed_seconds']
    seconds = spec['graphs'] * sum(per_task.values()) + loading
    return {'arm': arm, 'target': target, 'peak_vram_bytes': max(r['peak_memory_allocated_bytes'] for r in rows),
            'measured_seconds_per_task': per_task, 'expected_seconds': seconds,
            'guard_seconds': math.ceil(seconds * 1.5),
            'basis': 'Same-arm dry run, same images/processor/model and 1024-token generation path.'}


def execute(cmd,directory,log,deadline,stall_seconds=300,poll_seconds=1):
    """Stop on elapsed-time/output-stall limits; short primed answers are valid."""
    log.parent.mkdir(parents=True,exist_ok=True)
    print('Launch: '+shlex.join(cmd),flush=True)
    print('Monitor: tail -F '+shlex.quote(str(log))+'\nGPU: watch -n 30 nvidia-smi',flush=True)
    def sizes():return {p:p.stat().st_size for p in directory.glob('predictions*.jsonl')}
    previous=sizes();last_growth=time.monotonic();start=last_growth
    state,reason='FAILED',None
    with log.open('a',buffering=1) as handle:
        handle.write('\nLaunch: '+shlex.join(cmd)+'\n')
        process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                now=time.monotonic();current=sizes()
                if current!=previous:last_growth=now
                previous=current
                if now>=deadline:state,reason='PAUSED','Elapsed-time guard';stop_process(process);break
                if now-last_growth>=stall_seconds:state,reason='TRIPWIRE','No prediction growth for 5 minutes';stop_process(process);break
                code=process.poll()
                if code is not None:
                    state='COMPLETED' if code==0 else 'FAILED';reason=None if code==0 else f'exit code {code}';break
                time.sleep(poll_seconds)
        except BaseException:stop_process(process);raise
    return {'state':state,'reason':reason,'wall_seconds':time.monotonic()-start,'exit_code':process.returncode,'command':cmd}


def run(args):
    manifest=load_manifest(args.root)
    arms=list(ARMS) if args.arm=='all' else [args.arm]
    forecast=estimate(manifest,args.phase,arms)
    print(json.dumps(forecast,indent=2),flush=True)
    if args.plan:
        for arm in arms:print(shlex.join(command(args.root,args.workspace,manifest,args.phase,arm)))
        return True
    if not os.environ.get('TMUX') or not args.root.is_relative_to(args.workspace.resolve()):
        raise ValueError('Run diagnostic jobs inside tmux with repository/outputs on /workspace')
    for name in ('FULLRUN_CONTRACT','STAGE1_MATRIX_CELL_JSON','STAGE1_FROZEN_PROMPT_POLICY_JSON'):
        if os.environ.get(name):raise ValueError('Unexpected inherited experiment environment: '+name)
    # Every planned resume/dry-run proof is checked before weights load.
    for arm in arms:
        directory=output_dir(args.root,args.phase,arm)
        if directory.exists() and any(directory.iterdir()) and not (directory/'diagnostic_contract.json').exists():
            raise ValueError('Unknown diagnostic output collision')
        if args.phase=='pilot':
            verify_arm(args.root,manifest,'pilot-dry',arm)
        contract={'phase':args.phase,'arm':arm,'settings':manifest['arms'][arm],
                  'spec':manifest['phases'][args.phase],'manifest_sha256':sha(args.root/'outputs'/NAME/'manifest.json')}
        frozen_json(directory/'diagnostic_contract.json',contract)
        if (directory/'run_config.json').exists():verify_arm(args.root,manifest,args.phase,arm,complete=False)
    if args.phase=='pilot':
        forecasts=[dry_projection(args.root,manifest,args.phase,arm) for arm in arms]
        forecast={'phase':args.phase,'arms':arms,'expected_seconds':sum(f['expected_seconds'] for f in forecasts),
                  'guard_seconds':sum(f['guard_seconds'] for f in forecasts),'measured_dry_runs':forecasts}
        print('Dry-run extrapolation: '+json.dumps(forecast),flush=True)
    os.environ.update(HF_HOME=str(args.workspace/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    lock=(args.workspace/'.fullrun_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hardware_preflight(args.workspace,'llava',VERSIONS)
    seconds=args.max_hours*3600 if args.max_hours is not None else forecast['guard_seconds']
    deadline=time.monotonic()+seconds
    for arm in arms:
        directory=output_dir(args.root,args.phase,arm)
        status_path=args.root/'outputs'/NAME/'status'/(args.phase+'_'+arm+'.json')
        if status_path.exists() and json.loads(status_path.read_text())['state']=='COMPLETED':
            verify_arm(args.root,manifest,args.phase,arm);continue
        status=execute(command(args.root,args.workspace,manifest,args.phase,arm),directory,
                       args.workspace/'logs'/NAME/(args.phase+'_'+arm+'.log'),deadline)
        status.update(diagnostic_arm=arm,phase=args.phase)
        if status['state']=='COMPLETED':
            try:verify_arm(args.root,manifest,args.phase,arm)
            except Exception as exc:status.update(state='FAILED',reason='verification: '+str(exc))
        record_status(status_path,status)
        if status['state']!='COMPLETED':return False
    print('Diagnostic session completed. STOP the Pod; billing continues.',flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','verify','estimate'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--workspace',type=Path,default=Path('/workspace'))
    parser.add_argument('--measurements-dir',type=Path)
    parser.add_argument('--phase',choices=PHASES,default='pilot')
    parser.add_argument('--arm',choices=[*ARMS,'all'],default='all')
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--max-hours',type=float)
    parser.add_argument('--hourly-rate',type=float,help='Current user-supplied Pod rate for cost extrapolation')
    args=parser.parse_args();args.root=args.root.resolve()
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours<=0):parser.error('Invalid --max-hours')
    if args.action=='prepare':print(json.dumps(prepare(args.root,args.measurements_dir),indent=2));return
    if args.hourly_rate is not None and (not math.isfinite(args.hourly_rate) or args.hourly_rate<=0):parser.error('Invalid --hourly-rate')
    if args.action=='run':
        try:
            ok=run(args)
        finally:
            print('Job session ended. STOP the Pod; billing continues.',flush=True)
        raise SystemExit(0 if ok else 1)
    manifest=load_manifest(args.root);arms=list(ARMS) if args.arm=='all' else [args.arm]
    if args.action=='estimate':print(json.dumps(estimate(manifest,args.phase,arms),indent=2));return
    for arm in arms:
        rows=verify_arm(args.root,manifest,args.phase,arm)
        evidence={'arm':arm,'phase':args.phase,'rows':len(rows),
              'peak_vram_bytes':max(r['peak_memory_allocated_bytes'] for r in rows),
              'mean_seconds_per_item':sum(r['item_elapsed_seconds'] for r in rows)/len(rows)}
        if args.phase.endswith('-dry'):
            target=args.phase.removesuffix('-dry')
            projected=dry_projection(args.root,manifest,target,arm)
            evidence.update(projected)
            seconds=projected['expected_seconds']
            evidence.update(expected_cost_usd=seconds/3600*args.hourly_rate if args.hourly_rate else None,
                            with_margin_cost_usd=seconds/2400*args.hourly_rate if args.hourly_rate else None)
        print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
