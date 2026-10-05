#!/usr/bin/env python3
"""512-token reruns of every strict-64 ceiling, with an exhaustive exact-token prefix audit."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time
from fullrun_common import *
from fullrun_session import command, check_resume, hardware_preflight, watch_job, record_status


def rescue_spec(root,model,strict_spec,strict_directory,directory,prepare=True):
    current_code = strict_spec.get('code_files', {})
    strict_spec = saved_spec(root, strict_directory, strict_spec)
    verify_job(strict_directory,strict_spec)
    originals=prediction_rows(strict_directory,strict_spec)
    selected=[r['statement_idx'] for r in originals if r['hit_token_ceiling']]
    selection=directory/'selected_indices.json'
    if directory.exists() and not (directory/'contract.json').exists() and any(p.name!='selected_indices.json' for p in directory.iterdir()):
        raise ValueError('Unknown rescue output collision: '+str(directory))
    if prepare:
        frozen_json(selection,selected)
    elif not selection.is_file() or json.loads(selection.read_text())!=selected:
        raise ValueError('Missing/changed rescue selection')
    cfg = json.loads((strict_directory/'run_config.json').read_text())
    if cfg.get('seed') is None or 'RTX PRO 6000' not in cfg.get('gpu_type', ''):
        raise ValueError('Strict QA must record its seed and actual RTX PRO 6000 GPU type')
    execution = {k: cfg[k] for k in ('seed', 'gpu_type','image_processing','pan_and_scan','pan_and_scan_kwargs','dtype','attn_implementation','cache_implementation') if k in cfg}
    execution['generation'] = {**cfg['generation'], 'max_new_tokens':512}
    spec=copy.deepcopy(strict_spec)
    spec['code_files'] = current_code
    spec.update(label='rescue_'+strict_spec['condition'],effective_max_new_tokens=512,
                record_count=len(selected),expected_keys=[[idx,'obqa_answer'] for idx in selected])
    spec['input_files'][str(selection.relative_to(root))]=sha(selection)
    spec['input_files'][str((strict_directory/'predictions.jsonl').relative_to(root))]=sha(strict_directory/'predictions.jsonl')
    spec['rescue']={'label':'extended_512','selection':'every strict-64 hit_token_ceiling, irrespective of parsing',
                    'strict_predictions':str((strict_directory/'predictions.jsonl').relative_to(root)),
                    'prefix_tokens':64,'deepseek_ceiling_detection_approximate':model=='deepseek',
                    'strict_code_files':strict_spec.get('code_files',{}), 'strict_execution':execution,
                    'strict_config_sha256':sha(strict_directory/'run_config.json'),
                    'strict_contract_sha256':sha(strict_directory/'contract.json')}
    return spec,originals


def prefix_audit(originals,rescued):
    expected={r['statement_idx']:r for r in originals if r['hit_token_ceiling']}
    actual={r['statement_idx']:r for r in rescued}
    if len(actual)!=len(rescued) or set(expected)!=set(actual):
        raise ValueError('Rescue coverage/duplicates differ from all strict ceiling items')
    rows=[]
    for idx,original in expected.items():
        rescue=actual[idx]
        for field in ('model_id','model_revision','gold_option','image'):
            if original[field]!=rescue[field]:raise ValueError('Rescue identity mismatch: '+field)
        left,right=original.get('generated_token_ids'),rescue.get('generated_token_ids')
        valid=isinstance(left,list) and isinstance(right,list) and len(left)>=64 and len(right)>=64
        matched=valid and left[:64]==right[:64]
        rows.append({'statement_idx':idx,'prefix_tokens':64,'matched':bool(matched),
                     'strict_id_count':len(left) if isinstance(left,list) else None,
                     'rescue_id_count':len(right) if isinstance(right,list) else None,
                     'reason':None if matched else ('missing/short exact token IDs' if not valid else 'token prefix differs'),
                     'still_hit_512':bool(rescue['hit_token_ceiling'])})
    return {'selected':len(expected),'checked':len(rows),'matched':sum(r['matched'] for r in rows),
            'mismatches':sum(not r['matched'] for r in rows),'still_hit_512':sum(r['still_hit_512'] for r in rows),'items':rows}


def verify_rescue(root,model):
    base=root/'outputs'/(DATA_NAME+'_results')/model
    reports={}
    for strict_spec in specs(root,model,False)[1:]:
        directory=base/'rescue_512'/strict_spec['condition']
        spec,originals=rescue_spec(root,model,strict_spec,base/strict_spec['label'],directory,prepare=False)
        if not spec['record_count']:
            recorded=json.loads((directory/'contract.json').read_text())
            if {k:v for k,v in recorded.items() if k!='code_files'}!={k:v for k,v in spec.items() if k!='code_files'}:
                raise ValueError('Zero-selection rescue contract differs')
            validate_frozen_files(root,{**recorded,'code_files':{}})
            rescued=[]
        else:
            # Audit an existing rescue using its recorded code, even on a later laptop commit.
            spec = saved_spec(root,directory,spec)
            verify_job(directory,spec)
            cfg=json.loads((directory/'run_config.json').read_text())
            if {k:cfg.get(k) for k in spec['rescue']['strict_execution']} != spec['rescue']['strict_execution']:
                raise ValueError('Rescue seed/GPU/decoding differs from strict execution')
            rescued=prediction_rows(directory,spec)
            for row in rescued:
                link=row.get('code_attempt')
                if link:
                    path=Path(link['path'])
                    if len(path.parts)!=2 or path.parts[0]!='code_attempts' or path.parts[1] in ('.','..'):
                        raise ValueError('Invalid rescue code-attempt path')
                    if sha(base/'rescue_512'/path)!=link['sha256']:
                        raise ValueError('Rescue code-attempt hash differs')
        reports[strict_spec['condition']]=prefix_audit(originals,rescued)
        reports[strict_spec['condition']]['strict_predictions_sha256']=sha(base/strict_spec['label']/'predictions.jsonl')
        reports[strict_spec['condition']]['rescue_predictions_sha256']=sha(directory/'predictions.jsonl') if rescued else None
    return reports


def run_rescue(args,model,deadline):
    base=args.root/'outputs'/(DATA_NAME+'_results')/model
    prepared=[]
    attempt_code=specs(args.root,model,False)[1]["code_files"]
    # Validate all strict outputs and resume contracts before any weights load.
    for strict in specs(args.root,model,False)[1:]:
        directory=base/'rescue_512'/strict['condition']
        spec,originals=rescue_spec(args.root,model,strict,base/strict['label'],directory)
        if (directory/'contract.json').exists():
            recorded=json.loads((directory/'contract.json').read_text())
            if {k:v for k,v in recorded.items() if k!='code_files'}!={k:v for k,v in spec.items() if k!='code_files'}:
                raise ValueError('Incompatible rescue resume experimental settings')
            spec=recorded
            check_resume(directory,spec)
        frozen_json(directory/'contract.json',spec)
        prepared.append((directory,spec,originals))
    attempts=base/'rescue_512/code_attempts'
    attempt=attempts/('attempt_%04d.json' % (len(list(attempts.glob('attempt_*.json')))+1))
    frozen_json(attempt,{'code_files':attempt_code,'strict_code_files':{s['condition']:s['rescue']['strict_code_files'] for _,s,_ in prepared}})
    if any(s['record_count'] for _,s,_ in prepared):hardware_preflight(args.workspace,model,prepared[0][1]['versions'])
    for directory,spec,originals in prepared:
        status_path=base/'rescue_512/status'/(spec['condition']+'.json')
        if not spec['record_count']:
            record_status(status_path,{'state':'COMPLETED','reason':'No strict-64 ceilings'})
            continue
        if status_path.exists() and json.loads(status_path.read_text())['state']=='COMPLETED':
            verify_job(directory,spec);continue
        preflight=base/'preflight/preflight_report.json' if model=='gemma' else None
        cmd=command(args.root,args.workspace,spec,directory,preflight)+['--only-indices-json',str(directory/'selected_indices.json')]
        if '--seed' in cmd:
            cmd[cmd.index('--seed')+1]=str(spec['rescue']['strict_execution']['seed'])
        previous=os.environ.get('FULLRUN_CODE_ATTEMPT')
        os.environ['FULLRUN_CODE_ATTEMPT']=str(attempt)
        try:
            status=watch_job(cmd,directory,spec,args.workspace/'logs'/(DATA_NAME+'_results')/model/('rescue_'+spec['condition']+'.log'),deadline)
        finally:
            if previous is None:os.environ.pop('FULLRUN_CODE_ATTEMPT',None)
            else:os.environ['FULLRUN_CODE_ATTEMPT']=previous
        if status['state']=='COMPLETED':verify_job(directory,spec)
        record_status(status_path,status)
        if status['state']!='COMPLETED':return False
    report=verify_rescue(args.root,model)
    frozen_json(base/'rescue_512/prefix_report.json',report)
    print('RESCUE exact prefix audit (every selected item): '+json.dumps(report),flush=True)
    if model=='deepseek':print('DeepSeek ceiling detection is approximate (response retokenization); token-prefix audit uses actual generate IDs.',flush=True)
    from fullrun_consolidate import print_spot_check
    print_spot_check(args.root)
    return not any(r['mismatches'] for r in report.values())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=MODELS)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workspace',type=Path,default=Path('/workspace'))
    p.add_argument('--verify-only',action='store_true');p.add_argument('--max-hours',type=float,default=1)
    args=p.parse_args();args.root=args.root.resolve()
    if args.verify_only:
        report=verify_rescue(args.root,args.model)
        print(json.dumps(report,indent=2))
        raise SystemExit(1 if any(r['mismatches'] for r in report.values()) else 0)
    if not math.isfinite(args.max_hours) or args.max_hours<=0:p.error('Invalid --max-hours')
    if not os.environ.get('TMUX') or not args.root.is_relative_to(args.workspace.resolve()):raise ValueError('Run in tmux on /workspace')
    os.environ.update(HF_HOME=str(args.workspace/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    os.environ['FULLRUN_MONITOR_PATH']=str(args.workspace/'logs/fullrun_current_job.json')
    lock=(args.workspace/'.fullrun_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ok=run_rescue(args,args.model,time.monotonic()+args.max_hours*3600)
    print('Rescue session ended. STOP the Pod; billing continues.',flush=True)
    raise SystemExit(0 if ok else 1)


if __name__=='__main__':main()
