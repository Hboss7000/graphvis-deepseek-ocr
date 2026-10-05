#!/usr/bin/env python3
"""Read-only, per-model smoke gate with historical provenance and descriptive ceilings."""
import argparse
import json
from pathlib import Path
from fullrun_common import *
from fullrun_session import auxiliary_specs
from fullrun_s0b import historical_spec, proof, probe_specs
from report_fullrun_probes import paired_report


def evidence_roots(root, smoke=None, recovery=None):
    smoke=smoke or root/'outputs'/(DATA_NAME+'_smoke')
    recovery=recovery or root/'outputs'/(DATA_NAME+'_s0b')
    if not smoke.exists():smoke=root/'outputs/fullrun_fetch_20261004_183521.qtJrSg'/ (DATA_NAME+'_smoke')
    if not recovery.exists():recovery=root/'outputs/fullrun_s0b_fetch_jcjYPY6T'/ (DATA_NAME+'_s0b')
    return smoke,recovery


def verify_historical(root,directory,current):
    old=historical_spec(directory,current)
    validate_frozen_files(root,dict(old,code_files={}))
    result=verify_job(directory,old)
    status=directory.parent/'status'/(current['label']+'.json')
    state=json.loads(status.read_text())
    last=state['attempts'][-1]
    if state['state']!='COMPLETED':
        # The old n=5 final sanity rejection is superseded by the user's fixed policy.
        if state['state']!='FAILED' or not (last.get('reason') or '').startswith('Parse/ceiling sanity threshold exceeded:'):
            raise ValueError('Historical job did not finish: '+str(status))
        result['historical_sanity_superseded']=last['reason']
    result['directory']=str(directory.relative_to(root))
    result['evidence_files']={str(p.relative_to(root)):sha(p) for p in directory.iterdir() if p.is_file() and p.name not in ('.lock',)}
    return result


def model_gate(root,model,smoke_root=None,s0b_root=None,gemma_smoke_root=None):
    smoke,recovery=evidence_roots(root,smoke_root,s0b_root)
    report={'model':model,'mode':'per-model gate','jobs':[],'errors':[],'probe':{},'passed':False}
    try:
        manifest=json.loads((recovery/'reuse_manifest.json').read_text())
        for entry in manifest:
            if entry['model']!=model:continue
            prefix=Path(entry['directory'])
            for name,expected in entry['files'].items():
                suffix=Path(name).relative_to(prefix)
                actual=smoke/model/entry['label']/suffix
                if sha(actual)!=expected:raise ValueError('Reused smoke evidence hash mismatch: '+str(actual))
    except (OSError,ValueError,KeyError,TypeError) as exc:report['errors'].append('Reused hashes: '+str(exc))
    for current in specs(root,model,True):
        if model=='llava' or (model in ('qwen','gemma') and current['label']=='stage1'):
            directory=smoke/model/current['label']
        else:directory=recovery/model/current['label']
        if model=='gemma' and gemma_smoke_root and current['stage']=='qa':
            directory=gemma_smoke_root/model/current['label']
        try:report['jobs'].append(verify_historical(root,directory,current))
        except (OSError,ValueError,KeyError,TypeError) as exc:report['errors'].append(current['label']+': '+str(exc))
    try:
        qa_base=smoke if model=='llava' else (gemma_smoke_root if model=='gemma' and gemma_smoke_root else recovery)
        proof(qa_base/model/'qa_image')
        report['resume_proof']='PASSED'
    except (OSError,ValueError,KeyError,TypeError) as exc:report['errors'].append('Resume proof: '+str(exc))
    if model=='gemma':
        try:
            current,=auxiliary_specs(root,model,True)
            directory=smoke/model/'preflight';old=historical_spec(directory,current)
            validate_frozen_files(root,dict(old,code_files={}))
            for crop in ('pan_and_scan','no_pan_and_scan'):verify_job(directory/crop,old)
            report['preflight']=json.loads((directory/'preflight_report.json').read_text())['settings']
            report['preflight_nonblocking_D6']=True
        except (OSError,ValueError,KeyError,TypeError) as exc:report['errors'].append('Gemma preflight: '+str(exc))
    if model=='llava':
        try:
            arms={}
            for current in probe_specs(root):
                directory=recovery/model/current['label']
                verify_historical(root,directory,current)
                arms[current['label'][6:-2]]=directory/'predictions_llava_node_description.jsonl'
            report['probe']=paired_report(root/'outputs/fullrun_s0b_inputs_2026-10-04/reading_probe50.jsonl',
                 root/'outputs'/DATA_NAME/'test/graph_metadata_0_500.jsonl',{a:arms[a] for a in ('uniform','legacy','white')},50)
            if not report['probe']['gate']['passed']:raise ValueError('50-graph paired basic recall drop exceeds .05')
        except (OSError,ValueError,KeyError,TypeError) as exc:report['errors'].append('Reading gate: '+str(exc))
    report['passed']=not report['errors']
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=(*MODELS,'all'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--smoke-root',type=Path);p.add_argument('--s0b-root',type=Path)
    p.add_argument('--repair-gemma-proof',action='store_true');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();smoke,recovery=evidence_roots(args.root,args.smoke_root,args.s0b_root)
    if args.repair_gemma_proof:
        from fullrun_session import write_resume_proof
        directory=recovery/'gemma/qa_image';old=json.loads((directory/'contract.json').read_text())
        write_resume_proof(directory,old)
        print('Recovered Gemma resume proof from byte-for-byte unchanged first three rows: '+str(directory/'resume_verified.json'))
    reports={m:model_gate(args.root,m,smoke,recovery) for m in MODELS if args.model=='all' or m==args.model}
    from verify_fullrun import print_table
    for report in reports.values():print_table(report)
    if args.output_dir:
        frozen_json(args.output_dir/'per_model_gate.json',reports)
    raise SystemExit(0 if all(r['passed'] for r in reports.values()) else 1)


if __name__=='__main__':main()
