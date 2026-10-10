#!/usr/bin/env python3
"""Compare audit_v2 with accepted sealed Qwen Stage 1 claims; no new readings."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SCORER_DIR=ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0,str(SCORER_DIR))
import score_stage1_audit_v2 as scorer


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(task,value):
    if value is None:return None
    if task=='shortest_path_listing' and value=='NO_PATH':return []
    if task in ('node_description','neighbor_listing'):return sorted(set(value))
    if task=='triple_listing':return sorted(set(tuple(t) for t in value))
    if task=='relation_identification':return scorer.base.relation_key(value)
    return value


def historical_claim(task,row):
    if task in scorer.base.NUMERIC_TASKS:return row['parsed_integer']
    if task=='highest_node_degree':
        return [row['parsed_name'],row['parsed_degree']] if row['parsed_name'] is not None and row['parsed_degree'] is not None else None
    if task=='relation_identification':return row['parsed_relation']
    if task=='shortest_path_listing':return row['parsed_path']
    if task=='triple_listing':return row['parsed_predicted_triples']
    return row['parsed_predicted_nodes']


def validate(args):
    if args.output.exists():raise FileExistsError(f'Output must be new: {args.output}')
    counts=defaultdict(Counter);mismatches=[];excluded=[];decision_disagreements=[]
    before=sha(args.adjudications)
    for line in args.adjudications.open(encoding='utf-8'):
        row=json.loads(line)
        if row.get('model')!='qwen' or row.get('family')!='stage1':continue
        if not row.get('accepted'):
            excluded.append({k:row[k] for k in ('audit_id','arm','task','statement_idx')});continue
        task=row['task'];parsed=scorer.extract(task,row['raw_response'])
        expected=canonical(task,row['claim']);actual=canonical(task,parsed.value)
        old=canonical(task,historical_claim(task,row['scores']['template_v1']))
        # Historical list parsers represent an absent claim as an empty list.
        if expected is None and old==[]:old=None
        agree=actual==expected;previous=old==expected
        for key in ((row['arm'],task),('all',task)):
            counts[key].update(n=1,audit_v2_agree=int(agree),template_v1_agree=int(previous),
                               fixed=int(agree and not previous),regressed=int(previous and not agree))
        identity={k:row[k] for k in ('audit_id','arm','task','statement_idx')}
        if parsed.decision!=row['decision']:
            decision_disagreements.append({**identity,'reader_decision':row['decision'],'parser_decision':parsed.decision})
        if not agree:
            mismatches.append({**identity,'reader_claim':row['claim'],'parser_claim':parsed.value,
                'reader_decision':row['decision'],'parser_decision':parsed.decision,
                'parser_rule':parsed.rule,'raw_response':row['raw_response']})
    if before!=sha(args.adjudications):raise RuntimeError('Sealed evidence changed during validation')
    frozen=json.loads((args.adjudications.parent/'inputs_frozen.json').read_text())['inputs']
    exceptions=json.loads((args.adjudications.parent/'authorized_exceptions.json').read_text())
    for entry in frozen:
        path=ROOT/entry['path'];actual=sha(path)
        expected=exceptions.get(entry['path'],{}).get('authorized_current_sha256',entry['sha256'])
        if actual!=expected:raise ValueError(f'Frozen evidence changed: {path}')
    summary={'scope':'Accepted sealed Qwen Stage 1 claims; development regression evidence, not held-out validation.',
             'accepted_records':sum(v['n'] for (arm,_),v in counts.items() if arm=='all'),
             'claim_disagreements':len(mismatches),'decision_disagreements':len(decision_disagreements),
             'unresolved_records_excluded':len(excluded),'frozen_input_hashes_verified':len(frozen),
             'path_only_authorized_exceptions':len(exceptions),
             'global_audit_validation':'Prior LLaVA failure remains; this report is scoped to code regression and input integrity.',
             'adjudications_sha256':before,
             'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(scorer.__file__),Path(scorer.base.__file__),Path(scorer.template.__file__),Path(__file__))}}
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in (('validation.json',summary),('claim_disagreements.json',mismatches),('decision_disagreements.json',decision_disagreements),('unresolved_excluded.json',excluded)):
        (args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
    rows=[{'arm':arm,'task':task,**dict(count),'audit_v2_agreement':count['audit_v2_agree']/count['n'],'template_v1_agreement':count['template_v1_agree']/count['n']} for (arm,task),count in sorted(counts.items())]
    with (args.output/'claim_agreement.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjudications',type=Path,default=ROOT/'outputs/scorer_audit_continued_2026-10-09/adjudications.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate(args),indent=2))


if __name__=='__main__':main()
