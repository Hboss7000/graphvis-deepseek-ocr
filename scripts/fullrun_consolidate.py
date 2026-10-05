#!/usr/bin/env python3
"""Strict-64 and optional extended QA tables, with a fixed all-four-model headline rule."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
from fullrun_common import *
from fullrun_rescue import verify_rescue
from verify_fullrun import verify_model


def parsed_rows(rows):
    result={}
    for row in rows:
        idx=row['statement_idx']
        if idx in result:raise ValueError('Duplicate QA index')
        letter,tier=parse_answer(row['raw_response'],4)
        result[idx]={**row,'predicted_option':None if letter=='FAILED' else letter,'parse_tier':tier,
                     'is_correct':letter==row['gold_option']}
    return result


def metrics(rows):
    n=len(rows);parsed=sum(r['predicted_option'] is not None for r in rows);correct=sum(r['is_correct'] for r in rows)
    return {'n':n,'correct':correct,'parsed':parsed,'accuracy':correct/n if n else None,
            'accuracy_among_parsed':correct/parsed if parsed else None,
            'parse_failure_rate':(n-parsed)/n if n else None,
            'ceiling_count':sum(r['hit_token_ceiling'] for r in rows),
            'ceiling_rate':sum(r['hit_token_ceiling'] for r in rows)/n if n else None}


def paired(image,text,indices=None):
    ids=set(image) if indices is None else set(indices)
    if set(image)!=set(text) or not ids<=set(image):raise ValueError('Paired coverage mismatch')
    better=sum(image[i]['is_correct'] and not text[i]['is_correct'] for i in ids)
    worse=sum(text[i]['is_correct'] and not image[i]['is_correct'] for i in ids)
    discordant=better+worse
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(better,worse)+1))/2**discordant) if discordant else 1.
    return {'n':len(ids),'image_minus_text_accuracy':(better-worse)/len(ids) if ids else None,
            'image_only_correct':better,'text_only_correct':worse,'mcnemar_exact_two_sided_p':p if ids else None,
            'test':'exact McNemar; no multiplicity correction; descriptive strata'}


def graph_strata(metadata):
    counts=sorted(len(r['visible_nodes']) for r in metadata.values())
    # Empirical nearest-rank quantiles; tied counts stay together, strata may be uneven.
    cuts=[counts[math.ceil(len(counts)*q)-1] for q in (1/3,2/3)]
    result={}
    for idx,meta in metadata.items():
        nodes=meta['visible_nodes'];gold=meta['answerKey']
        question={n['cid'] for n in nodes if n['in_question']}
        target={n['cid'] for n in nodes if gold in n['in_choices']}
        adjacency={n['cid']:set() for n in nodes}
        for edge in meta['edges']:
            # Direction-neutral reachability, matching the preparation diagnostics.
            a,b=edge['source_cid'],edge['target_cid'];adjacency[a].add(b);adjacency[b].add(a)
        seen=set(question);queue=deque(question)
        while queue:
            for node in adjacency[queue.popleft()]-seen:seen.add(node);queue.append(node)
        count=len(nodes)
        result[idx]={'reachable':bool(target & seen),'node_count':count,
                     'node_tercile':'low' if count<=cuts[0] else ('middle' if count<=cuts[1] else 'high')}
    return result,{'cutpoints':cuts,'rule':'nearest-rank 1/3 and 2/3 across all 500 metadata graphs; ties stay together',
                   'reachability':'undirected visible graph; any correct-option node reachable from any question node; overlap counts as reachable'}


def rescue_state(root, model):
    base=root/'outputs'/(DATA_NAME+'_results')/model/'rescue_512'
    if not (base/'prefix_report.json').is_file():
        return {'complete':False,'reason':'Optional rescue not completed','audit':None}
    audit=verify_rescue(root,model)
    complete=set(audit)==set(CONDITIONS) and all(r['checked']==r['selected'] and r['mismatches']==0 for r in audit.values())
    return {'complete':complete,'reason':None if complete else 'Prefix mismatches require review','audit':audit}


def headline_rule(root):
    states={m:rescue_state(root,m) for m in MODELS}
    return ('extended_512' if all(s['complete'] for s in states.values()) else 'strict_64'),states


def consolidate_model(root,model,expected_count=500):
    if expected_count==500:
        verified=verify_model(root,model)
        if not verified['passed']:raise ValueError('Full strict verification failed: '+str(verified['errors']))
    primary,states=headline_rule(root)
    state=states[model]
    audit=state["audit"]
    base=root/'outputs'/(DATA_NAME+'_results')/model
    versions={'strict_64':{},'extended_512':{}}
    overridden_jobs = []
    for condition in CONDITIONS:
        config_path = base / ('qa_' + condition) / 'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text()).get('tripwire_overridden', False):
            overridden_jobs.append('qa_' + condition)
    for condition in CONDITIONS:
        original=parsed_rows(read_rows(base/('qa_'+condition)/'predictions.jsonl'))
        if set(original)!=set(range(expected_count)):raise ValueError('Expected exact QA coverage')
        replacements=parsed_rows(read_rows(base/'rescue_512'/condition/'predictions.jsonl')) if state['complete'] else {}
        wanted={i for i,r in original.items() if r['hit_token_ceiling']}
        if state['complete'] and set(replacements)!=wanted:raise ValueError('Extended replacements must equal all capped strict items')
        versions['strict_64'][condition]=original
        versions['extended_512'][condition]={i:replacements.get(i,r) for i,r in original.items()} if state['complete'] else {}
    metadata={r['statement_idx']:r for r in read_rows(root/'outputs'/DATA_NAME/'test/graph_metadata_0_500.jsonl')}
    strata,definitions=graph_strata(metadata)
    report={'model':model,'primary':primary,
            'tripwire_overridden': bool(overridden_jobs), 'tripwire_overridden_jobs': overridden_jobs,
            'strict_scoring': 'All answers in denominator; unparsed answers count as wrong. Parsed-only accuracy is secondary.','headline_rule':'extended only when all four models have completed verified rescues; otherwise strict-64',
            'all_model_rescue_states':states,'extended_available':state['complete'],
            'extended_note':None if state['complete'] else 'Unavailable: no complete verified rescue for this model; extended metrics are null, never copied from strict results.', 'parser':'explicit_v1','rescue_audit':audit,
            'deepseek_ceiling_detection_approximate':model=='deepseek','strata_definitions':definitions,'versions':{}}
    for label,conditions in versions.items():
        value={'available':label=='strict_64' or state['complete'],'conditions':{c:metrics(list(rows.values())) for c,rows in conditions.items()},
               'paired_image_vs_text':{c:paired(conditions['image'],conditions[c]) for c in CONDITIONS if c!='image'},'strata':{}}
        for condition, condition_metrics in value['conditions'].items():
            condition_metrics['tripwire_overridden'] = label == 'strict_64' and 'qa_' + condition in overridden_jobs
            if label == 'extended_512':
                condition_metrics['strict_source_tripwire_overridden'] = 'qa_' + condition in overridden_jobs
        for kind in ('reachable','node_tercile'):
            for group in ((False,True) if kind=='reachable' else ('low','middle','high')):
                indices={i for i in conditions['image'] if strata[i][kind]==group}
                value['strata'][kind+'='+str(group)]={
                    'conditions':{c:metrics([rows[i] for i in indices]) for c,rows in conditions.items()},
                    'paired_image_vs_text':{c:paired(conditions['image'],conditions[c],indices) for c in CONDITIONS if c!='image'}}
        report['versions'][label]=value
    return report,versions


def print_spot_check(root,n=30):
    """Round-robin over every available model/condition; prefer actual rescued outputs."""
    buckets=[]
    for model in MODELS:
        base=root/'outputs'/(DATA_NAME+'_results')/model
        for condition in CONDITIONS:
            strict=read_rows(base/('qa_'+condition)/'predictions.jsonl')
            rescue=read_rows(base/'rescue_512'/condition/'predictions.jsonl')
            if not strict:continue
            indexed=parsed_rows(strict);new=parsed_rows(rescue)
            # Rescued rows first, then remaining extended rows for diversity.
            rows=list(new.values())+[r for i,r in indexed.items() if i not in new]
            buckets.append((model,condition,rows))
    emitted=0;position=0
    print('EXTENDED ANSWER SPOT CHECK (up to 30; mixes available models and conditions):',flush=True)
    while emitted<n and any(position<len(rows) for _,_,rows in buckets):
        for model,condition,rows in buckets:
            if emitted>=n:break
            if position>=len(rows):continue
            row=rows[position];emitted+=1
            print(json.dumps({'spot':emitted,'model':model,'condition':condition,'statement_idx':row['statement_idx'],
                  'parsed_letter':row['predicted_option'],'raw_response':row['raw_response']},ensure_ascii=False),flush=True)
        position+=1
    if not buckets:print('No full-run predictions available; no fabricated examples.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=(*MODELS,'all'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    for model in MODELS if args.model=='all' else [args.model]:
        report,versions=consolidate_model(args.root,model)
        frozen_json(args.output_dir/model/'summary.json',report)
        for label,conditions in versions.items():
            for condition,rows in conditions.items():
                path=args.output_dir/model/label/(condition+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
                data=''.join(json.dumps(row,ensure_ascii=False)+'\n' for _,row in sorted(rows.items()))
                if path.exists() and path.read_text()!=data:raise ValueError('Consolidation output collision')
                if not path.exists():path.write_text(data)
        print(json.dumps(report,indent=2))
    print_spot_check(args.root)


if __name__=='__main__':main()
