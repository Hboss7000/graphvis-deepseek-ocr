#!/usr/bin/env python3
"""CPU pilot metrics from unchanged span_extended scoring, plus raw examples."""
import argparse
import json
from pathlib import Path
from llava_stage1_jobs import ROOT, NAME, NUMERIC_TASKS, ARMS, load_manifest, verify_arm, estimate
from fullrun_common import read_rows
from score_stage1 import score_record, gold_integer


def summarize_task(records, rows, metadata, task):
    lookup={(r['statement_idx'],r['task_type']):r for r in records}
    scores=[]
    for row in rows:
        key=(row['statement_idx'],row['task_type'])
        scored=score_record(lookup[key],row['raw_response'],metadata[row['statement_idx']],extractor='span_extended')
        parsed=scored.get('parsed_integer') if task!='highest_node_degree' else scored.get('parsed_degree')
        gold=gold_integer(task,lookup[key]['answer'])
        scores.append({'parsed':parsed,'correct':parsed==gold,'containment':scored['lenient_containment'],
                       'error':abs(parsed-gold) if parsed is not None else None})
    n=len(rows);errors=[s['error'] for s in scores if s['error'] is not None]
    return {'n':n,'answer_rate':sum(s['parsed'] is not None for s in scores)/n if n else None,
            'exact_accuracy':sum(s['correct'] for s in scores)/n if n else None,
            'lenient_containment':sum(float(s['containment']) for s in scores)/n if n else None,
            'mean_absolute_error':sum(errors)/len(errors) if errors else None,
            'mean_absolute_error_n':len(errors),
            'ceiling_rate':sum(r['hit_token_ceiling'] for r in rows)/n if n else None,
            'raw_answers':[{'statement_idx':r['statement_idx'],'raw_response':r['raw_response']} for r in rows[:3]]}


def summary(root,phase='pilot'):
    manifest=load_manifest(root)
    records=read_rows(root/manifest['phases'][phase]['input_jsonl'])
    metadata={r['statement_idx']:r for r in read_rows(root/manifest['graph_metadata'])}
    report={'phase':phase,'arms':{},'notes':{
        'exact_accuracy':'Parsed integer equals gold; highest_node_degree reports degree accuracy (name accuracy remains in the unchanged scorer).',
        'lenient_containment':'Unchanged scorer diagnostic; not exact accuracy.',
        'mae_denominator':'Parsed answers only; unparsed answers remain wrong for exact accuracy.'}}
    for arm in ARMS:
        rows=verify_arm(root,manifest,phase,arm)
        report['arms'][arm]={'tasks':{t:summarize_task(records,[r for r in rows if r['task_type']==t],metadata,t)
                                    for t in NUMERIC_TASKS},
            'peak_vram_bytes':max(r['peak_memory_allocated_bytes'] for r in rows),
            'mean_seconds_per_item':sum(r['item_elapsed_seconds'] for r in rows)/len(rows)}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--phase',choices=['pilot-dry','pilot'],default='pilot')
    args=parser.parse_args();report=summary(args.root,args.phase)
    print('arm task                    n answer-rate exact-accuracy containment MAE(parsed) ceiling-rate')
    for arm,data in report['arms'].items():
        print(f"{arm}: peak VRAM={data['peak_vram_bytes']/1024**3:.2f} GiB; seconds/item={data['mean_seconds_per_item']:.3f}")
        for task,m in data['tasks'].items():
            mae='n/a' if m['mean_absolute_error'] is None else f"{m['mean_absolute_error']:.3f}"
            print(f"{arm}  {task:23} {m['n']:3} {m['answer_rate']:.3f}       {m['exact_accuracy']:.3f}          {m['lenient_containment']:.3f}       {mae:>6}       {m['ceiling_rate']:.3f}")
            for example in m['raw_answers']:print('  '+json.dumps(example,ensure_ascii=False))
    print('Notes: '+json.dumps(report['notes']))


if __name__=='__main__':main()
