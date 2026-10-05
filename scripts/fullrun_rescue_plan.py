#!/usr/bin/env python3
"""CPU-only rescue planning from fetched strict predictions, without loading models."""
import argparse
import json
import math
from pathlib import Path
from fullrun_common import CONDITIONS, MODELS, read_rows


def plan(model, directory, mean_tokens=None):
    conditions={};examples=[]
    for condition in CONDITIONS:
        path=directory/('qa_'+condition)/'predictions.jsonl'
        if not path.is_file():raise ValueError('Missing strict predictions: '+str(path))
        rows=read_rows(path)
        if not rows or len({r['statement_idx'] for r in rows})!=len(rows):
            raise ValueError('Empty/duplicate strict predictions: '+str(path))
        capped=[r for r in rows if r['hit_token_ceiling']]
        field='generation_elapsed_seconds' if all('generation_elapsed_seconds' in r for r in rows) else 'item_elapsed_seconds'
        seconds=sum(r[field] for r in rows)
        tokens=sum(len(r['generated_token_ids']) if isinstance(r.get('generated_token_ids'),list) else r['generated_token_count'] for r in rows)
        if not math.isfinite(seconds) or seconds<=0 or tokens<=0:raise ValueError('Invalid measured speed')
        speed=tokens/seconds
        # Censored strict answers reveal no actual uncapped completion length.
        # Use observed rerun work as a transparent lower bound, not 512 per item.
        observed=sum(len(r['generated_token_ids']) if isinstance(r.get('generated_token_ids'),list) else r['generated_token_count'] for r in capped)
        estimate_tokens=len(capped)*mean_tokens if mean_tokens is not None else observed
        hours=estimate_tokens/speed/3600
        conditions[condition]={'strict_n':len(rows),'capped_answers':len(capped),'measured_tokens_per_second':speed,
          'measurement_time_field':field,'observed_capped_tokens':observed,
          'assumed_mean_rescue_tokens':mean_tokens,'estimated_rescue_hours':hours,'estimated_rescue_cost_usd':hours*2.09,
          'estimate_kind':'supplied mean length at measured speed' if mean_tokens is not None else 'observed token workload lower bound',
          'observed_workload_lower_bound_hours':observed/speed/3600}
        examples.extend({'model':model,'condition':condition,'statement_idx':r['statement_idx'],'response_first_200':r['raw_response'][:200]} for r in capped)
    # Spread the ten examples across conditions, rather than only showing image.
    buckets=[[r for r in examples if r['condition']==c] for c in CONDITIONS]
    sample=[]
    while len(sample)<10 and any(buckets):
        for bucket in buckets:
            if bucket and len(sample)<10:sample.append(bucket.pop(0))
    return {'model':model,'conditions':conditions,'estimated_rescue_hours':sum(c['estimated_rescue_hours'] for c in conditions.values()),
      'estimated_rescue_cost_usd':sum(c['estimated_rescue_cost_usd'] for c in conditions.values()),'capped_examples':sample,
      'deepseek_ceiling_detection_approximate':model=='deepseek',
      'note':'Strict-64 censors completion lengths. Default reports measured replay workload as a lower bound, not a prediction of 512 completion time. Supply --mean-rescue-tokens for a length assumption (64–512). Loading, prefill changes and fetch overhead are not included; item_elapsed_seconds includes preprocessing when generation timing is unavailable.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=MODELS)
    p.add_argument('--model-dir',type=Path,required=True,help='Fetched model directory containing qa_image etc.')
    p.add_argument('--mean-rescue-tokens',type=float);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mean_rescue_tokens is not None and (not math.isfinite(a.mean_rescue_tokens) or not 64<=a.mean_rescue_tokens<=512):p.error('Mean length must be between 64 and 512')
    report=plan(a.model,a.model_dir,a.mean_rescue_tokens)
    data=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:f.write(data)
    print(data,end='')


if __name__=='__main__':main()
