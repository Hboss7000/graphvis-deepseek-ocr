#!/usr/bin/env python3
"""Measured smoke extrapolation plus explicitly capped 512-token rerun allowances."""
import argparse
import json
from pathlib import Path
from fullrun_common import *
from fullrun_gate import model_gate,evidence_roots


def estimates(root):
    smoke,recovery=evidence_roots(root);reports={}
    for model in MODELS:
        gate=model_gate(root,model,smoke,recovery)
        if len(gate['jobs'])!=5:raise ValueError('Missing measurements: '+model)
        jobs={j['job']:j for j in gate['jobs']};full=specs(root,model,False)
        strict_seconds=sum(jobs[s['label']]['seconds_per_item']*s['record_count'] for s in full)
        conditions={};rescue_seconds=0
        for condition in CONDITIONS:
            directory=root/jobs['qa_'+condition]['directory'];rows=read_rows(directory/'predictions.jsonl')
            generation_seconds=sum(r.get('generation_elapsed_seconds',r['item_elapsed_seconds']) for r in rows)
            tokens=sum(r['generated_token_count'] for r in rows)
            rate=tokens/generation_seconds
            ceilings=sum(r['hit_token_ceiling'] for r in rows)
            estimated_n=500*ceilings/len(rows)
            seconds=estimated_n*512/rate
            rescue_seconds+=seconds
            conditions[condition]={'smoke_n':len(rows),'ceilings':ceilings,'generation_tokens_per_second':rate,'rate_time_field':'generation_elapsed_seconds' if all('generation_elapsed_seconds' in r for r in rows) else 'item_elapsed_seconds (includes preprocessing)',
                                    'estimated_full_rescue_items':estimated_n,'capped_rerun_seconds':seconds}
        total=strict_seconds+rescue_seconds
        reports[model]={'stage1_seconds_per_item':jobs['stage1']['seconds_per_item'],
          'strict_inference_hours':strict_seconds/3600,'rescue_capped_hours':rescue_seconds/3600,
          'total_hours':total/3600,'cost_usd':total/3600*2.09,
          'with_50_percent_margin_hours':total/2400,'with_50_percent_margin_cost_usd':total/2400*2.09,
          'conditions':conditions,'gate_passed':gate['passed'],
          'assumptions':'Five-item QA sample ceiling rates extrapolated to 500 per condition. Every selected item reruns from scratch up to 512 tokens (not 448). Uses measured generation TPS, not model-loading time. Actual stops, prefill and loading overhead unknown; 50% margin is an allowance, not a guarantee.',
          'deepseek_ceiling_detection_approximate':model=='deepseek'}
    totals={key:sum(r[key] for r in reports.values()) for key in ('strict_inference_hours','rescue_capped_hours','total_hours','cost_usd','with_50_percent_margin_hours','with_50_percent_margin_cost_usd')}
    return {'models':reports,'total':totals,'gemma_startup_smoke_allowance_minutes':3,
            'note':'Gemma’s requested ~3 min startup smoke adds $0.1045 before margin ($0.15675 with margin); not a measured new run. Listed totals exclude this fixed allowance.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();report=estimates(args.root);frozen_json(args.output_dir/'estimates.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
