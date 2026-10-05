#!/usr/bin/env python3
"""Measured smoke extrapolation for strict-only full sessions."""
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
        reports[model]={'stage1_seconds_per_item':jobs['stage1']['seconds_per_item'],
          'strict_inference_hours':strict_seconds/3600,'total_hours':strict_seconds/3600,
          'cost_usd':strict_seconds/3600*2.09,
          'with_50_percent_margin_hours':strict_seconds/2400,'with_50_percent_margin_cost_usd':strict_seconds/2400*2.09,
          'gate_passed':gate['passed'],'rescue_included':False,
          'assumptions':'Measured s0b seconds/item extrapolated to 900 Stage 1 and 500 QA items per condition. Strict QA64 only; optional rescue is planned after the strict run from actual capped counts and measured speed.'}
    totals={key:sum(r[key] for r in reports.values()) for key in ('strict_inference_hours','total_hours','cost_usd','with_50_percent_margin_hours','with_50_percent_margin_cost_usd')}
    all_in={key:value + (.05 if key.endswith('hours') else .1045) * (1.5 if key.startswith('with_50') else 1) for key,value in totals.items()}
    return {'models':reports,'total':totals,'total_including_gemma_startup':all_in,'sessions':{'S1':{'models':['llava'],'expected_hours':reports['llava']['total_hours']},'S2':{'models':['qwen','deepseek'],'expected_hours':reports['qwen']['total_hours']+reports['deepseek']['total_hours']},'S3':{'models':['gemma'],'expected_hours':reports['gemma']['total_hours']+.05}},'gemma_startup_smoke_allowance_minutes':3,
            'note':'Gemma’s requested ~3 min startup smoke adds $0.1045 before margin ($0.15675 with margin); not a measured new run. Listed totals exclude this fixed allowance.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();report=estimates(args.root);frozen_json(args.output_dir/'estimates.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
