#!/usr/bin/env python3
"""CPU LLaVA-NeXT expanded image + chat prompt + gold/EOS lengths, no model weights."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from fullrun_common import *
from llava_common import configure_processor,format_prompt,image_token_id
from stage1_common import raw_image_prompt


def percentiles(values):
    values=sorted(values)
    def percentile(q):
        position=(len(values)-1)*q;lo=math.floor(position);hi=math.ceil(position)
        return values[lo]*(hi-position)+values[hi]*(position-lo) if hi!=lo else values[lo]
    return {str(p):percentile(p/100) for p in (0,25,50,75,90,95,99,100)}


def count_lengths(root,processor,config):
    from PIL import Image
    configure_processor(processor,config)
    rows=ordered_stage(read_rows(root/'outputs'/DATA_NAME/'test/stage1_subset100.jsonl'))
    if len(rows)!=900:raise ValueError('Expected all 900 Stage 1 records')
    image_counts={};results=[];by_task=defaultdict(list)
    for row in rows:
        prompt=format_prompt(processor,raw_image_prompt(row,'none'),'image')
        complete=prompt+row['answer']+processor.tokenizer.eos_token
        plain=processor.tokenizer(complete,add_special_tokens=True)['input_ids']
        if plain.count(image_token_id(processor))!=1:raise ValueError('Expected one unexpanded image marker')
        idx=row['statement_idx']
        if idx not in image_counts:
            with Image.open(root/'outputs'/DATA_NAME/row['image']) as opened:
                inputs=processor(text=complete,images=opened.convert('RGB'),return_tensors='np')
            ids=inputs['input_ids'][0].tolist()
            expanded=sum(i==image_token_id(processor) for i in ids)
            if len(ids)!=len(plain)+expanded-1:raise ValueError('Processor expansion differs from exact tokenizer + image replacement')
            image_counts[idx]=expanded
        total=len(plain)-1+image_counts[idx]
        item={'statement_idx':idx,'task_type':row['task_type'],'image_tokens':image_counts[idx],
              'prompt_gold_eos_tokens_excluding_image':len(plain)-1,'combined_tokens':total,'exceeds_2048':total>2048}
        results.append(item);by_task[row['task_type']].append(total)
    values=[r['combined_tokens'] for r in results]
    return {'model_id':MODELS['llava'][0],'revision':MODELS['llava'][1], 'n':len(rows),'graphs':len(image_counts),
            'percentiles':percentiles(values),'maximum':max(values),'exceed_2048':sum(v>2048 for v in values),
            'per_task':{k:{'percentiles':percentiles(v),'maximum':max(v),'exceed_2048':sum(x>2048 for x in v)} for k,v in by_task.items()},
            'method':'Exact pinned LLaVA-NeXT processor image expansion; tokenizer chat generation prompt + gold + EOS. Each graph processed on CPU once; replacement identity checked against full processor output. No truncation.',
            'limitation':'This is the current NeXT inference chat/anyres path. GraphVis original LLaVA training may use a different image encoder/collator; validate its exact training path before fine-tuning.',
            'source_sha256':sha(root/'outputs'/DATA_NAME/'test/stage1_subset100.jsonl'),'items':results}


def main():
    import transformers,torch
    from transformers import AutoProcessor,AutoConfig
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    if transformers.__version__!='5.16.1':raise ValueError('Pinned processor requires transformers==5.16.1')
    model,revision=MODELS['llava']
    processor=AutoProcessor.from_pretrained(model,revision=revision,local_files_only=args.offline)
    config=AutoConfig.from_pretrained(model,revision=revision,local_files_only=args.offline)
    report=count_lengths(args.root,processor,config)
    report['versions']={'transformers':transformers.__version__,'torch':torch.__version__,'device':'CPU; processor/tokenizer only'}
    frozen_json(args.output_dir/'stage1_lengths.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='items'},indent=2))


if __name__=='__main__':main()
