#!/usr/bin/env python3
"""CPU-only validation and frozen three-arm reading input for session 0b."""
import json
from fullrun_common import *
from PIL import Image


def prepare(root=ROOT):
    out=root/'outputs/fullrun_s0b_inputs_2026-10-04'
    arms={'uniform':'outputs/'+DATA_NAME,'legacy':'outputs/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_tb30',
          'white':'outputs/'+DATA_NAME+'_uniform_white50'}
    legacy=root/arms['legacy'];white=root/arms['white']
    for name in ('stage1_graph_comprehension_subset50_seed13.jsonl','stage2_obqa_subset50_seed13.jsonl','graph_metadata_subset50_seed13.jsonl'):
        if (legacy/'test'/name).read_bytes() != (white/'test'/name).read_bytes():
            raise ValueError('White rendering changed records or metadata: '+name)
    source=legacy/'test/stage1_graph_comprehension_subset50_seed13.jsonl'
    data=b''.join(line+b'\n' for line in source.read_bytes().splitlines() if json.loads(line)['task_type']=='node_description')
    rows=[json.loads(line) for line in data.splitlines()]
    if len(rows)!=50:raise ValueError('Expected all 50 original graphs')
    out.mkdir(parents=True,exist_ok=True);p=out/'reading_probe50.jsonl'
    if p.exists() and p.read_bytes()!=data:raise ValueError('Reading probe input collision')
    if not p.exists():
        with p.open('xb') as f:f.write(data)
    files={str(p.relative_to(root)):sha(p)};sizes={}
    for label,name in arms.items():
        arm=root/name
        meta_path=arm/('test/graph_metadata_0_500.jsonl' if label=='uniform' else 'test/graph_metadata_subset50_seed13.jsonl')
        meta={r['statement_idx']:r for r in read_rows(meta_path)};files[str(meta_path.relative_to(root))]=sha(meta_path)
        for row in rows:
            idx=row['statement_idx'];path=arm/meta[idx]['image']
            with Image.open(path) as image:size=image.size
            if idx in sizes and sizes[idx]!=size:raise ValueError(f'Changed geometry: {label}/{idx}')
            sizes[idx]=size;files[str(path.relative_to(root))]=sha(path)
    frozen_json(out/'manifest.json',{'record_count':50,'records':str(p.relative_to(root)),'files':files,'arms':arms,
        'change':'white: node-fill #FFFFFF only; stage1, stage2, metadata byte-identical to legacy; solid outlines preserved',
        'renderer_sha256':sha(root/'scripts/generate_graphvis_datasets.py')})
    print('Verified 50 graphs × 3 arms; identical dimensions/records/metadata; frozen '+str(out))


if __name__=='__main__':prepare()
