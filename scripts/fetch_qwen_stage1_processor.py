#!/usr/bin/env python3
"""Fetch ONLY pinned processor/tokenizer metadata into a new laptop directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'experiments/2026-10-06_qwen_stage1_training/processor_assets_manifest.json'


def fetch(destination):
    manifest=json.loads(MANIFEST.read_text())
    destination.mkdir(parents=True,exist_ok=True)
    for name,digest in manifest['files_sha256'].items():
        if '/' in name or name.endswith('.safetensors') or '.bin' in name:
            raise ValueError('Only flat processor assets are permitted')
        target=destination/name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                raise ValueError('Existing asset differs; refusing overwrite: '+str(target))
            continue
        url=f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}/{name}"
        data=urllib.request.urlopen(url,timeout=90).read()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('Pinned asset checksum differs: '+name)
        with target.open('xb') as handle:handle.write(data)
        print(name,len(data),flush=True)
    target=destination/'processor_assets_manifest.json'
    encoded=MANIFEST.read_bytes()
    if target.exists():
        if target.read_bytes()!=encoded:raise ValueError('Existing manifest differs')
    else:
        with target.open('xb') as handle:handle.write(encoded)
    print('Processor assets verified; no weights or existing Hugging Face cache touched.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    fetch(p.parse_args().output_dir)
