"""Read-only completeness check of pinned safetensors snapshots; never download."""
import argparse
import json
from pathlib import Path
import struct

MODELS = {
    'llava': ('llava-hf/llava-v1.6-mistral-7b-hf', '2424fdd47412fccc66d91719126b420e9fbd7065'),
    'qwen': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
}


def inspect_cache(cache, backbone):
    model_id, revision = MODELS[backbone]
    snapshot = Path(cache) / 'hub' / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json'}
    if backbone == 'qwen':
        required.add('chat_template.json')
    index = snapshot / 'model.safetensors.index.json'
    if index.is_file():
        data = json.loads(index.read_text())
        required.add(index.name)
        weights = set(data['weight_map'].values())
        if any(Path(name).name != name or not name.endswith('.safetensors') for name in weights):
            raise ValueError('Invalid cached weight index')
        required.update(weights)
    else:
        required.add('model.safetensors')
    missing = sorted(name for name in required if not (snapshot / name).is_file()
                     or (snapshot / name).stat().st_size == 0)
    for name in sorted(required):
        path = snapshot / name
        if not name.endswith('.safetensors') or name in missing:
            continue
        try:
            with path.open('rb') as handle:
                length = struct.unpack('<Q', handle.read(8))[0]
                if not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError('Invalid safetensors header length')
                header = json.loads(handle.read(length))
            end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
            if path.stat().st_size != 8 + length + end:
                raise ValueError('Truncated/inconsistent shard size')
        except (ValueError, KeyError, TypeError, struct.error):
            missing.append(name + ' (invalid/truncated safetensors shard)')
    return {'backbone': backbone, 'model_id': model_id, 'revision': revision,
            'snapshot': str(snapshot), 'complete': not missing, 'missing': missing,
            'files_bytes': {name: (snapshot / name).stat().st_size for name in sorted(required)
                            if (snapshot / name).is_file()},
            'downloads': False, 'cache_writes': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('/workspace/.cache/huggingface'))
    p.add_argument('--backbone', choices=MODELS, default='llava')
    args = p.parse_args()
    report = inspect_cache(args.cache, args.backbone)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
