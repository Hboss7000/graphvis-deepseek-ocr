"""Verified adapter+projector loading shared by training smoke and inference."""
import hashlib
import json
import os
from pathlib import Path

from llava_stage1_training import file_sha, projector


def verify_checkpoint(source):
    source = Path(source)
    hashes = json.loads((source / 'checkpoint_manifest.json').read_text())
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual != set(hashes) | {'checkpoint_manifest.json'}:
        raise ValueError('Checkpoint has missing/unexpected files')
    for name, digest in hashes.items():
        if file_sha(source / name) != digest:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    return hashes


def adapter_provenance(source, model_id, revision):
    if os.environ.get('FULLRUN_CONTRACT'):
        raise ValueError('Adapter evaluation needs separate outputs outside the zero-shot fullrun contract')
    source = Path(source).resolve()
    hashes = verify_checkpoint(source)
    config = json.loads((source / 'run_config.json').read_text())
    if (config.get('model_id'), config.get('model_revision')) != (model_id, revision):
        raise ValueError('Adapter base model/revision differs from inference')
    required = ['projector.pt', 'run_config.json', 'adapter/adapter_config.json',
                'adapter/adapter_model.safetensors']
    if any(name not in hashes for name in required):
        raise ValueError('Missing adapter/projector artifact')
    digest = hashlib.sha256()
    artifact_files = sorted(name for name in hashes if name.startswith('adapter/') or name == 'projector.pt')
    for name in artifact_files:
        digest.update(name.encode() + b'\0' + hashes[name].encode() + b'\n')
    return {'path': str(source), 'sha256': digest.hexdigest(),
            'run_config_sha256': hashes['run_config.json'],
            'sha256_method': 'sorted adapter/projector relative paths and verified file SHA256 values',
            'files_sha256': {name: hashes[name] for name in artifact_files}}


def load_adapter(model, source, model_id, revision):
    from peft import PeftModel
    import torch
    provenance = adapter_provenance(source, model_id, revision)
    adapted = PeftModel.from_pretrained(model, Path(source) / 'adapter', is_trainable=False)
    projector(adapted).load_state_dict(torch.load(Path(source) / 'projector.pt',
                                                  map_location='cpu', weights_only=True), strict=True)
    adapted.requires_grad_(False)
    base = adapted.get_base_model()
    base.config.use_cache = True
    base.model.language_model.config.use_cache = True
    return adapted.eval(), provenance
