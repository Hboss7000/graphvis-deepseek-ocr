"""Checkpoint disk budgeting and retention shared by Stage 1 backbones."""
import json
from pathlib import Path
import re
import shutil
import time


class InsufficientCheckpointSpace(RuntimeError):
    """Pause before writing a checkpoint while retaining the last intact save."""


def tensor_bytes(value):
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return value.numel() * value.element_size()
    return 0


def checkpoint_budget(model, optimizer, projection, config):
    from peft import get_peft_model_state_dict
    adapter = tensor_bytes(get_peft_model_state_dict(model))
    bridge = tensor_bytes(projection.state_dict())
    state = tensor_bytes(optimizer.state_dict())
    # Include serialization/config/RNG overhead and round conservatively upward.
    raw = adapter + bridge + state
    expected = int(raw * 1.05) + 1024 * 1024 + len(json.dumps(config).encode()) * 2
    return {'adapter_bytes': adapter, 'projector_bytes': bridge,
            'optimizer_tensor_bytes': state, 'estimated_checkpoint_bytes': expected,
            'adapter_dtypes': sorted({str(p.dtype) for n, p in model.named_parameters() if 'lora_' in n}),
            'optimizer_dtypes': sorted({str(v.dtype) for state in optimizer.state.values()
                                       for v in state.values() if hasattr(v, 'dtype')})}


def check_free_space(parent, budget):
    parent = Path(parent)
    free = shutil.disk_usage(parent).free
    required = 2 * budget['estimated_checkpoint_bytes']
    if free < required:
        raise InsufficientCheckpointSpace(
            f'Checkpoint paused: free={free} bytes, required={required} bytes '
            '(2 x checkpoint budget). Last intact checkpoint retained; STOP the Pod.')
    return {**budget, 'free_bytes_before_save': free, 'required_free_bytes': required}


def retain_checkpoints(directory, current, config, keep=2):
    """Prune only this run's old saves after the newly published save verifies."""
    from llava_adapter import verify_checkpoint
    if keep < 2:
        raise ValueError('Retention must keep at least two intact checkpoints')
    directory, current = Path(directory).resolve(), Path(current).resolve()
    if current.parent != directory:
        raise ValueError('New checkpoint must belong to this output directory')
    verify_checkpoint(current)
    candidates = []
    for path in directory.iterdir():
        match = re.fullmatch(r'checkpoint-(\d{6})', path.name)
        if not match:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Checkpoint retention rejects symlinks/non-directories')
        verify_checkpoint(path)
        if json.loads((path / 'run_config.json').read_text()) != config:
            raise ValueError('Retention cannot prune a different run configuration')
        candidates.append((int(match[1]), path))
    candidates.sort()
    survivors = {p for _, p in candidates[-keep:]}
    survivors.add(current)
    survivors.update(p for step, p in candidates if step == config.get('optimizer_steps'))
    deleted = []
    for step, path in candidates:
        if path in survivors:
            continue
        manifest = json.loads((path / 'checkpoint_manifest.json').read_text())
        entry = {'time_ns': time.time_ns(), 'step': step, 'path': str(path),
                 'verified_newer_checkpoint': str(current), 'files_sha256': manifest,
                 'checkpoint_bytes': sum(p.stat().st_size for p in path.rglob('*') if p.is_file())}
        with (directory / 'checkpoint_retention.jsonl').open('a') as handle:
            handle.write(json.dumps(entry, sort_keys=True) + '\n')
        shutil.rmtree(path)
        deleted.append(entry)
    return deleted
