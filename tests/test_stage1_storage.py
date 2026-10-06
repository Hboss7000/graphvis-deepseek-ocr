import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage1_storage import check_free_space, InsufficientCheckpointSpace, retain_checkpoints
from stage1_weight_cache import inspect_cache, MODELS


def checkpoint(path, config, content=b'intact'):
    path.mkdir()
    (path / 'run_config.json').write_text(json.dumps(config))
    (path / 'projector.pt').write_bytes(content)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    (path / 'checkpoint_manifest.json').write_text(json.dumps(files))


def test_retention_keeps_latest_two_final_and_resume_until_new_save_verifies(tmp_path):
    config = {'optimizer_steps': 450}
    for step in (100, 200, 300):
        checkpoint(tmp_path / f'checkpoint-{step:06}', config)
    newest = tmp_path / 'checkpoint-000300'
    with (newest / 'projector.pt').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        retain_checkpoints(tmp_path, newest, config)
    assert (tmp_path / 'checkpoint-000100').exists()
    (newest / 'projector.pt').write_bytes(b'intact')
    deleted = retain_checkpoints(tmp_path, newest, config)
    assert [r['step'] for r in deleted] == [100]
    assert (tmp_path / 'checkpoint-000200').exists()
    checkpoint(tmp_path / 'checkpoint-000450', config)
    checkpoint(tmp_path / 'checkpoint-000500', config)
    checkpoint(tmp_path / 'checkpoint-000600', config)
    retain_checkpoints(tmp_path, tmp_path / 'checkpoint-000600', config)
    assert {p.name for p in tmp_path.glob('checkpoint-*')} == {'checkpoint-000450', 'checkpoint-000500', 'checkpoint-000600'}


def test_insufficient_space_stops_before_write(tmp_path, monkeypatch):
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: usage._replace(free=199))
    with pytest.raises(InsufficientCheckpointSpace, match='2 x checkpoint'):
        check_free_space(tmp_path, {'estimated_checkpoint_bytes': 100})
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: usage._replace(free=200))
    assert check_free_space(tmp_path, {'estimated_checkpoint_bytes': 100})['required_free_bytes'] == 200


def test_cache_checker_never_downloads_or_overwrites(tmp_path):
    import struct
    header = json.dumps({'a': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
    shard = struct.pack('<Q', len(header)) + header + b'x'
    model, revision = MODELS['llava']
    snapshot = tmp_path / 'hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    for name in ('config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json'):
        (snapshot / name).write_text('{}')
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
    (snapshot / 'one.safetensors').write_bytes(shard)
    assert inspect_cache(tmp_path, 'llava')['missing'] == ['two.safetensors']
    (snapshot / 'two.safetensors').write_bytes(shard)
    before = {p.name: p.read_bytes() for p in snapshot.iterdir()}
    report = inspect_cache(tmp_path, 'llava')
    assert report['complete'] and not report['cache_writes'] and not report['downloads']
    assert before == {p.name: p.read_bytes() for p in snapshot.iterdir()}
    (snapshot / 'two.safetensors').write_bytes(shard[:-1])
    assert not inspect_cache(tmp_path, 'llava')['complete']


def test_interpreter_uses_qwen_base_when_python312_absent(tmp_path):
    import os
    directory = tmp_path / 'bin'
    directory.mkdir()
    (directory / 'readlink').symlink_to('/usr/bin/readlink')
    qwen = tmp_path / 'venvs/venv_qwen/bin/python'
    qwen.parent.mkdir(parents=True)
    qwen.symlink_to(sys.executable)
    script = Path(__file__).resolve().parents[1] / 'scripts/stage1_train_interpreter.sh'
    result = subprocess.run(['/bin/bash',str(script),str(tmp_path)],env={**os.environ,'PATH':str(directory)},capture_output=True,text=True)
    assert result.returncode == 0
    assert Path(result.stdout.strip()) == Path(sys.executable).resolve()
