from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fullrun_common as common
import fullrun_session as session
import fullrun_runtime as runtime
import verify_fullrun as verifier


def spec(model='llava', stage='qa', n=5):
    return dict(model=model, stage=stage, label='qa_image' if stage == 'qa' else 'stage1',
                condition='image', mode='smoke', model_id=common.MODELS[model][0],
                model_revision=common.MODELS[model][1], effective_max_new_tokens=64 if stage == 'qa' else 1024,
                versions={'torch': '2.8.0+cu128', 'transformers': '4.46.3' if model == 'deepseek' else '5.16.1'},
                prompt_bodies_sha256='a' * 64, record_count=n,
                input_files={},
                expected_keys=[[i, 'obqa_answer' if stage == 'qa' else 'node_description'] for i in range(n)],
                seconds_per_item_reference=common.SECONDS[model])


def row(s, i=0):
    return dict(statement_idx=i, task_type='node_description', raw_response='A',
                model_id=s['model_id'], model_revision=s['model_revision'],
                generated_token_count=2, hit_token_ceiling=False,
                item_elapsed_seconds=1.0, peak_memory_allocated_bytes=1024,
                parse_tier='exact', predicted_option='A', timestamp_utc=str(i))


def config(s):
    return {**{k: s[k] for k in ('model_id', 'model_revision', 'effective_max_new_tokens', 'prompt_bodies_sha256')},
            'generation': dict(do_sample=False, num_beams=1, max_new_tokens=s['effective_max_new_tokens']),
            'torch_version': s['versions']['torch'], 'transformers_version': s['versions']['transformers'],
            'fullrun': s}


def write_output(directory, s):
    directory.mkdir(parents=True, exist_ok=True)
    common.frozen_json(directory / 'contract.json', s)
    common.frozen_json(directory / 'run_config.json', config(s))
    path = common.prediction_files(directory, s)[0]
    path.write_text(''.join(json.dumps(row(s, i)) + '\n' for i in range(s['record_count'])))
    return path


def test_verifier_good_and_bad(tmp_path):
    s = spec()
    path = write_output(tmp_path, s)
    assert common.verify_job(tmp_path, s)['rows'] == 5
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        common.verify_job(tmp_path, s)
    path.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
    with pytest.raises(ValueError, match='Missing predictions'):
        common.verify_job(tmp_path, s)
    path.write_text(original)
    bad = config(s)
    bad['effective_max_new_tokens'] = 8192
    (tmp_path / 'run_config.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='effective_max_new_tokens'):
        common.verify_job(tmp_path, s)
    bad = config(s)
    bad['prompt_bodies_sha256'] = 'b' * 64
    (tmp_path / 'run_config.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='prompt_bodies_sha256'):
        common.verify_job(tmp_path, s)


def test_resume_rejects_collisions_and_partial_lines(tmp_path):
    s = spec()
    (tmp_path / 'unrelated').write_text('preserve')
    with pytest.raises(ValueError, match='collision'):
        session.check_resume(tmp_path, s)
    common.frozen_json(tmp_path / 'contract.json', s)
    path = tmp_path / 'predictions.jsonl'
    path.write_text('{"statement_idx":')
    assert common.read_rows(path, growing=True) == []
    with pytest.raises(ValueError, match='Incomplete'):
        session.check_resume(tmp_path, s)


def test_tripwire_boundaries():
    s = spec(n=20)
    rows = [row(s, i) for i in range(20)]
    for r in rows[:10]:
        r['raw_response'] = ''
    assert common.tripwire(rows, 'qa', 3.5) is None
    rows[10]['raw_response'] = ''
    assert '50%' in common.tripwire(rows, 'qa', 3.5)
    assert common.tripwire(rows[:19], 'qa', 3.5) is None
    for r in rows:
        r.update(raw_response='two words', generated_token_count=2)
    for r in rows[:16]:
        r['generated_token_count'] = 1
    assert common.tripwire(rows, 'stage1', 3.5) is None
    rows[16]['generated_token_count'] = 1
    assert '80%' in common.tripwire(rows, 'stage1', 3.5)
    for r in rows:
        r.update(generated_token_count=2, item_elapsed_seconds=10.5)
    assert common.tripwire(rows, 'stage1', 3.5) is None
    rows[0]['item_elapsed_seconds'] = 11
    assert '3x' in common.tripwire(rows, 'stage1', 3.5)


STUB = '''import json, os, pathlib, sys, time
s=json.loads(pathlib.Path(os.environ['FULLRUN_CONTRACT']).read_text())
out=pathlib.Path(os.environ['FULLRUN_CONTRACT']).parent
if sys.argv[1] == 'stall':
    print('console chatter is not prediction progress', flush=True)
    time.sleep(30)
    sys.exit(0)
if sys.argv[1] == 'fail': sys.exit(7)
limit=int(sys.argv[1])
p=out/'predictions.jsonl'
done={json.loads(x)['statement_idx'] for x in p.read_text().splitlines()} if p.exists() else set()
with p.open('a') as f:
    for i in range(limit):
        if i in done: continue
        r=dict(statement_idx=i,raw_response='A', model_id=s['model_id'], model_revision=s['model_revision'],generated_token_count=2,hit_token_ceiling=False,item_elapsed_seconds=1.,peak_memory_allocated_bytes=1024,parse_tier='exact',predicted_option='A',timestamp_utc=str(i))
        f.write(json.dumps(r)+'\\n'); f.flush()
'''


@pytest.mark.parametrize('model', common.MODELS)
def test_real_watchdog_with_stub_resume_three_to_five(tmp_path, model):
    s = spec(model)
    common.frozen_json(tmp_path / 'contract.json', s)
    common.frozen_json(tmp_path / 'run_config.json', config(s))
    cmd = [sys.executable, '-c', STUB]
    first = session.watch_job(cmd + ['3'], tmp_path, s, tmp_path / 'job.log', time.monotonic() + 10, poll_seconds=.01)
    assert first['state'] == 'COMPLETED'
    old = (tmp_path / 'predictions.jsonl').read_bytes()
    assert len(common.prediction_rows(tmp_path, s)) == 3
    session.check_resume(tmp_path, s)
    second = session.watch_job(cmd + ['5'], tmp_path, s, tmp_path / 'job.log', time.monotonic() + 10, poll_seconds=.01)
    assert second['state'] == 'COMPLETED'
    assert (tmp_path / 'predictions.jsonl').read_bytes().startswith(old)
    assert common.verify_job(tmp_path, s)['rows'] == 5


def test_watchdog_stall_and_time_guard(tmp_path):
    s = spec()
    common.frozen_json(tmp_path / 'contract.json', s)
    stalled = session.watch_job([sys.executable, '-c', STUB, 'stall'], tmp_path, s,
                               tmp_path / 'log', time.monotonic() + 10, poll_seconds=.01, stall_seconds=.15)
    assert stalled['state'] == 'TRIPWIRE'
    paused = session.watch_job([sys.executable, '-c', STUB, 'stall'], tmp_path, s,
                              tmp_path / 'log', time.monotonic() + .15, poll_seconds=.01)
    assert paused['state'] == 'PAUSED'


def test_failed_job_does_not_block_next_jobs(tmp_path, monkeypatch):
    jobs = [spec() for _ in range(2)]
    jobs[0]['label'] = 'qa_image'
    jobs[1]['label'] = 'qa_text'
    monkeypatch.setattr(session, 'specs', lambda *a: jobs)
    monkeypatch.setattr(session, 'auxiliary_specs', lambda *a: [])
    monkeypatch.setattr(session, 'validate_inputs', lambda *a: None)
    monkeypatch.setattr(session, 'hardware_preflight', lambda *a: None)
    def make_command(root, workspace, s, directory, *unused):
        common.frozen_json(directory / 'run_config.json', config(s))
        return [sys.executable, '-c', STUB, 'fail' if s['label'] == 'qa_image' else '5']
    monkeypatch.setattr(session, 'command', make_command)
    args = SimpleNamespace(root=tmp_path, workspace=tmp_path, smoke=False, plan=False)
    assert not session.run_model(args, 'llava', time.monotonic() + 15)
    base = tmp_path / 'outputs' / (common.DATA_NAME + '_results') / 'llava/status'
    assert json.loads((base / 'qa_image.json').read_text())['state'] == 'FAILED'
    assert json.loads((base / 'qa_text.json').read_text())['state'] == 'COMPLETED'


def test_cross_model_prompt_mismatch(tmp_path, monkeypatch):
    s = spec()
    monkeypatch.setattr(verifier, 'specs', lambda *a: [s])
    base = tmp_path / 'outputs' / (common.DATA_NAME + '_smoke')
    write_output(base / 'llava/qa_image', s)
    common.frozen_json(base / 'llava/status/qa_image.json', {'state': 'COMPLETED'})
    assert verifier.verify_model(tmp_path, 'llava', True)['passed']
    common.frozen_json(base / 'qwen/qa_image/run_config.json', {'prompt_bodies_sha256': 'b' * 64})
    assert not verifier.verify_model(tmp_path, 'llava', True)['passed']


@pytest.mark.skipif(not (ROOT / 'outputs' / common.DATA_NAME / 'inputs_manifest.json').is_file(),
                    reason='Prepared fullrun data required for integration test')
def test_real_command_arguments_parse_for_every_runner_and_auxiliary(monkeypatch, tmp_path):
    import importlib.util
    import contextlib
    import io
    for model in common.MODELS:
        for smoke in (False, True):
            jobs = common.specs(ROOT, model, smoke) + session.auxiliary_specs(ROOT, model, smoke)
            for s in jobs:
                cmd = session.command(ROOT, Path('/workspace'), s, tmp_path / model / s['label'],
                                      tmp_path / 'preflight_report.json' if model == 'gemma' and s['label'] != 'preflight' else None,
                                      3 if smoke and s['stage'] == 'qa' else None)
                module_spec = importlib.util.spec_from_file_location('contract_test_runner', cmd[2])
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                monkeypatch.setattr(sys, 'argv', [cmd[2], *cmd[3:]])
                args = module.parse_args()
                assert args.resume
                assert args.max_new_tokens == (1024 if s['stage'] == 'stage1' else 64)
                assert args.revision == common.MODELS[model][1]
                session.validate_inputs(ROOT, s)
                if s['stage'] == 'qa' and smoke:
                    assert args.limit == 3
                if model == 'gemma':
                    assert args.pan_and_scan and args.pan_and_scan_min_crop_size == 256
                    assert args.pan_and_scan_max_num_crops == 4
                    assert args.pan_and_scan_min_ratio_to_activate == 1.2
                # Construct runner configs without GPU imports, using synthetic installed version modules.
                # The actual GPU-version gate is separately executed on the pod before model loading.


def test_preflight_detects_changed_input_before_launch(tmp_path):
    p = tmp_path / 'input.jsonl'
    p.write_text('{}\n')
    s = spec()
    s.update(input_files={'input.jsonl': common.sha(p)}, input_jsonl='input.jsonl', label='qa_image')
    p.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='changed input'):
        session.validate_inputs(tmp_path, s)


def test_historical_comparison_lists_changes_without_reinterpreting_old_parse():
    from compare_fullrun_deepseek import compare
    s = spec(n=500)
    old = [row(s, i) for i in range(500)]
    new = copy.deepcopy(old)
    new[10]['raw_response'] = 'The answer is B.'
    new[10]['predicted_option'] = 'B'
    differences = compare(old, new)
    assert {(d['statement_idx'], d['field']) for d in differences} == {(10, 'raw_response'), (10, 'predicted_option')}
    with pytest.raises(ValueError, match='Duplicate'):
        compare(old + [old[0]], new)


def test_fullrun_config_pins_inputs_prompts_and_versions(tmp_path, monkeypatch):
    s = spec()
    s.update(input_jsonl='input', graph_metadata='meta', input_files={'input': 'b' * 64, 'meta': 'c' * 64},
             data_provenance={'render': 'uniform'})
    common.frozen_json(tmp_path / 'contract.json', s)
    monkeypatch.setenv('FULLRUN_CONTRACT', str(tmp_path / 'contract.json'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__=s['versions']['torch']))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__=s['versions']['transformers']))
    c = config(s)
    c.update(input_jsonl={'sha256': 'b' * 64}, graph_metadata={'sha256': 'c' * 64})
    result = runtime.enrich_config(c)
    assert result['fullrun'] == s
    assert result['source_image_generation'] == s['data_provenance']
    c['generation']['do_sample'] = True
    with pytest.raises(ValueError, match='greedy'):
        runtime.enrich_config(c)
    c['generation']['do_sample'] = False
    c['torch_version'] = 'wrong'
    with pytest.raises(ValueError, match='version'):
        runtime.enrich_config(c)


def test_snapshot_shards_checked_before_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(session.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(session.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=10_000_000_000))
    with pytest.raises(ValueError, match='cached config'):
        session.hardware_preflight(tmp_path, 'llava', spec()['versions'])
    model, revision = common.MODELS['llava']
    snapshot = tmp_path / '.cache/huggingface/hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'absent.safetensors'}}))
    with pytest.raises(ValueError, match='weight shard'):
        session.hardware_preflight(tmp_path, 'llava', spec()['versions'])
