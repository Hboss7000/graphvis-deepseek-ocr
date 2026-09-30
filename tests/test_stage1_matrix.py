"""CPU-only contract, pairing, provenance and launcher tests for pruning matrices."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_pruning_matrix as build
import collect_stage1_matrix as collect
import generate_graphvis_datasets as gen
from stage1_common import write_run_config
from score_stage1 import aggregate_task, score_record


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def fake_matrix(tmp_path, complete=True, missing=False, illegible=False):
    cells = []
    for condition, indices in [('baseline', [0, 1]), ('nodes40', [1, 2])]:
        directory = tmp_path / condition
        directory.mkdir()
        metadata = []
        records = []
        pruning = {**build.PRUNING_DEFAULTS, **build.CONDITIONS[condition]}
        for idx in range(3):
            image = f'{idx}.png'
            (directory / image).write_bytes(b'image fixture')
            count = 1 if condition == 'baseline' else 2
            metadata.append(dict(statement_idx=idx, image=image,
                visible_nodes=[dict(cid=i, name=str(i), label=str(i), in_question=i == 0,
                                    in_choices=['A'] if i == 1 else []) for i in range(count)],
                edges=[], choices=[{'label': 'A'}, {'label': 'B'}], answerKey='A',
                pruning={**pruning, 'core_size': count, 'core_truncated': False,
                         'bridges_added': 0, 'edges_before_truncation': 0, 'edges_after': 0}))
            if idx in indices:
                records.append(dict(statement_idx=idx, task_type='node_number', image=image,
                                    prompt='How many nodes?', answer=f'There are {count} nodes in the graph.'))
        input_path, meta_path = directory / 'input.jsonl', directory / 'metadata.jsonl'
        dump_jsonl(input_path, records)
        dump_jsonl(meta_path, metadata)
        build.write_json(directory / 'stats.json', {'pruning_configurations': {json.dumps(pruning, sort_keys=True): 3}})
        build.write_json(directory / 'audit.json', {'min_label_px': 10, 'images': [
            {'statement_idx': i, 'estimated_label_cap_px_896': 2 if illegible else 20} for i in range(3)]})
        cell = dict(cell_id='qwen__' + condition, model='qwen', condition=condition,
            pruning_flags=pruning, generation={**build.FIXED, 'limit': 3, 'prompt_policy': 'fixed-template', 'min_label_px': 10},
            model_settings=copy.deepcopy(build.MODELS['qwen']), source_sha256={},
            input_jsonl=str(input_path), graph_metadata=str(meta_path), image_root=str(directory),
            expected_count=2, output_dir=str(directory / 'results'),
            input_jsonl_sha256=build.sha_file(input_path), graph_metadata_sha256=build.sha_file(meta_path),
            images_sha256=build.image_digest(metadata, directory), preflight_graphs=0,
            diagnostics={'pruning_stats': str(directory / 'stats.json'), 'image_audit': str(directory / 'audit.json')})
        cell['diagnostics_sha256'] = {k: build.sha_file(v) for k, v in cell['diagnostics'].items()}
        cell['contract_sha256'] = build.sha_object(cell)
        cells.append(cell)
        if missing and condition == 'nodes40':
            continue
        output = Path(cell['output_dir'])
        output.mkdir()
        config = valid_config(cell)
        build.write_json(output / 'run_config.json', config)
        rows = [dict(statement_idx=r['statement_idx'], task_type='node_number', raw_response='1') for r in records]
        if not complete:
            rows = rows[:1]
        dump_jsonl(output / 'predictions_qwen_node_number.jsonl', rows)
        if complete:
            build.write_json(output / 'metrics_qwen.json', {'input_record_count': 2})
    return cells


def valid_config(cell):
    return {**build.run_config_fields(cell), 'condition': 'image', 'model_id': cell['model_settings']['model_id'],
        'model_revision': cell['model_settings']['revision'], 'seed': 13, 'task_set': 'extended',
        'input_jsonl': {'sha256': cell['input_jsonl_sha256'], 'path': cell['input_jsonl']},
        'graph_metadata': {'sha256': cell['graph_metadata_sha256'], 'path': cell['graph_metadata'],
                           'path_sha256': build.sha_object(cell['graph_metadata'])},
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 1024},
        'attention_implementation': 'sdpa', 'image_processing': {'min_pixels': 262144, 'max_pixels': 1310720}}


@pytest.mark.parametrize('override', [{'dpi': 100}, {'seed': 2}, {'hide_relatedto_labels': True},
                                     {'max_nodes': -1}, {'lifelines': 0}, {'bridge_rule': 'invalid'}])
def test_conditions_reject_drift(override):
    with pytest.raises(ValueError):
        build.condition_settings({'test': override})


def test_baseline_cannot_be_redefined():
    with pytest.raises(ValueError, match='baseline'):
        build.condition_settings({'baseline': {'max_nodes': 40}})


def test_collector_paired_keys_and_condition_specific_gold(tmp_path):
    cells = fake_matrix(tmp_path)
    rows, coverage, models, conditions = collect.collect(cells)
    paired = [r for r in rows if r['scope'] == 'intersection' and r['task_type'] == 'node_number'
              and r['metric_name'] == 'strict_accuracy']
    assert [(r['condition'], r['n_items'], r['value']) for r in paired] == [('baseline', 1, 1.0), ('nodes40', 1, 0.0)]
    full = [r for r in rows if r['scope'] == 'full' and r['task_type'] == 'node_number'
            and r['metric_name'] == 'strict_accuracy']
    assert [r['n_items'] for r in full] == [2, 2]
    assert [r['only_this_condition_count'] for r in coverage] == [1, 1]
    assert [r['eligible_intersection_count'] for r in coverage] == [1, 1]
    assert models == ['qwen'] and conditions == ['baseline', 'nodes40']


def test_missing_cell_is_not_silently_removed_from_pairing(tmp_path):
    rows, coverage, _, _ = collect.collect(fake_matrix(tmp_path, missing=True))
    assert coverage[1]['status'] == 'missing'
    assert all(r['n_items'] == 0 and r['value'] is None for r in rows if r['scope'] == 'intersection')
    assert all(r['provisional_intersection'] for r in coverage)
    assert any(r['value'] == 1 for r in rows if r['scope'] == 'full' and r['condition'] == 'baseline')


def test_illegibility_masks_value_but_preserves_audit_score(tmp_path):
    rows, _, _, _ = collect.collect(
        fake_matrix(tmp_path, illegible=True), legibility_mask_threshold=0
    )
    row = next(r for r in rows if r['metric_name'] == 'strict_accuracy' and r['task_type'] == 'node_number')
    assert row['value'] is None and row['unmasked_value'] == 1
    assert row['status'] == 'illegible'
    assert collect.display(row).startswith('n/a (illegible')


def test_prediction_count_and_duplicates_fail(tmp_path):
    cells = fake_matrix(tmp_path, complete=False)
    with pytest.raises(ValueError, match='realised 1 != expected 2'):
        build.prediction_rows(cells[0], complete=True)
    path = Path(cells[0]['output_dir']) / 'predictions_qwen_node_number.jsonl'
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match='duplicated'):
        build.prediction_rows(cells[0])


def test_metadata_image_and_manifest_tampering_fail(tmp_path):
    cells = fake_matrix(tmp_path)
    manifest = tmp_path / 'manifest.jsonl'
    dump_jsonl(manifest, cells)
    assert len(build.load_manifest(manifest)) == 2
    changed = copy.deepcopy(cells)
    changed[0]['pruning_flags']['max_nodes'] = 99
    dump_jsonl(manifest, changed)
    with pytest.raises(ValueError, match='contract changed'):
        build.load_manifest(manifest)
    image = Path(cells[0]['image_root']) / '0.png'
    image.write_bytes(b'mutated image')
    with pytest.raises(ValueError, match='images changed'):
        build.validate_cell_data(cells[0])


def test_run_config_preserves_modality_and_is_resume_safe(tmp_path, monkeypatch):
    cell = fake_matrix(tmp_path)[0]
    config = valid_config(cell)
    config.pop('pruning')
    config.pop('pruning_condition')
    output = tmp_path / 'config'
    output.mkdir()
    monkeypatch.setenv('STAGE1_MATRIX_CELL_JSON', json.dumps(cell))
    write_run_config(output, config)
    write_run_config(output, config)
    written = json.loads((output / 'run_config.json').read_text())
    assert written['condition'] == 'image' and written['pruning_condition'] == 'baseline'
    build.validate_run_config(cell, written)
    written['pruning']['max_nodes'] = 40
    with pytest.raises(ValueError, match='differs in pruning'):
        build.validate_run_config(cell, written)


def test_fixed_prompt_templates_ignore_prior_rng_consumption():
    records = [dict(statement_idx=4, task_type='node_number', prompt=prompt) for prompt in gen.NODE_NUMBER_PROMPTS]
    build.normalize_prompts(records, gen)
    assert len({r['prompt'] for r in records}) == 1
    records = [dict(statement_idx=4, task_type='node_degree', prompt=prompt.format(node='a plant'))
               for prompt in gen.NODE_DEGREE_PROMPTS]
    build.normalize_prompts(records, gen)
    assert len({r['prompt'] for r in records}) == 1
    assert 'a plant' in records[0]['prompt']


def test_commands_use_existing_evaluators_and_cell_count(tmp_path):
    cell = fake_matrix(tmp_path)[0]
    for model, settings in build.MODELS.items():
        cell = {**cell, 'model': model, 'model_settings': {**settings,
            **{key: value for key, value in zip(build.CROP_KEYS, (256, 4, 1.2))}}}
        command = build.evaluator_command(cell, report='condition_preflight.json')
        assert command[0].startswith('/storage/home/hleonel/venv_')
        assert command[1] == settings['evaluator']
        assert command[command.index('--expected-count') + 1] == '2'
        assert command[command.index('--task-set') + 1] == 'extended'
        assert '--approve-prompts' in command and '--resume' in command
        if model == 'gemma3':
            assert '--preflight-report' in command
            assert '--preflight' in build.evaluator_command(cell, True, 'preflight')
        if model == 'deepseek':
            assert '--image-size' not in command  # existing evaluator uses verified constants


def test_unresolved_crops_cannot_submit(tmp_path):
    cell = fake_matrix(tmp_path)[0]
    cell = {**cell, 'model': 'gemma3', 'model_settings': {**build.MODELS['gemma3'], **dict.fromkeys(build.CROP_KEYS)}}
    with pytest.raises(ValueError, match='unresolved'):
        build.require_model_controls([cell])


def test_console_splits_wide_condition_sets(tmp_path, capsys):
    rows, coverage, models, conditions = collect.collect(fake_matrix(tmp_path, missing=True))
    collect.console_summary(collect.primary_index(rows), models, conditions, 3, 80)
    lines = capsys.readouterr().out.splitlines()
    table_lines = [line for line in lines if line.startswith(('task ', *collect.TASKS))]
    assert table_lines and all(len(line) <= 80 for line in table_lines)


def test_primary_keys_are_actual_aggregate_output_keys():
    for task, key in collect.PRIMARY.items():
        assert key in collect.flatten_metrics(aggregate_task(task, [])), (task, key)


def test_gemma_preflight_resumes_completed_items(tmp_path, monkeypatch):
    import eval_gemma3_stage1 as evaluator
    from PIL import Image
    monkeypatch.delenv('STAGE1_MATRIX_CELL_JSON', raising=False)
    records = []
    for idx in range(2):
        Image.new('RGB', (4, 4), 'white').save(tmp_path / f'{idx}.png')
        records.append(dict(statement_idx=idx, task_type='triple_listing', image=f'{idx}.png', prompt='List triples'))
    args = SimpleNamespace(output_dir=tmp_path / 'preflight', image_root=tmp_path, resume=True,
                           preflight_min_recall=.95, pan_and_scan=True)
    monkeypatch.setattr(evaluator.gemma, 'gate_identity', lambda *a: {})
    monkeypatch.setattr(evaluator.gemma, 'manifest_fields', lambda *a: {})
    monkeypatch.setattr(evaluator.gemma, 'prepare_inputs', lambda *a: {})
    monkeypatch.setattr(evaluator.gemma, 'image_budget_diagnostics', lambda *a: {})
    calls = []
    def infer(*arguments):
        calls.append(arguments[5].pan_and_scan)
        if len(calls) == 2:
            raise RuntimeError('simulated preemption')
        return 'answer', 1, False, 1, 256
    monkeypatch.setattr(evaluator.gemma, 'infer_one', infer)
    monkeypatch.setattr(evaluator, 'make_result', lambda record, *a: {
        'statement_idx': record['statement_idx'], 'task_type': record['task_type'],
        'gold_node_component_recall': {'raw': {'recall': 1.0}}})
    monkeypatch.setattr(evaluator, 'aggregate_task', lambda *a: {'gold_node_component_recall': {'raw': {'macro_recall': 1.0}}})
    config = {'resolved_dtype': 'bf16', 'image_processing': {}}
    with pytest.raises(RuntimeError, match='preemption'):
        evaluator.run_preflight(args, records, {0: {}, 1: {}}, None, None, {}, None, config)
    evaluator.run_preflight(args, records, {0: {}, 1: {}}, None, None, {}, None, config)
    assert calls == [True, True, True, False, False]
    evaluator.run_preflight(args, records, {0: {}, 1: {}}, None, None, {}, None, config)
    assert len(calls) == 5
    report = json.loads((args.output_dir / 'preflight_report.json').read_text())
    assert all(setting['passed'] for setting in report['settings'].values())


def test_cell_runner_checks_completion_and_offline_environment(tmp_path, monkeypatch):
    cell = fake_matrix(tmp_path, complete=False)[0]
    calls = []
    monkeypatch.setattr(build, 'check_call', lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    with pytest.raises(ValueError, match='realised 1 != expected 2'):
        build.run_cell(cell)
    assert len(calls) == 2
    for _, kwargs in calls:
        assert kwargs['env']['APPTAINERENV_HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['APPTAINERENV_TRANSFORMERS_OFFLINE'] == '1'
    assert '--resume' in calls[-1][0]
    assert not (Path(cell['output_dir']) / 'matrix_complete.json').exists()
    # Requeue after failure can take the released flock again.
    with pytest.raises(ValueError, match='realised'):
        build.run_cell(cell)


def test_collector_cli_writes_partial_matrix_without_dropping_cells(tmp_path, monkeypatch):
    cells = fake_matrix(tmp_path, complete=False, missing=True)
    manifest = tmp_path / 'manifest.jsonl'
    dump_jsonl(manifest, cells)
    output = tmp_path / 'collected'
    monkeypatch.setattr(sys, 'argv', ['collect', '--manifest', str(manifest),
                                     '--output-dir', str(output), '--no-plots'])
    monkeypatch.chdir(ROOT)
    collect.main()
    rows = json.loads((output / 'coverage.json').read_text())
    assert [(r['status'], r['realised_count']) for r in rows] == [('partial', 1), ('missing', 0)]
    assert all(r['intersection_count'] == 0 for r in rows)
    assert '— (n=0)' in (output / 'summary.md').read_text()
    assert (output / 'wide.csv').is_file() and (output / 'long.csv').is_file()


def test_missing_results_stay_missing_even_when_the_image_audit_fails(tmp_path):
    rows, _, _, _ = collect.collect(fake_matrix(tmp_path, missing=True, illegible=True))
    missing = [r for r in rows if r['condition'] == 'nodes40']
    assert all(r['status'] == 'missing' and r['value'] is None for r in missing)
    assert collect.display(missing[0]) == '— (n=0)'
