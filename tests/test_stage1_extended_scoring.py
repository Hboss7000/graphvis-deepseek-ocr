"""Extended scoring contracts and compatibility with completed paper outputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import gemma3_common
import score_stage1 as scorer
import stage1_common
import run_stage1_qwen
import run_stage1_deepseek
import eval_gemma3_stage1
from test_gemma3_evaluation import dataset, dump, cli


def source(task, **gold):
    return {'task_type': task, 'answer': 'Prose is intentionally not used as gold.', 'gold': gold}


def relation():
    return source('relation_identification', relation='atlocation', relation_text='at location',
                  node_a='garden', node_b='flower')


def neighbors():
    return source('neighbor_listing', node='center', neighbors=['garden [A,B]', 'flower', 'water'], degree=3)


def paths(truncated=False):
    return source('shortest_path_listing', node_a='start [A,B]', node_b='finish', distance=2,
                  paths=[['start [A,B]', 'garden', 'finish'], ['start [A,B]', 'water', 'finish']],
                  paths_truncated=truncated)


@pytest.mark.parametrize('response', ['at location', 'atlocation', 'AT LOCATION', 'Relation: atlocation',
    'The relation is "at location".', 'The relationship between garden and flower is at location.', 'The relation between "garden" and "flower" is "at location".'])
def test_relation_forms(response):
    row = scorer.score_record(relation(), response, {})
    assert all(row['relation_correct'].values())
    assert row['gold_relation'] == 'atlocation'


def test_relation_must_be_an_assertion():
    row = scorer.score_record(relation(), 'not at location; perhaps related to', {})
    assert not row['is_correct']
    record = relation()
    record['gold']['relation'] = 'relatedto'
    record['gold']['relation_text'] = 'related to'
    wrong = scorer.score_record(record, 'atlocation', {})
    aggregate = scorer.aggregate_task('relation_identification', [scorer.score_record(relation(), 'atlocation', {}), wrong])
    assert aggregate['raw']['accuracy'] == 0.5
    assert aggregate['per_relation']['relatedto']['raw']['accuracy'] == 0
    assert aggregate['per_relation']['atlocation']['raw']['accuracy'] == 1


@pytest.mark.parametrize('response', ['garden [A,B], flower',
    'The node "center" is directly connected to: garden [A,B], flower.',
    'The neighbors are: garden [A,B], flower.', 'The node is connected to garden [A,B], flower.', '1. garden [A,B]\n2. flower'])
def test_neighbor_partial_credit(response):
    row = scorer.score_record(neighbors(), response, {})
    assert row['set_metrics']['raw']['precision'] == 1
    assert row['set_metrics']['raw']['recall'] == pytest.approx(2 / 3)
    assert row['set_metrics']['raw']['f1'] == pytest.approx(0.8)
    assert row['gold_degree'] == 3
    metrics = scorer.aggregate_task('neighbor_listing', [row])
    assert metrics['raw']['mean_f1'] == pytest.approx(0.8)
    assert metrics['by_degree']['3']['raw']['mean_recall'] == pytest.approx(2 / 3)


def test_normalization_tiers():
    row = scorer.score_record(neighbors(), 'GARDEN, FLOWER, WATER', {})
    assert row['set_metrics']['raw']['recall'] == 0
    assert row['set_metrics']['basic']['recall'] == pytest.approx(2 / 3)
    assert row['set_metrics']['annotation_stripped']['f1'] == 1
    row = scorer.score_record(paths(), 'START -> WATER -> FINISH', {})
    assert row['path_exact'] == {'raw': False, 'basic': False, 'annotation_stripped': True}
    row = scorer.score_record(relation(), 'atlocation [A]', {})
    assert all(row['relation_correct'].values())
    legacy = scorer.score_record(relation(), 'atlocation [A]', {}, extractor='legacy')
    assert legacy['relation_correct'] == {'raw': False, 'basic': False, 'annotation_stripped': True}


@pytest.mark.parametrize('response', ['start [A,B] -> water -> finish',
    'start [A,B] → water → finish', 'start [A,B], water, finish',
    '1. start [A,B]\n2. water\n3. finish', '1) start [A,B] 2) water 3) finish',
    'The shortest path from "start [A,B]" to "finish" is: start [A,B] -> water -> finish.'])
def test_alternative_shortest_path(response):
    row = scorer.score_record(paths(), response, {})
    assert all(row['path_exact'].values())
    assert row['path_partial']['raw'] == {'hop_count_correct': True, 'node_overlap_f1': 1.0}


def test_wrong_path_correct_hops_and_truncated_gold():
    row = scorer.score_record(paths(True), 'start [A,B] -> elsewhere -> finish', {})
    assert row['path_exact']['raw'] is False
    assert row['path_partial']['raw']['hop_count_correct'] is True
    assert row['path_partial']['raw']['node_overlap_f1'] == pytest.approx(2 / 3)
    metrics = scorer.aggregate_task('shortest_path_listing', [row])
    assert metrics['paths_truncated_count'] == 1
    assert metrics['raw']['path_exact'] == 0 and metrics['raw']['hop_count_correct'] == 1
    empty = scorer.score_record(paths(), '', {})
    assert not empty['path_partial']['raw']['hop_count_correct']
    assert empty['path_partial']['raw']['node_overlap_f1'] == 0


def complete_predictions(dataset, out, records, tasks):
    out.mkdir()
    for task in tasks:
        dump(out / f'predictions_fixture_{task}.jsonl', [
            {'statement_idx': r['statement_idx'], 'task_type': task, 'raw_response': r['answer'],
             'hit_token_ceiling': False, 'gold': r['answer'], 'scoring_status': 'pending'}
            for r in records if r['task_type'] == task])
    return SimpleNamespace(input_jsonl=dataset.root / ('extended.jsonl' if len(tasks) == 9 else 'paper.jsonl'),
                           graph_metadata=dataset.root / 'metadata.jsonl', predictions_dir=out,
                           model_name='fixture', task_set='extended' if len(tasks) == 9 else 'paper')


def test_extended_files_gold_collision_and_all_aggregates(dataset):
    args = complete_predictions(dataset, dataset.root / 'complete', dataset.extended, scorer.EXTENDED_TASK_TYPES)
    metrics, rows = scorer.score_files(args)
    assert tuple(metrics['tasks']) == scorer.EXTENDED_TASK_TYPES
    for task in scorer.NEW_TASK_TYPES:
        row = rows[task][0]
        assert isinstance(row['gold'], str) and isinstance(row['structured_gold'], dict)
        assert row['is_correct'] and 'scoring_status' not in row
        assert task in metrics['scoring_notes']
    scorer.write_failure_cases(dataset.root / 'failures.md', 'fixture', rows, scorer.EXTENDED_TASK_TYPES)
    assert '## shortest_path_listing' in (dataset.root / 'failures.md').read_text()
    # CLI consumes the same fixtures, including all nine prediction files.
    subprocess.run([sys.executable, str(Path(scorer.__file__)), '--input-jsonl', str(args.input_jsonl),
        '--graph-metadata', str(args.graph_metadata), '--predictions-dir', str(args.predictions_dir),
        '--model-name', 'fixture', '--task-set', 'extended'], capture_output=True, check=True)


@pytest.mark.parametrize('runner', [run_stage1_qwen, run_stage1_deepseek, eval_gemma3_stage1])
def test_runner_rejects_extended_as_paper_before_heavy_imports(runner, dataset, monkeypatch):
    argv = cli(dataset, 'extended', dataset.root / 'out')
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match=r"relation_identification.*--task-set extended"):
        runner.main()


def test_paper_golden_metrics(dataset):
    args = complete_predictions(dataset, dataset.root / 'paper_complete', dataset.papers, scorer.PAPER_TASK_TYPES)
    # Mix correct, partial and unparseable responses to exercise all old branches.
    alter_paper_predictions(args.predictions_dir)
    args.extractor = "legacy"
    metrics, rows = scorer.score_files(args)
    path = dataset.root / 'metrics.json'
    scorer.write_json(path, metrics)
    assert path.read_bytes() == (Path(__file__).parent / 'fixtures/stage1_paper_metrics.json').read_bytes()
    assert scorer.TASK_TYPES is scorer.PAPER_TASK_TYPES
    assert not set(scorer.NEW_TASK_TYPES) & metrics['scoring_notes'].keys()


def alter_paper_predictions(directory):
    responses = {'node_description': 'sun flower, water bottle', 'node_number': 'There are 7 nodes.',
                 'edge_number': 'unknown', 'triple_listing': '(sun flower, is a, garden soil)',
                 'highest_node_degree': 'The node is "garden soil", with a degree of 2.',
                 'node_degree': '3'}
    for task, response in responses.items():
        path = directory / f'predictions_fixture_{task}.jsonl'
        rows = scorer.read_jsonl(path)
        rows[0]['raw_response'] = response
        dump(path, rows)


def test_new_gold_required():
    with pytest.raises(ValueError, match='structured gold'):
        scorer.score_record({'task_type': 'neighbor_listing', 'answer': 'a, b'}, 'a, b', {})


@pytest.mark.parametrize('runner', [run_stage1_qwen, run_stage1_deepseek])
def test_nine_task_runner_inference_and_resume(runner, dataset, monkeypatch):
    """Exercise real handle allocation, inline scoring, final scoring and resume."""
    answers = {r['prompt']: r['answer'] for r in dataset.extended}
    calls = []
    def respond(prompt):
        calls.append(prompt)
        return answers[prompt.removeprefix('<image>\n')]
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), __version__='test', bfloat16='bf16')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    if runner is run_stage1_qwen:
        transformers = SimpleNamespace(__version__='4.57.1', AutoProcessor=object, Qwen3VLForConditionalGeneration=object)
        monkeypatch.setattr(runner, 'load_processor', lambda *a: (object(), 'fixture'))
        monkeypatch.setattr(runner, 'format_qwen_prompt', lambda processor, body, condition: body)
        monkeypatch.setattr(runner, 'prepare_inputs', lambda *a: {})
        monkeypatch.setattr(runner, 'image_budget_diagnostics', lambda *a: {})
        monkeypatch.setattr(runner, 'load_model', lambda *a: (object(), 'dtype', 'bf16'))
        monkeypatch.setattr(runner, 'infer_one', lambda model, processor, prompt, *a: (respond(prompt), 20, False))
    else:
        model = SimpleNamespace(generation_config=SimpleNamespace())
        model.eval = lambda: model
        model.cuda = lambda: model
        model.to = lambda dtype: model
        model.infer = lambda tokenizer, **kw: respond(kw['prompt'])
        tokenizer = SimpleNamespace(encode=lambda response, **kw: list(range(20)))
        transformers = SimpleNamespace(__version__='4.46.3',
            AutoModel=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
    seeds = []
    transformers.set_seed = seeds.append
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    out = dataset.root / runner.MODEL_NAME
    argv = cli(dataset, 'extended', out, '--task-set', 'extended', '--approve-prompts')
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    assert len(calls) == len(dataset.extended)
    metrics = json.loads((out / f'metrics_{runner.MODEL_NAME}.json').read_text())
    assert tuple(metrics['task_record_counts']) == tuple(sorted(scorer.EXTENDED_TASK_TYPES))
    for task in scorer.NEW_TASK_TYPES:
        rows = scorer.read_jsonl(out / f'predictions_{runner.MODEL_NAME}_{task}.jsonl')
        assert len(rows) == 2 and all(row['is_correct'] for row in rows)
        assert isinstance(rows[0]['gold'], str) and isinstance(rows[0]['structured_gold'], dict)
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    runner.main()
    assert len(calls) == len(dataset.extended)
    assert seeds == [13, 13]


@pytest.mark.parametrize('model', ['qwen', 'deepseek'])
def test_completed_gpu_paper_metrics_byte_regression(model, tmp_path):
    root = ROOT / 'outputs/stage1_scoring_regression_2026-09-11'
    baseline = root / model / f'metrics_{model}.json'
    if not baseline.exists():
        pytest.skip('Optional completed-run artifacts have not been copied from the cluster')
    config = json.loads((root / model / 'run_config.json').read_text())
    input_jsonl = root / 'stage1_graph_comprehension_0_500.jsonl'
    metadata = root / 'graph_metadata_0_500.jsonl'
    assert gemma3_common.sha256_file(input_jsonl) == config['input_jsonl']['sha256']
    assert gemma3_common.sha256_file(metadata) == config['graph_metadata']['sha256']
    target = tmp_path / 'metrics.json'
    subprocess.run([sys.executable, scorer.__file__, '--task-set', 'paper', '--extractor', 'legacy',
        '--input-jsonl', str(input_jsonl), '--graph-metadata', str(metadata),
        '--predictions-dir', str(root / model), '--model-name', model,
        '--metrics-out', str(target), '--failure-cases-out', str(tmp_path / 'failures.md')],
        check=True, capture_output=True)
    # Recompute with today's legacy branch and compare bytes directly, without
    # trusting a cached verification from an earlier scorer revision.
    assert target.read_bytes() == baseline.read_bytes()
