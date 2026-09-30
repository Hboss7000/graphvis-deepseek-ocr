"""Contract tests with synthetic graphs and fake inference; no weights or GPU."""
import contextlib
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import gemma3_common as common
import eval_gemma3_stage1 as stage1
import eval_gemma3_stage2 as stage2
import generate_graphvis_datasets as generator
import run_zero_shot_qwen as qwen2
import run_stage1_qwen as qwen1
import score_stage1
from compare_gemma3_runs import compare
from prompt_common import IMAGE_REFERENCE_SENTENCE, format_kg_block, read_jsonl

REVISION = 'a' * 40


def dump(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def dataset(tmp_path):
    names = ['sun_flower', 'garden_soil', 'water_bottle', 'greenhouse']
    nodes = {i: {'name': name, 'in_question': i == 0, 'in_choices': set()}
             for i, name in enumerate(names)}
    graph = {'edges': [(0, 'isa', 1), (0, 'partof', 2), (1, 'atlocation', 3), (2, 'usedfor', 3)],
             'connected_nodes': list(range(4)), 'visible_nodes': list(range(4)),
             'disconnected_answers': [], 'disconnected_questions': [], 'q_cids': [0], 'a_cids': []}
    statement = {'answerKey': 'A', 'question': {'stem': 'Where does the flower grow?',
                 'choices': [{'label': c, 'text': str(i)} for i, c in enumerate('ABCD')]}}
    papers, extended, stage2_rows, metadata = [], [], [], []
    for idx in range(2):
        image = f'{idx}.png'
        Image.new('RGB', (100, 80), 'white').save(tmp_path / image)
        papers.extend(generator.build_stage1_records(image, 'test', idx, nodes, graph, random.Random(13), None))
        extended.extend(generator.build_stage1_records(image, 'test', idx, nodes, graph, random.Random(13), None,
                                                       stage1_task_set='extended'))
        stage2_rows.append(generator.build_stage2_record(image, 'test', idx, statement))
        metadata.append(generator.graph_metadata(idx, statement, nodes, graph, image))
    for name, rows in [('paper', papers), ('extended', extended), ('stage2', stage2_rows), ('metadata', metadata)]:
        dump(tmp_path / f'{name}.jsonl', rows)
    return SimpleNamespace(root=tmp_path, papers=papers, extended=extended, stage2=stage2_rows, metadata=metadata)


class ImageProcessor:
    def __init__(self, do_pan_and_scan=False, pan_and_scan_min_crop_size=256,
                 pan_and_scan_max_num_crops=4, pan_and_scan_min_ratio_to_activate=1.2):
        self.size = {'height': 896, 'width': 896}
        self.pan_and_scan_min_crop_size = pan_and_scan_min_crop_size
        self.pan_and_scan_max_num_crops = pan_and_scan_max_num_crops
        self.pan_and_scan_min_ratio_to_activate = pan_and_scan_min_ratio_to_activate

    def preprocess(self, images):
        pass


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class Processor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.tokenizer = SimpleNamespace(padding_side='right', image_token_id=100)
        self.image_token_id = 99
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not kwargs['tokenize']:
            return json.dumps(messages)
        has_image = any(block['type'] == 'image' for msg in messages for block in msg['content'])
        views = (3 if kwargs.get('do_pan_and_scan') else 1) if has_image else 0
        soft_tokens = views * 256
        return Inputs(input_ids=np.array(
            [[1, 2] + [99] * views + [100] * soft_tokens]
        ))

    def decode(self, generated, **kwargs):
        assert generated.tolist() == [7, 8], 'Prompt leaked into decode'
        assert kwargs == {'skip_special_tokens': True}
        return ' A '


def args_for_common(**updates):
    updates.setdefault('cache_implementation', 'dynamic')
    updates.setdefault('pan_and_scan', True)
    return SimpleNamespace(model_id=common.DEFAULT_MODEL_ID, revision=REVISION,
                           max_new_tokens=64, expected_count=2, batch_size=1, seed=13,
                           attn_impl='eager',
                           **{key: None for key in common.CROP_KEYS}, **updates)


def setup_processor(args):
    processor = Processor()
    loader = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)
    return common.load_processor(loader, args)


@pytest.fixture
def fake_runtime(monkeypatch, dataset):
    processor = Processor()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), __version__='fake',
                                 inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        __version__='fake', AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
        Gemma3ForConditionalGeneration=object))
    monkeypatch.setattr(common, 'load_model', lambda *a: (object(), 'dtype', 'torch.bfloat16'))
    answers = {r['prompt']: r['answer'] for r in dataset.extended}
    calls = []

    def infer(model, processor, body, condition, image, args, torch, details):
        calls.append((condition, image, args.pan_and_scan))
        if condition == 'image':
            response = answers.get(body.removeprefix('<image>\n'), 'A')
            image_views = 3 if args.pan_and_scan else 1
            image_soft_tokens = image_views * 256
        else:
            assert image is None
            response, image_views, image_soft_tokens = 'The answer is A.', 0, 0
        return response, 20, False, image_views, image_soft_tokens

    monkeypatch.setattr(common, 'infer_one', infer)
    return processor, calls


def cli(dataset, stage, output, *extra):
    args = ['eval', '--revision', REVISION, '--input-jsonl', str(dataset.root / f'{stage}.jsonl'),
            '--graph-metadata', str(dataset.root / 'metadata.jsonl'), '--image-root', str(dataset.root),
            '--expected-count', str(len(getattr(dataset, 'papers' if stage == 'paper' else stage)))]
    args.extend(['--output-jsonl' if stage == 'stage2' else '--output-dir', str(output)])
    return args + list(extra)


def test_prompt_bodies_identical(dataset):
    assert stage2.render_prompt is qwen2.render_prompt
    assert stage1.raw_image_prompt is qwen1.raw_image_prompt
    class Capture:
        def apply_chat_template(self, messages, **kwargs):
            return messages
    record = dataset.stage2[0]
    for condition in stage2.CONDITIONS:
        body = stage2.render_prompt(record['prompt'], condition, format_kg_block(dataset.metadata[0]))
        assert common.build_messages(body, condition) == qwen2.format_qwen_prompt(Capture(), body, condition)
    for record in dataset.papers:
        body = stage1.raw_image_prompt(record)
        assert common.build_messages(body, 'image') == qwen1.format_qwen_prompt(Capture(), body, 'image')


def test_processor_signature_and_resolved_controls():
    args = args_for_common()
    args.pan_and_scan_max_num_crops = 6
    processor, details = setup_processor(args)
    assert details['pan_and_scan_kwargs'] == dict(pan_and_scan_min_crop_size=256,
        pan_and_scan_max_num_crops=6, pan_and_scan_min_ratio_to_activate=1.2)
    assert processor.tokenizer.padding_side == 'left'
    processor.image_processor = SimpleNamespace(size={'height': 896, 'width': 896}, preprocess=lambda **kw: None)
    loader = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)
    with pytest.raises(RuntimeError, match='does not explicitly expose'):
        common.load_processor(loader, args)


def test_model_load_pin_and_greedy(monkeypatch):
    calls = []
    model = SimpleNamespace(generation_config=SimpleNamespace(), eval=lambda: model,
                            parameters=lambda: iter([SimpleNamespace(dtype='torch.bfloat16')]))
    loader = SimpleNamespace(from_pretrained=lambda *a, **kw: (calls.append(kw) or model))
    seeds = []
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(set_seed=seeds.append))
    common.load_model(loader, SimpleNamespace(bfloat16='bf16'), args_for_common())
    assert calls == [dict(revision=REVISION, dtype='bf16', device_map='auto', attn_implementation='eager')]
    assert seeds == [13]
    assert model.generation_config.do_sample is False
    assert model.generation_config.temperature is None


@pytest.mark.parametrize('cache', ['dynamic', 'static'])
@pytest.mark.parametrize('condition', stage2.CONDITIONS)
@pytest.mark.parametrize('pan_and_scan', [True, False])
def test_inference_slices_prompt_and_counts_realised_tokens(condition, cache, pan_and_scan):
    args = args_for_common(cache_implementation=cache, pan_and_scan=pan_and_scan)
    processor, details = setup_processor(args)
    kwargs_seen = []
    def generate(**kwargs):
        kwargs_seen.append(kwargs)
        return np.concatenate([kwargs['input_ids'], [[7, 8]]], axis=1)
    model = SimpleNamespace(device='fake-gpu', generate=generate)
    body = '<image>\nquestion' if condition == 'image' else 'question'
    response, tokens, ceiling, image_views, image_soft_tokens = common.infer_one(
        model, processor, body, condition,
        object() if condition == 'image' else None, args,
        SimpleNamespace(inference_mode=contextlib.nullcontext), details)
    assert (response, tokens, ceiling) == ('A', 2, False)
    expected_views = (3 if pan_and_scan else 1) if condition == 'image' else 0
    assert image_views == expected_views
    assert image_soft_tokens == expected_views * 256
    kwargs = kwargs_seen[0]
    assert kwargs['do_sample'] is False
    if cache == 'static':
        assert kwargs['cache_implementation'] == 'static'
    else:
        assert 'cache_implementation' not in kwargs
    assert not {'temperature', 'top_p', 'top_k'} & kwargs.keys()
    assert ('do_pan_and_scan' in processor.calls[-1][1]) == (condition == 'image')


def test_preflight_both_settings_and_full_paper_scoring(dataset, fake_runtime, monkeypatch):
    preflight_dir = dataset.root / 'gate'
    monkeypatch.setattr(sys, 'argv', cli(dataset, 'paper', preflight_dir, '--preflight', '2', '--approve-prompts'))
    stage1.main()
    report_path = preflight_dir / 'preflight_report.json'
    report = json.loads(report_path.read_text())
    assert report['graph_count'] == 2
    assert all(setting['passed'] for setting in report['settings'].values())
    for setting, views, soft_tokens in [('pan_and_scan', 3, 768),
                                        ('no_pan_and_scan', 1, 256)]:
        rows = read_jsonl(preflight_dir / setting / 'predictions_gemma3_triple_listing.jsonl')
        assert len(rows) == 2
        assert all(r['image_views'] == views for r in rows)
        assert all(r['image_soft_tokens'] == soft_tokens for r in rows)
    out = dataset.root / 'paper_results'
    monkeypatch.setattr(sys, 'argv', cli(dataset, 'paper', out, '--approve-prompts', '--preflight-report', str(report_path)))
    stage1.main()
    assert json.loads((out / 'metrics_gemma3.json').read_text())['input_record_count'] == 12
    config = json.loads((out / 'run_config.json').read_text())
    assert config['revision'] == REVISION and config['pan_and_scan'] is True
    assert 'not comparable' in config['resolution_control_note']
    assert 'min_pixels' not in config['image_processing']
    assert config['generation']['max_new_tokens'] == 1024
    rows = read_jsonl(out / 'predictions_gemma3_node_number.jsonl')
    assert list(rows[0])[:8] == ['statement_idx', 'task_type', 'image', 'prompt', 'gold', 'raw_response',
                               'generated_token_count', 'hit_token_ceiling']
    # Resume must not regenerate completed records.
    count = len(fake_runtime[1])
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--resume'])
    stage1.main()
    assert len(fake_runtime[1]) == count
    # Extended mode now scores all nine tasks through the shared scorer.
    extended_out = dataset.root / 'extended_results'
    monkeypatch.setattr(sys, 'argv', cli(dataset, 'extended', extended_out, '--task-set', 'extended',
                                       '--approve-prompts', '--preflight-report', str(report_path)))
    stage1.main()
    for name in stage1.EXTENDED_TASKS[6:]:
        row = read_jsonl(extended_out / f'predictions_gemma3_{name}.jsonl')[0]
        assert row['is_correct'] and 'structured_gold' in row and 'scoring_status' not in row
    assert json.loads((extended_out / 'metrics_gemma3.json').read_text())['input_record_count'] == 18


def test_preflight_failure_is_a_gate(dataset, fake_runtime, monkeypatch):
    monkeypatch.setattr(common, 'infer_one', lambda *a: ('unreadable', 2, False, 1, 256))
    out = dataset.root / 'failed_gate'
    monkeypatch.setattr(sys, 'argv', cli(dataset, 'paper', out, '--preflight', '2', '--approve-prompts'))
    with pytest.raises(SystemExit, match='Legibility gate failed'):
        stage1.main()
    report = out / 'preflight_report.json'
    args = stage1.parse_args()
    args.preflight_report = report
    _, details = setup_processor(args)
    with pytest.raises(ValueError, match='Legibility gate failed'):
        common.require_preflight(args, details)
    args.preflight_report = None
    with pytest.raises(ValueError, match='requires --preflight-report'):
        common.require_preflight(args, details)


def test_stage2_50_item_scorer_and_repeat(dataset, fake_runtime, monkeypatch):
    dataset.stage2 = [dict(dataset.stage2[0], statement_idx=i) for i in range(50)]
    dump(dataset.root / 'stage2.jsonl', dataset.stage2)
    dump(dataset.root / 'metadata.jsonl', [dict(dataset.metadata[0], statement_idx=i) for i in range(50)])
    # There are deliberately no readable images in this text condition.
    monkeypatch.setattr(Image, 'open', lambda *a: pytest.fail('Text condition tried to read an image'))
    outputs = []
    for trial in ('eager', 'eager_repeat', 'sdpa'):
        out = dataset.root / trial / 'predictions.jsonl'
        monkeypatch.setattr(sys, 'argv', cli(dataset, 'stage2', out, '--condition', 'text',
            '--approve-prompt-diff', '--attn-impl', trial.removesuffix('_repeat')))
        stage2.main()
        outputs.append(out)
        config = json.loads((out.parent / 'run_config.json').read_text())
        assert config['max_new_tokens'] == 64 and config['first_image_budget'] == {}
        subprocess.run([sys.executable, str(common.STAGE2_SCRIPTS / 'score_predictions.py'),
            '--predictions', str(out), '--expected-count', '50', '--metrics-out', str(out.parent / 'metrics.json')],
            check=True, capture_output=True)
        row = read_jsonl(out)[0]
        assert list(row) == ['statement_idx', 'image', 'gold_option', 'predicted_option', 'raw_response',
            'parse_tier', 'is_correct', 'model_id', 'model_revision', 'timestamp_utc',
            'image_views', 'image_soft_tokens']
        assert row['image_views'] == 0
        assert row['image_soft_tokens'] == 0
    assert compare(outputs[0], outputs[1], 50)['identical']
    assert compare(outputs[0], outputs[2], 50)['identical']
    rows = read_jsonl(outputs[2])
    rows[0]['raw_response'] = 'B'
    dump(outputs[2], rows)
    assert compare(outputs[0], outputs[2], 50)['different_items'] == 1


@pytest.mark.parametrize('revision', [None, 'main', 'abc123'])
def test_revision_required(revision):
    args = args_for_common()
    args.revision = revision
    with pytest.raises(ValueError, match='commit hash'):
        common.validate_args(args)


def test_cli_defaults_and_pixel_flags_removed(monkeypatch, dataset):
    for module, source in [(stage1, 'paper'), (stage2, 'stage2')]:
        argv = cli(dataset, source, dataset.root / 'out')
        if module is stage2:
            argv += ['--condition', 'text']
        monkeypatch.setattr(sys, 'argv', argv)
        args = module.parse_args()
        assert args.attn_impl == 'eager' and args.pan_and_scan and args.batch_size == 1
        assert args.max_new_tokens == (1024 if module is stage1 else 64)
        if module is stage1:
            assert args.task_set == 'paper'
        monkeypatch.setattr(sys, 'argv', argv + ['--min-pixels', '100'])
        with pytest.raises(SystemExit):
            module.parse_args()
