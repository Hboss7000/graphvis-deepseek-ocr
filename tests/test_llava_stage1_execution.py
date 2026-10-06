"""CPU contract tests for paid-run plans and unchanged evaluation arms."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'scripts', ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'):
    sys.path.insert(0, str(path))

from evaluate_llava_stage1_adapter import jobs, summarize
from llava_stage1_execution_plan import plan
from llava_stage1_prompt import format_llava_prompt
from llava_common import format_prompt
from run_zero_shot_llava import format_qa_prompt
from compare_llava_stage1_training import verify_config
from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION
from llava_stage1_training import file_sha


class Processor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return '[INST] ' + messages[0]['content'] + ' [/INST]'


@pytest.mark.parametrize('condition', ['image', 'text', 'text_noref', 'kg_text'])
def test_qa_default_unchanged_and_explicit_graphvis_template(condition):
    body = ('<image>\n' if condition == 'image' else '') + 'Question'
    processor = Processor()
    assert format_qa_prompt(processor, body, condition, SimpleNamespace()) == format_prompt(processor, body, condition)
    tuned = format_qa_prompt(processor, body, condition, SimpleNamespace(prompt_template='llava_v1'))
    assert tuned.endswith(' USER: ' + body + ' ASSISTANT:')
    assert tuned.count('<image>') == int(condition == 'image')
    if condition != 'image':
        with pytest.raises(ValueError, match='Text-only'):
            format_llava_prompt(processor, '<image>\nQuestion', condition, 'llava_v1')


@pytest.mark.parametrize('phase,stage_count,qa_count', [('dry', 9, 8), ('full', 900, 500)])
def test_evaluation_preserves_all_arms_and_strict_generation(phase, stage_count, qa_count):
    commands = dict(jobs(phase, Path('/adapter'), Path('/data'), Path('/dry'), Path('/output')))
    assert set(commands) == {'stage1', 'stage1_prefix', 'qa_image', 'qa_text', 'qa_text_noref', 'qa_kg_text'}
    for name, command in commands.items():
        value = lambda flag: command[command.index(flag) + 1]
        assert value('--adapter') == '/adapter'
        assert value('--prompt-template') == 'llava_v1'
        assert value('--expected-count') == str(stage_count if name.startswith('stage1') else qa_count)
        assert value('--max-new-tokens') == ('1024' if name.startswith('stage1') else '64')
        if name.startswith('stage1'):
            assert value('--assistant-prefix-mode') == ('gold-template' if name.endswith('prefix') else 'none')
            assert value('--extractor') == 'span_extended'
            assert value('--task-set') == 'extended'


def test_cpu_timing_cannot_authorize_gpu_estimates_and_guard_adds_half():
    evidence = {'passed': True, 'dtype': 'float32', 'gpu_type': 'CPU', 'selected_batch': 1,
                'seconds_per_optimizer_step': 10,
                'batch_probe_attempts': [{'wall_seconds': 40}]}
    with pytest.raises(ValueError, match='real bf16'):
        plan(evidence)
    evidence.update(dtype='bf16', gpu_type='NVIDIA RTX PRO 6000 Blackwell')
    result = plan(evidence)
    smoke = result['jobs']['smoke']
    assert smoke['expected_hours'] == pytest.approx(280 / 3600)
    assert smoke['guard_hours'] == pytest.approx(smoke['expected_hours'] * 1.5)
    assert smoke['estimated_cost_usd'] == pytest.approx(smoke['expected_hours'] * 2.09)
    with pytest.raises(ValueError, match='successful'):
        plan(evidence, evaluation_dry={'passed': False, 'phase': 'dry'})


def test_evaluation_dry_rejects_missing_rows_and_mixed_adapters(tmp_path):
    for name in ('stage1', 'stage1_prefix', 'qa_image', 'qa_text_noref', 'qa_text', 'qa_kg_text'):
        folder = tmp_path / name
        folder.mkdir()
        n = 9 if name.startswith('stage1') else 8
        (folder / 'run_config.json').write_text(json.dumps({'adapter': {'sha256': 'a' * 64}}))
        (folder / 'runtime_metrics.json').write_text(json.dumps({'loading_elapsed_seconds': 2}))
        rows = [{'statement_idx': i, 'task_type': str(i), 'item_elapsed_seconds': 1,
                 'peak_memory_allocated_bytes': 123, 'hit_token_ceiling': False} for i in range(n)]
        (folder / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = summarize(tmp_path, 'dry')
    assert result['full_extrapolated_seconds'] == 900 * 2 + 500 * 4 + 2 * 6
    assert result['peak_vram_bytes'] == 123
    (tmp_path / 'qa_image/run_config.json').write_text(json.dumps({'adapter': {'sha256': 'b' * 64}}))
    with pytest.raises(ValueError, match='different adapters'):
        summarize(tmp_path, 'dry')
    (tmp_path / 'qa_image/predictions.jsonl').write_text('')
    with pytest.raises(ValueError, match='coverage'):
        summarize(tmp_path, 'dry')


def test_comparison_rejects_changed_template_cap_and_sampling(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text('{}\n')
    config = {'model_id': DEFAULT_MODEL_ID, 'model_revision': DEFAULT_REVISION, 'seed': 13,
              'input_jsonl': {'sha256': file_sha(source)}, 'prompt_template': 'llava_v1',
              'adapter': {'sha256': 'a' * 64}, 'effective_max_new_tokens': 64,
              'generation': {'do_sample': False, 'num_beams': 1}}
    verify_config(config, source, 'llava_v1', 'none', True, 64)
    for key, value in [('effective_max_new_tokens', 1024), ('prompt_template', 'hf-chat'), ('adapter', None)]:
        with pytest.raises(ValueError):
            verify_config({**config, key: value}, source, 'llava_v1', 'none', True, 64)
    changed = copy.deepcopy(config)
    changed['generation']['do_sample'] = True
    with pytest.raises(ValueError, match='greedy'):
        verify_config(changed, source, 'llava_v1', 'none', True, 64)
