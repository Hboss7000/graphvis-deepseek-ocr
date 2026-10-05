"""CPU contracts for Stage 1 diagnostic formatting, leakage and job protocols."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT/'scripts', ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'):
    sys.path.insert(0, str(directory))
import llava_stage1_prompt as prompts
import run_stage1_llava as runner
from stage1_common import raw_image_prompt


class Processor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return '[INST] ' + messages[0]['content'] + ' [/INST]'


def test_graphvis_one_turn_is_exact_literal_and_bypasses_chat_template():
    prompt = 'How many nodes are there in the graph?'
    expected = ("A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                'USER: <image>\nHow many nodes are there in the graph? ASSISTANT:')
    assert prompts.format_stage1_prompt(object(), '<image>\n'+prompt, 'llava_v1') == expected
    assert '<s>' not in expected and '</s>' not in expected


def test_default_preserves_main_hf_chat_and_user_text():
    record={'task_type':'node_number', 'prompt':'  Unchanged question?\nSecond line.  '}
    args=SimpleNamespace(answer_format='none',prompt_template='hf-chat')
    body=raw_image_prompt(record,'none')
    assert runner.render_prompt(Processor(),record,args)=='[INST] '+body+' [/INST]'
    args.prompt_template='llava_v1'
    assert 'USER: '+body+' ASSISTANT:' in runner.render_prompt(object(),record,args)


@pytest.mark.parametrize('body',['Question','<image>\n<image>\nQuestion'])
def test_native_wrapper_rejects_missing_or_duplicate_image_marker(body):
    with pytest.raises(ValueError):prompts.format_stage1_prompt(object(),body,'llava_v1')
