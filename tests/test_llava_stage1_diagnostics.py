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


@pytest.mark.parametrize('task,question,expected',[
    ('node_number','How many nodes?','There are'),
    ('edge_number','How many edges?','There are'),
    ('node_degree','What is the degree of the node "room 101"?','The degree of the node with the name "room 101" is'),
    ('highest_node_degree','Which node has the highest degree?','One node with the highest degree is "'),
    ('node_description','List all nodes.','The image depicts the following nodes:'),
    ('triple_listing','List all triples.','The triples in the graph are listed as: ('),
    ('relation_identification','Which relation links "a" to "b"?','The relation between "a" and "b" is "'),
    ('neighbor_listing','Name every neighbor of "a".','The node "a" is directly connected to:'),
    ('shortest_path_listing','Trace from "a" to "b".','The shortest path from "a" to "b" is:'),
])
def test_prefix_literals(task,question,expected):
    assert prompts.gold_template_prefix(task,question)==expected


def test_900_gold_answers_do_not_enter_any_prefix():
    path=ROOT/'outputs/fullrun_2026-10-04_B/test/stage1_subset100.jsonl'
    if not path.exists():pytest.skip('Frozen 900-record dataset required')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==900 and len({r['statement_idx'] for r in rows})==100
    checked={}
    for record in rows:
        prefix=prompts.prefix_for_record(record,'gold-template')
        # Changing every gold-bearing field cannot influence the prefix.
        poisoned={**record,'answer':'SECRET ANSWER 987654321','gold':{'node':'SECRET','relation':'SECRET',
                 'neighbors':['SECRET'],'paths':[['SECRET']]},'target_node':'SECRET'}
        assert prompts.prefix_for_record(poisoned,'gold-template')==prefix
        assert 'SECRET' not in prefix and '987654321' not in prefix
        if record['task_type']=='node_degree':
            name=record['prompt'].split('"')[1]
            canonical=f'The degree of the node "{name}" is'
            assert record['answer'].startswith(canonical+' ')
            assert record['answer'][len(canonical):].strip()
        else:
            assert record['answer'].startswith(prefix)
            assert record['answer'][len(prefix):].strip()
        checked[record['task_type']]=checked.get(record['task_type'],0)+1
    assert set(checked.values())=={100} and len(checked)==9


@pytest.mark.parametrize('template',['hf-chat','llava_v1'])
def test_prefix_is_assistant_side_and_user_turn_is_unchanged(template):
    record={'task_type':'node_number','prompt':'How many nodes?'}
    args=SimpleNamespace(answer_format='none',prompt_template=template,assistant_prefix_mode='gold-template')
    rendered=runner.render_prompt(Processor(),record,args)
    boundary='[/INST]' if template=='hf-chat' else 'ASSISTANT:'
    assert rendered.endswith(boundary+' There are')
    assert '<image>\n'+record['prompt'] in rendered
    args.assistant_prefix_mode='none'
    assert runner.render_prompt(Processor(),record,args).endswith(boundary)


def test_scoring_uses_prefix_plus_continuation_and_counts_only_new_tokens():
    record={'task_type':'node_number','statement_idx':1,'image':'q.png','prompt':'How many nodes?',
            'answer':'There are 18 nodes in the graph.'}
    args=SimpleNamespace(answer_format='none',assistant_prefix_mode='gold-template',extractor='span_extended',
                         model_id='m',revision='r')
    result=runner.make_result(record,' 18 nodes in the graph.',6,False,100,1.,1234,{},args)
    assert result['raw_response']=='There are 18 nodes in the graph.'
    assert result['continuation_response']==' 18 nodes in the graph.'
    assert result['parsed_integer']==18 and result['is_correct']
    assert result['generated_token_count']==6
    args.assistant_prefix_mode='none'
    result=runner.make_result(record,'18',1,False,100,1.,1234,{},args)
    assert result['raw_response']=='18' and 'assistant_prefix' not in result


def test_prefix_rejects_question_without_unambiguous_named_targets():
    with pytest.raises(ValueError):prompts.gold_template_prefix('node_degree','Look at the node.')
    with pytest.raises(ValueError):prompts.gold_template_prefix('relation_identification','Only "a" is given.')
