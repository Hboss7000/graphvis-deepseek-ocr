"""CPU selection checks; these need no model/training libraries."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_llava_stage1_training import select
from llava_stage1_prompt import format_stage1_prompt
from llava_stage1_training import training_text


def test_selection_samples_sorted_question_ids_and_excludes_validation():
    statements = [{'id': name} for name in ['z', 'b', 'a', 'e', 'd']]
    rng = random.Random(13)
    expected_train = rng.sample(sorted(row['id'] for row in statements), 3)
    expected_val = rng.sample(sorted(set(row['id'] for row in statements) - set(expected_train)), 1)
    selected = select(statements, [{'id': 'test'}], 3, 1)
    assert [statements[i]['id'] for i in selected['training']] == [
        row['id'] for row in statements if row['id'] in expected_train]
    assert [statements[i]['id'] for i in selected['validation']] == expected_val
    assert not set(selected['training']) & set(selected['validation'])


def test_duplicate_and_test_overlap_fail():
    with pytest.raises(ValueError, match='Duplicate'):
        select([{'id': 'a'}, {'id': 'a'}], [], 1, 1)
    with pytest.raises(ValueError, match='overlap'):
        select([{'id': 'a'}, {'id': 'b'}], [{'id': 'a'}], 1, 1)


def test_training_text_uses_the_existing_evaluation_template_and_one_final_eos():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token='</s>'))
    row = {'prompt': 'How many nodes are there in the graph?',
           'answer': 'There are 3 nodes in the graph.'}
    prefix, full = training_text(processor, row)
    assert prefix == format_stage1_prompt(processor, '<image>\n' + row['prompt'], 'llava_v1')
    assert full == prefix + ' ' + row['answer'] + '</s>'
    assert full.count('<image>') == full.count('</s>') == 1
    for answer in ('', 'Oops</s>', '<image> hidden marker'):
        with pytest.raises(ValueError):
            training_text(processor, {**row, 'answer': answer})
