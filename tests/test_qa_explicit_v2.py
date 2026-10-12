"""Opt-in first-line QA parsing: frozen v1/default behavior must survive."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'))
from prompt_common import PARSER_PROTOCOL, parse_answer


@pytest.mark.parametrize('response,letter', [
    ("B.\n\nHere's the reasoning: ... The choices are:\n*   A. a marsh", 'B'),
    ('**C**\n\nExplanation: ...', 'C'),
    ('(D)\nBecause ...', 'D'),
    ('A. \n\n**Explanation:** ...', 'A'),
    ('\n  \nB:\nThe options include **A** and **D**.', 'B'),
])
def test_positive_first_nonempty_line(response, letter):
    before = parse_answer(response, 4, parser_version='explicit_v1')
    after = parse_answer(response, 4, parser_version='explicit_v2')
    assert after[0] == letter
    assert after[1] == ('first_line_letter' if before[0] == 'FAILED' else before[1])
    assert parse_answer(response, 4) == before


@pytest.mark.parametrize('response', [
    "A tree's roots grow ...",
    '*   A. a marsh\n*   B. a desert',
    'None of the provided options (A, B, C, or D) are ...',
    "Here's the breakdown ...",
    'b.\nBecause ...',
    'Answer: B\nBecause ...',
    'Explanation first\nB.',
])
def test_failed_nonanswers_remain_failed(response):
    assert parse_answer(response, 4, parser_version='explicit_v1') == ('FAILED', 'FAILED')
    assert parse_answer(response, 4, parser_version='explicit_v2') == ('FAILED', 'FAILED')


@pytest.mark.parametrize('response', [
    'A', 'D.', '(B)', 'A. apple', 'The answer is D.',
    '**C**\nExplanation', 'B.\nThe answer is D.',
    '**A** is wrong, so the answer is D', 'E',
])
def test_v1_success_has_precedence_and_keeps_tier(response):
    before = parse_answer(response, 5, parser_version='explicit_v1')
    assert before[0] != 'FAILED'
    assert parse_answer(response, 5, parser_version='explicit_v2') == before
    assert parse_answer(response, 5) == before


def test_default_protocol_and_choice_count():
    assert PARSER_PROTOCOL == 'explicit_v1'
    assert parse_answer('(D)\nBecause ...', 3, parser_version='explicit_v2') == ('FAILED', 'FAILED')
    with pytest.raises(ValueError, match='Unknown QA parser version'):
        parse_answer('B', 4, parser_version='unknown')
