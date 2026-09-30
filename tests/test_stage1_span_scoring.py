"""Span extraction regression cases; no inference or GPU dependencies."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rescore_stage1 as offline
from test_stage1_extended_scoring import relation, neighbors, paths, complete_predictions
from test_gemma3_evaluation import dataset

scorer = offline.scorer


@pytest.mark.parametrize('text, expected', [
    ('1. foo\n2. bar\n**Final Answer: 100**', 100), ('42', 42),
    ('1. foo\nTherefore, 9 nodes and 11 edges.', 11),
    ('17 candidates\nAnswer: unknown', 17), ('unknown', None),
])
def test_numeric_extraction(text, expected):
    assert scorer.span_integer(text) == expected


def test_numeric_strict_and_standalone_containment():
    row = scorer.score_record({'task_type': 'node_number', 'answer': 'There are 42 nodes.'},
                              '42 candidates. Final Answer: 7', {})
    assert row['parsed_integer'] == 7 and row['signed_error'] == -35
    assert row['lenient_containment'] is True and not row['is_correct']
    assert scorer.integer_containment('42.', 42)
    for text in ('142', '42.5', 'x42', '-42', '0.42'):
        assert not scorer.integer_containment(text, 42)
    aggregate = scorer.aggregate_task('node_number', [row])
    assert aggregate['strict_accuracy'] == 0
    assert aggregate['lenient_containment'] == 1
    assert aggregate['mean_absolute_error'] == 35


@pytest.mark.parametrize('response', ['Final Answer: Node "node 17", with a degree of 3.',
    'Node "node 17" has degree 3.', 'Node "node 17" has a degree of 3.',
    'Node "node 17": degree of 3; index 17'])
def test_highest_degree_cue(response):
    assert scorer.span_degree(response) == 3
    assert scorer.span_degree('The selected node is "node 17".') is None


def test_highest_degree_strict_name_does_not_use_all_mentions():
    meta = {'visible_nodes': [{'cid': 1, 'name': 'hub'}, {'cid': 2, 'name': 'leaf'},
                              {'cid': 3, 'name': 'other'}],
            'edges': [{'source_cid': 1, 'target_cid': 2}, {'source_cid': 1, 'target_cid': 3}]}
    record = {'task_type': 'highest_node_degree', 'answer': 'hub with a degree of 2'}
    row = scorer.score_record(record, '"hub" has degree 2. Final Answer: "leaf" with a degree of 2', meta)
    assert row['degree_correct'] and not row['is_correct'] and row['lenient_containment']


@pytest.mark.parametrize('tail', ['', '\n\nThis connection is labeled "related to".\n\n'
    'There are no other nodes directly connected to "looking" in the graph.'])
def test_looking_neighbor(tail):
    record = {'task_type': 'neighbor_listing', 'answer': 'look',
              'gold': {'node': 'looking', 'neighbors': ['look'], 'degree': 1}}
    text = ('Based on the provided graph, the node "looking" has only one adjacent node.\n\n'
            'The adjacent node to "looking" is:\n- **look**' + tail)
    row = scorer.score_record(record, text, {})
    assert row['parsed_predicted_nodes'] == ['look']
    assert row['set_metrics']['raw']['precision'] == row['set_metrics']['raw']['recall'] == 1


def test_lists_markers_and_non_answers():
    assert scorer.answer_span('First line\nLast line') == 'Last line'
    assert scorer.answer_span('**Final Answer:**\n* garden\n* flower') == '- garden\n- flower'
    row = scorer.score_record(neighbors(), 'Analysis\n- irrelevant\nFinal Answer:\n'
        '* garden [A,B]\n* related to\n* the and\n* flower\nExplanation afterwards.', {})
    assert row['parsed_predicted_nodes'] == ['garden [A,B]', 'flower']
    assert scorer.span_node_items('Nodes:\n- garden\nCategory two:\n- flower\nDone.') == ['garden', 'flower']


def test_relation_prose_and_multiple_candidates():
    record = relation()
    record['gold'].update(relation='hassubevent', relation_text='has subevent')
    text = 'Based on the graph, the relation that links **"eat"** to **"have lunch"** is **"has subevent"**.'
    assert scorer.score_record(record, text, {})['parsed_relation'] == 'hassubevent'
    row = scorer.score_record(record, 'has subevent, at location, related to', {})
    assert row['parsed_relation'] == 'relatedto' and not row['is_correct'] and row['lenient_containment']
    row = scorer.score_record(record, 'has subevent\nFinal Answer: unknown', {})
    assert row['parsed_relation'] is None and row['lenient_containment']
    assert scorer.span_relation('not capable of') == 'notcapableof'
    assert scorer.span_relation('notdesires') == 'notdesires'
    assert scorer.span_relation('AT-LOCATION') == 'atlocation'
    assert scorer.span_relation('causesomething') is None


def test_relatedto_excluded_and_empty():
    related = relation()
    related['gold'].update(relation='relatedto', relation_text='related to')
    rows = [scorer.score_record(related, 'related to', {}), scorer.score_record(relation(), 'unknown', {})]
    metrics = scorer.aggregate_task('relation_identification', rows)
    assert metrics['strict_accuracy'] == .5
    assert metrics['relatedto_excluded']['strict_accuracy'] == 0
    assert metrics['relatedto_excluded']['record_count'] == 1
    assert scorer.aggregate_task('relation_identification', rows[:1])['relatedto_excluded']['strict_accuracy'] is None


def test_last_complete_path_and_cycle():
    row = scorer.score_record(paths(), 'start [A,B] -> garden -> finish\n'
        'Final Answer: start [A,B] -> water -> finish\nA later incomplete trace: start [A,B] -> garden', {})
    assert row['parsed_path'] == ['start [A,B]', 'water', 'finish']
    assert row['path_exact']['raw'] and not row['degenerate_output']
    parsed, cycle = scorer.span_path('a -> related to -> b -> c -> b', 'a', 'z')
    assert parsed == ['a', 'b', 'c', 'b'] and cycle
    parsed, cycle = scorer.span_path('a -> b -> c -> b\nThere is no route, unfortunately.', 'a', 'z')
    assert parsed == ['a', 'b', 'c', 'b'] and cycle
    record = {'task_type': 'shortest_path_listing', 'answer': '', 'gold': {
        'node_a': 'a', 'node_b': 'b', 'paths': [['a', 'b']], 'distance': 1, 'paths_truncated': False}}
    row = scorer.score_record(record, 'a -> b -> c -> b', {})
    assert row['degenerate_output'] and not row['path_exact']['raw']
    assert scorer.aggregate_task('shortest_path_listing', [row])['degenerate_output_fraction'] == 1


@pytest.mark.parametrize('final', ['start [A,B], water, finish',
    '1. start [A,B]\n2. water\n3. finish'])
def test_final_comma_or_numbered_path_supersedes_arrow_trace(final):
    row = scorer.score_record(paths(), 'start [A,B] -> garden -> finish\nFinal Answer:\n' + final, {})
    assert row['parsed_path'] == ['start [A,B]', 'water', 'finish']


def test_triples_keep_relation_component_and_parse_backticks():
    record = {'task_type': 'triple_listing', 'answer': '(garden, related to, flower)'}
    row = scorer.score_record(record, 'Explanation\n1. `garden` `related to` `flower`\nDone.', {})
    assert row['set_metrics']['raw']['f1'] == 1
    assert row['lenient_containment'] == 1


def test_offline_new_directory_and_immutable_predictions(dataset):
    args = complete_predictions(dataset, dataset.root / 'predictions', dataset.extended, scorer.EXTENDED_TASK_TYPES)
    before = {p: p.read_bytes() for p in args.predictions_dir.iterdir()}
    output = dataset.root / 'rescore'
    command = [sys.executable, offline.__file__, '--predictions-dir', str(args.predictions_dir),
        '--input-jsonl', str(args.input_jsonl), '--graph-metadata', str(args.graph_metadata),
        '--model-name', args.model_name, '--task-set', 'extended', '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert before == {p: p.read_bytes() for p in args.predictions_dir.iterdir()}
    metrics = json.loads((output / 'metrics_fixture.json').read_text())
    assert metrics['scoring_notes']['extractor'] == 'span'
    assert (output / 'scored_fixture_shortest_path_listing.jsonl').exists()
    assert len(json.loads((output / 'comparison.json').read_text())['rows']) == 9
    retry = subprocess.run(command, capture_output=True, text=True)
    assert retry.returncode != 0 and 'new directory' in retry.stderr
