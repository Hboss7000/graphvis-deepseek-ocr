"""Template scoring contracts: asserted answers, errors and immutable legacy API."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/score_stage1_template_v1.py'
spec = importlib.util.spec_from_file_location('template_scorer', PATH)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

META = {'visible_nodes': [{'cid': 1, 'name': 'hub'}, {'cid': 2, 'name': 'leaf'}, {'cid': 3, 'name': 'other'}],
        'edges': [{'source_cid': 1, 'target_cid': 2}, {'source_cid': 1, 'target_cid': 3}]}
GOLD = {
    'node_description': 'The image depicts the following nodes: hub, leaf, other.',
    'node_number': 'There are 3 nodes in the graph.',
    'edge_number': 'There are 2 edges in the graph.',
    'triple_listing': 'The triples in the graph are listed as: (hub, related to, leaf), (hub, related to, other).',
    'highest_node_degree': 'One node with the highest degree is "hub" with a degree of 2.',
    'node_degree': 'The degree of the node "leaf" is 1.',
    'relation_identification': 'The relation on the edge between "hub" and "leaf" is "related to".',
    'neighbor_listing': 'The node "hub" is directly connected to: leaf, other.',
    'shortest_path_listing': 'One shortest path from "leaf" to "other" is: leaf -> hub -> other.',
}
STRUCTURED = {
    'relation_identification': {'node_a': 'hub', 'node_b': 'leaf', 'relation': 'relatedto', 'relation_text': 'related to'},
    'neighbor_listing': {'node': 'hub', 'neighbors': ['leaf', 'other'], 'degree': 2},
    'shortest_path_listing': {'node_a': 'leaf', 'node_b': 'other', 'distance': 2,
                            'paths': [['leaf', 'hub', 'other']], 'paths_truncated': False},
}


def record(task):
    return {'task_type': task, 'answer': GOLD[task], **({'gold': STRUCTURED[task]} if task in STRUCTURED else {})}


@pytest.mark.parametrize('task', scorer.TASK_TYPES)
def test_all_nine_canonical_templates_score_correctly(task):
    row = scorer.score_record(record(task), GOLD[task], META)
    assert row['is_correct']
    assert row['scorer_version'] == 'template_v1'
    metrics = scorer.aggregate_task(task, [row])
    if task in {'node_description', 'triple_listing', 'neighbor_listing'}:
        assert row['set_metrics']['raw']['f1'] == 1
    else:
        assert metrics['strict_accuracy'] == 1


@pytest.mark.parametrize('text, names', [
    ('The degree of the node "hub" is 2, and its neighbors are "leaf" and "other".', ['leaf', 'other']),
    ('The degree of the node "hub" is 2. Its neighbors are: leaf, other.', ['leaf', 'other']),
    ('The degree of the node "hub" is 2, and it is connected to "leaf", and "other".', ['leaf', 'other']),
    ('The degree of the node "hub" is 2, and it is connected to the node "leaf".', ['leaf']),
    ('The image depicts connections for "hub": "hub" is connected to "leaf" and "other".', ['leaf', 'other']),
    ('One node that shares an edge with "hub" is "leaf".', ['leaf']),
    ('The degree of the node "hub" is 2.', []),
    ('The neighbors are: "leaf", "other", "wrong".', ['leaf', 'other', 'wrong']),
    ('Final Answer: The neighbors are: "leaf", "other". Explanation: "wrong" is not adjacent.', ['leaf', 'other']),
])
def test_neighbor_payload_excludes_query_degree_and_explanation(text, names):
    row = scorer.score_record(record('neighbor_listing'), text, META)
    assert row['parsed_predicted_nodes'] == names
    assert row['set_metrics']['raw']['false_positive'] == int('wrong' in names)
    assert row['set_metrics']['raw']['false_negative'] == 2 - len({'leaf', 'other'} & set(names))


def test_no_gold_lookup_or_stopword_removal_and_quote_syntax():
    text = 'The neighbors are: "fish and chips", "label, with comma", "related to", "node [A,B]".'
    expected = ['fish and chips', 'label, with comma', 'related to', 'node [A,B]']
    assert scorer.parse_template('neighbor_listing', text)[1] == expected
    for gold in (['fish and chips'], ['unmentioned']):
        r = record('neighbor_listing')
        r['gold'] = {**r['gold'], 'neighbors': gold}
        scored = scorer.score_record(r, text, META)
        assert scored['parsed_predicted_nodes'] == expected


def test_highest_assertion_is_selected_not_any_valid_name_mentioned():
    text = 'The node with the highest degree is "hub" with a degree of 2.\nFinal Answer: One node with the highest degree is "leaf" with a degree of 2.'
    row = scorer.score_record(record('highest_node_degree'), text, META)
    assert row['parsed_name'] == 'leaf'
    assert row['degree_correct'] and not row['name_correct_raw'] and not row['is_correct']
    row = scorer.score_record(record('highest_node_degree'), 'One node with the highest degree is "hub" with a degree of 7.', META)
    assert row['name_correct_raw'] and not row['degree_correct'] and not row['is_correct']


def test_numeric_label_is_not_degree_or_count():
    r = record('node_degree')
    row = scorer.score_record(r, 'The degree of the node "node 99" is 1.', META)
    assert row['parsed_integer'] == 1
    row = scorer.score_record(record('node_number'), 'There are 7 nodes in the graph.', META)
    assert row['parsed_integer'] == 7 and row['signed_error'] == 4 and not row['is_correct']


def test_normalization_remains_raw_basic_annotation_stripped():
    row = scorer.score_record(record('highest_node_degree'), 'One node with the highest degree is "HUB [A]" with a degree of 2.', META)
    assert not row['name_correct_raw'] and not row['name_correct_basic']
    assert row['name_correct_annotation_stripped']
    row = scorer.score_record(record('neighbor_listing'), 'The neighbors are: "LEAF", "OTHER".', META)
    assert row['set_metrics']['raw']['f1'] == 0
    assert row['set_metrics']['basic']['f1'] == 1


def test_ties_and_strict_selected_name():
    meta = {'visible_nodes': META['visible_nodes'][:2], 'edges': META['edges'][:1]}
    source = {**record('highest_node_degree'), 'answer': 'One node with the highest degree is "hub" with a degree of 1.'}
    row = scorer.score_record(source, 'One node with the highest degree is "leaf" with a degree of 1.', meta)
    assert row['is_correct'] and set(row['valid_highest_names']) == {'hub', 'leaf'}


def test_relation_assertion_precedes_conflicting_explanation():
    text = GOLD['relation_identification'] + ' It is not "at location".'
    row = scorer.score_record(record('relation_identification'), text, META)
    assert row['parsed_relation'] == 'relatedto' and row['is_correct']
    row = scorer.score_record(record('relation_identification'), 'The relation on the edge between "hub" and "leaf" is "unknown".', META)
    assert row['parsed_relation'] is None and not row['is_correct']


def test_malformed_triples_still_penalized_and_quotes_are_syntax():
    text = 'The triples in the graph are listed as: ("hub", "related to", "leaf"), (hub, related to).'
    row = scorer.score_record(record('triple_listing'), text, META)
    assert row['parsed_predicted_triples'] == [['hub', 'related to', 'leaf']]
    assert row['malformed_triple_line_count'] == 1
    assert row['set_metrics']['raw']['false_positive'] == 1
    assert row['set_metrics']['raw']['false_negative'] == 1


def test_path_order_cycle_and_quoted_labels():
    row = scorer.score_record(record('shortest_path_listing'), 'One shortest path from "leaf" to "other" is: "leaf" -> "hub" -> "other".', META)
    assert row['parsed_path'] == ['leaf', 'hub', 'other'] and row['is_correct']
    row = scorer.score_record(record('shortest_path_listing'), 'One shortest path from "leaf" to "other" is: leaf -> hub -> leaf -> other.', META)
    assert row['degenerate_output'] and not row['is_correct']
    row = scorer.score_record(record('shortest_path_listing'), 'One shortest path from "leaf" to "other" is: other -> hub -> leaf.', META)
    assert not row['is_correct']


def test_baseline_bullet_lists_and_trailing_prose_use_compatibility_fallback():
    text = 'The adjacent node is:\n- **leaf**\n\nThis is the only node directly connected to "hub" in the graph.'
    row = scorer.score_record(record('neighbor_listing'), text, META)
    assert row['parsed_predicted_nodes'] == ['leaf']
    assert row['parser_rule'] == 'list_syntax_fallback'


def test_intro_and_commentary_do_not_supplant_bullet_answers():
    for intro in ['The node "hub" is connected to the following nodes:',
                  'The node that shares an edge with "hub" is:']:
        text = intro + '\n- leaf\n- other\n\nThese neighbors are listed above.'
        row = scorer.score_record(record('neighbor_listing'), text, META)
        assert row['parsed_predicted_nodes'] == ['leaf', 'other']
        assert row['is_correct']


def test_literal_labels_preserved_in_both_template_and_baseline_bullets():
    expected = ['being', 'made of', 'but', 'and pieces']
    for text in ['The image depicts the following nodes: being, made of, but, and pieces.',
                 'Nodes:\n- being\n- made of\n- but\n- and pieces\n\nThese are concepts.']:
        row = scorer.score_record(record('node_description'), text, META)
        assert row['parsed_predicted_nodes'] == expected


def test_original_scorer_unchanged_in_memory_and_on_disk():
    path = Path(scorer.base.__file__)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    old = scorer.base.score_record(record('highest_node_degree'), GOLD['highest_node_degree'], META, extractor='span_extended')
    new = scorer.score_record(record('highest_node_degree'), GOLD['highest_node_degree'], META)
    again = scorer.base.score_record(record('highest_node_degree'), GOLD['highest_node_degree'], META, extractor='span_extended')
    assert old == again and before == hashlib.sha256(path.read_bytes()).hexdigest()
    assert old['parsed_name'] == 'hub" with a degree of 2'
    assert new['parsed_name'] == 'hub'


def test_all_nine_files_recompute_stale_fields_and_validate_keys(tmp_path):
    source = tmp_path/'source.jsonl'
    meta = tmp_path/'metadata.jsonl'
    folder = tmp_path/'predictions'
    folder.mkdir()
    records = [{**record(task), 'statement_idx': 1, 'prompt': 'fixture', 'image': 'fixture.png'} for task in scorer.TASK_TYPES]
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    meta.write_text(json.dumps({**META, 'statement_idx': 1})+'\n')
    for r in records:
        (folder/f"predictions_fixture_{r['task_type']}.jsonl").write_text(json.dumps({
            'statement_idx': 1, 'task_type': r['task_type'], 'raw_response': r['answer'],
            'hit_token_ceiling': False, 'generated_token_count': 30,
            'parsed_name': 'stale', 'is_correct': False})+'\n')
    opts = SimpleNamespace(input_jsonl=source, graph_metadata=meta, predictions_dir=folder,
        task_set='extended', model_name='fixture', extractor=scorer.VERSION)
    original_hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,meta,*folder.iterdir()]}
    metrics, rows = scorer.score_files(opts)
    assert tuple(rows) == scorer.TASK_TYPES and metrics['scorer_version'] == scorer.VERSION
    assert all(rs[0]['is_correct'] for rs in rows.values())
    assert rows['highest_node_degree'][0]['parsed_name'] == 'hub'
    assert rows['highest_node_degree'][0]['raw_response'] == GOLD['highest_node_degree']
    assert not rows['highest_node_degree'][0]['hit_token_ceiling']
    assert original_hashes == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in original_hashes}
    path = folder/'predictions_fixture_node_number.jsonl'
    path.write_text(path.read_text()*2)
    with pytest.raises(ValueError, match='Duplicate prediction key'):
        scorer.score_files(opts)


def test_paired_cli_refuses_existing_output_before_any_evidence_read(tmp_path):
    path = ROOT/'scripts/rescore_stage1_template_v1.py'
    spec = importlib.util.spec_from_file_location('paired_template_rescore',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentinel = tmp_path/'evidence'
    sentinel.write_text('preserve me')
    with pytest.raises(FileExistsError, match='new directory'):
        module.rescore(SimpleNamespace(output=tmp_path))
    assert sentinel.read_text() == 'preserve me'


def test_quoted_comma_in_triple_label_and_unquoted_leading_and_are_literal():
    parsed = scorer.parse_template('triple_listing', 'The triples in the graph are listed as: ("label, with comma", "related to", "other").')
    assert parsed[1] == ([('label, with comma', 'related to', 'other')],0)
    assert scorer.label_list('leaf, and pieces.') == ['leaf','and pieces']
