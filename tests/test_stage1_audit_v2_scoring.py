"""Regression contracts for gold-independent extraction and corrected scoring."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from test_stage1_template_v1_scoring import META, GOLD, STRUCTURED, record
import score_stage1_audit_v2 as scorer
import stage1_scoring as dispatch


@pytest.mark.parametrize('task', scorer.TASK_TYPES)
def test_nine_canonical_answers(task):
    row=scorer.score_record(record(task),GOLD[task],META)
    assert row['is_correct']
    assert row['extraction_decision']=='COMMITTED'
    assert row['scorer_version']=='audit_v2'


@pytest.mark.parametrize('task,text,value', [
    ('edge_number','1. sun -> moon\n2. moon -> earth\nWait, let us count again.\n3. sun -> earth',None),
    ('node_degree','In-degree: 9\nOut-degree: 0\nTherefore the degree (total = in',None),
    ('node_number','There are 31 nodes.\nLet us list them again:\n1. bear\n2. mammal',31),
    ('node_number','There are 31 nodes.\nFinal Answer: unknown',None),
    ('node_number','Final Answer: 8\nExplanation: the first cluster has 2 nodes.',8),
    ('node_number','Final Answer: 8\nFinal Answer: 9',9),
    ('node_degree','✅ **Answer: 3**',3),
    ('highest_node_degree','Answer: earth, degree 10',['earth',10]),
    ('highest_node_degree','oil has the highest degree, with a degree of 8.',['oil',8]),
    ('highest_node_degree','The node with the most connections is "power".\nIt has a total of 10 connections.',['power',10]),
    ('highest_node_degree','- hub: degree 2\n- leaf: degree 1',None),
    ('neighbor_listing','The degree of the node "hub" is 2, and its neighbors are "leaf", and "other".',['leaf','other']),
    ('neighbor_listing','- **use** (connected via "related to")\n- **needed** (connected via "related to")',['use','needed']),
    ('neighbor_listing','- **"bear"**\n- **"baby"**',['bear','baby']),
    ('neighbor_listing','The only node adjacent to "involve" is **involves**.',['involves']),
    ('neighbor_listing','The neighbors are: "fish and chips", "label, with comma", "related to", "node [A,B]".',['fish and chips','label, with comma','related to','node [A,B]']),
    ('neighbor_listing','The degree of "hub" is 2.',None),
    ('node_description','- being\n- but\n- related to (appears multiple times as an edge label)',['being','but','related to']),
    ('relation_identification','The relation that connects "human" and "animal" is "is a".\nAnother arrow is labeled "related to".','isa'),
    ('relation_identification','There is no direct edge connecting "may" and "june"; both are related to summer.','NO_EDGE'),
    ('relation_identification','There are two edges: one is labeled "is a", the other is labeled "related to".',None),
    ('relation_identification','The relation between "hub" and "leaf" is "invented label".','inventedlabel'),
    ('shortest_path_listing','Candidate: leaf -> wrong -> other\nFinal Answer: unknown',None),
    ('shortest_path_listing','Candidate: leaf -> wrong -> other\nFinal Answer: There is no path.',[]),
    ('shortest_path_listing','There is no direct path.\nFinal Answer: leaf -> hub -> other',['leaf','hub','other']),
    ('shortest_path_listing','Answer: "related to" -> "being" -> "St. Louis"',['related to','being','St. Louis']),
    ('triple_listing','1. day - related to - rotation\n2. `sun` `is a` `body`\n3. (moon, part of, earth)',[('day','related to','rotation'),('sun','is a','body'),('moon','part of','earth')]),
])
def test_audited_answer_formats(task,text,value):
    assert scorer.extract(task,text).value==value


def test_highest_headline_requires_name_and_degree_and_accepts_ties():
    row=scorer.score_record(record('highest_node_degree'),'Answer: leaf, degree 2',META)
    metrics=scorer.aggregate_task('highest_node_degree',[row])
    assert row['degree_correct'] and not row['is_correct']
    assert metrics['strict_accuracy']==0 and metrics['strict_degree_accuracy']==1
    meta={'visible_nodes':META['visible_nodes'][:2],'edges':META['edges'][:1]}
    r={**record('highest_node_degree'),'answer':'One node with the highest degree is "hub" with a degree of 1.'}
    assert scorer.score_record(r,'Answer: leaf, degree 1',meta)['is_correct']


def test_wrong_labels_retained_and_parsing_does_not_depend_on_gold():
    task='neighbor_listing';raw='The neighbors are: "wrong", "related to", "being".'
    for gold in (['wrong'],['unmentioned']):
        r=record(task);r['gold']={**r['gold'],'neighbors':gold}
        row=scorer.score_record(r,raw,META)
        assert row['parsed_predicted_nodes']==['wrong','related to','being']
    task='shortest_path_listing';raw='Answer: leaf -> wrong -> other'
    row=scorer.score_record(record(task),raw,META)
    assert row['parsed_path']==['leaf','wrong','other'] and not row['is_correct']


def test_closed_malformed_triple_penalized_incomplete_tail_reported():
    row=scorer.score_record(record('triple_listing'),'(hub, related to, leaf)\n(hub, related to, other)\n(hub, related to,)\n(hub, related to',META)
    assert row['malformed_triple_line_count']==1
    assert row['incomplete_triple_line_count']==1
    assert row['set_metrics']['raw']['false_positive']==1
    assert row['set_metrics']['raw']['f1']==pytest.approx(.8)
    assert not row['is_correct']


def test_version_dispatch_preserves_historical_results():
    for version in ('legacy','span','span_extended'):
        assert dispatch.score_record(record('node_number'),'unknown, Step 17',META,extractor=version)==scorer.base.score_record(record('node_number'),'unknown, Step 17',META,extractor=version)
    assert dispatch.score_record(record('node_number'),'unknown, Step 17',META)['parsed_integer'] is None
    with pytest.raises(ValueError):dispatch.implementation('missing')


def fixture_files(tmp_path):
    data=tmp_path/'data.jsonl';meta=tmp_path/'meta.jsonl';pred=tmp_path/'pred';pred.mkdir()
    data.write_text(''.join(json.dumps({**record(t),'statement_idx':0,'image':'dummy.png','prompt':'Question'})+'\n' for t in scorer.TASK_TYPES))
    meta.write_text(json.dumps({**META,'statement_idx':0})+'\n')
    for t in scorer.TASK_TYPES:
        (pred/f'predictions_test_{t}.jsonl').write_text(json.dumps({'statement_idx':0,'task_type':t,'raw_response':GOLD[t],'is_correct':False,'parsed_integer':999})+'\n')
    return SimpleNamespace(input_jsonl=data,graph_metadata=meta,predictions_dir=pred,model_name='test',task_set='extended',output=tmp_path/'new')


def test_offline_rescore_recomputes_stale_scores_preserves_inputs_refuses_overwrite(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from rescore_stage1_audit_v2 import rescore
    args=fixture_files(tmp_path)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*.jsonl')}
    manifest=rescore(args)
    assert manifest['inputs_unchanged']
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    for t in scorer.TASK_TYPES:
        row=json.loads((args.output/f'scored_test_{t}.jsonl').read_text())
        assert row['is_correct'] and row['scorer_version']=='audit_v2'
    with pytest.raises(FileExistsError):rescore(args)


def test_file_scoring_rejects_duplicate_prediction_identity(tmp_path):
    args=fixture_files(tmp_path);args.extractor='audit_v2'
    p=args.predictions_dir/'predictions_test_node_number.jsonl';p.write_text(p.read_text()*2)
    with pytest.raises(ValueError,match='[Dd]uplicate'):scorer.score_files(args)


def test_explicit_empty_neighbors_get_full_credit_but_missing_answer_does_not():
    r=record('neighbor_listing');r['gold']={**r['gold'],'neighbors':[],'degree':0}
    empty=scorer.score_record(r,'Answer: No neighbors.',META)
    missing=scorer.score_record(r,'Answer: unknown',META)
    assert empty['is_correct'] and empty['set_metrics']['raw']['f1']==1
    assert not missing['is_correct'] and missing['set_metrics']['raw']['f1']==0


@pytest.mark.parametrize('raw,expected', [
    ('The image depicts the relation "related to" between "bird" and "generic".','relatedto'),
    ('The relationship between "ball" and "bounce" is:\n"ball" is "capable of" "bounce".','capableof'),
    ('The specific, direct relationship indicated by the labeled edge is "at location".','atlocation'),
])
def test_relation_labels_and_triples_are_distinct(raw,expected):
    assert scorer.extract('relation_identification',raw).value==expected


def test_count_subtotals_and_triple_explanations_are_not_predictions():
    assert scorer.extract('edge_number','From "sun":\n- sun -> earth\nThat\'s 1 connection.').value is None
    parsed=scorer.extract('triple_listing','1. `sun` `is a` `body`\n2. `other` (This node is isolated, no connections found in KG, so no triples can be formed.)\n* Related to: This is a general relationship indicating a connection.')
    assert parsed.value==[('sun','is a','body')]


def test_corrected_default_is_available_in_all_stage1_runners(monkeypatch):
    import importlib
    for name in ('run_stage1_qwen','run_stage1_deepseek','run_stage1_llava','eval_gemma3_stage1'):
        module=importlib.import_module(name)
        assert module.DEFAULT_EXTRACTOR=='audit_v2'
        assert module.score_record is dispatch.score_record
    import collect_stage1_matrix
    assert collect_stage1_matrix.DEFAULT_EXTRACTOR=='audit_v2'
