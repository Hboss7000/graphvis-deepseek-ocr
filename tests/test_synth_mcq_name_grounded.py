"""Name-grounded schema, independent solver, question rows and graph-cluster checks."""
import copy
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from synth_mcq_common import (DEFAULT_DATASET, LETTERS, load_graphs, questions_for, dataset_stats,
    solver_check, wrong_graph_pairing, evaluation_items, permuted_choices, question_type,
    logit_scores, read_predictions, evidence_prompt, shuffled_triples, prompt_body, graph_metadata,
    format_kg_block, expand_conditions)
from run_synth_mcq import parse_args, run_identity
from analyze_synth_mcq import analyze


@pytest.fixture(scope='module')
def grounded():
    return load_graphs(DEFAULT_DATASET)


def test_loader_has_50_graphs_100_unique_questions_no_compatibility_double_count(grounded):
    stats = dataset_stats(grounded)
    assert (stats['graphs'], stats['questions'], stats['triples']) == (50, 100, 2169)
    assert (stats['triples_min'], stats['triples_max']) == (40, 49)
    assert len({q['id'] for g in grounded for q in questions_for(g)}) == 100
    assert all(g['question'] == questions_for(g)[0] for g in grounded)
    for g in grounded:
        people = {n['id'] for n in g['nodes'] if n['kind'] == 'person_name'}
        assert all(set(q['choices'].values()) == people for q in questions_for(g))


@pytest.mark.parametrize('corruption', ['graph_count', 'compatibility', 'questions_count', 'choice_keys',
    'duplicate_choice', 'concept_choice', 'gold', 'support', 'duplicate_question', 'node_kind', 'unused_node'])
def test_new_loader_rejects_corrupted_schema(tmp_path, corruption):
    payload = json.loads(DEFAULT_DATASET.read_text())
    g = payload['graphs'][0]; q = g['questions'][1]
    if corruption == 'graph_count': payload['graphs'].pop()
    elif corruption == 'compatibility': g['question']['text'] += ' changed'
    elif corruption == 'questions_count': g['questions'].pop()
    elif corruption == 'choice_keys': q['choices']['E'] = q['choices'].pop('A')
    elif corruption == 'duplicate_choice': q['choices']['A'] = q['choices']['B']
    elif corruption == 'concept_choice':
        letter = next(l for l in LETTERS if l != q['correct_answer'])
        q['choices'][letter] = next(n['id'] for n in g['nodes'] if n['kind'] == 'concept')
    elif corruption == 'gold': q['correct_answer'] = next(l for l in LETTERS if l != q['correct_answer'])
    elif corruption == 'support': q['supporting_triples'][1]['relation'] = 'missing'
    elif corruption == 'duplicate_question': q['id'] = g['questions'][0]['id']
    elif corruption == 'node_kind': g['nodes'][0]['kind'] = 'concept'
    elif corruption == 'unused_node':
        node = next(n['id'] for n in g['nodes'] if n['kind'] == 'concept' and n['id'] not in {t['source'] for q in g['questions'] for t in q['supporting_triples']} and n['id'] not in {t['target'] for q in g['questions'] for t in q['supporting_triples']})
        g['triples'] = [t for t in g['triples'] if node not in (t['source'], t['target'])]
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): load_graphs(path)


def test_solver_reproduces_100_two_two_one_gold_matches_and_11_types(grounded):
    report = solver_check(grounded)
    assert report['n_questions'] == report['passed'] == 100 and report['mismatches'] == 0
    assert report['clue_count_distribution'] == {'2,2,1': 100}
    assert len({r['question_type'] for r in report['questions']}) == 11
    assert all(r['conjunction_matches'] == [r['correct_node']] for r in report['questions'])
    corrupt = copy.deepcopy(grounded)
    # A fact outside supporting_triples creates a second conjunction match.
    g = corrupt[0]; q = g['questions'][0]
    artifact = next(t['source'] for t in q['supporting_triples'] if t['relation'] == 'createdby')
    other = next(n['id'] for n in g['nodes'] if n['kind'] == 'person_name' and n['id'] != q['correct_node'])
    g['triples'].append(dict(source=artifact, relation='createdby', target=other))
    with pytest.raises(ValueError, match='solver mismatch'): solver_check(corrupt)


def test_derangement_is_seeded_collision_free_and_reports_redraws(grounded):
    a = wrong_graph_pairing(grounded, 0); b = wrong_graph_pairing(grounded, 0)
    assert a == b and a['redraws'] >= 0 and a['draws'] == a['redraws'] + 1
    assert a['mapping'] != wrong_graph_pairing(grounded, 1)['mapping']
    indexed = {g['id']: g for g in grounded}
    assert set(a['mapping']) == set(a['mapping'].values()) == set(indexed)
    for gid, shown in a['mapping'].items():
        assert gid != shown
        people = {n['id'] for n in indexed[shown]['nodes'] if n['kind'] == 'person_name'}
        assert all(q['correct_node'] not in people for q in questions_for(indexed[gid]))
    impossible = copy.deepcopy(grounded[:2]); impossible[1]['nodes'] = copy.deepcopy(impossible[0]['nodes'])
    impossible[1]['questions'] = copy.deepcopy(impossible[0]['questions'])
    with pytest.raises(ValueError, match='No valid wrong-graph derangement'): wrong_graph_pairing(impossible, 0)


def test_dataset_aliases_default_and_resolved_hash_identity(tmp_path):
    common = ['--out', str(tmp_path), '--models', 'base']
    default = parse_args(common)
    assert default.graphs == default.dataset_path == DEFAULT_DATASET.resolve()
    old = parse_args([*common, '--graphs', str(DEFAULT_DATASET)])
    new = parse_args([*common, '--dataset-path', str(DEFAULT_DATASET)])
    both = parse_args([*common, '--graphs', str(DEFAULT_DATASET), '--dataset-path', str(DEFAULT_DATASET)])
    assert run_identity(old) == run_identity(new) == run_identity(both)
    identity = run_identity(new)
    assert identity['dataset_path'] == str(DEFAULT_DATASET.resolve())
    assert identity['dataset_sha256'] == identity['input_sha256']
    with pytest.raises(SystemExit):
        parse_args([*common, '--graphs', str(ROOT.parent / 'knowledge_graphs_50_triples_questions.json'), '--dataset-path', str(DEFAULT_DATASET)])


def test_full_grid_counts_and_wrong_graph_modalities(grounded, tmp_path):
    args = parse_args(['--out', str(tmp_path), '--adapter', '/fixture', '--prompt-styles', 'obqa,needed'])
    items = evaluation_items(grounded, args.models, args.conditions, args.prompt_styles, args.n_perms)
    assert len(items) == len(set(items)) == 8000
    assert len(evaluation_items(grounded, args.models, args.conditions, ['obqa'], 4)) == 4000
    assert set(args.conditions) == {'image', 'text', 'none', 'wrong_image', 'wrong_text'}
    assert expand_conditions(['wrong_graph'], 'image') == ['wrong_image']
    assert expand_conditions(['wrong_graph'], 'text') == ['wrong_text']
    assert expand_conditions(['wrong_graph'], 'both') == ['wrong_image', 'wrong_text']


def test_wrong_evidence_preserves_question_choices_and_uses_shown_triple_order(grounded):
    graph, shown = grounded[:2]; manifest = {'graphs': {}}
    for g in (graph, shown):
        triples = shuffled_triples(g, 0)
        bodies = {q['id']: {str(p): {c: prompt_body(g, permuted_choices(g, p, q), c, triples, question=q)
                  for c in ('image', 'text', 'none')} for p in range(4)} for q in questions_for(g)}
        manifest['graphs'][g['id']] = dict(triples=triples, question_prompt_bodies=bodies,
            prompt_bodies=bodies[g['questions'][0]['id']])
    for q in questions_for(graph):
        for p in range(4):
            right, _ = evidence_prompt(manifest, graph, q, p, 'image', 'obqa', graph)
            wrong, modality = evidence_prompt(manifest, graph, q, p, 'wrong_image', 'obqa', shown)
            assert right == wrong and modality == 'image'
            wrong_text, modality = evidence_prompt(manifest, graph, q, p, 'wrong_text', 'obqa', shown)
            block = format_kg_block(graph_metadata(shown, manifest['graphs'][shown['id']]['triples'])[2], preserve_order=True)
            assert block in wrong_text and q['text'] in wrong_text and modality == 'text'
            for l, name in permuted_choices(graph, p, q).items(): assert f'{l}. {name}' in wrong_text


def name_grounded_fake_rows(graphs, styles=('obqa',), known=False):
    rows = []
    for style, model, condition, gid, qid, p in evaluation_items(graphs, ('base', 'tuned'),
            ('image', 'text', 'none', 'wrong_image', 'wrong_text'), styles, 4):
        g = next(g for g in graphs if g['id'] == gid); q = next(q for q in questions_for(g) if q['id'] == qid)
        choices = permuted_choices(g, p, q); gold = next(l for l in LETTERS if choices[l] == q['correct_node'])
        if known:
            correct = condition == 'text' or (condition == 'image' and (
                (model == 'base' and q == questions_for(g)[1 if gid == graphs[0]['id'] else 0]) or
                (model == 'tuned' and gid == graphs[0]['id'])))
        else: correct = (p + int(qid[1:]) + (model == 'base')) % 3 != 0
        letter = gold if correct else next(l for l in LETTERS if l != gold)
        logits = {l: float(l == letter) for l in LETTERS}
        rows.append(dict(prompt_style=style, model=model, condition=condition, graph_id=gid, question_id=qid,
            question_ids=[q['id'] for q in questions_for(g)], questions_per_graph=2, dataset='name_grounded_50',
            question_type=question_type(q), perm=p, choices=choices, correct_node=q['correct_node'], correct_letter=gold,
            predicted_letter=letter, predicted_node=choices[letter], is_correct=correct, parse_ok=True,
            parse_tier='exact', parser_protocol='explicit_v1', render_style='training', render_fontsize=30,
            first_letter_logits=logits, **logit_scores(logits, choices, q['correct_node'])))
    return rows


def test_known_paired_graph_table_clusters_two_questions_and_all_controls(grounded, tmp_path):
    rows = name_grounded_fake_rows(grounded[:2], known=True)
    pred = tmp_path / 'predictions.jsonl'; pred.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert len(read_predictions(pred)) == 160
    summary, pairs = analyze(pred, tmp_path / 'analysis', repetitions=500, n_perms=4)
    base = next(r for r in summary if r['scope'] == 'overall' and r['model'] == 'base' and r['condition'] == 'image' and r['metric'] == 'parsed_letter')
    assert base['n_graphs'] == 2 and base['n_rows'] == 16 and base['accuracy'] == .5
    delta = next(r for r in pairs if r['left'] == 'tuned_image' and r['right'] == 'base_image' and r['metric'] == 'parsed_letter')
    assert delta['mean_difference'] == 0 and delta['ci_low'] == -.5 and delta['ci_high'] == .5
    assert (delta['graphs_improved'], delta['graphs_worsened'], delta['graphs_tied']) == (1, 1, 0)
    for control in ('none', 'wrong_image', 'wrong_text'):
        comparison = next(r for r in pairs if r['left'] == 'base_image' and r['right'] == f'base_{control}' and r['metric'] == 'parsed_letter')
        assert comparison['mean_difference'] == comparison['ci_low'] == comparison['ci_high'] == .5
    with (tmp_path / 'analysis/per_graph.csv').open() as handle: per_graph = list(csv.DictReader(handle))
    assert all(int(r['n_rows']) == 8 and int(r['n_questions']) == 2 for r in per_graph)
    with (tmp_path / 'analysis/question_type_breakdown.csv').open() as handle:
        assert all(r['label'] == 'descriptive, small n' for r in csv.DictReader(handle))
    # An entire missing second question must exclude its graph, not turn it into a 4-row cluster.
    damaged = [r for r in rows if not (r['graph_id'] == grounded[0]['id'] and r['question_id'] == grounded[0]['questions'][1]['id'] and r['model'] == 'base' and r['condition'] == 'image')]
    pred.write_text(''.join(json.dumps(r)+'\n' for r in damaged))
    summaries, _ = analyze(pred, tmp_path / 'partial', repetitions=200, n_perms=4)
    assert next(r for r in summaries if r['scope'] == 'overall' and r['model'] == 'base' and r['condition'] == 'image' and r['metric'] == 'parsed_letter')['n_graphs'] == 1


def test_image_gate_blocks_missing_failed_or_unmatched_audit_before_inference(tmp_path):
    from run_synth_mcq import validate_image_gate
    from synth_mcq_common import sha256_file, MIN_PIXELS, MAX_PIXELS
    args = parse_args(['--out', str(tmp_path), '--models', 'base'])
    with pytest.raises(ValueError, match='requires --image-audit'): validate_image_gate(args, 'manifest')
    args.image_audit = tmp_path / 'gate.json'
    gate = dict(dataset_sha256=sha256_file(DEFAULT_DATASET), render_manifest_sha256='manifest',
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, n_images=50, render_style='training',
                node_fontsize=30, edge_fontsize=24, threshold_px=10,
                nodes_below_10px=0, edges_below_10px=0, passed=True)
    args.image_audit.write_text(json.dumps(gate)); validate_image_gate(args, 'manifest')
    gate.update(passed=False, edges_below_10px=1); args.image_audit.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='STOP'): validate_image_gate(args, 'manifest')
    gate.update(passed=True, edges_below_10px=0, dataset_sha256='another_dataset'); args.image_audit.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='differs'): validate_image_gate(args, 'manifest')


def test_new_question_condition_style_rows_config_and_resume_cpu_only(grounded, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import weakref
    import run_synth_mcq as runner
    import run_zero_shot_qwen as qa
    from synth_mcq_common import sha256_file, MIN_PIXELS, MAX_PIXELS
    manifest = {'graphs': {}}
    for g in grounded:
        triples = shuffled_triples(g, 0)
        bodies = {q['id']: {str(p): {c: prompt_body(g, permuted_choices(g, p, q), c, triples, question=q)
                  for c in ('image', 'text', 'none')} for p in range(4)} for q in questions_for(g)}
        manifest['graphs'][g['id']] = dict(triples=triples, question_prompt_bodies=bodies,
            prompt_bodies=bodies[g['questions'][0]['id']])
    (tmp_path / 'render_manifest.json').write_text(json.dumps(manifest))
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps(dict(dataset_sha256=sha256_file(DEFAULT_DATASET),
        render_manifest_sha256=sha256_file(tmp_path / 'render_manifest.json'),
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, n_images=50, render_style='training',
        node_fontsize=30, edge_fontsize=24, threshold_px=10,
        nodes_below_10px=0, edges_below_10px=0, passed=True)))
    monkeypatch.setattr(runner, 'verify_renders', lambda *args: manifest)
    events, refs, seen_inputs = [], [], []
    class Model:
        def named_parameters(self): return iter([('weights', None)])
        def named_modules(self): return iter([('', SimpleNamespace())])
    def load(*args):
        assert all(ref() is None for ref in refs)
        model = Model(); refs.append(weakref.ref(model)); events.append('pristine')
        return model, 'dtype', 'torch.bfloat16'
    def adapter(model, *args):
        events.append('adapter'); return model, {'sha256': 'fixture'}
    def infer(model, processor, body, modality, image_path, args, torch):
        seen_inputs.append((modality, image_path.stem, body))
        return dict(predicted_letter='A', parse_ok=True, parse_tier='exact', parser_protocol='explicit_v1',
            first_letter_logits={l: float(l == 'A') for l in LETTERS}, vision_tokens=123 if modality == 'image' else 0,
            item_elapsed_seconds=.01, peak_memory_allocated_bytes=0)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='cpu stand-in', cuda=SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True, empty_cache=lambda: None, memory_allocated=lambda: 0)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='5.16.1', AutoProcessor=object,
        Qwen3VLForConditionalGeneration=object, set_seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, 'llava_adapter', SimpleNamespace(load_adapter=adapter, adapter_provenance=lambda *args: {'sha256': 'fixture'}))
    monkeypatch.setattr(qa, 'load_model', load)
    monkeypatch.setattr(qa, 'load_processor', lambda *args: (SimpleNamespace(), 'fixture'))
    monkeypatch.setattr(runner, 'infer_one', infer)
    monkeypatch.setattr(runner, 'load_sanity', lambda model, name, path: dict(pristine_base=name == 'base', lora_tensors=0 if name == 'base' else 2, merger_tensors=0 if name == 'base' else 4))
    common = ['--dataset-path', str(DEFAULT_DATASET), '--out', str(tmp_path), '--image-audit', str(gate),
              '--prompt-styles', 'obqa,needed', '--n-perms', '1', '--adapter', '/fixture']
    runner.main([*common, '--models', 'base', '--limit', '1'])
    assert len(read_predictions(tmp_path / 'predictions.jsonl')) == 20
    runner.main([*common, '--models', 'tuned', '--limit', '2', '--resume'])
    runner.main([*common, '--limit', '2', '--resume'])
    assert events == ['pristine', 'pristine', 'adapter', 'pristine']
    rows = read_predictions(tmp_path / 'predictions.jsonl'); assert len(rows) == len(seen_inputs) == 80
    runner.main([*common, '--limit', '2', '--resume']); assert len(events) == 4
    assert {r['question_id'] for r in rows} == {'Q01', 'Q02', 'Q51', 'Q52'}
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['dataset_path'] == config['identity']['dataset_path'] == str(DEFAULT_DATASET.resolve())
    assert config['dataset_sha256'] == config['identity']['dataset_sha256'] == sha256_file(DEFAULT_DATASET)
    assert config['solver_check']['passed'] == 100 and config['wrong_graph_pairing']['gold_name_collisions'] == 0
    pairing = config['wrong_graph_pairing']['mapping']
    assert all(r['shown_graph_id'] == (pairing[r['graph_id']] if r['condition'].startswith('wrong_') else r['graph_id']) for r in rows)
    assert all(bool(r['vision_tokens']) == (r['condition'] in ('image', 'wrong_image')) for r in rows)
