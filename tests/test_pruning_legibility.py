"""Separate graph-content, renderer, report and audit tests."""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_graphvis_datasets as gen
import image_audit
import legibility_sweep
import pruning_stats

spec = importlib.util.spec_from_file_location('legacy_graph', Path(__file__).parent / 'fixtures/legacy_pruning_render.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def hub_graph():
    # Bridge leaves have a question hub and an answer anchor as neighbours.
    # Unlike marking every leaf as core, this does not force every hub edge to
    # survive as a leaf's lifeline (the cap intentionally remains soft).
    nodes = {i: {'name': f'concept_{i}', 'in_question': i == 0,
                 'in_choices': {'A'} if i == 10 else set()} for i in range(11)}
    edges = {(0, 'relatedto', i) for i in range(1, 10)} | {(i, 'isa', 10) for i in range(1, 10)}
    return nodes, edges


def test_degree_cap_disabled_and_branch_not_entered():
    nodes, edges = hub_graph()
    visited = []
    def trace(frame, event, arg):
        if frame.f_code.co_name == 'edge_rank':
            visited.append(event)
        return trace
    sys.settrace(trace)
    try:
        uncapped = gen.prune_graph(nodes, edges, 18, 60, 0)
    finally:
        sys.settrace(None)
    assert not visited
    assert gen.edge_degree(uncapped)[0] == 9
    assert len(uncapped['edges']) == 18
    capped = gen.prune_graph(nodes, edges, 18, 60, 5)
    assert gen.edge_degree(capped)[0] == 5
    assert uncapped['pruning'] == {'max_nodes': 18, 'max_edges': 60, 'max_degree': 0,
                                    'edges_before_truncation': 18, 'edges_after': 18,
                                    'bridge_rule': 'qa-bridge', 'lifelines': True,
                                    'core_size': 2, 'core_truncated': False, 'bridges_added': 9}


def test_pruning_hash_seed_determinism():
    code = '''
import json, sys
sys.path.insert(0, 'scripts')
import generate_graphvis_datasets as gen
nodes = {i: {'name': str(i), 'in_question': i % 2 == 0, 'in_choices': {'A'} if i % 3 == 0 else set()} for i in range(16)}
edges = {(a, rel, b) for a in nodes for b in nodes if a != b for rel in ('isa','partof','relatedto')}
print(json.dumps([gen.prune_graph(nodes, edges, n, e, cap, bridge_rule=rule, lifelines=lifelines)
    for n in (0,8,18) for e in (0,9) for cap in (0,5)
    for rule in ('qa-bridge','core-neighbor','any','none') for lifelines in (True,False)], sort_keys=True))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code], cwd=ROOT,
                                      env={**os.environ, 'PYTHONHASHSEED': str(seed)}) for seed in (0, 1)]
    assert outputs[0] == outputs[1]


def test_truncation_metadata_and_warning(capsys):
    nodes, edges = hub_graph()
    graph = gen.prune_graph(nodes, edges, 18, 3, 0)
    assert graph['pruning']['edges_before_truncation'] == 18
    assert graph['pruning']['edges_after'] == len(graph['edges']) == 3
    assert '18 -> 3 edges' in capsys.readouterr().out
    statement = {'answerKey': 'A', 'question': {'stem': 'test', 'choices': []}}
    assert gen.graph_metadata(0, statement, nodes, graph, 'x.png')['pruning'] == graph['pruning']


def test_legacy_flags_unchanged_without_ambiguous_ties(tmp_path):
    nodes = {i: {'name': str(i), 'in_question': True, 'in_choices': set()} for i in range(3)}
    edges = {(0, 'isa', 1), (1, 'relatedto', 2)}
    old = legacy.prune_graph(nodes, edges, 18, 30, 5)
    new = gen.prune_graph(nodes, edges, 18, 30, 5)
    assert {k: v for k, v in new.items() if k != 'pruning'} == old
    for builder in (gen.build_stage1_records,):
        a = builder('test.png', 'test', 0, nodes, old, random.Random(13), None)
        b = builder('test.png', 'test', 0, nodes, new, random.Random(13), None)
        assert json.dumps(a) == json.dumps(b)
    # Same graph/default attributes, actual PNG byte equality in this Graphviz installation.
    a = legacy.render_graph(tmp_path / 'old', nodes, old, 'A', 'dot', False)
    b = gen.render_graph(tmp_path / 'new', nodes, new, 'A', 'dot', False)
    assert a.read_bytes() == b.read_bytes()


def test_render_flags_only_change_attributes(tmp_path, monkeypatch):
    nodes, edges = hub_graph()
    graph = gen.prune_graph(nodes, edges, 18, 60, 0)
    before = json.dumps(graph, sort_keys=True)
    sources = []
    monkeypatch.setattr(gen.graphviz.Digraph, 'render', lambda self, *a, **kw: sources.append(self.source))
    gen.render_graph(tmp_path / 'a', nodes, graph, 'A', 'dot', False)
    gen.render_graph(tmp_path / 'b', nodes, graph, 'A', 'dot', False,
                     node_fontsize=30, edge_fontsize=20, nodesep=0.2, ranksep=0.4,
                     graph_size='10,10', graph_ratio='fill', rankdir='TB')
    assert ' size=' not in sources[0] and 'ratio=' not in sources[0]
    for attribute in ('fontsize=30', 'fontsize=20', 'nodesep=0.2', 'ranksep=0.4',
                      'size="10,10"', 'ratio=fill', 'rankdir=TB'):
        assert attribute in sources[1]
    assert json.dumps(graph, sort_keys=True) == before


def test_wrap_labels_changes_image_label_only_at_word_boundary():
    assert gen.wrap_node_label('jewelry box organizer', 14) == 'jewelry box\\norganizer'
    assert gen.wrap_node_label('short label', 14) == 'short label'
    assert gen.wrap_node_label('extraordinarilylongtoken', 8) == 'extraordinarilylongtoken'


def test_auto_orient_is_deterministic_and_prefers_tb_on_tie(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    sizes = {'LR': (1800, 700), 'TB': (700, 1800)}

    def fake_render(stem, *args, rankdir='LR', **kwargs):
        path = Path(stem).with_suffix('.png')
        PILImage.new('RGB', sizes[rankdir], 'white').save(path)
        return path

    monkeypatch.setattr(gen, 'render_graph', fake_render)
    args = ({}, {'connected_nodes': [], 'disconnected_answers': [], 'edges': []}, 'A',
            'dot', False, False, 200, 3, 30, 24, 0.5, 0.4, None, None, 'TB', 'llava', 0)
    path1, meta1 = gen.render_selected_orientation(tmp_path / 'first', *args)
    path2, meta2 = gen.render_selected_orientation(tmp_path / 'second', *args)
    assert meta1 == meta2
    assert meta1['rankdir_selected'] == 'TB'
    assert path1.read_bytes() == path2.read_bytes()


def test_render_options_do_not_change_stage1_records():
    nodes, edges = hub_graph()
    graph = gen.prune_graph(nodes, edges, 18, 60, 0)
    baseline = gen.build_stage1_records('test/images/q.png', 'test', 7, nodes, graph,
                                        random.Random(13), None, 'extended', 'per-task')
    rerendered = gen.build_stage1_records('test/images/q.png', 'test', 7, nodes, graph,
                                          random.Random(13), None, 'extended', 'per-task')
    assert json.dumps(baseline, ensure_ascii=False) == json.dumps(rerendered, ensure_ascii=False)


def test_stats_node_and_option_denominators():
    meta = {'visible_nodes': [
        {'cid': 0, 'in_question': True, 'in_choices': []},
        {'cid': 1, 'in_question': False, 'in_choices': ['A']},
        {'cid': 2, 'in_question': False, 'in_choices': ['A']},
        {'cid': 3, 'in_question': False, 'in_choices': ['B']}],
        'edges': [{'source_cid': 0, 'target_cid': 1}],
        'choices': [{'label': 'A'}, {'label': 'B'}, {'label': 'C'}], 'answerKey': 'A'}
    result = pruning_stats.summarize([meta])
    assert result['correct_answer_nodes']['degree_zero_fraction'] == 0.5
    assert result['correct_answer_nodes']['reachable_fraction'] == 0.5
    assert result['correct_answer_options']['degree_zero_fraction'] == 0
    assert result['correct_answer_options']['reachable_distance']['mean'] == 1
    assert result['best_distractor_options']['degree_zero_fraction'] == 1
    assert result['max_edges_truncation']['fired_count'] is None
    assert result['node_count']['mean'] == 4


def test_audit_measured_height(tmp_path):
    root = tmp_path
    split = root / 'test'
    split.mkdir()
    Image.new('RGB', (1792, 896), 'white').save(split / 'x.png')
    meta = {'statement_idx': 0, 'image': 'test/x.png', 'visible_nodes': [], 'edges': []}
    (split / 'graph_metadata_0_1.jsonl').write_text(json.dumps(meta) + '\n')
    result = image_audit.audit_split(split)
    reference = image_audit.reference_cap_height()
    assert reference > 0
    assert result['images'][0]['estimated_label_cap_px_896'] == reference / 2
    assert result['images'][0]['llava_selected_grid'] == {'height': 336, 'width': 672}
    assert result['images'][0]['estimated_label_cap_px_llava_anyres'] == reference * 0.375
    assert result['images'][0]['aspect_ratio'] == 2
    assert image_audit.reference_cap_height(30) > reference


def test_cli_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['generate'])
    args = gen.parse_args()
    assert (args.max_nodes, args.max_edges, args.max_degree) == (18, 60, 0)
    assert args.bridge_rule == 'qa-bridge' and args.lifelines is True
    assert args.core_policy == 'truncate' and args.max_bridges == 0
    assert (args.node_fontsize, args.edge_fontsize, args.nodesep, args.ranksep, args.rankdir) == (18, 14, 0.5, 0.7, 'LR')
    assert args.graph_size is None and args.graph_ratio is None



def capture_keep(nodes, edges, *args, **kwargs):
    # Retention precedes the unchanged visible-node policy, which omits isolated
    # fillers/questions. Inspect retention directly without changing public output.
    kept = set()
    def trace(frame, event, arg):
        if frame.f_code is gen.prune_graph.__code__ and event == 'return':
            kept.update(frame.f_locals['keep'])
        return trace
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        graph = gen.prune_graph(nodes, edges, *args, **kwargs)
    finally:
        sys.settrace(previous)
    return kept, graph


def test_default_hub_graph_exact_regression():
    nodes, edges = hub_graph()
    graph = gen.prune_graph(nodes, edges, 18, 60, 0)
    expected = {
        'connected_nodes': list(range(11)), 'visible_nodes': list(range(11)),
        'edges': sorted(edges), 'disconnected_answers': [], 'disconnected_questions': [],
        'q_cids': [0], 'a_cids': [10],
        'pruning': {'max_nodes': 18, 'max_edges': 60, 'max_degree': 0,
                    'edges_before_truncation': 18, 'edges_after': 18},
    }
    new_keys = {'bridge_rule', 'lifelines', 'core_size', 'core_truncated', 'bridges_added'}
    graph['pruning'] = {k: v for k, v in graph['pruning'].items() if k not in new_keys}
    assert graph == expected


@pytest.mark.parametrize('rule,expected', [
    ('qa-bridge', {0, 1, 2}), ('core-neighbor', {0, 1, 2, 3}),
    ('any', set(range(7))), ('none', {0, 1}),
])
def test_bridge_eligibility_and_unlimited_retention(rule, expected):
    nodes = {i: {'name': str(i), 'in_question': i == 0,
                 'in_choices': {'A'} if i == 1 else set()} for i in range(7)}
    # 2 spans Q/A; 3 touches only Q; 4/5 form a separate component; 6 is isolated.
    edges = {(0, 'isa', 1), (0, 'isa', 2), (2, 'isa', 1),
             (0, 'isa', 3), (4, 'isa', 5)}
    keep, graph = capture_keep(nodes, edges, 0, 0, 0, bridge_rule=rule)
    assert keep == expected
    assert graph['pruning']['bridges_added'] == len(expected) - 2
    assert graph['pruning']['core_size'] == 2
    assert graph['pruning']['core_truncated'] is False


@pytest.mark.parametrize('rule,expected_filler', [
    ('qa-bridge', 2), ('core-neighbor', 2), ('any', 3),
])
def test_filler_degree_tiebreak_is_only_used_for_any(rule, expected_filler):
    nodes = {i: {'name': str(i), 'in_question': i == 0,
                 'in_choices': {'A'} if i == 1 else set()} for i in range(5)}
    edges = {(0, 'isa', 2), (2, 'isa', 1), (0, 'isa', 3),
             (3, 'isa', 1), (3, 'isa', 4)}
    keep, graph = capture_keep(nodes, edges, 3, 0, 0, bridge_rule=rule)
    assert keep == {0, 1, expected_filler}
    assert graph['pruning']['bridges_added'] == 1


def test_any_ranks_zero_core_neighbors_by_degree_then_cid():
    nodes = {i: {'name': str(i), 'in_question': i == 0, 'in_choices': set()}
             for i in range(6)}
    edges = {(3, 'isa', 4), (3, 'isa', 5)}
    keep, _ = capture_keep(nodes, edges, 3, 0, 0, bridge_rule='any')
    assert keep == {0, 3, 4}


@pytest.mark.parametrize('budget,truncated,size', [(0, False, 22), (18, True, 18)])
def test_core_retention_metadata(budget, truncated, size):
    nodes = {i: {'name': str(i), 'in_question': i < 20,
                 'in_choices': {'A'} if i >= 20 else set()} for i in range(22)}
    keep, graph = capture_keep(nodes, set(), budget, 0, 0)
    assert len(keep) == size
    if not truncated:
        assert keep == set(nodes)
    assert graph['pruning']['core_size'] == 22
    assert graph['pruning']['core_truncated'] is truncated
    assert graph['pruning']['bridges_added'] == 0
    assert graph['pruning']['max_nodes'] == budget


def test_keep_core_policy_never_drops_core_when_core_exceeds_total_budget():
    nodes = {i: {'name': str(i), 'in_question': i < 20,
                 'in_choices': {'A'} if i >= 20 else set()} for i in range(27)}
    keep, graph = capture_keep(
        nodes, set(), 3, 0, 0, core_policy='keep', max_bridges=10
    )
    assert keep == set(nodes)
    assert graph['pruning']['core_size'] == 27
    assert graph['pruning']['core_truncated'] is False
    assert graph['pruning']['core_policy'] == 'keep'
    assert graph['pruning']['max_nodes'] == 3
    assert graph['pruning']['bridge_budget'] == 0
    assert 'max_nodes_ignored' not in graph['pruning']


def test_keep_core_policy_uses_remaining_total_budget_then_optional_bridge_cap():
    nodes, edges = hub_graph()
    keep, graph = capture_keep(
        nodes, edges, 6, 0, 0, core_policy='keep', max_bridges=4
    )
    assert {0, 10} <= keep
    assert len(keep - {0, 10}) == 4
    assert graph['pruning']['bridges_added'] == 4
    assert graph['pruning']['max_bridges'] == 4
    assert graph['pruning']['bridge_budget'] == 4


def test_keep_core_policy_total_budget_fills_only_remaining_slots():
    nodes, edges = hub_graph()
    keep, graph = capture_keep(nodes, edges, 5, 0, 0, core_policy='keep')
    core = {0, 10}
    assert core <= keep
    assert len(keep) == len(core) + min(9, max(0, 5 - len(core)))
    assert graph['pruning']['bridges_added'] == 3
    assert graph['pruning']['bridge_budget'] == 3


def test_explicit_default_core_policy_is_byte_identical():
    nodes, edges = hub_graph()
    implicit = gen.prune_graph(nodes, edges, 18, 60, 0)
    explicit = gen.prune_graph(
        nodes, edges, 18, 60, 0, core_policy='truncate', max_bridges=0
    )
    assert json.dumps(implicit, sort_keys=True).encode() == json.dumps(
        explicit, sort_keys=True
    ).encode()


def test_unlimited_edges_skips_truncation(capsys):
    nodes, edges = hub_graph()
    graph = gen.prune_graph(nodes, edges, 0, 0, 0)
    assert graph['edges'] == sorted(edges)
    assert graph['pruning']['max_edges'] == 0
    assert graph['pruning']['edges_after'] == graph['pruning']['edges_before_truncation'] == 18
    assert 'max_edges truncation' not in capsys.readouterr().out


def test_no_lifelines_enforces_hard_degree_cap():
    nodes = {i: {'name': str(i), 'in_question': i == 0,
                 'in_choices': {'A'} if i else set()} for i in range(6)}
    edges = {(0, 'isa', i) for i in range(1, 6)}
    soft = gen.prune_graph(nodes, edges, 0, 0, 2)
    hard = gen.prune_graph(nodes, edges, 0, 0, 2, lifelines=False)
    assert gen.edge_degree(soft)[0] == 5
    assert max(gen.edge_degree(hard).values()) == 2
    assert hard['pruning']['lifelines'] is False
    # Disabling exemptions alone does not change edge selection/truncation priority.
    a = gen.prune_graph(nodes, edges, 0, 2, 0)
    b = gen.prune_graph(nodes, edges, 0, 2, 0, lifelines=False)
    assert a['edges'] == b['edges']


def test_hard_degree_cap_counts_self_loop_twice():
    nodes = {0: {'name': 'loop', 'in_question': True, 'in_choices': set()}}
    graph = gen.prune_graph(nodes, {(0, 'isa', 0)}, 0, 0, 1, lifelines=False)
    assert graph['edges'] == []


@pytest.mark.parametrize('flag', ['--max-nodes', '--max-edges', '--max-degree'])
@pytest.mark.parametrize('module', [gen, legibility_sweep])
def test_negative_caps_rejected_before_loading_data(monkeypatch, tmp_path, flag, module):
    output = tmp_path / 'unused'
    monkeypatch.setattr(sys, 'argv', ['generate', '--out-dir', str(output), flag, '-1'])
    with pytest.raises(ValueError, match='max_nodes >= 0'):
        module.main()
    assert not output.exists()


def test_sweep_cli_defaults_and_ablation_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(legibility_sweep, 'run_sweep', calls.append)
    monkeypatch.setattr(sys, 'argv', ['sweep', '--out-dir', 'unused'])
    legibility_sweep.main()
    assert (calls[0].max_nodes, calls[0].max_edges, calls[0].max_degree) == (18, 60, 0)
    assert calls[0].bridge_rule == 'qa-bridge' and calls[0].lifelines is True
    monkeypatch.setattr(sys, 'argv', ['sweep', '--out-dir', 'unused', '--max-nodes', '0',
                                     '--max-edges', '0', '--bridge-rule', 'any', '--no-lifelines'])
    legibility_sweep.main()
    assert (calls[1].max_nodes, calls[1].max_edges) == (0, 0)
    assert calls[1].bridge_rule == 'any' and calls[1].lifelines is False
    args = gen.parse_args()
    assert (args.max_nodes, args.max_edges) == (0, 0)
    assert args.bridge_rule == 'any' and args.lifelines is False


def test_stats_separates_ablation_configurations_and_accepts_older_metadata():
    pruning = {'max_nodes': 18, 'max_edges': 60, 'max_degree': 0,
               'edges_before_truncation': 0, 'edges_after': 0}
    base = {'visible_nodes': [], 'edges': [], 'choices': [{'label': 'A'}, {'label': 'B'}],
            'answerKey': 'A'}
    configs = [pruning, {**pruning, 'bridge_rule': 'qa-bridge', 'lifelines': True},
               {**pruning, 'bridge_rule': 'core-neighbor', 'lifelines': True},
               {**pruning, 'bridge_rule': 'qa-bridge', 'lifelines': False}]
    result = pruning_stats.summarize([{**base, 'pruning': config} for config in configs])
    buckets = {(c['bridge_rule'], c['lifelines']): count
               for key, count in result['pruning_configurations'].items()
               for c in [json.loads(key)]}
    assert buckets == {('qa-bridge', True): 2, ('core-neighbor', True): 1,
                       ('qa-bridge', False): 1}
