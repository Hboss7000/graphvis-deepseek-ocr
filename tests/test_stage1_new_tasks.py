"""Synthetic pruned graphs; no dataset pickles or Graphviz executable required."""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('graphvis_generator', ROOT / 'scripts/generate_graphvis_datasets.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def fixture_graph(edges=None):
    names = ['sun_flower', 'garden_soil', 'water_bottle', 'greenhouse', 'mountain', 'ocean', 'isolated_answer']
    nodes = {i: {'name': name, 'in_question': i == 0,
                 'in_choices': {'B', 'A'} if i == 0 else set()} for i, name in enumerate(names)}
    if edges is None:
        edges = [(0, 'atlocation', 1), (0, 'relatedto', 2),
                 (1, 'usedfor', 3), (2, 'partof', 3), (4, 'isa', 5)]
    connected = sorted({cid for s, _, t in edges for cid in (s, t)})
    return nodes, {'edges': edges, 'connected_nodes': connected,
                   'visible_nodes': sorted(set(connected) | {6}),
                   'disconnected_answers': [6], 'disconnected_questions': [],
                   'q_cids': [0], 'a_cids': [0, 6]}


def records(seed=13, edges=None, **kwargs):
    nodes, graph = fixture_graph(edges)
    return gen.build_stage1_records('test.png', 'train', 0, nodes, graph,
                                    random.Random(seed), kwargs.pop('tasks_per_graph', None),
                                    **{'stage1_task_set': 'extended', **kwargs})


def task(items, name):
    return next((r for r in items if r['task_type'] == name), None)


def test_diamond():
    _, graph = fixture_graph()
    assert gen.all_shortest_paths(gen.adjacency(graph), 0, 3) == [[0, 1, 3], [0, 2, 3]]
    assert gen.all_shortest_paths(gen.adjacency(graph), 0, 5) == []


def test_path_selection():
    nodes, graph = fixture_graph()
    labels = {gen.label_for_node(n['name']): cid for cid, n in nodes.items()}
    adj = gen.adjacency(graph)
    for seed in range(60):
        gold = task(records(seed), 'shortest_path_listing')['gold']
        a, b = labels[gold['node_a']], labels[gold['node_b']]
        assert b not in adj[a]
        assert gold['distance'] == 2
        assert gold['paths'] == sorted([[gen.label_for_node(nodes[c]['name']) for c in p]
                                       for p in gen.all_shortest_paths(adj, a, b)])
        assert not gold['paths_truncated']
    assert task(records(edges=[(0, 'isa', 1), (4, 'isa', 5)]), 'shortest_path_listing') is None


def test_neighbor_gold():
    edges = [(0, 'isa', i) for i in (1, 2, 3)]
    found = False
    for seed in range(60):
        gold = task(records(seed, edges), 'neighbor_listing')['gold']
        assert gold['node'] != 'isolated answer'
        if gold['node'] == 'sun flower':
            assert gold == {'node': 'sun flower', 'degree': 3,
                            'neighbors': ['garden soil', 'greenhouse', 'water bottle']}
            found = True
    assert found


def test_relatedto_guard_and_only_relatedto():
    edges = [(0, 'relatedto', 1)]
    assert task(records(edges=edges, hide_relatedto_labels=True), 'relation_identification') is None
    assert task(records(edges=edges), 'relation_identification')['gold']['relation'] == 'relatedto'


def test_relation_balancing(monkeypatch):
    monkeypatch.setattr(gen, 'RELATEDTO_KEEP_RATE', 0)
    for seed in range(20):
        assert task(records(seed), 'relation_identification')['gold']['relation'] != 'relatedto'


def test_rendered_labels():
    # User clarified that current suffix-free labels must be preserved.
    nodes, _ = fixture_graph()
    expected = gen.node_style(0, nodes, 'A')[0]
    record = task(records(edges=[(0, 'atlocation', 1)]), 'relation_identification')
    assert expected == 'sun flower'
    assert expected in record['prompt'] and expected in record['answer']
    assert record['gold']['node_a'] == expected


def test_near_duplicates():
    assert gen._too_similar('jewelry box [A,B]', 'jewlery box [C]')
    assert not gen._too_similar('garden soil', 'water bottle')
    nodes, graph = fixture_graph([(0, 'isa', 1), (1, 'isa', 2)])
    nodes[0]['name'], nodes[2]['name'] = 'jewelry_box', 'jewlery_box'
    items = gen.build_stage1_extended_tasks(nodes, graph, gen.adjacency(graph), random.Random(13), False)
    assert task(items, 'shortest_path_listing') is None
    nodes[1]['name'] = 'jewelry_box'
    items = gen.build_stage1_extended_tasks(nodes, graph, gen.adjacency(graph), random.Random(13), False)
    assert task(items, 'relation_identification') is None


def test_path_cap(monkeypatch):
    monkeypatch.setattr(gen, 'MAX_SHORTEST_PATHS', 1)
    paths = gen.all_shortest_paths(gen.adjacency(fixture_graph()[1]), 0, 3)
    assert len(paths) == 1
    assert task(records(), 'shortest_path_listing')['gold']['paths_truncated']


def test_empty_graph():
    items = records(edges=[])
    assert all('gold' not in r for r in items)


def test_balancing_and_original_candidates():
    assert len(records(tasks_per_graph=2)) == 2
    extended = records(tasks_per_graph=2, stage1_balance='per-task')
    assert len(extended) == 9
    assert len({r['task_type'] for r in extended}) == 9
    paper = records(stage1_task_set='paper')
    assert [r for r in extended if 'gold' not in r] == paper


def hash_seed_payload():
    nodes, graph = fixture_graph()
    # Exercise unordered edge input for the new builders as well as records
    # built from the sorted lists returned by prune_graph.
    graph['edges'] = set(graph['edges'])
    extended = [gen.build_stage1_extended_tasks(nodes, graph, gen.adjacency(graph),
                                               random.Random(seed), False)
                for seed in range(20)]
    return [records(seed) for seed in range(20)], extended


def test_determinism():
    assert records() == records()
    code = ("import runpy,json; m=runpy.run_path('tests/test_stage1_new_tasks.py'); "
            "print(json.dumps(m['hash_seed_payload'](),sort_keys=True))")
    outputs = [subprocess.check_output([sys.executable, '-c', code], cwd=ROOT,
                                      env={**os.environ, 'PYTHONHASHSEED': str(seed)})
               for seed in (0, 1)]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('count', [None, 2])
def test_paper_regression(count):
    """Compare a fixed synthetic slice to JSONL captured before modification."""
    rng = random.Random(13)
    nodes, graph = fixture_graph()
    items = []
    for idx in range(3):
        items.extend(gen.build_stage1_records('test.png', 'train', idx, nodes, graph, rng, count,
                                              stage1_task_set='paper'))
    actual = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items).encode()
    expected = Path(__file__).with_name('fixtures') / f'stage1_paper_{count}.jsonl'
    assert actual == expected.read_bytes()


def test_cli_defaults_and_flags(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['generate'])
    args = gen.parse_args()
    assert (args.stage1_task_set, args.stage1_balance) == ('paper', 'pool')
    monkeypatch.setattr(sys, 'argv', ['generate', '--stage1-task-set', 'extended',
                                    '--stage1-balance', 'per-task'])
    args = gen.parse_args()
    assert (args.stage1_task_set, args.stage1_balance) == ('extended', 'per-task')
