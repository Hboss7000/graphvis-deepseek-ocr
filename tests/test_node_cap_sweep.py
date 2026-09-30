from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_graphvis_datasets as gen
import node_cap_sweep as sweep


def typed_graph():
    nodes = {
        i: {'name': str(i), 'in_question': i == 0,
            'in_choices': {'A'} if i == 1 else set()}
        for i in range(6)
    }
    # 2 touches Q and A; 3 only Q; 4 only A; 5 no core node but is
    # visible through a self-loop. The direct Q/A edge keeps the core visible.
    edges = {
        (0, 'isa', 1), (0, 'isa', 2), (2, 'isa', 1),
        (0, 'isa', 3), (1, 'isa', 4), (5, 'isa', 5),
    }
    return nodes, edges


@pytest.mark.parametrize('rule,expected', [
    ('qa-bridge', {'q_and_a': 1, 'q_only': 0, 'a_only': 0, 'neither': 0}),
    ('core-neighbor', {'q_and_a': 1, 'q_only': 1, 'a_only': 1, 'neither': 0}),
    ('any', {'q_and_a': 1, 'q_only': 1, 'a_only': 1, 'neither': 1}),
    ('none', {'q_and_a': 0, 'q_only': 0, 'a_only': 0, 'neither': 0}),
])
def test_each_rule_admits_exact_added_node_types(rule, expected):
    nodes, edges = typed_graph()
    with redirect_stdout(io.StringIO()):
        graph = gen.prune_graph(nodes, edges, 0, 0, 0, bridge_rule=rule)
    breakdown = sweep.added_node_breakdown(nodes, edges, graph)
    assert {kind: breakdown[kind] for kind in sweep.ADDED_TYPES} == expected


@pytest.mark.parametrize('rule', sweep.BRIDGE_RULES)
def test_added_node_breakdown_sums_to_bridges_added(rule):
    nodes, edges = typed_graph()
    with redirect_stdout(io.StringIO()):
        graph = gen.prune_graph(nodes, edges, 0, 0, 0, bridge_rule=rule)
    breakdown = sweep.added_node_breakdown(nodes, edges, graph)
    assert breakdown['total'] == graph['pruning']['bridges_added']


class QuietLog:
    def line(self, _message):
        pass


def test_qa_bridge_new_columns_preserve_legacy_csv_bytes_on_20_questions(monkeypatch):
    monkeypatch.setattr(sweep, 'tqdm', lambda iterable, **_kwargs: iterable)
    args = SimpleNamespace(
        data_root=ROOT / 'data_preprocessed_release', split='test', start=0, limit=20,
    )
    merged, _sizes = sweep.build_merged(args, QuietLog())

    legacy_fixed, extended_fixed = [], []
    for cap in sweep.FIXED_CAPS:
        metadata, prunings, skipped, breakdowns = sweep.prune_condition(
            f'qa-bridge_fixed_{cap}', merged, QuietLog(), sweep.fixed_cap_fn(cap),
            max_edges=0, max_degree=0, bridge_rule='qa-bridge', collect_breakdowns=True,
        )
        kwargs = dict(max_nodes=cap, max_edges=0, max_degree=0,
                      bridge_rule='qa-bridge', skipped=skipped)
        legacy_fixed.append(sweep.metric_row(
            'fixed_uncapped' if cap == 0 else f'fixed_{cap}', 'A1',
            metadata, prunings, **kwargs,
        ))
        extended_fixed.append(sweep.metric_row(
            'fixed_uncapped' if cap == 0 else f'fixed_{cap}', 'A1',
            metadata, prunings, breakdowns=breakdowns, **kwargs,
        ))

    legacy_adaptive, extended_adaptive = [], []
    for B in sweep.ADAPTIVE_BUDGETS:
        metadata, prunings, skipped, breakdowns = sweep.prune_condition(
            f'qa-bridge_adaptive_B{B}', merged, QuietLog(), sweep.adaptive_cap_fn(B),
            max_edges=0, max_degree=0, bridge_rule='qa-bridge', collect_breakdowns=True,
        )
        kwargs = dict(max_nodes='core+B', B=B, max_edges=0, max_degree=0,
                      bridge_rule='qa-bridge', skipped=skipped)
        legacy_adaptive.append(sweep.metric_row(
            f'adaptive_B{B}', 'A2', metadata, prunings, **kwargs,
        ))
        extended_adaptive.append(sweep.metric_row(
            f'adaptive_B{B}', 'A2', metadata, prunings,
            breakdowns=breakdowns, **kwargs,
        ))

    fixed_fields = list(legacy_fixed[0])
    adaptive_fields = list(legacy_adaptive[0])
    assert sweep.csv_text(legacy_fixed, fixed_fields).encode() == \
        sweep.csv_text(extended_fixed, fixed_fields).encode()
    assert sweep.csv_text(legacy_adaptive, adaptive_fields).encode() == \
        sweep.csv_text(extended_adaptive, adaptive_fields).encode()
