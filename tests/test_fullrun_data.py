from pathlib import Path
import json
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_graphvis_datasets as gen
import build_fullrun_data as build


def test_subset100_is_explicit_superset():
    original = json.loads((ROOT / 'subsets/obqa_test_subset50_seed13.json').read_text())
    subset, extra = build.extend_subset(original)
    expected = random.Random(13).sample(sorted(set(range(500)) - set(original['indices'])), 50)
    assert extra['indices'] == sorted(expected)
    assert subset['indices'] == sorted(original['indices'] + expected)
    assert len(set(subset['indices'])) == 100


def test_composition_preserves_bytes_and_rejects_duplicates(tmp_path):
    a, b, c = [tmp_path / name for name in ('a', 'b', 'out')]
    a.write_bytes(b'{"statement_idx": 7, "task_type": "node_number", "prompt": "old"}\n')
    b.write_bytes(b'{"statement_idx":2,"task_type":"node_number","prompt":"new"}\n')
    build.compose(a, b, c, [2, 7])
    assert c.read_bytes() == b.read_bytes() + a.read_bytes()
    with pytest.raises(FileExistsError):
        build.compose(a, b, c, [2, 7])
    with pytest.raises(ValueError, match='Duplicate'):
        build.compose(a, a, tmp_path / 'dup', [7])


def test_uniform_all_node_attributes_in_dot_and_legacy_bytes(tmp_path, monkeypatch):
    nodes = {
        1: dict(name='question', in_question=True, in_choices=set()),
        2: dict(name='answer', in_question=False, in_choices={'A'}),
        3: dict(name='bridge', in_question=False, in_choices=set()),
        4: dict(name='isolated', in_question=False, in_choices={'B'}),
    }
    graph = dict(connected_nodes=[1, 2, 3], disconnected_answers=[4], edges=[(1, 'isa', 3), (3, 'isa', 2)])
    sources = []
    monkeypatch.setattr(gen.graphviz.Digraph, 'render', lambda self, *a, **kw: sources.append(self.source))
    for style in (None, 'legacy', 'uniform'):
        gen.render_graph(tmp_path / 'x', nodes, graph, 'A', 'dot', False,
                         **({'node_style_mode': style} if style else {}))
    assert sources[0] == sources[1]
    parsed = json.loads(gen.graphviz.Source(sources[2]).pipe(format='json'))
    attrs = [{k: obj[k] for k in ('fillcolor', 'color', 'penwidth', 'style', 'shape', 'fontname', 'fontsize')}
             for obj in parsed['objects'] if obj['name'].isdigit()]
    assert len(attrs) == 4
    assert all(a == attrs[0] for a in attrs)
    assert attrs[0]['fillcolor'] == '#ADD8E6'
    assert attrs[0]['color'] == 'black'
    assert 'solid' in attrs[0]['style']
    cluster = next(o for o in parsed['objects'] if o['name'] == 'cluster_no_evidence')
    assert cluster['style'] == 'dashed'
    with pytest.raises(ValueError, match='reveal'):
        gen.render_graph(tmp_path / 'x', nodes, graph, 'A', 'dot', False,
                         reveal_correct_answer=True, node_style_mode='uniform')


@pytest.mark.skipif(not (ROOT / 'data_preprocessed_release/obqa/graph/test.graph.adj.pk').is_file(),
                    reason='QA-GNN release required for renderer integration test')
def test_uniform_jsonl_and_dimensions_unchanged_and_reuse_does_not_render(tmp_path):
    from PIL import Image
    common = [sys.executable, str(ROOT / 'scripts/generate_graphvis_datasets.py'), *build.FLAGS,
              '--data-root', str(ROOT / 'data_preprocessed_release'), '--limit', '3']
    for style in ('legacy', 'uniform'):
        subprocess.run(common + ['--node-style', style, '--out-dir', str(tmp_path / style)], check=True, capture_output=True)
    for name in ('stage1_graph_comprehension', 'stage2_obqa', 'graph_metadata'):
        assert (tmp_path / 'legacy/test' / f'{name}_0_3.jsonl').read_bytes() == (tmp_path / 'uniform/test' / f'{name}_0_3.jsonl').read_bytes()
    for i in range(3):
        with Image.open(tmp_path / f'legacy/test/images/q{i:05d}_clean.png') as a, Image.open(tmp_path / f'uniform/test/images/q{i:05d}_clean.png') as b:
            assert a.size == b.size
    subprocess.run(common + ['--reuse-render-from', str(tmp_path / 'uniform'), '--out-dir', str(tmp_path / 'reuse')], check=True, capture_output=True)
    assert not list((tmp_path / 'reuse/test/images').glob('*.png'))
    assert (tmp_path / 'reuse/test/stage1_graph_comprehension_0_3.jsonl').read_bytes() == (tmp_path / 'uniform/test/stage1_graph_comprehension_0_3.jsonl').read_bytes()


@pytest.mark.skipif(not (ROOT / 'outputs/fullrun_2026-10-04_B/consistency.json').exists(),
                    reason='Prepared 500-graph data required')
def test_prepared_500_matches_baseline_and_preserves_original_stage1():
    out = ROOT / 'outputs/fullrun_2026-10-04_B'
    result = build.compare_baseline(ROOT / build.BASELINE, out)
    assert result['passed']
    old = (ROOT / build.BASELINE / 'test/stage1_graph_comprehension_subset50_seed13.jsonl').read_bytes().splitlines(keepends=True)
    old_ids = {json.loads(line)['statement_idx'] for line in old}
    composed = (out / 'test/stage1_subset100.jsonl').read_bytes().splitlines(keepends=True)
    assert [line for line in composed if json.loads(line)['statement_idx'] in old_ids] == old
    assert len(composed) == 900
