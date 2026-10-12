#!/usr/bin/env python3
"""CPU replay of the frozen evaluation graphs; no retrieval, re-pruning or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from generate_graphvis_datasets import render_graph


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/fullrun_2026-10-04_B')
    parser.add_argument('--limit', type=int, help='CPU smoke only; full inference needs all 500 images')
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 500:
        parser.error('--limit must be between 1 and 500')
    archive = ROOT / 'reproduction'
    manifest = json.loads((archive / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if sha(archive / name) != entry['sha256']:
            raise ValueError(f'Frozen artifact changed: {name}')
    if args.out.exists():
        raise FileExistsError(f'Use a fresh output directory: {args.out}')
    target = args.out / 'test'
    target.mkdir(parents=True)
    for name in ('stage1_subset100.jsonl', 'stage2_obqa_0_500.jsonl', 'graph_metadata_0_500.jsonl'):
        shutil.copyfile(archive / 'inputs' / name, target / name)
    metadata = [json.loads(line) for line in (target / 'graph_metadata_0_500.jsonl').read_text().splitlines()]
    matches = []
    for row in metadata[:args.limit]:
        nodes = {int(n['cid']): {**n, 'in_choices': set(n['in_choices'])} for n in row['visible_nodes']}
        graph = dict(connected_nodes=[int(n['cid']) for n in row['visible_nodes'] if n['connected']],
                     visible_nodes=list(nodes), disconnected_answers=row['disconnected_answer_cids'],
                     edges=[(int(e['source_cid']), e['relation'], int(e['target_cid'])) for e in row['edges']])
        image = args.out / row['image']
        image.parent.mkdir(parents=True, exist_ok=True)
        render_graph(image.with_suffix(''), nodes, graph, row['answerKey'], 'dot', False,
                     dpi=200, node_fontsize=30, edge_fontsize=24, nodesep=.5, ranksep=.4,
                     rankdir=row['rankdir_selected'], node_style_mode='uniform', node_fill='#ADD8E6')
        matches.append(dict(image=row['image'], sha256=sha(image),
                            original_sha256=manifest['test_image_sha256'][row['image']]))
        print(image, flush=True)
    report = dict(rendered=len(matches), complete=len(matches) == 500,
                  byte_identical=all(r['sha256'] == r['original_sha256'] for r in matches), images=matches)
    (args.out / 'render_replay.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['byte_identical']:
        raise SystemExit('Image hashes differ: inspect Graphviz/fonts before claiming exact replay; see render_replay.json.')
    print(f'Rendered {len(matches)}/500 images; original PNG hashes match.')


if __name__ == '__main__':
    main()
