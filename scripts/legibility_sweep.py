#!/usr/bin/env python3
"""Render one fixed graph slice under the 12 requested layout configurations.

Pruning occurs once per statement, so only rendering changes across the grid.
Rank by fewest images below the threshold, then largest median scaled cap height.
The ranking selects a Gemma preflight candidate, not a proven readability winner.
"""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import pickle
import random

import generate_graphvis_datasets as gen
from image_audit import audit_split


def run_sweep(args):
    if args.max_degree < 0 or args.max_nodes < 0 or args.max_edges < 0:
        raise ValueError('Require max_degree >= 0, max_nodes >= 0 and max_edges >= 0')
    # Exclusive output root: never overwrite previous sweep data.
    args.out_dir.mkdir(parents=True, exist_ok=False)
    concepts = (args.data_root / 'cpnet/concept.txt').read_text().splitlines()
    statements = gen.load_jsonl(args.data_root / f'obqa/statement/{args.split}.statement.jsonl')
    with (args.data_root / f'obqa/graph/{args.split}.graph.adj.pk').open('rb') as handle:
        entries = pickle.load(handle)
    indices = range(args.start, args.start + args.limit)
    if args.limit <= 0 or args.start < 0 or args.start + args.limit > len(statements):
        raise ValueError('Requested slice must fit the input statements')
    graphs = []
    for idx in indices:
        nodes, edges, correct = gen.merge_choice_graphs(idx, entries, statements[idx], concepts)
        graph = gen.prune_graph(nodes, edges, args.max_nodes, args.max_edges, args.max_degree,
                                bridge_rule=args.bridge_rule, lifelines=args.lifelines)
        graphs.append((idx, nodes, graph, correct))
    del entries
    comparison = []
    for rankdir, fontsize, ranksep in product(('LR', 'TB'), (18, 24, 30), (0.7, 0.4)):
        name = f'{rankdir}_font{fontsize}_ranksep{ranksep}'
        root = args.out_dir / name
        split_out = root / args.split
        (split_out / 'images').mkdir(parents=True)
        (split_out / 'graphs').mkdir()
        rng = random.Random(args.seed)
        stage1, stage2, metadata = [], [], []
        for idx, nodes, graph, correct in graphs:
            image = gen.render_graph(split_out / 'images' / f'q{idx:05d}_clean', nodes, graph, correct,
                                     'dot', False, dpi=args.dpi, node_fontsize=fontsize,
                                     ranksep=ranksep, rankdir=rankdir)
            image = image.relative_to(root)
            meta = gen.graph_metadata(idx, statements[idx], nodes, graph, image)
            metadata.append(meta)
            gen.write_jsonl(split_out / 'graphs' / f'q{idx:05d}.jsonl', [meta])
            stage1.extend(gen.build_stage1_records(image, args.split, idx, nodes, graph, rng, None))
            stage2.append(gen.build_stage2_record(image, args.split, idx, statements[idx]))
        suffix = f'{args.start}_{args.start + args.limit}'
        for prefix, records in [('graph_metadata', metadata), ('stage1_graph_comprehension', stage1), ('stage2_obqa', stage2)]:
            gen.write_jsonl(split_out / f'{prefix}_{suffix}.jsonl', records)
        config = {**vars(args), 'node_fontsize': fontsize, 'rankdir': rankdir, 'ranksep': ranksep,
                  'edge_fontsize': 14, 'nodesep': 0.5, 'graph_size': None, 'graph_ratio': None}
        (root / 'sweep_config.json').write_text(json.dumps(config, default=str, indent=2) + '\n')
        audit = audit_split(split_out, root, fontsize, args.dpi, args.min_label_px)
        (root / 'image_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        height = audit['summary']['scaled_label_height']
        row = {'configuration': name, 'rankdir': rankdir, 'node_fontsize': fontsize, 'ranksep': ranksep,
               'images': len(graphs), 'below_threshold': audit['summary']['below_threshold'],
               'mean_label_px': height['mean'], 'median_label_px': height['median'],
               'min_label_px': min(r['estimated_label_cap_px_896'] for r in audit['images']),
               'max_label_px': height['max']}
        comparison.append(row)
        print(json.dumps(row), flush=True)
    comparison.sort(key=lambda r: (r['below_threshold'], -r['median_label_px'], r['configuration']))
    with (args.out_dir / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (args.out_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    header = '| Configuration | Below threshold | Mean cap px | Median cap px | Min cap px |'
    lines = [header, '| --- | ---: | ---: | ---: | ---: |']
    for row in comparison:
        lines.append(f"| {row['configuration']} | {row['below_threshold']} | {row['mean_label_px']:.2f} | "
                     f"{row['median_label_px']:.2f} | {row['min_label_px']:.2f} |")
    lines += ['', f"Preflight candidate: {comparison[0]['configuration']}. Confirm with Gemma triple-listing recall."]
    (args.out_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data_preprocessed_release'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'dev', 'test'), default='test')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--max-nodes', type=int, default=18, help='Node cap; 0 disables it.')
    parser.add_argument('--max-edges', type=int, default=60, help='Edge cap; 0 disables it.')
    parser.add_argument('--max-degree', type=int, default=0, help='Degree cap; 0 disables it.')
    parser.add_argument('--bridge-rule', choices=['qa-bridge', 'core-neighbor', 'any', 'none'],
                        default='qa-bridge', help='Eligibility rule for filler nodes.')
    parser.add_argument('--no-lifelines', dest='lifelines', action='store_false',
                        help='Enforce a hard degree cap by disabling lifeline exemptions.')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--min-label-px', type=float, default=10)
    run_sweep(parser.parse_args())


if __name__ == '__main__':
    main()
