#!/usr/bin/env python3
"""Compare pruned splits on identical statement IDs, using undirected edges.

Report both concept-node and question/option denominators. An option's degree is
its maximum concept degree; its distance is the minimum over its concepts. The
best distractor is the option with maximum degree (then minimum reachable
distance, then choice letter). Missing option concepts are counted separately;
the legacy question-level zero-degree statistic includes missing concepts as 0.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import statistics


def load_split(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob('graph_metadata_*.jsonl'))
    if not paths and path.is_dir():
        paths = sorted((path / 'graphs').glob('*.jsonl'))
    if not paths:
        raise ValueError(f'No graph metadata found in {path}; pass the split directory or metadata JSONL')
    rows = {}
    for filename in paths:
        for line in filename.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row['statement_idx'])
            if idx in rows:
                raise ValueError(f'Duplicate metadata for statement_idx={idx}')
            rows[idx] = row
    return rows


def distribution(values):
    values = list(values)
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None,
            'histogram': {str(key): count for key, count in sorted(Counter(values).items())}}


def graph_connectivity(meta):
    adj, degree = defaultdict(set), Counter()
    for edge in meta['edges']:
        a, b = int(edge['source_cid']), int(edge['target_cid'])
        adj[a].add(b)
        adj[b].add(a)
        degree[a] += 1
        degree[b] += 1
    sources = {int(n['cid']) for n in meta['visible_nodes'] if n['in_question']}
    sources.update(map(int, meta.get('disconnected_question_cids', [])))
    distances = {cid: 0 for cid in sources}
    queue = deque(sorted(sources))
    while queue:
        cid = queue.popleft()
        for other in sorted(adj[cid]):
            if other not in distances:
                distances[other] = distances[cid] + 1
                queue.append(other)
    return degree, distances


def summarize(rows):
    degrees, nodes, edges, correct_nodes, distractor_nodes = [], [], [], [], []
    correct_options, distractor_options = [], []
    known_truncation = fired = 0
    configs = Counter()
    for row in rows:
        degree, distances = graph_connectivity(row)
        degrees.extend(degree[int(n['cid'])] for n in row['visible_nodes'])
        nodes.append(len(row['visible_nodes']))
        edges.append(len(row['edges']))
        options = {}
        for choice in row['choices']:
            label = choice['label']
            concepts = [(degree[int(n['cid'])], distances.get(int(n['cid'])))
                        for n in row['visible_nodes'] if label in n['in_choices']]
            options[label] = {'concepts': concepts, 'degree': max((d for d, _ in concepts), default=0),
                              'distance': min((d for _, d in concepts if d is not None), default=None)}
        correct = options[row['answerKey']]
        distractors = [key for key in options if key != row['answerKey']]
        best = min(distractors, key=lambda key: (-options[key]['degree'],
                   options[key]['distance'] if options[key]['distance'] is not None else float('inf'), key))
        correct_nodes.extend(correct['concepts'])
        distractor_nodes.extend(options[best]['concepts'])
        correct_options.append(correct)
        distractor_options.append(options[best])
        pruning = row.get('pruning')
        if pruning:
            known_truncation += 1
            fired += pruning['edges_before_truncation'] > pruning['edges_after']
            config = {k: pruning[k] for k in ('max_nodes', 'max_edges', 'max_degree')}
            # Older metadata predates these flags and used their current defaults.
            config.update(bridge_rule=pruning.get('bridge_rule', 'qa-bridge'),
                          lifelines=pruning.get('lifelines', True))
            configs[json.dumps(config, sort_keys=True)] += 1

    def node_summary(items):
        return {'node_count': len(items),
                'degree_zero_fraction': sum(d == 0 for d, _ in items) / len(items) if items else None,
                'reachable_fraction': sum(dist is not None for _, dist in items) / len(items) if items else None,
                'reachable_distance': distribution(dist for _, dist in items if dist is not None)}

    def option_summary(items):
        return {'question_count': len(items),
                'no_concept_count': sum(not item['concepts'] for item in items),
                'degree_zero_fraction': sum(item['degree'] == 0 for item in items) / len(items) if items else None,
                'reachable_fraction': sum(item['distance'] is not None for item in items) / len(items) if items else None,
                'reachable_distance': distribution(item['distance'] for item in items if item['distance'] is not None)}

    return {'graphs': len(rows), 'visible_node_degree': distribution(degrees),
            'node_count': distribution(nodes), 'edge_count': distribution(edges),
            'correct_answer_nodes': node_summary(correct_nodes),
            'best_distractor_nodes': node_summary(distractor_nodes),
            'correct_answer_options': option_summary(correct_options),
            'best_distractor_options': option_summary(distractor_options),
            'max_edges_truncation': {'known_graphs': known_truncation, 'unknown_graphs': len(rows) - known_truncation,
                                     'fired_count': fired if known_truncation else None,
                                     'fraction_of_known': fired / known_truncation if known_truncation else None},
            'pruning_configurations': dict(configs)}


def compare_splits(old, new, start=None, limit=None):
    old_rows, new_rows = load_split(old), load_split(new)
    def selected(rows):
        return {idx for idx in rows if (start is None or idx >= start)
                and (limit is None or idx < (start if start is not None else 0) + limit)}
    ids = selected(old_rows)
    if ids != selected(new_rows) or not ids:
        raise ValueError('Splits must contain the same nonempty statement range; use --start and --limit')
    return {'statement_indices': sorted(ids), 'old': summarize([old_rows[i] for i in sorted(ids)]),
            'new': summarize([new_rows[i] for i in sorted(ids)]),
            'definitions': __doc__,
            'legacy_truncation_note': 'Without pruning metadata, truncation frequency is unknown, not zero.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--start', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = compare_splits(args.old, args.new, args.start, args.limit)
    text = json.dumps(report, indent=2) + '\n'
    print(text, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == '__main__':
    main()
