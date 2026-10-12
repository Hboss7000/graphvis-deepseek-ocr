#!/usr/bin/env python3
"""Does estimated label height predict gold-node recall? (post-hoc check of the 10 px threshold)

Joins per-graph label height (from image_audit.py output) with per-graph gold-node
recall (from prediction JSONL rows) and reports, per prediction file ("arm"):

  * mean recall for graphs below vs. at/above the threshold, with a permutation p-value
  * Spearman correlation between label height and recall
  * the same correlation controlling for node count (partial Spearman), because
    graphs with small labels are also graphs with many nodes
  * recall in four equal-count bins of label height
  * a CSV with one row per graph, and a scatter plot if matplotlib is installed

This is a post-hoc check. It cannot validate the threshold unless graphs exist on
both sides of it. Pure standard library (matplotlib optional).

Example:
  python3 scripts/legibility_vs_recall.py \\
      --audit outputs/<run>/image_audit.json \\
      --predictions llava=outputs/<probe>/predictions_llava_node_description.jsonl \\
      --task node_description --variant basic --out-dir outputs/threshold_check
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_path(row, dotted):
    current = row
    for part in dotted.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def mean(values):
    return sum(values) / len(values) if values else None


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            result[order[k]] = average
        i = j + 1
    return result


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def partial_spearman(x, y, z):
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    if None in (rxy, rxz, ryz):
        return None
    denominator = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / denominator


def permutation_p(low, high, draws=10000, seed=13):
    """Two-sided permutation test on the difference of means."""
    if not low or not high:
        return None
    observed = abs(mean(high) - mean(low))
    pooled = list(low) + list(high)
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        rng.shuffle(pooled)
        if abs(mean(pooled[len(low):]) - mean(pooled[:len(low)])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


def load_audit(path, x_key):
    audit = json.loads(Path(path).read_text())
    graphs = {}
    for image in audit['images']:
        if x_key not in image:
            raise KeyError(f"'{x_key}' not in audit image rows; available: {sorted(image)}")
        graphs[int(image['statement_idx'])] = {'x': float(image[x_key]),
                                              'node_count': image.get('node_count')}
    return graphs


def load_recall(path, task, recall_key):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f'{path}: no rows')
    recalls, seen = {}, set()
    for row in rows:
        if task and row.get('task_type') not in (None, task):
            continue
        idx = int(row['statement_idx'])
        if idx in seen:
            raise ValueError(f'{path}: duplicate statement_idx {idx} for task {task}; '
                             'pass a file with one row per graph')
        seen.add(idx)
        value = get_path(row, recall_key)
        if value is not None:
            recalls[idx] = value
    if not recalls:
        raise KeyError(f"{path}: no numeric value at '{recall_key}'. First row keys: {sorted(rows[0])}")
    return recalls


def analyse(arm, graphs, recalls, threshold, draws):
    idxs = sorted(set(graphs) & set(recalls))
    x = [graphs[i]['x'] for i in idxs]
    y = [recalls[i] for i in idxs]
    nodes = [graphs[i]['node_count'] for i in idxs]
    has_nodes = all(isinstance(n, (int, float)) for n in nodes)
    low = [r for r, v in zip(y, x) if v < threshold]
    high = [r for r, v in zip(y, x) if v >= threshold]
    out = {'arm': arm, 'n_joined': len(idxs), 'n_audit_only': len(set(graphs) - set(recalls)),
           'n_recall_only': len(set(recalls) - set(graphs)),
           'x_min': min(x) if x else None, 'x_max': max(x) if x else None,
           'threshold': threshold, 'n_below': len(low), 'n_at_or_above': len(high),
           'mean_recall_below': mean(low), 'mean_recall_at_or_above': mean(high),
           'perm_p_two_sided': permutation_p(low, high, draws),
           'spearman_x_recall': spearman(x, y) if len(x) > 2 else None,
           'partial_spearman_controlling_node_count':
               partial_spearman(x, y, [float(n) for n in nodes]) if has_nodes and len(x) > 2 else None,
           'spearman_nodecount_x': spearman([float(n) for n in nodes], x) if has_nodes and len(x) > 2 else None}
    bins = []
    if len(idxs) >= 8:
        order = sorted(range(len(idxs)), key=lambda i: x[i])
        size = len(order) / 4
        for b in range(4):
            part = order[round(b * size):round((b + 1) * size)]
            bins.append({'bin': b + 1, 'n': len(part), 'x_range': [x[part[0]], x[part[-1]]],
                         'mean_recall': mean([y[i] for i in part])})
    out['quartile_bins'] = bins
    rows = [{'arm': arm, 'statement_idx': i, 'label_px': graphs[i]['x'],
             'node_count': graphs[i]['node_count'], 'recall': recalls[i]} for i in idxs]
    return out, rows


def fmt(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def print_summary(s):
    print(f"\n=== {s['arm']} ===")
    print(f"graphs joined: {s['n_joined']} (audit only: {s['n_audit_only']}, recall only: {s['n_recall_only']})")
    print(f"label px range: {fmt(s['x_min'], 2)} - {fmt(s['x_max'], 2)}; threshold {s['threshold']}")
    print(f"below {s['n_below']} graphs, mean recall {fmt(s['mean_recall_below'])}; "
          f"at/above {s['n_at_or_above']} graphs, mean recall {fmt(s['mean_recall_at_or_above'])}; "
          f"permutation p = {fmt(s['perm_p_two_sided'])}")
    print(f"Spearman(label px, recall) = {fmt(s['spearman_x_recall'])}; "
          f"controlling node count = {fmt(s['partial_spearman_controlling_node_count'])}; "
          f"Spearman(node count, label px) = {fmt(s['spearman_nodecount_x'])}")
    for b in s['quartile_bins']:
        print(f"  bin {b['bin']}: n={b['n']}, label px {b['x_range'][0]:.2f}-{b['x_range'][1]:.2f}, "
              f"mean recall {fmt(b['mean_recall'])}")
    if min(s['n_below'], s['n_at_or_above']) < 5:
        print('  WARNING: fewer than 5 graphs on one side of the threshold. '
              'This arm says nothing about whether 10 px is a good cutoff.')


def plot(all_rows, threshold, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping plot')
        return
    arms = sorted({r['arm'] for r in all_rows})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in arms:
        pts = [r for r in all_rows if r['arm'] == arm]
        ax.scatter([r['label_px'] for r in pts], [r['recall'] for r in pts], label=arm, alpha=0.7)
    ax.axvline(threshold, linestyle='--', color='black', linewidth=1)
    ax.set(xlabel='Estimated label cap height (px)', ylabel='Gold-node recall', ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f'wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audit', required=True, help='image_audit.json from image_audit.py')
    parser.add_argument('--predictions', nargs='+', required=True, help='[name=]path.jsonl, one per arm')
    parser.add_argument('--task', default='node_description')
    parser.add_argument('--variant', default='basic', help="recall variant, e.g. basic or raw")
    parser.add_argument('--recall-key', default='gold_node_component_recall.{variant}.recall',
                        help='dotted path to the per-graph recall value in each prediction row')
    parser.add_argument('--x-key', default='estimated_label_cap_px_896',
                        help='audit field used as label height. For LLaVA, use a LLaVA-scaled field if the audit has one')
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    graphs = load_audit(args.audit, args.x_key)
    recall_key = args.recall_key.format(variant=args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, all_rows = [], []
    for spec in args.predictions:
        name, _, path = spec.partition('=')
        if not path:
            name, path = Path(spec).stem, spec
        recalls = load_recall(path, args.task, recall_key)
        summary, rows = analyse(name, graphs, recalls, args.threshold, args.draws)
        summaries.append(summary)
        all_rows += rows
        print_summary(summary)

    with (args.out_dir / 'per_graph.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['arm', 'statement_idx', 'label_px', 'node_count', 'recall'])
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out_dir / 'summary.json').write_text(json.dumps(
        {'task': args.task, 'recall_key': recall_key, 'x_key': args.x_key, 'arms': summaries}, indent=2) + '\n')
    plot(all_rows, args.threshold, args.out_dir / 'label_px_vs_recall.png')
    print(f'\nwrote {args.out_dir}/per_graph.csv and summary.json')


if __name__ == '__main__':
    main()
