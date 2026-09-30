#!/usr/bin/env python3
"""Collect Stage 1 pruning cells using their own gold and paired item intersections.

The manifest is the declared matrix, including missing cells. Full-set scores are
only descriptive. Unpruned still applies per-pair relation deduplication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import textwrap

from build_pruning_matrix import (ROOT, MODELS, load_manifest, validate_cell_data,
                                 validate_run_config, prediction_rows, read_jsonl, write_json, sha_file)
from score_stage1 import EXTENDED_TASK_TYPES, aggregate_task, score_record

TASKS = EXTENDED_TASK_TYPES
PRIMARY = {
    'node_description': 'strict.raw.exact_set_equality',
    'node_number': 'strict_accuracy', 'edge_number': 'strict_accuracy',
    'triple_listing': 'strict.raw.exact_set_equality',
    'highest_node_degree': 'strict_joint_accuracy', 'node_degree': 'strict_accuracy',
    'relation_identification': 'strict_accuracy',
    'neighbor_listing': 'strict.raw.mean_f1', 'shortest_path_listing': 'strict_accuracy',
}
TREND_ORDER = ('baseline', 'nodes40', 'nodes40_edges_off')
CONFOUNDS = (
    'All comparable scores use the intersection of (statement_idx, task_type) prediction keys '
    'across ALL declared conditions for a model. Full-set scores are descriptive. '
    'With incomplete cells the intersection is provisional; a missing cell makes it empty.\n'
    'Legibility versus content: by default no condition is masked for legibility. illegible_fraction '
    '(share of images below the declared label threshold) is reported as a covariate column and marked '
    'with a dagger on the figures; the scatter plot shows it directly. Pass --legibility-mask-threshold '
    'to blank conditions whose illegible_fraction exceeds it (status becomes n/a (illegible)); a failed ' 
    'height after whole-image fit into 896 pixels; they do not measure OCR accuracy or overlap.\n'
    'DeepSeek uses fixed image_size=768 with crops; its 768/896 scaled cap-height estimate is a '
    'proxy, not a simulation of its crop pipeline. Gemma pan-and-scan and Qwen dynamic resolution '
    'are different image budgets, so the legibility confound is model-dependent.\n'
    'Gold answers and sampled target nodes/pairs change with pruning. Each cell is rescored '
    'against its own hashed graph metadata and structured gold. Prompt template policy and '
    'realised prompt differences are recorded by the builder.\n'
    'Even unpruned keeps per-pair relation deduplication (best_for_pair); it is not the raw union.\n'
    'Nine-task macro scores are an unweighted mean of the nine declared primary scorer keys '
    '(including neighbor-listing F1); they are unavailable if any task has no paired items.'
)


def flatten_metrics(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten_metrics(item, name))
        elif item is None or (isinstance(item, (float, int)) and not isinstance(item, bool)):
            result[name] = item
    return result


def graph_diagnostics(cell, metadata):
    files = cell['diagnostics']
    stats = json.loads(Path(files['pruning_stats']).read_text())
    audit = json.loads(Path(files['image_audit']).read_text())
    buckets = stats['pruning_configurations']
    if len(buckets) != 1 or json.loads(next(iter(buckets))) != cell['pruning_flags']:
        raise ValueError('Diagnostic pruning bucket differs from manifest')
    if {r['statement_idx'] for r in audit['images']} != {r['statement_idx'] for r in metadata}:
        raise ValueError('Image audit has a different statement slice')
    heights = [r['estimated_label_cap_px_896'] for r in audit['images']]
    factor = 768 / 896 if cell['model'] == 'deepseek' else 1
    failed = sum(height * factor < audit['min_label_px'] for height in heights)
    n = len(metadata)
    return dict(n_graphs=n, median_node_count=statistics.median(len(r['visible_nodes']) for r in metadata),
                median_edge_count=statistics.median(len(r['edges']) for r in metadata),
                core_truncated_count=sum(r['pruning']['core_truncated'] for r in metadata),
                core_truncated_fraction=sum(r['pruning']['core_truncated'] for r in metadata) / n,
                bridges_added_median=statistics.median(r['pruning']['bridges_added'] for r in metadata),
                min_label_px_896=min(heights), median_label_px_896=statistics.median(heights),
                model_proxy_min_label_px=min(heights) * factor,
                label_threshold_px=audit['min_label_px'], illegible_graphs=failed,
                illegible_fraction=failed / n,
                overlap_statistics=audit.get('overlap_statistics'), overlap_note='Not measured by image_audit.py')


def load_cell(cell):
    records, metadata = validate_cell_data(cell)
    by_key = {(r['statement_idx'], r['task_type']): r for r in records}
    by_idx = {r['statement_idx']: r for r in metadata}
    diagnostics = graph_diagnostics(cell, metadata)
    path = Path(cell['output_dir'])
    predictions = prediction_rows(cell)
    config_path = path / 'run_config.json'
    if predictions and not config_path.exists():
        raise ValueError(f"{cell['cell_id']}: predictions lack provenance")
    if config_path.exists():
        validate_run_config(cell, json.loads(config_path.read_text()))
    scored = {}
    for key, prediction in predictions.items():
        if 'raw_response' not in prediction:
            raise ValueError(f'Missing raw response: {cell["cell_id"]} {key}')
        scored[key] = {**prediction, **score_record(by_key[key], prediction['raw_response'], by_idx[key[0]])}
    complete = len(scored) == cell['expected_count']
    if complete:
        metrics_path = path / f"metrics_{cell['model']}.json"
        if not metrics_path.exists():
            raise ValueError(f'Complete predictions have no scored output: {metrics_path}')
        metrics = json.loads(metrics_path.read_text())
        if metrics['input_record_count'] != cell['expected_count']:
            raise ValueError('Scored output count differs from manifest')
    preflights = sorted(path.glob('preflight/attempt_*/preflight_report.json'))
    preflight_failed = bool(preflights and not json.loads(preflights[-1].read_text())['settings']['pan_and_scan']['passed'])
    status = 'complete' if complete else ('partial' if predictions else 'missing')
    return dict(cell=cell, keys=set(by_key), scored=scored, status=status, diagnostics=diagnostics,
                preflight_failed=preflight_failed)


def metric_denominator(name, aggregate, n_items):
    # Item counts are always reported; some metrics use a smaller/different denominator.
    if name == 'mean_absolute_error':
        return aggregate.get('parsed_count', aggregate.get('parsed_degree_count', n_items))
    parts = name.split('.')
    node = aggregate
    for part in parts[:-1]:
        if not isinstance(node, dict):
            break
        if 'record_count' in node:
            n_items = node['record_count']
        node = node.get(part, {})
    if isinstance(node, dict):
        n_items = node.get('record_count', n_items)
        if parts[-1] == 'micro_recall' and 'gold_node_count' in node:
            return node['gold_node_count']
    return n_items


def collect(cells, legibility_mask_threshold=None):
    loaded = [load_cell(cell) for cell in cells]
    models = [m for m in MODELS if any(c['model'] == m for c in cells)]
    conditions = list(dict.fromkeys(c['condition'] for c in cells))
    long, coverage = [], []
    for model in models:
        group = [data for data in loaded if data['cell']['model'] == model]
        intersection = set.intersection(*(set(data['scored']) for data in group))
        eligible_intersection = set.intersection(*(data['keys'] for data in group))
        provisional = any(data['status'] != 'complete' for data in group)
        for data in group:
            cell = data['cell']
            others = set().union(*(d['keys'] for d in group if d is not data))
            only = data['keys'] - others
            legibility_masked = (legibility_mask_threshold is not None
                and data['diagnostics']['illegible_fraction'] > legibility_mask_threshold)
            masked = data['preflight_failed'] or legibility_masked
            coverage.append(dict(model=model, condition=cell['condition'], status=data['status'],
                expected_count=cell['expected_count'], realised_count=len(data['scored']),
                intersection_count=len(intersection), eligible_intersection_count=len(eligible_intersection),
                only_this_condition_count=len(only), nonshared_eligible_count=len(data['keys'] - eligible_intersection),
                only_this_condition_by_task={t: sum(key[1] == t for key in only) for t in TASKS},
                provisional_intersection=provisional, preflight_failed=data['preflight_failed'],
                legibility_masked=legibility_masked, **data['diagnostics']))
            for scope, keys in [('intersection', intersection), ('full', set(data['scored']))]:
                for task in TASKS:
                    rows = [data['scored'][k] for k in sorted(keys) if k[1] == task]
                    metrics = aggregate_task(task, rows)
                    for metric, value in flatten_metrics(metrics).items():
                        status = data['status']
                        if masked and status != 'missing':
                            status = 'illegible'
                        if not rows and status not in ('missing', 'illegible'):
                            status = 'no_items'
                        unmasked = value if rows else None
                        long.append(dict(model=model, condition=cell['condition'], task_type=task,
                            n_items=len(rows), metric_name=metric,
                            value=unmasked if not masked else None,
                            unmasked_value=unmasked, scope=scope, status=status,
                            metric_n_items=metric_denominator(metric, metrics, len(rows)),
                            expected_items=sum(k[1] == task for k in data['keys']),
                            provisional_intersection=provisional if scope == 'intersection' else False,
                            **data['diagnostics']))
    return long, coverage, models, conditions


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()})


def primary_index(rows):
    return {(r['model'], r['condition'], r['task_type'], r['scope']): r
            for r in rows if r['metric_name'] == PRIMARY[r['task_type']]}


def display(row, percent=True):
    if row['status'] == 'illegible':
        return f"n/a (illegible; n={row['n_items']})"
    if row['value'] is None:
        return f"— (n={row['n_items']})"
    value = f"{100 * row['value']:.1f}%" if percent else f"{row['value']:.3f}"
    return f"{value} (n={row['n_items']})"


def macro_rows(index, coverage, models, conditions):
    result = []
    lookup = {(r['model'], r['condition']): r for r in coverage}
    for scope in ('intersection', 'full'):
        for model in models:
            for condition in conditions:
                rows = [index[(model, condition, task, scope)] for task in TASKS]
                values = [r['value'] for r in rows]
                result.append(dict(model=model, condition=condition, scope=scope,
                    value=statistics.mean(values) if all(v is not None for v in values) else None,
                    n_items=sum(r['n_items'] for r in rows), n_tasks=sum(v is not None for v in values),
                    status='illegible' if any(r['status'] == 'illegible' for r in rows) else lookup[model, condition]['status'],
                    metric_name='mean of nine primary scorer keys (see primary_metrics)',
                    **{k: lookup[model, condition][k] for k in ('n_graphs', 'median_node_count', 'median_edge_count',
                        'core_truncated_count', 'core_truncated_fraction', 'bridges_added_median',
                        'model_proxy_min_label_px', 'illegible_graphs', 'label_threshold_px')}))
    return result


def console_summary(index, models, conditions, n, width):
    # Full condition names and explicit n/a markers fit through column blocks.
    for model in models:
        print(f'\n{model} — paired intersection; N={n} statements; primary scorer keys:')
        for task in TASKS:
            print(f'  {task}: {PRIMARY[task]}')
        columns, used = [], 27
        blocks = []
        for condition in conditions:
            size = max(len(condition), max(len(display(index[model, condition, t, 'intersection'], False)) for t in TASKS)) + 2
            if columns and used + size > width:
                blocks.append(columns)
                columns, used = [], 27
            columns.append((condition, size))
            used += size
        if columns:
            blocks.append(columns)
        for block in blocks:
            print('task'.ljust(27) + ''.join(c.rjust(w) for c, w in block))
            for task in TASKS:
                print(task.ljust(27) + ''.join(display(index[model, c, task, 'intersection'], False).rjust(w) for c, w in block))


def summary_markdown(index, macros, coverage, models, conditions, n):
    lines = ['# Stage 1 pruning-condition comparison', '',
        f'Zero-shot image inference on the same **N={n} test statements** per condition. '
        'Nine tasks test whether a model can read the graph. Only pruning varies within each model.', '',
        CONFOUNDS, '', 'Primary metric paths from `score_stage1.aggregate_task`:', '',
        '| Task | Scorer key |', '| --- | --- |']
    lines += [f'| {task} | `{PRIMARY[task]}` |' for task in TASKS]
    lines += ['', '## Headline scores and graph diagnostics', '',
        'Macro scores average all nine primary keys. `n` is the sum of task-item denominators; '
        'it is not a count of distinct statements. Full-set rows are descriptive, not paired.', '',
        '| Model | Condition | Scope | Macro score | Tasks | Median nodes / edges (graphs) | Core truncated | Median fillers (graphs) | Min label px (proxy; graphs) | Illegible images |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in macros:
        g = row['n_graphs']
        lines.append(f"| {row['model']} | {row['condition']} | {row['scope']} | {display(row)} | {row['n_tasks']}/9 | "
                     f"{row['median_node_count']:g} / {row['median_edge_count']:g} (n={g}) | "
                     f"{row['core_truncated_count']}/{g} | {row['bridges_added_median']:g} (n={g}) | "
                     f"{row['model_proxy_min_label_px']:.2f} (n={g}) | {row['illegible_graphs']}/{g} |")
    for scope in ('intersection', 'full'):
        for model in models:
            lines += ['', f'## {model}: {scope}', '',
                      'Signed deltas are percentage points against baseline, with current/baseline item counts. '
                      'Full-set deltas can reflect different eligible statements.', '',
                      '| Task | ' + ' | '.join(f'{c} | Δ baseline (pp)' for c in conditions) + ' |',
                      '| --- | ' + ' | '.join('--- | ---' for _ in conditions) + ' |']
            for task in TASKS:
                baseline = index.get((model, 'baseline', task, scope))
                cells = []
                for condition in conditions:
                    row = index[model, condition, task, scope]
                    delta = '—'
                    if row['status'] == 'illegible':
                        delta = 'n/a (illegible)'
                    elif baseline and row['value'] is not None and baseline['value'] is not None:
                        delta = f"{100 * (row['value'] - baseline['value']):+.1f} (n={row['n_items']}/{baseline['n_items']})"
                    cells += [display(row), delta]
                lines.append('| ' + task + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Coverage and eligibility', '',
              'Only-this-condition counts use generated eligible keys, not completion progress. '
              'Nonshared counts include keys absent from at least one other condition.', '',
              '| Model | Condition | Status | Realised / expected | Paired / eligible paired | Only this condition | Nonshared | Provisional pairing |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in coverage:
        lines.append(f"| {r['model']} | {r['condition']} | {r['status']} | {r['realised_count']}/{r['expected_count']} | "
                     f"{r['intersection_count']}/{r['eligible_intersection_count']} | {r['only_this_condition_count']}/{r['expected_count']} | "
                     f"{r['nonshared_eligible_count']}/{r['expected_count']} | {r['provisional_intersection']} |")
    lines += ['', 'Overlap statistics are **not measured** by the current image audit; missing values are not zero.',
              'Machine-readable outputs include every numeric scorer key for both scopes, metric denominators, '
              'unmasked values, status, and graph diagnostics. Figures show only the paired intersection.', '']
    return '\n'.join(lines)


def plot_figures(out_dir, index, macros, models, conditions, coverage):
    # Reuse the existing palette, hatches, configuration, and PNG/PDF writer.
    import plot_results as style
    plt = style.configure_matplotlib()
    plt.rcParams['savefig.dpi'] = 300
    colors = list(style.CONDITION_COLORS.values()) + ['#D55E00', '#CC79A7', '#56B4E9', '#000000']
    hatches = list(style.CONDITION_HATCHES.values()) + ['..', '++', 'oo', '--']
    macro_lookup = {(r['model'], r['condition']): r for r in macros if r['scope'] == 'intersection'}
    marks = ['o', 's', '^']
    # Scatter first: this directly exposes the content/readability confound.
    fig, ax = plt.subplots(figsize=(10, 5))
    missing = []
    for mi, model in enumerate(models):
        for ci, condition in enumerate(conditions):
            row = macro_lookup[model, condition]
            if row['value'] is None:
                missing.append(f"{model}/{condition}: {display(row)}")
                continue
            ax.scatter(row['model_proxy_min_label_px'], row['value'], color=colors[ci % len(colors)],
                       marker=marks[mi], hatch=hatches[ci % len(hatches)], edgecolor='black', label=f"{model}/{condition} (n={row['n_items']})")
    ax.set(xlabel='Estimated minimum label cap height (px; model proxy)',
           ylabel='Mean of nine primary scorer keys', ylim=(0, 1), title='Legibility — paired intersection')
    if ax.collections:
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    if missing:
        fig.text(0.02, -0.03, '\n'.join(missing), fontsize=7, va='top')
    style.save_figure(fig, out_dir, 'legibility_scatter', plt)

    fig, axes = plt.subplots(len(models), 1, figsize=(max(14, len(conditions) * 2.3), 4 * len(models)), squeeze=False)
    bar_width = .8 / len(conditions)
    for mi, model in enumerate(models):
        ax = axes[mi, 0]
        for ci, condition in enumerate(conditions):
            positions = [i - .4 + bar_width * (ci + .5) for i in range(len(TASKS))]
            rows = [index[model, condition, task, 'intersection'] for task in TASKS]
            ax.bar(positions, [r['value'] if r['value'] is not None else math.nan for r in rows],
                   width=bar_width, color=colors[ci % len(colors)], hatch=hatches[ci % len(hatches)],
                   edgecolor='black', linewidth=.5, label=condition)
            for x, row in zip(positions, rows):
                label = f"n={row['n_items']}" if row['value'] is not None else display(row, False)
                if row['illegible_fraction'] > 0:
                    label += ' †'
                ax.text(x, (row['value'] + .02) if row['value'] is not None else .03,
                        label, rotation=90, fontsize=6, ha='center', va='bottom')
        ax.set_xticks(range(len(TASKS)), [f'{t}\n{PRIMARY[t]}' for t in TASKS], rotation=25, ha='right')
        ax.set(ylabel='Primary scorer value', ylim=(0, 1.3), title=f'{model} — paired intersection')
        ax.legend(ncol=min(4, len(conditions)), fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.01, 0.01, '† at least one image below the legibility threshold (see legibility_scatter)', fontsize=7)
    fig.tight_layout()
    style.save_figure(fig, out_dir, 'task_comparison', plt)

    ordered = [c for c in TREND_ORDER if c in conditions]
    headline_tasks = ('node_description', 'triple_listing', 'shortest_path_listing')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    for ti, task in enumerate(headline_tasks):
        ax = axes[0, ti]
        for mi, model in enumerate(models):
            rows = [index[model, c, task, 'intersection'] for c in ordered]
            ax.plot(range(len(ordered)), [r['value'] if r['value'] is not None else math.nan for r in rows],
                    color=colors[mi], marker=marks[mi], linestyle=('-', '--', ':')[mi], label=model)
            for x, row in enumerate(rows):
                label = f"n={row['n_items']}" if row['value'] is not None else display(row, False)
                if row['illegible_fraction'] > 0:
                    label += ' †'
                ax.annotate(label, (x, row['value'] if row['value'] is not None else .04 + .08 * mi),
                            xytext=(0, 7 + 10 * mi), textcoords='offset points', fontsize=6, rotation=35)
        ax.set_xticks(range(len(ordered)), ordered, rotation=25, ha='right')
        ax.set(title=f'{task}\npaired intersection', ylabel=PRIMARY[task], ylim=(0, 1.15))
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle('Declared retention order (not a claim that realised graph sizes are monotonic)\n'
                 '† at least one image below the legibility threshold (see legibility_scatter)')
    fig.tight_layout()
    style.save_figure(fig, out_dir, 'retention_trend', plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/pruning_matrix/manifest.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/stage1_matrix/collected'))
    parser.add_argument('--terminal-width', type=int, default=120)
    parser.add_argument('--no-plots', action='store_true', help='Explicitly omit derived figures')
    parser.add_argument('--legibility-mask-threshold', type=float, default=None,
        help='Blank a condition (status n/a (illegible)) when illegible_fraction exceeds this value. '
             'Default: never mask for legibility; illegible_fraction stays a covariate column and figure marker.')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.terminal_width < 70:
        raise ValueError('Terminal width must be at least 70')
    if args.legibility_mask_threshold is not None and not 0 <= args.legibility_mask_threshold <= 1:
        raise ValueError('--legibility-mask-threshold must be within [0, 1]')
    cells = load_manifest(args.manifest, check_sources=True)
    rows, coverage, models, conditions = collect(cells, args.legibility_mask_threshold)
    index = primary_index(rows)
    macros = macro_rows(index, coverage, models, conditions)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'long.json', rows)
    csv_write(out / 'long.csv', rows)
    wide = []
    for model in models:
        for task in TASKS:
            metrics = list(dict.fromkeys(r['metric_name'] for r in rows if r['task_type'] == task))
            lookup = {(r['condition'], r['scope'], r['metric_name']): r for r in rows
                      if r['model'] == model and r['task_type'] == task}
            for scope in ('intersection', 'full'):
                for metric in metrics:
                    row = dict(model=model, task_type=task, scope=scope, metric_name=metric)
                    for condition in conditions:
                        item = lookup.get((condition, scope, metric))
                        row[condition] = item['value'] if item else None
                        row[condition + '__n_items'] = item['n_items'] if item else 0
                        row[condition + '__status'] = item['status'] if item else 'metric_not_applicable'
                    wide.append(row)
    write_json(out / 'wide.json', wide)
    csv_write(out / 'wide.csv', wide)
    write_json(out / 'coverage.json', coverage)
    csv_write(out / 'coverage.csv', coverage)
    write_json(out / 'macro.json', {'primary_metrics': PRIMARY, 'rows': macros})
    csv_write(out / 'macro.csv', macros)
    (out / 'summary.md').write_text(summary_markdown(index, macros, coverage, models, conditions,
                                                   cells[0]['generation']['limit']))
    if not args.no_plots:
        # Stable PDF timestamps, including plot_results' existing save helper.
        os.environ['SOURCE_DATE_EPOCH'] = '0'
        plot_figures(out, index, macros, models, conditions, coverage)
    print('\n'.join(textwrap.fill(paragraph, width=args.terminal_width) for paragraph in CONFOUNDS.splitlines()))
    console_summary(index, models, conditions, cells[0]['generation']['limit'], args.terminal_width)
    print(f'Wrote {out}/summary.md, long/wide tables, coverage and macro scores'
          + ('. Figures omitted by --no-plots.' if args.no_plots else ', plus PNG/PDF figures.'))


if __name__ == '__main__':
    main()
