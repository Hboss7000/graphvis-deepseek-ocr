#!/usr/bin/env python3
"""Measure node-cap coverage and rendering-legibility curves on OBQA test.

Part A merges each question once and performs every pruning condition in memory;
it never renders. Part B deliberately reuses the existing generator and audit
CLIs for the requested 100-question rendering grid.

Run from the repository root:

    PYTHONHASHSEED=0 myvenv/bin/python scripts/node_cap_sweep.py \\
      --out-dir outputs/node_cap_sweep_2026-09-24 --part-a --part-b 2>&1 | tee /tmp/cap_sweep.log

Bridge-rule coverage sweep (no rendering):

    PYTHONHASHSEED=0 myvenv/bin/python scripts/node_cap_sweep.py \\
      --out-dir outputs/bridge_rule_sweep_2026-09-24 --part-a \\
      --bridge-rules qa-bridge,core-neighbor,any,none 2>&1 | tee /tmp/bridge_sweep.log

Monitor from a second terminal:

    tail -f outputs/node_cap_sweep_2026-09-24/logs/sweep.log

Open results:

    xdg-open outputs/node_cap_sweep_2026-09-24/figures/overview.png
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from tqdm import tqdm

import generate_graphvis_datasets as gen
import pruning_stats


FIXED_CAPS = (10, 14, 18, 24, 30, 40, 50, 60, 80, 0)
ADAPTIVE_BUDGETS = (0, 2, 4, 6, 8, 10, 15, 20, 30)
REFERENCE_CONDITIONS = (
    ('legacy_18_30_5', 18, 30, 5, 'qa-bridge'),
    ('ref_18_60', 18, 60, 0, 'qa-bridge'),
    ('ref_40_60', 40, 60, 0, 'qa-bridge'),
    ('unpruned_2hop', 0, 0, 0, 'any'),
)
RENDER_CONFIGS = {
    'default': {'rankdir': 'LR', 'node_fontsize': 18, 'ranksep': 0.7},
    'candidate': {'rankdir': 'TB', 'node_fontsize': 30, 'ranksep': 0.4},
}
RENDER_CAPS = (14, 18, 24, 30, 40, 60)
BRIDGE_RULES = ('qa-bridge', 'core-neighbor', 'any', 'none')
ADDED_TYPES = ('q_and_a', 'q_only', 'a_only', 'neither')
ADDED_FIELDS = tuple(f'added_{kind}' for kind in ADDED_TYPES)
ADDED_FRACTION_FIELDS = tuple(f'added_{kind}_mean_frac' for kind in ADDED_TYPES)
RULE_STYLES = {
    'qa-bridge': ('#1f77b4', 'o'),
    'core-neighbor': ('#2ca02c', 's'),
    'any': ('#9467bd', '^'),
    'none': ('#7f7f7f', 'D'),
}
METRIC_FIELDS = (
    'correct_opt_deg0', 'correct_opt_reach', 'correct_opt_dist_median',
    'distractor_opt_deg0', 'distractor_opt_reach', 'deg0_gap',
    'correct_node_deg0', 'correct_node_reach', 'core_truncated_frac',
    'edge_trunc_frac', 'nodes_median', 'nodes_p90', 'edges_median',
    'edges_p90', 'bridges_added_median', 'skipped',
)
PERCENT_FIELDS = {
    'correct_opt_deg0', 'correct_opt_reach', 'distractor_opt_deg0',
    'distractor_opt_reach', 'deg0_gap', 'correct_node_deg0',
    'correct_node_reach', 'core_truncated_frac', 'edge_trunc_frac',
    'below_10px_frac', *ADDED_FRACTION_FIELDS,
}


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open('a', encoding='utf-8', buffering=1)

    def line(self, message: str) -> None:
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        text = f'[{stamp}] {message}'
        print(text, flush=True)
        self.handle.write(text + '\n')

    def close(self) -> None:
        self.handle.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True
    ).strip()


def percentile(values, q: float):
    values = list(values)
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q, method='linear'))


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    if not rows:
        raise ValueError(f'Cannot write empty table: {path}')
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def visible_sizes(metadata: list[dict]) -> tuple[list[int], list[int]]:
    return ([len(row['visible_nodes']) for row in metadata],
            [len(row['edges']) for row in metadata])


def added_node_breakdown(merged_nodes: dict, merged_edges: set, graph: dict) -> dict:
    """Classify visible non-core nodes by adjacency to question/answer core nodes."""
    q_cids = {cid for cid, node in merged_nodes.items() if node['in_question']}
    a_cids = {cid for cid, node in merged_nodes.items() if node['in_choices']}
    core = q_cids | a_cids
    neighbors = {cid: set() for cid in merged_nodes}
    for src, _rel, tgt in merged_edges:
        neighbors.setdefault(src, set()).add(tgt)
        neighbors.setdefault(tgt, set()).add(src)
    counts = {kind: 0 for kind in ADDED_TYPES}
    for cid in graph['visible_nodes']:
        if cid in core:
            continue
        touches_q = bool(neighbors.get(cid, set()) & q_cids)
        touches_a = bool(neighbors.get(cid, set()) & a_cids)
        if touches_q and touches_a:
            counts['q_and_a'] += 1
        elif touches_q:
            counts['q_only'] += 1
        elif touches_a:
            counts['a_only'] += 1
        else:
            counts['neither'] += 1
    counts['total'] = sum(counts.values())
    return counts


def metric_row(name: str, section: str, metadata: list[dict], prunings: list[dict],
               *, max_nodes=None, max_edges=None, max_degree=None,
               bridge_rule=None, B=None, skipped=0, breakdowns=None) -> dict:
    report = pruning_stats.summarize(metadata)
    correct_opt = report['correct_answer_options']
    distractor_opt = report['best_distractor_options']
    correct_node = report['correct_answer_nodes']
    nodes, edges = visible_sizes(metadata)
    count = len(prunings)
    row = {
        'section': section,
        'condition': name,
        'max_nodes': max_nodes,
        'B': B,
        'max_edges': max_edges,
        'max_degree': max_degree,
        'bridge_rule': bridge_rule,
        'graphs': len(metadata),
        'correct_opt_deg0': correct_opt['degree_zero_fraction'],
        'correct_opt_reach': correct_opt['reachable_fraction'],
        'correct_opt_dist_median': correct_opt['reachable_distance']['median'],
        'distractor_opt_deg0': distractor_opt['degree_zero_fraction'],
        'distractor_opt_reach': distractor_opt['reachable_fraction'],
        'deg0_gap': correct_opt['degree_zero_fraction'] - distractor_opt['degree_zero_fraction'],
        'correct_node_deg0': correct_node['degree_zero_fraction'],
        'correct_node_reach': correct_node['reachable_fraction'],
        'core_truncated_frac': sum(p['core_truncated'] for p in prunings) / count if count else None,
        'edge_trunc_frac': sum(p['edges_before_truncation'] > p['edges_after'] for p in prunings) / count if count else None,
        'nodes_median': statistics.median(nodes) if nodes else None,
        'nodes_p90': percentile(nodes, 90),
        'edges_median': statistics.median(edges) if edges else None,
        'edges_p90': percentile(edges, 90),
        'bridges_added_median': statistics.median(p['bridges_added'] for p in prunings) if prunings else None,
        'skipped': skipped,
    }
    if breakdowns is not None:
        with_additions = [item for item in breakdowns if item['total']]
        for kind in ADDED_TYPES:
            row[f'added_{kind}'] = statistics.median(
                item[kind] for item in breakdowns
            ) if breakdowns else None
            row[f'added_{kind}_mean_frac'] = statistics.mean(
                item[kind] / item['total'] for item in with_additions
            ) if with_additions else 0.0
    return row


def question_sizes(idx: int, nodes: dict, edges: set) -> dict:
    q_cids = {cid for cid, node in nodes.items() if node['in_question']}
    a_cids = {cid for cid, node in nodes.items() if node['in_choices']}
    core = q_cids | a_cids
    neighbors = {cid: set() for cid in nodes}
    for src, _rel, tgt in edges:
        neighbors.setdefault(src, set()).add(tgt)
        neighbors.setdefault(tgt, set()).add(src)
    bridge_candidates = sum(
        cid not in core
        and bool(neighbors.get(cid, set()) & q_cids)
        and bool(neighbors.get(cid, set()) & a_cids)
        for cid in nodes
    )
    return {'statement_idx': idx, 'core_size': len(core),
            'qa_bridge_candidates': bridge_candidates,
            'merged_nodes': len(nodes), 'merged_edges': len(edges)}


def prune_condition(label: str, merged: list[tuple[int, dict, set, dict]], log: RunLog,
                    cap_fn, *, max_edges: int, max_degree: int,
                    bridge_rule: str, collect_breakdowns=False):
    start = time.monotonic()
    metadata, prunings, breakdowns = [], [], []
    skipped = 0
    for idx, nodes, edges, statement in tqdm(
            merged, desc=f'Part A {label}', unit='question', dynamic_ncols=True):
        cap = int(cap_fn(nodes))
        if cap < 0:
            raise ValueError(f'{label}: negative node cap')
        if getattr(cap_fn, 'adaptive', False) and cap == 0:
            skipped += 1
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            graph = gen.prune_graph(
                nodes, edges, cap, max_edges, max_degree,
                bridge_rule=bridge_rule, lifelines=True,
            )
        metadata.append(gen.graph_metadata(idx, statement, nodes, graph, 'not_rendered'))
        prunings.append(graph['pruning'])
        if collect_breakdowns:
            breakdowns.append(added_node_breakdown(nodes, edges, graph))
    elapsed = time.monotonic() - start
    log.line(f'Part A condition={label} graphs={len(metadata)} skipped={skipped} elapsed_seconds={elapsed:.2f}')
    return metadata, prunings, skipped, breakdowns


def fixed_cap_fn(cap: int):
    def choose(_nodes):
        return cap
    return choose


def adaptive_cap_fn(B: int):
    def choose(nodes):
        return sum(n['in_question'] or bool(n['in_choices']) for n in nodes.values()) + B
    choose.adaptive = True
    return choose


def build_merged(args, log: RunLog):
    concept_path = args.data_root / 'cpnet/concept.txt'
    statement_path = args.data_root / f'obqa/statement/{args.split}.statement.jsonl'
    graph_path = args.data_root / f'obqa/graph/{args.split}.graph.adj.pk'
    log.line('Loading concept, statement, and adjacency inputs once for Part A')
    concepts = concept_path.read_text(encoding='utf-8').splitlines()
    statements = gen.load_jsonl(statement_path)
    with graph_path.open('rb') as handle:
        graph_entries = pickle.load(handle)
    end = args.start + args.limit
    if args.start < 0 or args.limit <= 0 or end > len(statements):
        raise ValueError('Requested Part A range must fit the input statements')
    merged, sizes = [], []
    for idx in tqdm(range(args.start, end), desc='Part A merge', unit='question', dynamic_ncols=True):
        nodes, edges, _correct = gen.merge_choice_graphs(
            idx, graph_entries, statements[idx], concepts
        )
        merged.append((idx, nodes, edges, statements[idx]))
        sizes.append(question_sizes(idx, nodes, edges))
    del graph_entries
    log.line(f'Part A merged_questions={len(merged)}')
    return merged, sizes


def size_percentiles(sizes: list[dict]) -> list[dict]:
    rows = []
    for metric in ('core_size', 'qa_bridge_candidates', 'merged_nodes', 'merged_edges'):
        values = [row[metric] for row in sizes]
        rows.append({'metric': metric, 'p50': percentile(values, 50),
                     'p75': percentile(values, 75), 'p90': percentile(values, 90),
                     'p95': percentile(values, 95), 'max': max(values)})
    return rows


def sanity_payload(ref18_meta, ref18_prune, ref40_meta, ref40_prune) -> dict:
    report18 = pruning_stats.summarize(ref18_meta)
    report40 = pruning_stats.summarize(ref40_meta)
    edge18 = sum(p['edges_before_truncation'] > p['edges_after'] for p in ref18_prune)
    edge40 = sum(p['edges_before_truncation'] > p['edges_after'] for p in ref40_prune)
    core18 = sum(p['core_truncated'] for p in ref18_prune)

    def values(report):
        return {
            'option_degree_zero_pct': 100 * report['correct_answer_options']['degree_zero_fraction'],
            'option_reachable_pct': 100 * report['correct_answer_options']['reachable_fraction'],
            'node_degree_zero_pct': 100 * report['correct_answer_nodes']['degree_zero_fraction'],
            'node_reachable_pct': 100 * report['correct_answer_nodes']['reachable_fraction'],
        }

    def match_label(observed, target):
        matches = [name for name in ('option', 'node')
                   if round(observed[f'{name}_degree_zero_pct'], 1) == target]
        return ','.join(matches) if matches else 'neither'

    observed18, observed40 = values(report18), values(report40)
    counts_pass = (edge18, edge40, core18) == (6, 422, 176)
    return {
        'status': 'PASS' if counts_pass else 'FAIL',
        'expected': {'ref_18_60_edge_truncation': 6, 'ref_40_60_edge_truncation': 422,
                     'core_truncation_at_18': 176},
        'observed': {'ref_18_60_edge_truncation': edge18,
                     'ref_40_60_edge_truncation': edge40,
                     'core_truncation_at_18': core18,
                     'ref_18_60': observed18, 'ref_40_60': observed40},
        'degree_zero_denominator_matching_42.6_pct': match_label(observed18, 42.6),
        'degree_zero_denominator_matching_15.0_pct': match_label(observed40, 15.0),
    }


def log_sanity(sanity: dict, log: RunLog) -> None:
    observed = sanity['observed']
    for key in ('ref_18_60', 'ref_40_60'):
        values = observed[key]
        log.line(
            f"Sanity {key}: option_deg0={values['option_degree_zero_pct']:.1f}% "
            f"option_reach={values['option_reachable_pct']:.1f}% "
            f"node_deg0={values['node_degree_zero_pct']:.1f}% "
            f"node_reach={values['node_reachable_pct']:.1f}%"
        )
    log.line(
        f"Sanity {sanity['status']}: edge_truncation="
        f"{observed['ref_18_60_edge_truncation']}/500," 
        f"{observed['ref_40_60_edge_truncation']}/500; "
        f"core_truncation_at_18={observed['core_truncation_at_18']}/500; "
        f"degree_zero_matches={sanity['degree_zero_denominator_matching_42.6_pct']},"
        f"{sanity['degree_zero_denominator_matching_15.0_pct']}"
    )


def run_node_cap_part_a(args, dirs, config: dict, log: RunLog):
    if args.split != 'test' or args.start != 0 or args.limit != 500:
        raise ValueError('The declared sanity gate requires --split test --start 0 --limit 500')
    merged, sizes = build_merged(args, log)
    write_csv(dirs['tables'] / 'sizes.csv', sizes)
    size_pct = size_percentiles(sizes)
    write_csv(dirs['tables'] / 'sizes_percentiles.csv', size_pct)

    # The two conditions needed for the mandatory gate run before the sweep.
    ref_cache = {}
    for name, cap in (('ref_18_60', 18), ('ref_40_60', 40)):
        ref_cache[name] = prune_condition(
            name, merged, log, fixed_cap_fn(cap), max_edges=60,
            max_degree=0, bridge_rule='qa-bridge',
        )
    sanity = sanity_payload(
        ref_cache['ref_18_60'][0], ref_cache['ref_18_60'][1],
        ref_cache['ref_40_60'][0], ref_cache['ref_40_60'][1],
    )
    config['sanity_check'] = sanity
    write_run_config(dirs['root'] / 'run_config.json', config)
    log_sanity(sanity, log)
    if sanity['status'] != 'PASS':
        write_failure_summary(dirs['root'], config)
        raise RuntimeError('Sanity truncation counts differ; stopped before the sweep')

    fixed_rows = []
    for cap in FIXED_CAPS:
        label = 'fixed_uncapped' if cap == 0 else f'fixed_{cap}'
        metadata, prunings, skipped, _breakdowns = prune_condition(
            label, merged, log, fixed_cap_fn(cap), max_edges=0,
            max_degree=0, bridge_rule='qa-bridge',
        )
        fixed_rows.append(metric_row(
            label, 'A1', metadata, prunings, max_nodes=cap,
            max_edges=0, max_degree=0, bridge_rule='qa-bridge', skipped=skipped,
        ))

    reference_rows = []
    for name, cap, edge_cap, degree_cap, bridge_rule in REFERENCE_CONDITIONS:
        if name in ref_cache:
            metadata, prunings, skipped, _breakdowns = ref_cache[name]
        else:
            metadata, prunings, skipped, _breakdowns = prune_condition(
                name, merged, log, fixed_cap_fn(cap), max_edges=edge_cap,
                max_degree=degree_cap, bridge_rule=bridge_rule,
            )
        reference_rows.append(metric_row(
            name, 'A3', metadata, prunings, max_nodes=cap,
            max_edges=edge_cap, max_degree=degree_cap,
            bridge_rule=bridge_rule, skipped=skipped,
        ))
    all_fixed = fixed_rows + reference_rows
    write_csv(dirs['tables'] / 'fixed_caps.csv', all_fixed,
              fieldnames=list(all_fixed[0]))

    adaptive_rows = []
    for B in ADAPTIVE_BUDGETS:
        label = f'adaptive_B{B}'
        metadata, prunings, skipped, _breakdowns = prune_condition(
            label, merged, log, adaptive_cap_fn(B), max_edges=0,
            max_degree=0, bridge_rule='qa-bridge',
        )
        adaptive_rows.append(metric_row(
            label, 'A2', metadata, prunings, max_nodes='core+B', B=B,
            max_edges=0, max_degree=0, bridge_rule='qa-bridge', skipped=skipped,
        ))
    write_csv(dirs['tables'] / 'adaptive.csv', adaptive_rows,
              fieldnames=list(adaptive_rows[0]))
    return {'sizes': sizes, 'size_percentiles': size_pct,
            'fixed': fixed_rows, 'references': reference_rows,
            'adaptive': adaptive_rows, 'sanity': sanity,
            'experiment': 'node-cap'}


def csv_text(rows: list[dict], fieldnames) -> str:
    """Serialize rows exactly as write_csv does, for byte-regression checks."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def compare_prior_csv(rows: list[dict], path: Path, *, section=None) -> dict:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        previous = list(reader)
    if section is not None:
        previous = [row for row in previous if row['section'] == section]
    expected = csv_text(previous, fields)
    observed = csv_text(rows, fields)
    if observed == expected:
        return {'status': 'PASS', 'rows': len(rows), 'result': 'byte-for-byte identical'}
    diff = list(difflib.unified_diff(
        expected.splitlines(), observed.splitlines(),
        fromfile=str(path), tofile='new qa-bridge rows', lineterm='', n=2,
    ))
    return {'status': 'FAIL', 'rows': len(rows), 'result': '\n'.join(diff[:80])}


def validate_breakdown_sums(label: str, prunings: list[dict], breakdowns: list[dict]):
    mismatches = [
        (idx, item['total'], pruning['bridges_added'])
        for idx, (item, pruning) in enumerate(zip(breakdowns, prunings))
        if item['total'] != pruning['bridges_added']
    ]
    if mismatches:
        raise ValueError(
            f'{label}: visible added-node breakdown does not sum to bridges_added; '
            f'first mismatch={mismatches[0]}'
        )


def run_bridge_rule_condition(merged, log, rule, *, cap=None, B=None):
    if (cap is None) == (B is None):
        raise ValueError('Specify exactly one of cap or B')
    adaptive = B is not None
    value = B if adaptive else cap
    suffix = f'adaptive_B{value}' if adaptive else ('fixed_uncapped' if value == 0 else f'fixed_{value}')
    metadata, prunings, skipped, breakdowns = prune_condition(
        f'{rule}_{suffix}', merged, log,
        adaptive_cap_fn(B) if adaptive else fixed_cap_fn(cap),
        max_edges=0, max_degree=0, bridge_rule=rule, collect_breakdowns=True,
    )
    validate_breakdown_sums(f'{rule}_{suffix}', prunings, breakdowns)
    return metric_row(
        suffix, 'A2' if adaptive else 'A1', metadata, prunings,
        max_nodes='core+B' if adaptive else cap, B=B,
        max_edges=0, max_degree=0, bridge_rule=rule,
        skipped=skipped, breakdowns=breakdowns,
    ), breakdowns


def write_bridge_failure_summary(root: Path, config: dict):
    sanity = config['bridge_rule_sanity']
    lines = [
        '# Bridge-rule sweep — sanity check failed', '',
        f"Date: {config['date']}", '',
        f"Git commit: `{config['git_commit']}`", '',
        f"Prior-table comparison: **{sanity['prior_csv_match']['status']}**", '',
        f"qa-bridge added-node categories: **{sanity['qa_bridge_categories']['status']}**", '',
        'The sweep stopped before running the remaining bridge rules.',
    ]
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_bridge_rule_part_a(args, dirs, config: dict, log: RunLog):
    if args.split != 'test' or args.start != 0 or args.limit != 500:
        raise ValueError('The bridge-rule sanity checks require --split test --start 0 --limit 500')
    if 'qa-bridge' not in args.bridge_rules:
        raise ValueError('Multi-rule sweeps must include qa-bridge for the declared sanity checks')
    merged, _sizes = build_merged(args, log)

    # Run every qa-bridge row first; no other rule is evaluated before both gates pass.
    qa_fixed, qa_adaptive, qa_breakdowns = [], [], []
    for cap in FIXED_CAPS:
        row, breakdowns = run_bridge_rule_condition(merged, log, 'qa-bridge', cap=cap)
        qa_fixed.append(row)
        qa_breakdowns.extend(breakdowns)
    for B in ADAPTIVE_BUDGETS:
        row, breakdowns = run_bridge_rule_condition(merged, log, 'qa-bridge', B=B)
        qa_adaptive.append(row)
        qa_breakdowns.extend(breakdowns)

    previous = Path('outputs/node_cap_sweep_2026-09-24/tables')
    fixed_check = compare_prior_csv(qa_fixed, previous / 'fixed_caps.csv', section='A1')
    adaptive_check = compare_prior_csv(qa_adaptive, previous / 'adaptive.csv')
    prior_status = 'PASS' if fixed_check['status'] == adaptive_check['status'] == 'PASS' else 'FAIL'
    prohibited = sum(
        item['q_only'] + item['a_only'] + item['neither']
        for item in qa_breakdowns
    )
    category_check = {
        'status': 'PASS' if prohibited == 0 else 'FAIL',
        'added_q_only_plus_a_only_plus_neither': prohibited,
        'question_conditions': len(qa_breakdowns),
    }
    sanity = {
        'prior_csv_match': {'status': prior_status, 'fixed_caps': fixed_check,
                            'adaptive': adaptive_check},
        'qa_bridge_categories': category_check,
    }
    config['bridge_rule_sanity'] = sanity
    write_run_config(dirs['root'] / 'run_config.json', config)
    log.line(
        f"Bridge sanity prior_csv_match={prior_status}: "
        f"fixed={fixed_check['result']}; adaptive={adaptive_check['result']}"
    )
    log.line(
        f"Bridge sanity qa_bridge_categories={category_check['status']}: "
        f"prohibited_added_nodes={prohibited}"
    )
    if prior_status != 'PASS' or category_check['status'] != 'PASS':
        write_bridge_failure_summary(dirs['root'], config)
        raise RuntimeError('Bridge-rule sanity check failed; stopped before remaining rules')

    fixed_rows, adaptive_rows = list(qa_fixed), list(qa_adaptive)
    for rule in args.bridge_rules:
        if rule == 'qa-bridge':
            continue
        for cap in FIXED_CAPS:
            row, _breakdowns = run_bridge_rule_condition(merged, log, rule, cap=cap)
            fixed_rows.append(row)
        if rule != 'none':
            for B in ADAPTIVE_BUDGETS:
                row, _breakdowns = run_bridge_rule_condition(merged, log, rule, B=B)
                adaptive_rows.append(row)

    write_csv(dirs['tables'] / 'fixed_caps_by_rule.csv', fixed_rows,
              fieldnames=list(fixed_rows[0]))
    write_csv(dirs['tables'] / 'adaptive_by_rule.csv', adaptive_rows,
              fieldnames=list(adaptive_rows[0]))
    reference = next(
        row for row in fixed_rows
        if row['bridge_rule'] == 'any' and int(row['max_nodes']) == 0
    )
    return {'experiment': 'bridge-rule', 'fixed': fixed_rows,
            'adaptive': adaptive_rows, 'reference': reference,
            'sanity': sanity, 'rules': list(args.bridge_rules)}


def run_part_a(args, dirs, config: dict, log: RunLog):
    if args.bridge_rules == ['qa-bridge']:
        return run_node_cap_part_a(args, dirs, config, log)
    return run_bridge_rule_part_a(args, dirs, config, log)


def run_subprocess(command: list[str], root: Path, log: RunLog, *, env=None) -> None:
    log.handle.write('$ ' + ' '.join(command) + '\n')
    log.handle.flush()
    result = subprocess.run(command, cwd=root, stdout=log.handle,
                            stderr=subprocess.STDOUT, text=True, env=env)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_part_b(args, dirs, log: RunLog):
    if args.split != 'test':
        raise ValueError('Part B is fixed to the OBQA test split')
    rows, per_image = [], []
    repo_root = Path(__file__).resolve().parents[1]
    # Preserve the virtual-environment launcher. Resolving this symlink would
    # select the system interpreter and lose the venv's installed packages.
    python = sys.executable
    part_b_root = dirs['root'] / 'partB'
    if args.part_b_storage_dir is not None:
        storage = args.part_b_storage_dir.resolve()
        storage.mkdir(parents=True, exist_ok=False)
        part_b_root.symlink_to(storage, target_is_directory=True)
        log.line(f'Part B physical_storage={storage} logical_path={part_b_root}')
    else:
        part_b_root.mkdir()
    audit_env = os.environ.copy()
    support_dir = Path(__file__).resolve().parent / 'node_cap_sweep_support'
    audit_env['PYTHONPATH'] = str(support_dir) + os.pathsep + audit_env.get('PYTHONPATH', '')
    for render_name, render in RENDER_CONFIGS.items():
        for cap in RENDER_CAPS:
            condition = f'{render_name}_cap{cap}'
            out_dir = part_b_root / condition
            started = time.monotonic()
            log.line(f'Part B condition={condition} status=start')
            generate = [
                python, 'scripts/generate_graphvis_datasets.py',
                '--split', 'test', '--start', '0', '--limit', '100',
                '--max-nodes', str(cap), '--max-edges', '0', '--max-degree', '0',
                '--rankdir', render['rankdir'],
                '--node-fontsize', str(render['node_fontsize']),
                '--ranksep', str(render['ranksep']),
                '--out-dir', str(out_dir),
            ]
            run_subprocess(generate, repo_root, log)
            audit_path = out_dir / 'audit.json'
            audit = [
                python, 'scripts/image_audit.py', '--split', str(out_dir / 'test'),
                '--node-fontsize', str(render['node_fontsize']),
                '--output', str(audit_path),
            ]
            run_subprocess(audit, repo_root, log, env=audit_env)
            report = json.loads(audit_path.read_text(encoding='utf-8'))
            images = report['images']
            label_px = [item['estimated_label_cap_px_896'] for item in images]
            widths = [item['width'] for item in images]
            heights = [item['height'] for item in images]
            aspects = [item['aspect_ratio'] for item in images]
            rows.append({
                'render': render_name, 'max_nodes': cap,
                'rankdir': render['rankdir'],
                'node_fontsize': render['node_fontsize'],
                'ranksep': render['ranksep'], 'images': len(images),
                'label_cap_px_median': statistics.median(label_px),
                'label_cap_px_p10': percentile(label_px, 10),
                'below_10px_frac': sum(value < 10 for value in label_px) / len(label_px),
                'image_width_median': statistics.median(widths),
                'image_height_median': statistics.median(heights),
                'aspect_ratio_median': statistics.median(aspects),
            })
            for item in images:
                per_image.append({
                    'cap': cap, 'render': render_name,
                    'statement_idx': item['statement_idx'],
                    'node_count': item['node_count'],
                    'estimated_label_cap_px_896': item['estimated_label_cap_px_896'],
                })
            elapsed = time.monotonic() - started
            log.line(f'Part B condition={condition} status=complete elapsed_seconds={elapsed:.2f}')
    write_csv(dirs['tables'] / 'legibility.csv', rows)
    write_csv(dirs['tables'] / 'legibility_per_image.csv', per_image)
    return {'legibility': rows, 'per_image': per_image}


def setup_axes(ax, ylabel, *, percent=False):
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    if percent:
        ax.set_ylim(bottom=0)


def mark_caps(ax):
    for cap in (18, 40):
        ax.axvline(cap, color='0.4', linestyle='--', linewidth=1)
        ax.text(cap, 0.98, str(cap), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=10,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7})


def fixed_xy(rows, field, percent=False):
    capped = [row for row in rows if int(row['max_nodes']) > 0]
    uncapped = next(row for row in rows if int(row['max_nodes']) == 0)
    scale = 100 if percent else 1
    return ([int(row['max_nodes']) for row in capped],
            [scale * row[field] for row in capped],
            90, scale * uncapped[field])


def plot_fixed_series(ax, rows, field, *, label, color, marker, percent=False):
    x, y, ux, uy = fixed_xy(rows, field, percent)
    ax.plot(x, y, color=color, marker=marker, linewidth=2, label=label)
    ax.scatter([ux], [uy], color=color, marker=marker, s=60, zorder=3)


def reference_value(references, field, percent=False):
    row = next(row for row in references if row['condition'] == 'unpruned_2hop')
    return row[field] * (100 if percent else 1)


def finish_cap_axis(ax):
    ticks = list(FIXED_CAPS[:-1]) + [90]
    labels = [str(value) for value in FIXED_CAPS[:-1]] + ['∞']
    ax.set_xticks(ticks, labels)
    ax.set_xlim(8, 93)
    ax.set_xlabel('max_nodes')
    mark_caps(ax)


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_part_a_figures(result, dirs):
    plt.rcParams.update({'font.size': 12})
    fixed, references = result['fixed'], result['references']
    blue, orange = '#1f77b4', '#ff7f0e'

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    panels = [('correct_opt_deg0', 'degree-zero (%)'),
              ('correct_opt_reach', 'reachable (%)'),
              ('core_truncated_frac', 'core truncated (%)')]
    for ax, (field, ylabel) in zip(axes, panels):
        plot_fixed_series(ax, fixed, field, label='correct answer', color=blue,
                          marker='o', percent=True)
        ax.axhline(reference_value(references, field, True), color='0.35',
                   linestyle=':', label='unpruned 2-hop')
        setup_axes(ax, ylabel, percent=True)
        mark_caps(ax)
    finish_cap_axis(axes[-1])
    axes[0].legend()
    savefig(fig, dirs['figures'] / 'fig1_coverage_vs_cap.png')

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    plot_fixed_series(axes[0], fixed, 'correct_opt_deg0', label='correct answer',
                      color=blue, marker='o', percent=True)
    plot_fixed_series(axes[0], fixed, 'distractor_opt_deg0', label='best distractor',
                      color=orange, marker='s', percent=True)
    setup_axes(axes[0], 'degree-zero (%)', percent=True)
    axes[0].legend()
    plot_fixed_series(axes[1], fixed, 'deg0_gap', label='correct − distractor',
                      color=blue, marker='D', percent=True)
    axes[1].axhline(0, color='0.35', linestyle=':')
    setup_axes(axes[1], 'gap (percentage points)')
    for ax in axes:
        mark_caps(ax)
    finish_cap_axis(axes[-1])
    savefig(fig, dirs['figures'] / 'fig2_correct_vs_distractor.png')

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, median_field, p90_field, ylabel in (
            (axes[0], 'nodes_median', 'nodes_p90', 'visible nodes'),
            (axes[1], 'edges_median', 'edges_p90', 'visible edges')):
        capped = [row for row in fixed if int(row['max_nodes']) > 0]
        x = [int(row['max_nodes']) for row in capped]
        med = [row[median_field] for row in capped]
        p90 = [row[p90_field] for row in capped]
        ax.plot(x, med, color=blue, marker='o', label='median')
        ax.fill_between(x, med, p90, color=blue, alpha=0.2, label='median to p90')
        uncapped = next(row for row in fixed if int(row['max_nodes']) == 0)
        ax.scatter([90], [uncapped[median_field]], color=blue, marker='o', s=60)
        ax.vlines(90, uncapped[median_field], uncapped[p90_field], color=blue,
                  alpha=0.35, linewidth=8)
        ax.axhline(reference_value(references, median_field), color='0.35',
                   linestyle=':', label='unpruned 2-hop median')
        setup_axes(ax, ylabel)
        mark_caps(ax)
    axes[0].legend()
    finish_cap_axis(axes[-1])
    savefig(fig, dirs['figures'] / 'fig3_graph_size_vs_cap.png')

    adaptive = result['adaptive']
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    for ax, (field, ylabel) in zip(axes, panels):
        ax.plot([row['B'] for row in adaptive], [100 * row[field] for row in adaptive],
                color=blue, marker='o')
        ax.axhline(reference_value(references, field, True), color='0.35', linestyle=':')
        setup_axes(ax, ylabel, percent=True)
    axes[-1].set_xlabel('bridge budget B (nodes beyond core)')
    savefig(fig, dirs['figures'] / 'fig4_adaptive_B.png')

    core = [row['core_size'] for row in result['sizes']]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(min(core) - 0.5, max(core) + 1.5, 1)
    ax.hist(core, bins=bins, color=blue, edgecolor='white')
    for cap, style in ((18, '--'), (40, '--')):
        ax.axvline(cap, color='0.3', linestyle=style, label=f'cap {cap}')
    for q, color in ((90, orange), (95, '#2ca02c')):
        value = percentile(core, q)
        ax.axvline(value, color=color, linestyle=':', label=f'p{q}={value:g}')
    ax.set_xlabel('core_size')
    setup_axes(ax, 'questions')
    ax.legend()
    savefig(fig, dirs['figures'] / 'fig5_core_size_hist.png')


def bridge_cap_xy(rows, rule, field, percent=False):
    selected = [row for row in rows if row['bridge_rule'] == rule]
    capped = [row for row in selected if int(row['max_nodes']) > 0]
    uncapped = next(row for row in selected if int(row['max_nodes']) == 0)
    scale = 100 if percent else 1
    return ([int(row['max_nodes']) for row in capped],
            [scale * row[field] for row in capped],
            scale * uncapped[field])


def make_bridge_rule_figures(result, dirs):
    plt.rcParams.update({'font.size': 12})
    fixed, adaptive, rules = result['fixed'], result['adaptive'], result['rules']
    reference = result['reference']

    for filename, field, ylabel in (
            ('fig1_reach_by_rule.png', 'correct_opt_reach', 'correct-option reachable (%)'),
            ('fig2_deg0_by_rule.png', 'correct_opt_deg0', 'correct-option degree-zero (%)')):
        fig, ax = plt.subplots(figsize=(8, 5))
        for rule in rules:
            x, y, uncapped = bridge_cap_xy(fixed, rule, field, percent=True)
            color, marker = RULE_STYLES[rule]
            ax.plot(x, y, color=color, marker=marker, linewidth=2, label=rule)
            ax.scatter([90], [uncapped], color=color, marker=marker, s=60, zorder=3)
        ax.axhline(100 * reference[field], color='0.35', linestyle=':',
                   label='unpruned 2-hop')
        setup_axes(ax, ylabel, percent=True)
        finish_cap_axis(ax)
        ax.legend()
        savefig(fig, dirs['figures'] / filename)

    # Rule identity uses the required fixed colour; hatches identify stack type.
    caps = (18, 30, 40)
    hatches = {'q_and_a': '', 'q_only': '//', 'a_only': 'xx', 'neither': '..'}
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = np.arange(len(caps), dtype=float)
    width = 0.18
    offsets = (np.arange(len(rules)) - (len(rules) - 1) / 2) * width
    for rule_idx, rule in enumerate(rules):
        color, _marker = RULE_STYLES[rule]
        bottom = np.zeros(len(caps))
        rule_rows = {
            int(row['max_nodes']): row for row in fixed
            if row['bridge_rule'] == rule and int(row['max_nodes']) in caps
        }
        for kind in ADDED_TYPES:
            values = 100 * np.asarray(
                [rule_rows[cap][f'added_{kind}_mean_frac'] for cap in caps], dtype=float
            )
            ax.bar(centers + offsets[rule_idx], values, width=width, bottom=bottom,
                   color=color, edgecolor='black', linewidth=0.6,
                   hatch=hatches[kind], alpha=0.85)
            bottom += values
    ax.set_xticks(centers, [str(cap) for cap in caps])
    ax.set_xlabel('max_nodes')
    setup_axes(ax, 'mean added-node composition (%)', percent=True)
    rule_handles = [Line2D([0], [0], color=RULE_STYLES[rule][0], marker=RULE_STYLES[rule][1],
                           linewidth=4, label=rule) for rule in rules]
    type_handles = [Patch(facecolor='white', edgecolor='black', hatch=hatches[kind],
                          label=kind) for kind in ADDED_TYPES]
    first = ax.legend(handles=rule_handles, loc='upper left', fontsize=9, title='rule')
    ax.add_artist(first)
    ax.legend(handles=type_handles, loc='upper right', fontsize=9, title='added type')
    savefig(fig, dirs['figures'] / 'fig3_added_node_types.png')

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    adaptive_rules = [rule for rule in rules if rule != 'none']
    for rule in adaptive_rules:
        selected = [row for row in adaptive if row['bridge_rule'] == rule]
        color, marker = RULE_STYLES[rule]
        x = [int(row['B']) for row in selected]
        axes[0].plot(x, [100 * row['correct_opt_reach'] for row in selected],
                     color=color, marker=marker, label=rule)
        axes[1].plot(x, [100 * row['correct_opt_deg0'] for row in selected],
                     color=color, marker=marker, label=rule)
    axes[0].axhline(100 * reference['correct_opt_reach'], color='0.35', linestyle=':')
    axes[1].axhline(100 * reference['correct_opt_deg0'], color='0.35', linestyle=':')
    setup_axes(axes[0], 'reachable (%)', percent=True)
    setup_axes(axes[1], 'degree-zero (%)', percent=True)
    axes[0].legend()
    axes[-1].set_xlabel('bridge budget B (nodes beyond core)')
    savefig(fig, dirs['figures'] / 'fig4_adaptive_by_rule.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    for rule in rules:
        x, y, uncapped = bridge_cap_xy(fixed, rule, 'edges_median')
        color, marker = RULE_STYLES[rule]
        ax.plot(x, y, color=color, marker=marker, linewidth=2, label=rule)
        ax.scatter([90], [uncapped], color=color, marker=marker, s=60, zorder=3)
    ax.axhline(reference['edges_median'], color='0.35', linestyle=':',
               label='unpruned 2-hop')
    setup_axes(ax, 'median visible edges')
    finish_cap_axis(ax)
    ax.legend()
    savefig(fig, dirs['figures'] / 'fig5_size_by_rule.png')


def make_part_b_figures(result, dirs):
    plt.rcParams.update({'font.size': 12})
    blue, orange = '#1f77b4', '#ff7f0e'
    styles = {'default': (blue, 'o'), 'candidate': (orange, 's')}
    rows = result['legibility']
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for render in ('default', 'candidate'):
        selected = [row for row in rows if row['render'] == render]
        color, marker = styles[render]
        x = [row['max_nodes'] for row in selected]
        axes[0].plot(x, [row['label_cap_px_median'] for row in selected],
                     color=color, marker=marker, label=render)
        axes[1].plot(x, [100 * row['below_10px_frac'] for row in selected],
                     color=color, marker=marker, label=render)
    axes[0].axhline(10, color='0.35', linestyle=':', label='10 px threshold')
    setup_axes(axes[0], 'median label cap (px)')
    setup_axes(axes[1], 'below 10 px (%)', percent=True)
    for ax in axes:
        mark_caps(ax)
    axes[-1].set_xlabel('max_nodes')
    axes[0].legend()
    savefig(fig, dirs['figures'] / 'fig6_legibility_vs_cap.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    for render in ('default', 'candidate'):
        selected = [row for row in result['per_image'] if row['render'] == render]
        color, marker = styles[render]
        ax.scatter([row['node_count'] for row in selected],
                   [row['estimated_label_cap_px_896'] for row in selected],
                   color=color, marker=marker, s=18, alpha=0.45, label=render)
    ax.axhline(10, color='0.35', linestyle=':')
    ax.set_xlabel('actual visible node count')
    setup_axes(ax, 'estimated label cap at 896×896 (px)')
    ax.legend()
    savefig(fig, dirs['figures'] / 'fig7_labelpx_vs_nodecount.png')


def make_overview(dirs):
    names = ('fig1_coverage_vs_cap.png', 'fig3_graph_size_vs_cap.png',
             'fig5_core_size_hist.png', 'fig6_legibility_vs_cap.png')
    if not all((dirs['figures'] / name).exists() for name in names):
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, name in zip(axes.flat, names):
        with Image.open(dirs['figures'] / name) as image:
            ax.imshow(image)
        ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                        wspace=0.02, hspace=0.02)
    fig.savefig(dirs['figures'] / 'overview.png', dpi=150)
    plt.close(fig)


def format_cell(field, value):
    if value is None or value == '':
        return '—'
    if field in PERCENT_FIELDS:
        return f'{100 * float(value):.1f}%'
    if isinstance(value, float):
        return f'{value:.1f}'
    return str(value)


def markdown_table(rows, fields, bold_predicate=lambda _row: False):
    lines = ['| ' + ' | '.join(fields) + ' |',
             '| ' + ' | '.join('---:' if field not in ('condition', 'metric', 'render', 'bridge_rule') else '---'
                                for field in fields) + ' |']
    for row in rows:
        cells = [format_cell(field, row.get(field)) for field in fields]
        if bold_predicate(row):
            cells = [f'**{cell}**' for cell in cells]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def pct(value):
    return f'{100 * value:.1f}%'


def by_cap(rows, cap):
    return next(row for row in rows if int(row['max_nodes']) == cap)


def write_failure_summary(root: Path, config: dict):
    sanity = config['sanity_check']
    observed = sanity['observed']
    text = [
        '# Node-cap sweep — sanity check failed', '',
        f"Date: {config['date']}", '',
        f"Git commit: `{config['git_commit']}`", '',
        'Sanity check: **FAIL**', '',
        f"Observed edge truncation counts: ref_18_60={observed['ref_18_60_edge_truncation']}/500; "
        f"ref_40_60={observed['ref_40_60_edge_truncation']}/500. Core truncation at 18: "
        f"{observed['core_truncation_at_18']}/500.", '',
        'The sweep stopped before A1/A2/remaining A3 and Part B, as required.',
    ]
    (root / 'SUMMARY.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def bridge_comparison_table(rows, rules, field, *, best):
    caps = (14, 18, 24, 30, 40, 0)
    header = ['max_nodes', *rules]
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join(['---:', *['---:' for _ in rules]]) + ' |']
    lookup = {(row['bridge_rule'], int(row['max_nodes'])): row for row in rows}
    for cap in caps:
        values = [lookup[(rule, cap)][field] for rule in rules]
        optimum = best(values)
        cells = ['uncapped' if cap == 0 else str(cap)]
        for value in values:
            formatted = pct(value)
            cells.append(f'**{formatted}**' if value == optimum else formatted)
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def bridge_values_sentence(rows, rules, field, cap, label):
    lookup = {(row['bridge_rule'], int(row['max_nodes'])): row for row in rows}
    values = ', '.join(f"{rule}={pct(lookup[(rule, cap)][field])}" for rule in rules)
    return f'{label}: {values}.'


def write_bridge_summary(root: Path, config: dict, result: dict):
    rules, fixed, adaptive = result['rules'], result['fixed'], result['adaptive']
    sanity = result['sanity']
    lines = ['# Bridge-rule sweep', '', f"Date: {config['date']}", '',
             f"Git commit: `{config['git_commit']}`", '', 'Input SHA-256:', '']
    for name, details in config['input_files'].items():
        lines.append(f"- `{name}`: `{details['sha256']}`")
    prior = sanity['prior_csv_match']
    categories = sanity['qa_bridge_categories']
    lines += [
        '', 'Sanity checks:', '',
        f"- Prior qa-bridge CSV comparison: **{prior['status']}** "
        f"(fixed: {prior['fixed_caps']['result']}; adaptive: {prior['adaptive']['result']}).",
        f"- qa-bridge added-node categories: **{categories['status']}** "
        f"({categories['added_q_only_plus_a_only_plus_neither']} prohibited nodes across "
        f"{categories['question_conditions']} question-conditions).",
        '', ('Added-node fraction columns are means of each question\'s added-node '
             'composition, calculated over questions with at least one added node; '
             'conditions with no additions report 0%.'),
        '', '## Main comparison — correct-option reachability', '',
        bridge_comparison_table(fixed, rules, 'correct_opt_reach', best=max), '',
        bridge_values_sentence(fixed, rules, 'correct_opt_reach', 18, 'Cap 18'),
        bridge_values_sentence(fixed, rules, 'correct_opt_reach', 0, 'Uncapped'),
        '', '## Main comparison — correct-option degree-zero', '',
        bridge_comparison_table(fixed, rules, 'correct_opt_deg0', best=min), '',
        bridge_values_sentence(fixed, rules, 'correct_opt_deg0', 18, 'Cap 18'),
        bridge_values_sentence(fixed, rules, 'correct_opt_deg0', 0, 'Uncapped'),
        '', '## Full fixed-cap results', '',
    ]
    fields = ('bridge_rule', 'max_nodes') + METRIC_FIELDS + ADDED_FIELDS + ADDED_FRACTION_FIELDS
    lines.append(markdown_table(fixed, fields))
    lookup = {(row['bridge_rule'], int(row['max_nodes'])): row for row in fixed}
    edges30 = ', '.join(f"{rule}={lookup[(rule, 30)]['edges_median']:.1f}" for rule in rules)
    added30 = ', '.join(
        f"{rule}={sum(lookup[(rule, 30)][field] for field in ADDED_FIELDS):.1f}"
        for rule in rules
    )
    lines += [
        '', f'Median visible edges at cap 30: {edges30}.',
        f'Sum of median added-node-type counts at cap 30: {added30}.',
        '', '## Full adaptive-cap results', '',
    ]
    adaptive_fields = ('bridge_rule', 'B') + METRIC_FIELDS + ADDED_FIELDS + ADDED_FRACTION_FIELDS
    lines.append(markdown_table(adaptive, adaptive_fields))
    adaptive_lookup = {(row['bridge_rule'], int(row['B'])): row for row in adaptive}
    adaptive_rules = [rule for rule in rules if rule != 'none']
    for B in (0, 30):
        values = ', '.join(
            f"{rule}={pct(adaptive_lookup[(rule, B)]['correct_opt_reach'])}"
            for rule in adaptive_rules
        )
        lines += ['', f'Correct-option reachability at B={B}: {values}.']
    lines += [
        '', '![](figures/fig1_reach_by_rule.png)', '',
        '![](figures/fig2_deg0_by_rule.png)', '',
        '![](figures/fig3_added_node_types.png)', '',
        '![](figures/fig4_adaptive_by_rule.png)', '',
        '![](figures/fig5_size_by_rule.png)',
    ]
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_summary(root: Path, config: dict, part_a, part_b):
    lines = ['# Node-cap sweep', '', f"Date: {config['date']}", '',
             f"Git commit: `{config['git_commit']}`", '', 'Input SHA-256:', '']
    for name, details in config['input_files'].items():
        lines.append(f"- `{name}`: `{details['sha256']}`")
    lines += ['', f"Sanity check: **{config.get('sanity_check', {}).get('status', 'NOT RUN')}**"]
    if part_a:
        sanity = part_a['sanity']
        obs = sanity['observed']
        lines += [
            '',
            f"Reference counts: ref_18_60 edge truncation {obs['ref_18_60_edge_truncation']}/500; "
            f"ref_40_60 edge truncation {obs['ref_40_60_edge_truncation']}/500; "
            f"core truncation at 18 {obs['core_truncation_at_18']}/500.",
            f"At 18, option/node degree-zero were "
            f"{obs['ref_18_60']['option_degree_zero_pct']:.1f}%/"
            f"{obs['ref_18_60']['node_degree_zero_pct']:.1f}%; at 40 they were "
            f"{obs['ref_40_60']['option_degree_zero_pct']:.1f}%/"
            f"{obs['ref_40_60']['node_degree_zero_pct']:.1f}%. The earlier degree-zero values match "
            f"the {sanity['degree_zero_denominator_matching_42.6_pct']} denominator at 18 and "
            f"the {sanity['degree_zero_denominator_matching_15.0_pct']} denominator at 40.",
            '', '## A0 — merged size distributions', '',
            markdown_table(part_a['size_percentiles'], ('metric', 'p50', 'p75', 'p90', 'p95', 'max')),
        ]
        sizes = part_a['sizes']
        over18 = sum(row['core_size'] > 18 for row in sizes) / len(sizes)
        over40 = sum(row['core_size'] > 40 for row in sizes) / len(sizes)
        core_pct = next(row for row in part_a['size_percentiles'] if row['metric'] == 'core_size')
        lines += ['', f"Core size exceeded 18 for {pct(over18)} of questions and exceeded 40 for {pct(over40)}.",
                  f"Core size p90/p95/max were {core_pct['p90']:.1f}/{core_pct['p95']:.1f}/{core_pct['max']} nodes.",
                  '', '![](figures/fig5_core_size_hist.png)', '',
                  '## A1 — fixed caps, edge cap off', '']
        fields = ('max_nodes',) + METRIC_FIELDS
        lines.append(markdown_table(part_a['fixed'], fields,
                                    lambda row: int(row['max_nodes']) in (18, 40)))
        row18, row40 = by_cap(part_a['fixed'], 18), by_cap(part_a['fixed'], 40)
        lines += [
            '',
            f"Correct-option degree-zero was {pct(row18['correct_opt_deg0'])} at 18 and {pct(row40['correct_opt_deg0'])} at 40.",
            f"Correct-option reachability was {pct(row18['correct_opt_reach'])} at 18 and {pct(row40['correct_opt_reach'])} at 40.",
            f"Median visible nodes/edges were {row18['nodes_median']:.1f}/{row18['edges_median']:.1f} at 18 and {row40['nodes_median']:.1f}/{row40['edges_median']:.1f} at 40.",
            '', '## A3 — reference conditions', '',
        ]
        ref_fields = ('condition', 'max_nodes', 'max_edges', 'max_degree', 'bridge_rule') + METRIC_FIELDS
        lines.append(markdown_table(part_a['references'], ref_fields,
                                    lambda row: row['condition'] in ('ref_18_60', 'ref_40_60')))
        r18 = next(row for row in part_a['references'] if row['condition'] == 'ref_18_60')
        r40 = next(row for row in part_a['references'] if row['condition'] == 'ref_40_60')
        unpruned = next(row for row in part_a['references'] if row['condition'] == 'unpruned_2hop')
        lines += [
            '',
            f"Reference edge truncation was {pct(r18['edge_trunc_frac'])} at ref_18_60 and {pct(r40['edge_trunc_frac'])} at ref_40_60.",
            f"Unpruned 2-hop correct-option degree-zero/reachability were {pct(unpruned['correct_opt_deg0'])}/{pct(unpruned['correct_opt_reach'])}.",
            '', '![](figures/fig1_coverage_vs_cap.png)', '',
            '![](figures/fig2_correct_vs_distractor.png)', '',
            '![](figures/fig3_graph_size_vs_cap.png)',
            '', '## A2 — adaptive cap', '',
        ]
        adaptive_fields = ('B',) + METRIC_FIELDS
        lines.append(markdown_table(part_a['adaptive'], adaptive_fields))
        b0, b30 = part_a['adaptive'][0], part_a['adaptive'][-1]
        lines += [
            '',
            f"Correct-option degree-zero was {pct(b0['correct_opt_deg0'])} at B=0 and {pct(b30['correct_opt_deg0'])} at B=30.",
            f"Correct-option reachability was {pct(b0['correct_opt_reach'])} at B=0 and {pct(b30['correct_opt_reach'])} at B=30.",
            f"The adaptive sweep skipped {sum(row['skipped'] for row in part_a['adaptive'])} question-condition pairs.",
            '', '![](figures/fig4_adaptive_B.png)',
        ]
    if part_b:
        lines += ['', '## Part B — rendering legibility proxy', '']
        fields = ('render', 'max_nodes', 'label_cap_px_median', 'label_cap_px_p10',
                  'below_10px_frac', 'image_width_median', 'image_height_median',
                  'aspect_ratio_median')
        lines.append(markdown_table(part_b['legibility'], fields,
                                    lambda row: int(row['max_nodes']) in (18, 40)))
        for render in ('default', 'candidate'):
            selected = [row for row in part_b['legibility'] if row['render'] == render]
            first, last = selected[0], selected[-1]
            lines += ['',
                      f"For {render}, median label cap was {first['label_cap_px_median']:.1f} px at cap 14 and {last['label_cap_px_median']:.1f} px at cap 60.",
                      f"For {render}, the fraction below 10 px was {pct(first['below_10px_frac'])} at cap 14 and {pct(last['below_10px_frac'])} at cap 60."]
        lines += [
            '',
            'The audit models an aspect-preserving fit into 896×896 (Gemma input size). It is a geometric proxy, not model recall, and does not model DeepSeek-OCR-2 crops or Qwen dynamic resolution.',
            '', '![](figures/fig6_legibility_vs_cap.png)', '',
            '![](figures/fig7_labelpx_vs_nodecount.png)',
        ]
    if (root / 'figures/overview.png').exists():
        lines += ['', '## Overview', '', '![](figures/overview.png)']
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_run_config(path: Path, config: dict):
    path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data_preprocessed_release'))
    parser.add_argument('--split', choices=('test',), default='test')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument(
        '--bridge-rules', default='qa-bridge',
        help='Comma-separated prune_graph bridge rules (default: qa-bridge).',
    )
    parser.add_argument(
        '--part-b-storage-dir', type=Path,
        help='Optional physical directory for partB; --out-dir/partB becomes a symlink to it.',
    )
    parser.add_argument('--part-a', action='store_true')
    parser.add_argument('--part-b', action='store_true')
    args = parser.parse_args()
    args.bridge_rules = [value.strip() for value in args.bridge_rules.split(',') if value.strip()]
    invalid = [value for value in args.bridge_rules if value not in BRIDGE_RULES]
    if not args.bridge_rules or invalid or len(set(args.bridge_rules)) != len(args.bridge_rules):
        parser.error(f'--bridge-rules must be unique values from {",".join(BRIDGE_RULES)}')
    if not args.part_a and not args.part_b:
        parser.error('select --part-a and/or --part-b')
    if args.part_b and args.bridge_rules != ['qa-bridge']:
        parser.error('Part B rendering is defined only for --bridge-rules qa-bridge')
    return args


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    args.data_root = args.data_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    dirs = {'root': args.out_dir,
            'tables': args.out_dir / 'tables',
            'figures': args.out_dir / 'figures',
            'logs': args.out_dir / 'logs'}
    for path in (dirs['tables'], dirs['figures'], dirs['logs']):
        path.mkdir()
    log = RunLog(dirs['logs'] / 'sweep.log')
    concept_path = args.data_root / 'cpnet/concept.txt'
    statement_path = args.data_root / f'obqa/statement/{args.split}.statement.jsonl'
    graph_path = args.data_root / f'obqa/graph/{args.split}.graph.adj.pk'
    inputs = {'concept.txt': concept_path, 'test.statement.jsonl': statement_path,
              'test.graph.adj.pk': graph_path}
    config = {
        'date': '2026-09-24',
        'git_commit': git_commit(repo_root),
        'argv': sys.argv,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'python': sys.version,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'input_files': {name: {'path': str(path), 'sha256': sha256(path)}
                        for name, path in inputs.items()},
        'fixed_caps': list(FIXED_CAPS),
        'adaptive_budgets': list(ADAPTIVE_BUDGETS),
        'render_caps': list(RENDER_CAPS),
        'render_configurations': RENDER_CONFIGS,
        'status': 'running',
    }
    write_run_config(dirs['root'] / 'run_config.json', config)
    experiment_name = 'bridge-rule' if args.bridge_rules != ['qa-bridge'] else 'node-cap'
    log.line(f"Started {experiment_name} sweep git_commit={config['git_commit']}")
    part_a = part_b = None
    try:
        if args.part_a:
            part_a = run_part_a(args, dirs, config, log)
            if part_a['experiment'] == 'bridge-rule':
                make_bridge_rule_figures(part_a, dirs)
            else:
                make_part_a_figures(part_a, dirs)
        if args.part_b:
            part_b = run_part_b(args, dirs, log)
            make_part_b_figures(part_b, dirs)
        make_overview(dirs)
        config['status'] = 'complete'
        config['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_run_config(dirs['root'] / 'run_config.json', config)
        if part_a and part_a['experiment'] == 'bridge-rule':
            write_bridge_summary(dirs['root'], config, part_a)
        else:
            write_summary(dirs['root'], config, part_a, part_b)
        log.line('Bridge-rule sweep complete' if part_a and part_a['experiment'] == 'bridge-rule'
                 else 'Node-cap sweep complete')
    except BaseException as exc:
        config['status'] = 'failed'
        config['error'] = f'{type(exc).__name__}: {exc}'
        write_run_config(dirs['root'] / 'run_config.json', config)
        log.line(f"Node-cap sweep failed: {config['error']}")
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
