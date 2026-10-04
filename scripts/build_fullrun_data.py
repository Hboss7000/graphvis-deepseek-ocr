#!/usr/bin/env python3
"""CPU-only fullrun preparation. Refuse collisions; stop on baseline drift."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'outputs/inference50_2026-10-03_corekeep_budget18_e30_auto_orient_tb30'
FLAGS = ['--split', 'test', '--seed', '13', '--stage1-task-set', 'extended',
         '--stage1-balance', 'per-task', '--core-policy', 'keep', '--max-nodes', '18',
         '--max-edges', '30', '--max-degree', '0', '--bridge-rule', 'qa-bridge',
         '--rankdir', 'TB', '--node-fontsize', '30', '--edge-fontsize', '24',
         '--ranksep', '0.4', '--auto-orient', 'llava', '--wrap-labels', '0',
         '--edge-label-style', 'plain', '--node-style', 'uniform']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def lines(path):
    return [(json.loads(line), line) for line in path.read_bytes().splitlines(keepends=True) if line.strip()]


def extend_subset(original):
    old = original['indices']
    if len(old) != 50 or len(set(old)) != 50 or original['source_n'] != 500:
        raise ValueError('Expected 50 unique indices from a 500-question source')
    remaining = sorted(set(range(500)) - set(old))
    extra = sorted(random.Random(13).sample(remaining, 50))
    result = dict(original, indices=sorted(old + extra), n=100, seed=13)
    return result, dict(original, indices=extra, n=50, seed=13)


def compare_baseline(baseline, rendered):
    from PIL import Image
    suffix = 'subset50_seed13'
    report = {'checks': {}, 'stage1_regenerated_differences': []}
    for prefix in ('stage2_obqa', 'graph_metadata'):
        old = lines(baseline / 'test' / f'{prefix}_{suffix}.jsonl')
        new = {r['statement_idx']: (r, b) for r, b in lines(rendered / 'test' / f'{prefix}_0_500.jsonl')}
        mismatches = [r['statement_idx'] for r, b in old if new.get(r['statement_idx'], (None, None))[1] != b]
        report['checks'][prefix + '_byte_differences'] = mismatches
    old_meta = lines(baseline / 'test' / f'graph_metadata_{suffix}.jsonl')
    new_meta = {r['statement_idx']: r for r, _ in lines(rendered / 'test/graph_metadata_0_500.jsonl')}
    dimensions, orientations, pruning = [], [], []
    for row, _ in old_meta:
        idx = row['statement_idx']
        with Image.open(baseline / row['image']) as a, Image.open(rendered / row['image']) as b:
            if a.size != b.size:
                dimensions.append({'statement_idx': idx, 'old': a.size, 'new': b.size})
        if row['rankdir_selected'] != new_meta[idx]['rankdir_selected']:
            orientations.append(idx)
        if row['pruning'] != new_meta[idx]['pruning']:
            pruning.append(idx)
    report['checks'].update(dimensions=dimensions, orientations=orientations, pruning=pruning)
    regenerated = {(r['statement_idx'], r['task_type']): r for r, _ in
                   lines(rendered / 'test/stage1_graph_comprehension_0_500.jsonl')}
    for row, _ in lines(baseline / 'test' / f'stage1_graph_comprehension_{suffix}.jsonl'):
        key = (row['statement_idx'], row['task_type'])
        other = regenerated.get(key, {})
        fields = sorted(k for k in set(row) | set(other) if row.get(k) != other.get(k))
        if fields:
            report['stage1_regenerated_differences'].append({'key': key, 'fields': fields})
    report['passed'] = not any(report['checks'].values())
    return report


def compose(original_path, extra_path, destination, expected_ids):
    entries = lines(original_path) + lines(extra_path)
    keys = [(r['statement_idx'], r['task_type']) for r, _ in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate Stage 1 keys')
    if {r['statement_idx'] for r, _ in entries} != set(expected_ids):
        raise ValueError('Stage 1 graph selection differs from subset100')
    # Stable sort preserves each graph's original task order and exact line bytes.
    entries.sort(key=lambda pair: pair[0]['statement_idx'])
    with destination.open('xb') as f:
        for _, raw in entries:
            if not raw.endswith(b'\n'):
                raise ValueError('Source JSONL must end each record with newline')
            f.write(raw)
    return Counter(r['task_type'] for r, _ in entries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--name', default='fullrun_2026-10-04_B')
    args = p.parse_args()
    root = args.root.resolve()
    out = root / 'outputs' / args.name
    if out.exists():
        raise FileExistsError(f'Fresh directory required: {out}')
    subset_path = root / 'subsets/obqa_test_subset100_seed13.json'
    baseline = root / BASELINE
    original_manifest = root / 'subsets/obqa_test_subset50_seed13.json'
    subset, extra = extend_subset(json.loads(original_manifest.read_text()))
    if subset_path.exists() and json.loads(subset_path.read_text()) != subset:
        raise ValueError(f'Existing subset100 manifest differs: {subset_path}')
    source = root / 'data_preprocessed_release/obqa/statement/test.statement.jsonl'
    if sha(source) != subset['sha256_of_statement_file']:
        raise ValueError('Source statement hash differs')
    generator = root / 'scripts/generate_graphvis_datasets.py'
    command = [sys.executable, str(generator), *FLAGS, '--data-root', str(root / 'data_preprocessed_release')]
    out.mkdir(parents=True)
    save(out / 'preparation.json', {'generator_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'generator_sha256': sha(generator), 'flags': FLAGS,
        'sources': {str(path.relative_to(root)): sha(path) for path in [
            source, original_manifest, root / 'data_preprocessed_release/cpnet/concept.txt',
            root / 'data_preprocessed_release/obqa/graph/test.graph.adj.pk']}})
    subprocess.run(command + ['--limit', '500', '--out-dir', str(out)], check=True, cwd=root)
    consistency = compare_baseline(baseline, out)
    save(out / 'consistency.json', consistency)
    if not consistency['passed']:
        raise SystemExit(f'FAILED consistency gate: {out / "consistency.json"}; stop and report')
    extra_manifest = out / 'obqa_test_extra50_seed13.json'
    save(extra_manifest, extra)
    extra_out = out / 'extra_records'
    extra_command = command + ['--indices-file', str(extra_manifest), '--out-dir', str(extra_out),
                               '--reuse-render-from', str(out)]
    subprocess.run(extra_command, check=True, cwd=root)
    original = baseline / 'test/stage1_graph_comprehension_subset50_seed13.jsonl'
    additional = extra_out / 'test/stage1_graph_comprehension_extra50_seed13.jsonl'
    composed = out / 'test/stage1_subset100.jsonl'
    counts = compose(original, additional, composed, subset['indices'])
    if not subset_path.exists():
        save(subset_path, subset)
    save(out / 'stage1_provenance.json', {
        'sources': {str(path.relative_to(root)): sha(path) for path in [original, additional, subset_path]},
        'output_sha256': sha(composed), 'task_counts': counts,
        'generator': json.loads((out / 'preparation.json').read_text()),
        'extra_command': extra_command, 'images_from': str(out.relative_to(root)),
        'original_records_preserved_byte_for_byte': True})
    from pruning_stats import summarize
    from image_audit import audit_split
    meta = [r for r, _ in lines(out / 'test/graph_metadata_0_500.jsonl')]
    save(out / 'pruning_diagnostics.json', summarize(meta))
    save(out / 'image_audit.json', audit_split(out / 'test/graph_metadata_0_500.jsonl', out, 30))
    save(out / 'orientation_counts.json', Counter(r['rankdir_selected'] for r in meta))
    print(f'COMPLETED CPU dataset: {out}; Stage 1 {sum(counts.values())}, QA {len(meta)}')


if __name__ == '__main__':
    main()
