#!/usr/bin/env python3
"""Print paired per-graph recall and response diagnostics for probe arms A-D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def rows(path: Path) -> dict[int, dict]:
    return {int(x['statement_idx']): x for x in
            (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-root', type=Path, required=True)
    p.add_argument('--graph-metadata', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    arms = {a: rows(args.probe_root / f'llava_{a}/predictions_llava_node_description.jsonl') for a in 'ABCD'}
    metadata = rows(args.graph_metadata)
    ids = list(arms['A'])
    if any(list(arms[a]) != ids for a in 'BCD'):
        raise ValueError('A-D probe arms do not contain identical ordered graph IDs')
    recall = lambda arm, idx: arms[arm][idx]['set_metrics']['raw']['recall']
    lines = ['| statement_idx | B rankdir | A recall | B recall | C recall | D recall | token-ceiling hits |',
             '|---:|:---:|---:|---:|---:|---|']
    for idx in ids:
        values = [recall(a, idx) for a in 'ABCD']
        hits = ', '.join(a for a in 'ABCD' if arms[a][idx].get('hit_token_ceiling')) or '—'
        lines.append(f"| {idx} | {metadata[idx].get('rankdir_selected', 'n/a')} | " +
                     ' | '.join(f'{v:.4f}' for v in values) + f' | {hits} |')
    means = {a: mean(recall(a, idx) for idx in ids) for a in 'ABCD'}
    changed = [idx for idx in ids if metadata[idx].get('rankdir_selected') not in (None, 'TB')]
    gain = means['B'] - means['A']
    changed_contribution = sum(recall('B', i)-recall('A', i) for i in changed) / len(ids)
    lines += ['', 'Macro raw recall: ' + ', '.join(f'{a}={means[a]:.4f}' for a in 'ABCD'),
              f"B-A gain={gain:.4f}; orientation-changed IDs={changed}; their contribution to the full macro gain="
              f'{changed_contribution:.4f} ({changed_contribution/gain:.1%} when gain is nonzero).']
    paren_ids = [i for i in ids if any(ch in arms['D'][i].get('raw_response', '') for ch in '()')]
    lines.append(f'D responses containing parentheses: {len(paren_ids)}/{len(ids)} ({len(paren_ids)/len(ids):.1%}); IDs={paren_ids}.')
    drops = sorted(((recall('B', i)-recall('D', i), i) for i in ids), reverse=True)[:3]
    lines += ['', '## Three largest D recall drops vs B', '']
    for drop, idx in drops:
        lines += [f'### statement_idx {idx} (drop {drop:.4f})', '', '**B raw response:**', '', '```text',
                  arms['B'][idx].get('raw_response', ''), '```', '', '**D raw response:**', '', '```text',
                  arms['D'][idx].get('raw_response', ''), '```', '']
    result = '\n'.join(lines) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
    print(result, end='')


if __name__ == '__main__':
    main()
