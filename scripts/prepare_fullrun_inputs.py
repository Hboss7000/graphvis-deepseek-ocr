#!/usr/bin/env python3
"""Freeze CPU-built inputs and their image hashes for fail-fast pod preflight."""
import argparse
import json
from pathlib import Path
from fullrun_common import ROOT, DATA_NAME, TASKS, read_rows, sha, frozen_json
from build_fullrun_data import BASELINE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--data-name', default=DATA_NAME)
    args = p.parse_args()
    root = args.root.resolve()
    data = root / 'outputs' / args.data_name
    if (data / 'inputs_manifest.json').exists():
        raise FileExistsError('Inputs already frozen; do not overwrite')
    if not json.loads((data / 'consistency.json').read_text())['passed']:
        raise ValueError('Consistency gate failed')
    inputs = data / 'inputs'
    inputs.mkdir(exist_ok=False)
    stage = read_rows(data / 'test/stage1_subset100.jsonl')
    qa = read_rows(data / 'test/stage2_obqa_0_500.jsonl')
    by_idx = {}
    for r in stage:
        by_idx.setdefault(r['statement_idx'], set()).add(r['task_type'])
    ids = sorted(i for i, tasks in by_idx.items() if tasks == set(TASKS))[:5]
    if len(ids) != 5:
        raise ValueError('Need five graphs eligible for all nine smoke tasks')
    def write(name, rows):
        path = inputs / name
        with path.open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return str(path.relative_to(root))
    selected = {
        'full': {'stage1': str((data / 'test/stage1_subset100.jsonl').relative_to(root)),
                 'qa': str((data / 'test/stage2_obqa_0_500.jsonl').relative_to(root))},
        'smoke': {'stage1': write('stage1_smoke.jsonl', [r for r in stage if r['statement_idx'] in ids]),
                  'qa': write('qa_smoke.jsonl', [r for r in qa if r['statement_idx'] in ids])},
    }
    probe_source = root / 'outputs/phase1_reading_probe/data/stage1_node_description_first20.jsonl'
    probe = inputs / 'reading_probe.jsonl'
    probe.write_bytes(probe_source.read_bytes())
    probe_rows = read_rows(probe)
    if len(probe_rows) != 20 or {r['task_type'] for r in probe_rows} != {'node_description'}:
        raise ValueError('Invalid original reading probe')
    paths = [data / 'test/graph_metadata_0_500.jsonl', probe,
             data / 'preparation.json', data / 'stage1_provenance.json', data / 'consistency.json',
             root / 'subsets/obqa_test_subset100_seed13.json']
    paths += [root / path for group in selected.values() for path in group.values()]
    paths += [data / r['image'] for r in qa]
    paths += [root / BASELINE / r['image'] for r in probe_rows]
    frozen_json(data / 'inputs_manifest.json', {
        'schema': 1, 'selections': selected, 'smoke_ids': ids,
        'probe_input': str(probe.relative_to(root)), 'legacy_image_root': BASELINE,
        'files': {str(path.relative_to(root)): sha(path) for path in sorted(set(paths))},
        'provenance': {'render': json.loads((data / 'preparation.json').read_text()),
                       'stage1': json.loads((data / 'stage1_provenance.json').read_text())},
    })
    print(f'COMPLETED input manifest: {data / "inputs_manifest.json"}')


if __name__ == '__main__':
    main()
