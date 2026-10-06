#!/usr/bin/env python3
"""Laptop-only exact smoke subsets from the frozen 100-graph and 500-QA inputs."""
import argparse
import json
from pathlib import Path
from llava_stage1_training import file_sha


def prepare(source, destination):
    stage_path = source / 'test/stage1_subset100.jsonl'
    qa_path = source / 'test/stage2_obqa_0_500.jsonl'
    metadata = source / 'test/graph_metadata_0_500.jsonl'
    stage = [json.loads(line) for line in stage_path.read_text().splitlines()]
    qa = [json.loads(line) for line in qa_path.read_text().splitlines()]
    if len(stage) != 900 or len(qa) != 500:
        raise ValueError('Expected frozen 900 Stage 1 / 500 QA records')
    graph = min(r['statement_idx'] for r in stage)
    stage_dry = [r for r in stage if r['statement_idx'] == graph]
    qa_dry = sorted(qa, key=lambda r: r['statement_idx'])[:8]
    if len(stage_dry) != 9 or len({r['task_type'] for r in stage_dry}) != 9:
        raise ValueError('Dry graph must exercise all nine evaluation tasks')
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in [('stage1_dry.jsonl', stage_dry), ('qa_dry.jsonl', qa_dry)]:
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        path = destination / name
        if path.exists() and path.read_text() != text:
            raise ValueError('Evaluation subset collision')
        path.write_text(text)
    manifest = {'stage1_dry_count': 9, 'qa_dry_count': 8, 'stage1_full_count': 900, 'qa_full_count': 500,
                'source_files_sha256': {str(p.relative_to(source)): file_sha(p)
                                        for p in (stage_path, qa_path, metadata)},
                'dry_files_sha256': {p.name: file_sha(p) for p in destination.glob('*.jsonl')}}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/fullrun_2026-10-04_B'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/llava_stage1_eval_inputs'))
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
