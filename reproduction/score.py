#!/usr/bin/env python3
"""Score fresh predictions with an explicitly selected thesis parser; CPU only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
QA = ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'
sys.path[:0] = [str(S1), str(QA)]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    stage = sub.add_parser('stage1')
    stage.add_argument('--input-jsonl', type=Path, required=True)
    stage.add_argument('--graph-metadata', type=Path, required=True)
    stage.add_argument('--predictions-dir', type=Path, required=True)
    stage.add_argument('--model-name', required=True, help='Prediction filename prefix: qwen, llava, gemma3 or deepseek')
    stage.add_argument('--extractor', choices=('template_v1', 'audit_v3', 'audit_v4_proposed'), required=True)
    stage.add_argument('--task-set', choices=('paper', 'extended'), default='extended')
    qa = sub.add_parser('qa')
    qa.add_argument('--input-jsonl', type=Path, required=True)
    qa.add_argument('--predictions', type=Path, required=True)
    qa.add_argument('--parser-version', choices=('explicit_v1', 'explicit_v2'), required=True)
    for p in (stage, qa):
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Use a new output file: {args.output}')
    if args.kind == 'stage1':
        if args.extractor == 'audit_v4_proposed':
            import score_stage1_audit_v4_proposed as scorer
        else:
            import stage1_scoring
            scorer = stage1_scoring.implementation(args.extractor)
        metrics, _ = scorer.score_files(args)
        files = [args.input_jsonl, args.graph_metadata, Path(scorer.__file__)]
        files += [S1 / name for name in ('score_stage1.py', 'score_stage1_template_v1.py',
                  'score_stage1_audit_v2.py', 'score_stage1_audit_v3.py', 'stage1_scoring.py')]
        files += sorted(args.predictions_dir.glob(f'predictions_{args.model_name}_*.jsonl'))
    else:
        from prompt_common import parse_answer
        import prompt_common
        gold_rows, predictions = read(args.input_jsonl), read(args.predictions)
        gold = {int(r['statement_idx']): r['answer'] for r in gold_rows}
        ids = [int(r['statement_idx']) for r in predictions]
        if not gold or len(gold) != len(gold_rows) or len(set(ids)) != len(ids) or set(ids) != set(gold):
            raise ValueError('QA prediction keys must cover exactly the unique input question keys')
        letters = [parse_answer(r.get('raw_response', ''), 4, parser_version=args.parser_version)[0] for r in predictions]
        correct = sum(letter == gold[idx] for letter, idx in zip(letters, ids))
        metrics = dict(n=len(ids), correct=correct, accuracy=correct / len(ids),
                       unparsed=sum(l not in {'A', 'B', 'C', 'D'} for l in letters), parser_version=args.parser_version,
                       letter_counts=dict(Counter(letters)))
        files = [args.input_jsonl, args.predictions, Path(prompt_common.__file__)]
    metrics['reproduction_source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(metrics, handle, indent=2)
        handle.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
