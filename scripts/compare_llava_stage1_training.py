#!/usr/bin/env python3
"""Re-score main/P2/P3/tuned/transfer arms using the existing unchanged scorers."""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'))
sys.path.insert(0, str(ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'))
from score_stage1 import score_files, TASK_SETS
from prompt_common import parse_answer
from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION
from llava_stage1_training import file_sha

PAPER = {'node_description': (1.4, 12.8), 'node_degree': (15.3, 27.),
         'highest_node_degree': (3.3, 11.6), 'node_number': (16.7, 27.5),
         'edge_number': (9.7, 16.2), 'triple_listing': (.6, 8.2)}
ARMS = {'main': ('hf-chat', 'none', False), 'P2': ('llava_v1', 'none', False),
        'P3': ('llava_v1', 'gold-template', False),
        'tuned': ('llava_v1', 'none', True), 'tuned-prefix': ('llava_v1', 'gold-template', True)}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_config(config, input_path, template, prefix, tuned, cap, backbone='llava'):
    model_id, revision = DEFAULT_MODEL_ID, DEFAULT_REVISION
    if backbone == 'qwen':
        from qwen_stage1_training import DEFAULT_MODEL_ID as model_id, DEFAULT_REVISION as revision
        bounds = config.get('image_processing', config)
        if (bounds.get('min_pixels'), bounds.get('max_pixels')) != (262144, 1310720):
            raise ValueError('Qwen pixel bounds differ')
        if config.get('default_system_prompt_injected') is not False:
            raise ValueError('Qwen must use no default system prompt')
    if (config['model_id'], config['model_revision'], config.get('seed')) != (
            model_id, revision, 13):
        raise ValueError('Comparison model/revision/seed differs')
    if config['input_jsonl']['sha256'] != file_sha(input_path):
        raise ValueError('Comparison input differs')
    if config.get('prompt_template', 'qwen-native-no-system' if backbone == 'qwen' else 'hf-chat') != template or config.get('assistant_prefix_mode', 'none') != prefix:
        raise ValueError('Comparison arm prompt differs')
    if bool(config.get('adapter')) != tuned:
        raise ValueError('Comparison adapter status differs')
    if config.get('effective_max_new_tokens') != cap:
        raise ValueError('Comparison token ceiling differs')
    generation = config['generation']
    if generation.get('do_sample') is not False or generation.get('num_beams') != 1:
        raise ValueError('Comparison requires the same greedy decoding')
    if tuned and not config['adapter'].get('sha256'):
        raise ValueError('Missing tuned adapter checksum')


def compare_stage(directories, source, metadata, backbone='llava'):
    arms = ARMS if backbone == 'llava' else {'main': ('qwen-native-no-system','none',False), 'tuned': ('qwen-native-no-system','none',True)}
    results = {}
    for name, (template, prefix, tuned) in arms.items():
        directory = directories[name]
        if not directory.exists():
            results[name] = {'status': 'missing', 'path': str(directory), 'metrics': None}
            continue
        config = json.loads((directory / 'run_config.json').read_text())
        verify_config(config, source, template, prefix, tuned, 1024, backbone)
        if config['graph_metadata']['sha256'] != file_sha(metadata):
            raise ValueError('Comparison metadata differs')
        if backbone == 'qwen' and (config.get('extractor') != 'span_extended'
                or config.get('answer_format', {}).get('mode') != 'none'):
            raise ValueError('Qwen comparison requires the unchanged extractor/answer format')
        metrics, _ = score_files(SimpleNamespace(input_jsonl=source, graph_metadata=metadata,
            predictions_dir=directory, model_name=backbone, task_set='extended', extractor='span_extended'))
        if metrics['input_record_count'] != 900 or any(n != 100 for n in metrics['task_record_counts'].values()):
            raise ValueError('Expected all 100 graphs x nine tasks in comparison')
        results[name] = {'status': 'complete', 'path': str(directory), 'metrics': metrics,
                         'adapter_sha256': config.get('adapter', {}).get('sha256'),
                         'prompt_bodies_sha256': config.get('prompt_bodies_sha256'),
                         'run_config_sha256': file_sha(directory / 'run_config.json')}
    if backbone == 'qwen':
        hashes = {r['prompt_bodies_sha256'] for r in results.values() if r['status'] == 'complete'}
        if None in hashes or len(hashes) > 1:
            raise ValueError('Qwen zero-shot/tuned prompt bodies differ')
    tuned_hashes = {value['adapter_sha256'] for name, value in results.items()
                   if name.startswith('tuned') and value['status'] == 'complete'}
    if len(tuned_hashes) > 1:
        raise ValueError('Tuned Stage 1 arms use different adapters')
    return results


def compare_qa(directory, source, tuned, backbone='llava'):
    if not directory.exists() or not any((directory / f'qa_{c}').exists() for c in ('image','text_noref','text','kg_text')):
        return {'status': 'missing', 'metrics': None}
    expected = {int(r['statement_idx']): r for r in read_rows(source)}
    if len(expected) != 500:
        raise ValueError('QA comparison requires all 500 source questions')
    metrics = {}
    adapter_hashes = set()
    for condition in ('image', 'text_noref', 'text', 'kg_text'):
        folder = directory / f'qa_{condition}'
        config = json.loads((folder / 'run_config.json').read_text())
        if backbone == 'qwen' and config['graph_metadata']['sha256'] != file_sha(
                source.parent / 'graph_metadata_0_500.jsonl'):
            raise ValueError('QA comparison metadata differs')
        verify_config(config, source, ('qwen-native-no-system' if backbone == 'qwen' else
            ('llava_v1' if tuned else 'hf-chat')), 'none', tuned, 64, backbone)
        if tuned:
            adapter_hashes.add(config['adapter']['sha256'])
        if config['condition'] != condition:
            raise ValueError('QA condition differs')
        paths = list(folder.glob('predictions*.jsonl'))
        if len(paths) != 1:
            raise ValueError('Expected exactly one QA prediction file per condition')
        rows = read_rows(paths[0])
        ids = [int(r['statement_idx']) for r in rows]
        if len(ids) != 500 or len(set(ids)) != 500 or set(ids) != set(expected):
            raise ValueError('Missing/duplicate/unexpected QA predictions')
        parsed = [parse_answer(r['raw_response'], n_choices=4)[0] for r in rows]
        if any(r.get('gold_option') != expected[int(r['statement_idx'])]['answer'] for r in rows):
            raise ValueError('QA gold answer differs')
        correct = sum(letter == expected[int(row['statement_idx'])]['answer'] for row, letter in zip(rows, parsed))
        metrics[condition] = {'n': 500, 'accuracy': correct / 500,
            'parse_failure_fraction': sum(p == 'FAILED' for p in parsed) / 500,
            'ceiling_fraction': sum(r['hit_token_ceiling'] for r in rows) / 500,
            'prompt_bodies_sha256': config.get('prompt_bodies_sha256'),
            'run_config_sha256': file_sha(folder / 'run_config.json')}
    if len(adapter_hashes) > 1:
        raise ValueError('Tuned QA conditions use different adapters')
    return {'status': 'complete', 'metrics': metrics,
            'adapter_sha256': next(iter(adapter_hashes), None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=('llava','qwen'), default='llava')
    p.add_argument('--stage1-only', action='store_true', help='QA was deliberately cut; still compare all nine Stage 1 tasks')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'outputs/fullrun_2026-10-04_B')
    baseline = ROOT / 'outputs/fullrun_2026-10-04_B_results/llava'
    p.add_argument('--main', type=Path)
    for name in ('P2', 'P3'):
        p.add_argument('--' + name.lower(), type=Path,
            default=ROOT / 'outputs/llava_stage1_diagnostics_2026-10-05/full' / name)
    p.add_argument('--tuned', type=Path, required=True, help='Fetched evaluation root containing stage1/ and qa_CONDITION/')
    p.add_argument('--qa-main', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    baseline = ROOT / f'outputs/fullrun_2026-10-04_B_results/{args.backbone}'
    args.main = args.main or baseline / 'stage1'
    args.qa_main = args.qa_main or baseline
    directories = {'main': args.main, 'P2': args.p2, 'P3': args.p3,
                   'tuned': args.tuned / 'stage1', 'tuned-prefix': args.tuned / 'stage1_prefix'}
    results = compare_stage(directories, args.data_dir / 'test/stage1_subset100.jsonl',
                            args.data_dir / 'test/graph_metadata_0_500.jsonl', args.backbone)
    report = {'stage1': results, 'qa': {
        'main': ({'status':'cut','metrics':None} if args.stage1_only else compare_qa(args.qa_main, args.data_dir / 'test/stage2_obqa_0_500.jsonl', False, args.backbone)),
        'tuned': ({'status':'cut','metrics':None} if args.stage1_only else compare_qa(args.tuned, args.data_dir / 'test/stage2_obqa_0_500.jsonl', True, args.backbone))},
        'paper_table4_percent': {task: {'original': v[0], 'after_printed': v[1]} for task, v in PAPER.items()},
        'paper_edge_number_discrepancy': {'printed': 16.2, 'request_gain_derived': 19.4, 'printed_gain': 9.7},
        'notes': ['Existing span_extended Stage 1 scorer and shared strict-64 QA parser; no scorer changes.',
                  'Held-out relation/neighbor/path tasks measure transfer.',
                  'Paper CSQA exact-match scores are directional references, not directly comparable headline metrics.',
                  'Unavailable arms are explicit missing results, never zeros.']}
    report['backbone'] = args.backbone
    if args.backbone == 'qwen' and all(arm['status'] == 'complete' for arm in report['qa'].values()):
        for condition in ('image', 'text_noref', 'text', 'kg_text'):
            hashes = {arm['metrics'][condition]['prompt_bodies_sha256'] for arm in report['qa'].values()}
            if None in hashes or len(hashes) != 1:
                raise ValueError('Qwen zero-shot/tuned QA prompt bodies differ')
    adapters = {arm['adapter_sha256'] for arm in results.values()
                if arm['status'] == 'complete' and arm.get('adapter_sha256')}
    if report['qa']['tuned'].get('adapter_sha256'):
        adapters.add(report['qa']['tuned']['adapter_sha256'])
    if len(adapters) > 1:
        raise ValueError('Tuned Stage 1 and QA used different adapters')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    def flatten(value, prefix=''):
        result = {}
        for name, item in value.items():
            path = prefix + name
            if isinstance(item, dict):
                result.update(flatten(item, path + '.'))
            elif isinstance(item, (int, float)):
                result[path] = item
        return result
    with (args.output_dir / 'stage1_comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task', 'metric', 'held_out_transfer', *results])
        for task in TASK_SETS['extended']:
            values = {name: flatten(result['metrics']['tasks'][task]) if result['metrics'] else {}
                      for name, result in results.items()}
            for metric in sorted({key for value in values.values() for key in value}):
                writer.writerow([task, metric, task not in PAPER,
                                 *(values[name].get(metric, '') for name in results)])
    with (args.output_dir / 'paper_reference.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task', 'original_percent', 'after_printed_percent', 'after_request_percent'])
        for task, values in PAPER.items():
            writer.writerow([task, *values, 19.4 if task == 'edge_number' else values[1]])
    print(json.dumps({name: data['status'] for name, data in results.items()}, indent=2))


if __name__ == '__main__':
    main()
