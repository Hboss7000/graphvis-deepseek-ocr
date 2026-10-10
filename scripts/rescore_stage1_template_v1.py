#!/usr/bin/env python3
"""CPU-only paired Stage 1 template_v1 rescore into a strictly new directory."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0, str(SCORER_DIR))
import score_stage1_template_v1 as scorer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n'


def csv_text(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def primary(task, metrics):
    if task in {'node_description', 'triple_listing'}:
        return 'macro_f1', metrics['strict']['raw']['macro_f1']
    if task == 'neighbor_listing':
        return 'macro_f1', metrics['strict']['raw']['mean_f1']
    return ('strict_degree_accuracy' if task == 'highest_node_degree' else 'strict_accuracy', metrics['strict_accuracy'])


def table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' +
            ''.join('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', '<br>') for v in row) + ' |\n' for row in rows))


def collect(args):
    folders = {'main': args.main, 'tuned': args.tuned}
    previous = json.loads(args.previous_comparison.read_text())
    inputs = [args.input_jsonl, args.graph_metadata, args.previous_comparison, args.tuned_evaluation_report,
              Path(scorer.base.__file__), Path(scorer.__file__), Path(__file__).resolve(),
              ROOT / 'scripts/fullrun_consolidate.py', ROOT / 'scripts/fullrun_rescue.py']
    for folder in folders.values():
        inputs.extend(sorted(p for p in folder.iterdir() if p.is_file()))
    # Protect all earlier comparison/headline/diagnostic evidence as well.
    inputs.extend(sorted(p for p in args.previous_comparison.parent.rglob('*') if p.is_file()))
    rescue_dir = ROOT / 'outputs/fullrun_consolidate_qwen_1/qwen/extended_512'
    rescue_files = sorted(rescue_dir.glob('*.jsonl'))
    inputs.extend(rescue_files)
    hashes = {str(p.resolve()): sha(p) for p in inputs}
    before, after, old_rows, new_rows, configs = {}, {}, {}, {}, {}
    for arm, folder in folders.items():
        options = SimpleNamespace(input_jsonl=args.input_jsonl, graph_metadata=args.graph_metadata,
            predictions_dir=folder, model_name='qwen', task_set='extended', extractor='span_extended')
        before[arm], old_rows[arm] = scorer.base.score_files(options)
        if before[arm] != previous['stage1'][arm]['metrics']:
            raise ValueError(f'Original comparison cannot be reproduced: {arm}')
        options.extractor = scorer.VERSION
        after[arm], new_rows[arm] = scorer.score_files(options)
        configs[arm] = json.loads((folder / 'run_config.json').read_text())
        if configs[arm]['input_jsonl']['sha256'] != sha(args.input_jsonl) or configs[arm]['graph_metadata']['sha256'] != sha(args.graph_metadata):
            raise ValueError(f'Recorded input hashes differ: {arm}')
        if configs[arm]['extractor'] != 'span_extended':
            raise ValueError(f'Unexpected original extractor: {arm}')
        if sha(folder / 'run_config.json') != previous['stage1'][arm]['run_config_sha256']:
            raise ValueError(f'Original run config changed: {arm}')
    sets = {arm: {task: {int(r['statement_idx']) for r in rows} for task, rows in tasks.items()}
            for arm, tasks in new_rows.items()}
    cohort = sets['main']['node_description']
    if len(cohort) != 100 or sets['main'] != sets['tuned'] or any(ids != cohort for ids in sets['main'].values()):
        raise ValueError('Expected identical 100 graphs in each of the nine tasks')
    if any(len(rows) != 100 for tasks in new_rows.values() for rows in tasks.values()):
        raise ValueError('Missing/duplicate records')
    base_sha = sha(Path(scorer.base.__file__))
    scorer_rel = str(Path(scorer.base.__file__).relative_to(ROOT))
    evaluation = json.loads(args.tuned_evaluation_report.read_text())
    if configs['main']['fullrun']['code_files'][scorer_rel] != base_sha or evaluation['code_sha256'][scorer_rel] != base_sha:
        raise ValueError('Original scorer revisions differ')
    if configs['main']['prompt_bodies_sha256'] != configs['tuned']['prompt_bodies_sha256']:
        raise ValueError('Prompt-body hashes differ')
    provenance = {'statement_idx': sorted(cohort), 'same_100_statement_idx_in_all_nine_tasks': True,
        'same_900_unique_record_keys': True, 'original_comparison_reproduced_exactly': True,
        'original_extractor': 'span_extended', 'original_scorer_sha256': base_sha,
        'new_scorer_version': scorer.VERSION, 'new_scorer_sha256': sha(Path(scorer.__file__)),
        'same_new_parser_in_both_arms': True, 'input_sha256': sha(args.input_jsonl),
        'metadata_sha256': sha(args.graph_metadata), 'prompt_bodies_sha256': configs['main']['prompt_bodies_sha256'],
        'explicit_v1': 'QA parser label, not the Stage 1 extractor.',
        'source_directories': {arm: str(folder.resolve()) for arm, folder in folders.items()},
        'raw_generation_and_token_ceiling_flags_unchanged': True}
    headline, comparison = [], []
    for task in scorer.TASK_TYPES:
        bm, tm = after['main']['tasks'][task], after['tuned']['tasks'][task]
        metric, value = primary(task, bm)
        headline.append({'task': task, 'metric': metric, 'baseline': value, 'tuned': primary(task, tm)[1],
            'baseline_hit_token_ceiling_fraction': bm['hit_token_ceiling_fraction'],
            'tuned_hit_token_ceiling_fraction': tm['hit_token_ceiling_fraction'], 'n_baseline': 100, 'n_tuned': 100,
            'baseline_name_accuracy': bm.get('raw', {}).get('name_accuracy', ''),
            'tuned_name_accuracy': tm.get('raw', {}).get('name_accuracy', ''),
            'baseline_joint_accuracy': bm.get('strict_joint_accuracy', ''),
            'tuned_joint_accuracy': tm.get('strict_joint_accuracy', '')})
        comparison.append({'task': task, 'metric': metric, 'n_per_arm': 100,
            'baseline_before': primary(task, before['main']['tasks'][task])[1], 'baseline_after': value,
            'tuned_before': primary(task, before['tuned']['tasks'][task])[1], 'tuned_after': primary(task, tm)[1],
            **{f'{arm}_{version}_{measure}_accuracy': results[arm]['tasks'][task].get('raw', {}).get('name_accuracy', '')
                if measure == 'name' else results[arm]['tasks'][task].get('strict_joint_accuracy', '')
               for arm in ('main', 'tuned') for version, results in [('before', before), ('after', after)] for measure in ('name', 'joint')}})
    # Explain the original FN increase and also report the corrected counts.
    base_nodes = {r['statement_idx']: r for r in old_rows['main']['node_description']}
    capped = [r for r in old_rows['tuned']['node_description'] if r['hit_token_ceiling']]
    loops = []
    for row in capped:
        counts = Counter(row['parsed_predicted_nodes'])
        duplicate = 1 - len(counts) / len(row['parsed_predicted_nodes'])
        if duplicate <= .95:
            raise ValueError('Repetition classification no longer supported by stored response')
        loops.append({'statement_idx': row['statement_idx'], 'raw_response': row['raw_response'], 'gold': row['gold'],
            'classification': 'repetition loop', 'parsed_items': len(row['parsed_predicted_nodes']),
            'unique_items': len(counts), 'most_common_items': counts.most_common(5), 'duplicate_fraction': duplicate,
            'baseline_before_false_negative': base_nodes[row['statement_idx']]['set_metrics']['raw']['false_negative'],
            'tuned_before_false_negative': row['set_metrics']['raw']['false_negative']})
    loop_ids = {r['statement_idx'] for r in capped}
    node_counts = {version: {arm: {'full_false_negative': results[arm]['tasks']['node_description']['strict']['raw']['micro_false_negative'],
        'ceiling_six_false_negative': sum(r['set_metrics']['raw']['false_negative'] for r in rows[arm]['node_description'] if r['statement_idx'] in loop_ids)}
        for arm in folders} for version, results, rows in [('before', before, old_rows), ('after', after, new_rows)]}
    sensitivity, sensitivity_details = [], {}
    for task in ('edge_number', 'triple_listing', 'highest_node_degree'):
        kept = {r['statement_idx'] for r in old_rows['main'][task] if not r['hit_token_ceiling']}
        sensitivity_details[task] = {'retained_statement_idx': sorted(kept), 'excluded_statement_idx': sorted(cohort-kept)}
        for version, results, rows in [('before', before, old_rows), ('after', after, new_rows)]:
            for arm in folders:
                subset = [r for r in rows[arm][task] if r['statement_idx'] in kept]
                matched = scorer.base.aggregate_task(task, subset, extractor='span_extended')
                sensitivity_details[task][f'{arm}_{version}'] = matched
                sensitivity.append({'task': task, 'arm': arm, 'scorer': 'span_extended' if version == 'before' else scorer.VERSION,
                    'metric': primary(task, matched)[0], 'full_n': 100, 'excluded_baseline_capped_n': 100-len(kept),
                    'matched_n': len(kept), 'full_score': primary(task, results[arm]['tasks'][task])[1],
                    'matched_uncapped_score': primary(task, matched)[1],
                    'matched_name_accuracy': matched.get('raw', {}).get('name_accuracy', ''),
                    'matched_joint_accuracy': matched.get('strict_joint_accuracy', '')})
    rescue = {'fetched_stage1_512_rescue_available': False,
        'available_qa_extended_512_rows': {str(p.resolve()): len(scorer.base.read_jsonl(p)) for p in rescue_files},
        'scope': 'Existing extended_512 rescue is QA 64→512, not Stage 1 1024-token evaluation.',
        'queued_execution_status': 'unknown; no pod contacted'}
    return before, after, old_rows, new_rows, provenance, headline, comparison, loops, node_counts, sensitivity, sensitivity_details, rescue, hashes


def rescore(args):
    # Refuse before reading evidence, and use mkdir/open exclusivity again when
    # writing so an existing result cannot be replaced accidentally.
    if args.output.exists():
        raise FileExistsError(f'Output must be a new directory: {args.output}')
    before, after, old_rows, new_rows, provenance, headline, comparison, loops, node_counts, sensitivity, sensitivity_details, rescue, hashes = collect(args)
    outputs = {}
    for arm in ('main', 'tuned'):
        outputs[f'{arm}/metrics_qwen_template_v1.json'] = json_text(after[arm])
        for task, rows in new_rows[arm].items():
            outputs[f'{arm}/scored_qwen_{task}.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows)
    outputs['stage1_headline.csv'] = csv_text(headline)
    headline_md = '# Qwen Stage 1 — template_v1\n\n' + table(
        ['Task', 'Metric', 'Baseline', 'Tuned', 'Baseline ceiling', 'Tuned ceiling', 'N per arm'],
        [[r['task'], r['metric'], f"{r['baseline']:.4f}", f"{r['tuned']:.4f}",
          f"{r['baseline_hit_token_ceiling_fraction']:.0%}", f"{r['tuned_hit_token_ceiling_fraction']:.0%}", 100] for r in headline])
    headline_md += '\nRaw normalization. Set F1 is macro F1. Highest-degree headline is degree-only; selected-name and joint accuracy are separately included in the CSV and report. Both arms use the same template_v1 parser and unchanged scoring formulas.\n'
    outputs['stage1_headline.md'] = headline_md
    outputs['stage1_before_after.csv'] = csv_text(comparison)
    outputs['comparison.json'] = json_text({'before_scorer': 'span_extended', 'after_scorer': scorer.VERSION,
        'before': before, 'after': after, 'provenance': provenance,
        'parser_rule_counts': {arm: {task: dict(Counter(r['parser_rule'] for r in rows)) for task, rows in tasks.items()} for arm,tasks in new_rows.items()}})
    outputs['provenance.json'] = json_text(provenance)
    outputs['node_description_ceiling_audit.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in loops)
    outputs['node_description_counts.json'] = json_text(node_counts)
    outputs['baseline_ceiling_sensitivity.csv'] = csv_text(sensitivity)
    outputs['baseline_ceiling_sensitivity.json'] = json_text(sensitivity_details)
    outputs['rescue_status.json'] = json_text(rescue)
    report = '# Qwen Stage 1 template_v1 — findings\n\n'
    report += '## 1. Headline and before/after — scorer issue corrected\n\n'
    report += 'Nine tasks; N=100 per arm. `stage1_headline.csv` / `.md` contain corrected scores and unchanged token-ceiling fractions. `stage1_before_after.csv` contains all per-task primary before/after scores, with additional highest-degree name/joint columns.\n\n'
    report += table(['Task','Metric','Baseline before → after','Tuned before → after'],
        [[r['task'],r['metric'],f"{r['baseline_before']:.4f} → {r['baseline_after']:.4f}",f"{r['tuned_before']:.4f} → {r['tuned_after']:.4f}"] for r in comparison])
    report += '\nParser changes: consume explicit task-template syntax before fallback extraction; pair highest-degree name/value; split only the neighbor payload; retain literal node-list/tuple names including `being`, `but`, `made of`; quoted tuple/list names are syntax. Explicit relation assertions take precedence over explanatory relation mentions. No fuzzy matching, no gold-member lookup for parsing, no changed scoring formulas or normalization tiers. Ordered paths, tie handling and malformed-triple penalties are preserved.\n\n'
    report += '## 2. Highest degree — scorer issue fixed\n\n'
    report += table(['Arm','Selected-name accuracy before → after','Joint accuracy before → after','Degree-only accuracy before → after'],
        [[arm, f"{before[arm]['tasks']['highest_node_degree']['raw']['name_accuracy']:.2f} → {after[arm]['tasks']['highest_node_degree']['raw']['name_accuracy']:.2f}",
          f"{before[arm]['tasks']['highest_node_degree']['strict_joint_accuracy']:.2f} → {after[arm]['tasks']['highest_node_degree']['strict_joint_accuracy']:.2f}",
          f"{before[arm]['tasks']['highest_node_degree']['strict_accuracy']:.2f} → {after[arm]['tasks']['highest_node_degree']['strict_accuracy']:.2f}"] for arm in ('main','tuned')])
    report += '\nThe right name merely appearing somewhere is never strict credit: the selected explicit answer must name a metadata-valid maximum-degree node. Complete rescored records: `main/scored_qwen_highest_node_degree.jsonl`, `tuned/scored_qwen_highest_node_degree.jsonl`.\n\n'
    report += '## 3. Node description — model behavior: six repetition loops\n\n'
    report += 'Ceiling IDs: ' + ', '.join(str(r['statement_idx']) for r in loops) + '. All six are repetition loops, with >95% duplicated parsed entries, not unique long lists.\n\n'
    report += table(['Scorer','Baseline full FN','Tuned full FN','Baseline six FN','Tuned six FN','Share of paired FN increase explained by six'],
        [[version,node_counts[version]['main']['full_false_negative'],node_counts[version]['tuned']['full_false_negative'],
          node_counts[version]['main']['ceiling_six_false_negative'],node_counts[version]['tuned']['ceiling_six_false_negative'],
          f"{(node_counts[version]['tuned']['ceiling_six_false_negative']-node_counts[version]['main']['ceiling_six_false_negative'])/(node_counts[version]['tuned']['full_false_negative']-node_counts[version]['main']['full_false_negative']):.2%}"] for version in ('before','after')])
    report += '\nThe six explain 84/85 additional FN (98.82%) under both scorer versions. Original other-94 FN: 13→14. The revised parser removes five erroneous filtered-label FN from each arm; corrected other-94 FN: 8→9. Full verbatim looping responses and gold: `node_description_ceiling_audit.jsonl`.\n\n'
    report += '## 4. Neighbor listing — scorer issue fixed; omissions remain model behavior\n\n'
    report += table(['Arm','All-100 F1 before → after','Degree-1 F1 before → after (N=36)'],
        [[arm,f"{before[arm]['tasks']['neighbor_listing']['raw']['mean_f1']:.4f} → {after[arm]['tasks']['neighbor_listing']['raw']['mean_f1']:.4f}",
          f"{before[arm]['tasks']['neighbor_listing']['by_degree']['1']['raw']['mean_f1']:.4f} → {after[arm]['tasks']['neighbor_listing']['by_degree']['1']['raw']['mean_f1']:.4f}"] for arm in ('main','tuned')])
    omissions = [r['statement_idx'] for r in new_rows['tuned']['neighbor_listing'] if r['parser_rule']=='degree_only_no_neighbors']
    report += f'\n{len(omissions)} tuned replies only give a degree and assert no neighbor names, so the new parser returns an empty set. Actual extra/wrong assertions still count as false positives. All corrected per-record parsed sets: `tuned/scored_qwen_neighbor_listing.jsonl`; baseline uses the identical parser.\n\n'
    report += '## 5. Provenance — cohort/version mismatch ruled out\n\n'
    report += f"Same 100 statement_idx in all nine tasks, same 900 unique task/graph keys; original input/metadata/prompt-body hashes match. Recorded old scorer hashes in both runs equal `{provenance['original_scorer_sha256']}` (span_extended). New scorer SHA256 in both rescoring arms: `{provenance['new_scorer_sha256']}` ({scorer.VERSION}). `explicit_v1` belongs to QA, not Stage 1. Exact IDs/hashes: `provenance.json`.\n\n"
    report += '## 6. Token-ceiling sensitivity — causal comprehension gain unknown\n\n'
    report += 'Exclude baseline-capped IDs and score both arms on the identical retained subset. Scores below use template_v1; `baseline_ceiling_sensitivity.csv` also contains span_extended before values.\n\n'
    sr=[]
    for task in ('edge_number','triple_listing','highest_node_degree'):
        r=[x for x in sensitivity if x['task']==task and x['scorer']==scorer.VERSION]
        sr.append([task,r[0]['metric'],r[0]['matched_n'],f"{r[0]['full_score']:.4f} → {r[1]['full_score']:.4f}",f"{r[0]['matched_uncapped_score']:.4f} → {r[1]['matched_uncapped_score']:.4f}"])
    report += table(['Task','Metric','Retained N','Full baseline → tuned','Uncapped baseline → tuned'],sr)
    report += '\nEdge-count margin shrinks 37→5.13 percentage points after excluding capped baseline cases. Triple F1 and highest-degree degree-only improvements persist. Exclusion is nonrandom and cannot isolate concise formatting from comprehension causally. No fetched Stage 1 512-token rescue is available: existing extended_512 rescue targets QA 64→512 and the four local Qwen JSONL files are empty. Queued execution status remains unknown; no pod contacted.\n'
    outputs['report.md'] = report
    if hashes != {p: sha(Path(p)) for p in hashes}:
        raise RuntimeError('Input evidence changed during analysis')
    manifest = {'provenance': provenance, 'input_sha256_before_and_after': hashes,
        'original_evidence_unchanged': True, 'new_output_sha256': {p:hashlib.sha256(content.encode()).hexdigest() for p,content in outputs.items()}}
    outputs['rescore_manifest.json'] = json_text(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    for relative, content in outputs.items():
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            handle.write(content)
    print(f'Created {len(outputs)} new files in {args.output}; original evidence unchanged.')
    print(table(['Task','Metric','Baseline before → after','Tuned before → after'],
        [[r['task'],r['metric'],f"{r['baseline_before']:.4f} → {r['baseline_after']:.4f}",f"{r['tuned_before']:.4f} → {r['tuned_after']:.4f}"] for r in comparison]))
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    data = ROOT / 'outputs/fullrun_2026-10-04_B/test'
    p.add_argument('--input-jsonl', type=Path, default=data/'stage1_subset100.jsonl')
    p.add_argument('--graph-metadata', type=Path, default=data/'graph_metadata_0_500.jsonl')
    p.add_argument('--main', type=Path, default=ROOT/'outputs/fullrun_2026-10-04_B_results/qwen/stage1')
    p.add_argument('--tuned', type=Path, default=ROOT/'outputs/qwen_stage1_fetched/eval_trained_stage1/stage1')
    p.add_argument('--previous-comparison', type=Path, default=ROOT/'outputs/qwen_stage1_comparison/comparison.json')
    p.add_argument('--tuned-evaluation-report', type=Path, default=ROOT/'outputs/qwen_stage1_fetched/eval_trained_stage1/evaluation_report.json')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


if __name__ == '__main__':
    rescore(parse_args())
