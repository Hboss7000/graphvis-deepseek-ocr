#!/usr/bin/env python3
"""Official zero-shot tables: Stage 1 rescored with frozen audit_v3, OBQA recomputed.

CPU only, no inference. Reads canonical predictions, writes only to OUT.
Stage 1 uses the unmodified audit_v3 score_record; the cross-check compares it
with the separately computed outputs/parser_v3_rescore_2026-10-10 CSVs.
"""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
QA_DIR = ROOT/'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'
sys.path[:0] = [str(PARSER_DIR), str(QA_DIR)]
import score_stage1_audit_v2 as v2
import score_stage1_audit_v3 as v3
from prompt_common import parse_answer

base = v2.base
OUT = ROOT/'outputs/zero_shot_official_v3_2026-10-10'
AUDIT = ROOT/'outputs/parser_v3_rescore_2026-10-10'
RESULTS = ROOT/'outputs/fullrun_2026-10-04_B_results'
DATA = ROOT/'outputs/fullrun_2026-10-04_B/test/stage1_subset100.jsonl'
META = DATA.parent/'graph_metadata_0_500.jsonl'
QA_DATA = DATA.parent/'stage2_obqa_0_500.jsonl'
V3_SHA = '71e3294af69a99adc3bb6352e5051131d322c5c8d52ad9e9ba56016024081d96'
MODELS = ('qwen', 'gemma', 'llava', 'deepseek')
PREFIX = {'gemma': 'gemma3'}
LIST_TASKS = ('node_description', 'triple_listing', 'neighbor_listing')
TASKS = ('node_description', 'node_number', 'edge_number', 'triple_listing', 'highest_node_degree',
         'node_degree', 'relation_identification', 'neighbor_listing', 'shortest_path_listing')
QA_CONDITIONS = ('image', 'text', 'text_noref', 'kg_text')
# Qwen base on 500 white images: the only full zero-shot control-image run.
QA_CONTROLS = {('qwen', 'image_blank'): (
    'outputs/qwen_qa_analysis_2026-10-08/task4/fetched/eval_base_blank/qa_image_blank/predictions.jsonl',
    'outputs/qwen_qa_analysis_2026-10-08/task4/fetched/eval_base_blank/evaluation_report.json')}
Z = 1.959963984540054


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def rel(path):
    return str(Path(path).relative_to(ROOT))


def wilson(k, n):
    p = k/n; d = 1+Z*Z/n
    c = (p+Z*Z/(2*n))/d; h = Z*math.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def secondary_f1(task, row):
    """Lowercase + whitespace-collapse only (the documented v3-rescore secondary).

    score_stage1's 'basic' variant also strips trailing .!?;: and is NOT used here.
    Malformed-triple penalty and the committed-empty rule match the primary metric.
    """
    norm = lambda x: ' '.join(x.strip().lower().split())
    claim = row['extracted_claim'] or []
    if task == 'triple_listing':
        gold = {tuple(map(norm, t)) for t in row['parsed_gold_triples']}; pred = {tuple(map(norm, t)) for t in claim}
    else:
        gold = {norm(x) for x in row['parsed_gold_nodes']}; pred = {norm(x) for x in claim}
    c = base.set_counts(gold, pred)
    tp, fn = c['true_positive'], c['false_negative']
    fp = c['false_positive'] + (row.get('malformed_triple_line_count', 0) if task == 'triple_listing' else 0)
    if row['extraction_decision'] == 'COMMITTED' and not (tp or fp or fn):
        return 1.0
    precision = tp/(tp+fp) if tp+fp else 0.0
    return 2*precision*c['recall']/(precision+c['recall']) if precision+c['recall'] else 0.0


def install_impossible_path_match_guard():
    """Same operational shortcut as scripts/rescore_stage1_audit_v3.py (see its proof).

    A string ending in -> or → cannot fullmatch the frozen path pattern; return the
    non-match immediately instead of backtracking. Parser source is unchanged.
    """
    target = r'\s*(?:\d+[.)]\s*)?(?:' + v2.Q + r'|[\w [\],.\'-]+)(?:\s*(?:->|→)\s*(?:' + v2.Q + r'|[\w [\],.\'-]+))+\s*'
    original = re.fullmatch
    def guarded(pattern, string, flags=0):
        if pattern == target and isinstance(string, str) and string.endswith(('->', '→')):
            return None
        return original(pattern, string, flags)
    re.fullmatch = guarded


def main():
    OUT.mkdir(parents=False, exist_ok=False)
    assert sha(PARSER_DIR/'score_stage1_audit_v3.py') == V3_SHA
    freeze = json.loads((AUDIT/'parser_freeze.json').read_text())
    # The parser and every module it imports must be byte-identical to the freeze.
    # stage1_scoring.py (the dispatcher) is excluded: v3 does not import it and its
    # only change since the freeze is DEFAULT_EXTRACTOR = 'audit_v3'.
    for name, digest in freeze['dependency_sha256'].items():
        if not name.endswith('/stage1_scoring.py'):
            assert sha(ROOT/name) == digest, name
    install_impossible_path_match_guard()
    records, metadata = base.validate_inputs(read(DATA), read(META), base.EXTENDED_TASK_TYPES)
    inputs = {rel(p): sha(p) for p in (DATA, META, QA_DATA)}
    listing = []

    # ---- Stage 1 inputs: the canonical base cohorts used by the v3 rescore.
    cohorts = {(c['model'], c['arm']): c for c in json.loads((AUDIT/'cohorts.json').read_text())['cohorts']}
    audit_hashes = json.loads((AUDIT/'input_hashes_before.json').read_text())
    stage1_files = {}
    for m in MODELS:
        d = ROOT/cohorts[(m, 'base')]['directory']
        assert d == RESULTS/m/'stage1'
        cfg = json.loads((d/'run_config.json').read_text())
        assert cfg['input_jsonl']['sha256'] == sha(DATA) and cfg['graph_metadata']['sha256'] == sha(META)
        for t in TASKS:
            p = d/f'predictions_{PREFIX.get(m, m)}_{t}.jsonl'
            rows = read(p)
            assert audit_hashes[rel(p)] == sha(p), f'{p} differs from the v3 rescore input'
            n_unique = len({int(r['statement_idx']) for r in rows})
            listing.append(dict(kind='stage1', model=m, cell=t, path=rel(p), sha256=sha(p), records=len(rows), unique=n_unique))
            stage1_files[(m, t)] = (p, rows)

    # ---- QA inputs: every full zero-shot condition.
    qa_files = {}
    for m in MODELS:
        for c in QA_CONDITIONS:
            p = RESULTS/m/f'qa_{c}'/'predictions.jsonl'
            if p.is_file(): qa_files[(m, c)] = p
    for key, (p, _) in QA_CONTROLS.items():
        qa_files[key] = ROOT/p
    for (m, c), p in qa_files.items():
        rows = read(p)
        listing.append(dict(kind='qa', model=m, cell=c, path=rel(p), sha256=sha(p), records=len(rows),
                            unique=len({int(r['statement_idx']) for r in rows})))

    print(f"{'kind':6} {'model':9} {'cell':24} {'n':>4} {'uniq':>4} sha256[:16]  path")
    for r in listing:
        print(f"{r['kind']:6} {r['model']:9} {r['cell']:24} {r['records']:4} {r['unique']:4} {r['sha256'][:16]} {r['path']}")
    bad = [r for r in listing if r['records'] != r['unique'] or r['records'] != (100 if r['kind'] == 'stage1' else 500)]
    if bad:
        sys.exit(f'STOP: incomplete cohorts: {bad}')

    # ---- Stage 1 rescore with the unmodified audit_v3 scorer.
    ref = {(r['model'], r['task']): r for r in csv.DictReader((AUDIT/'per_task_v2_vs_v3.csv').open()) if r['arm'] == 'base'}
    ref2 = {(r['model'], r['task']): r for r in csv.DictReader((AUDIT/'per_task_v2_vs_v3_secondary_lists.csv').open()) if r['arm'] == 'base'}
    stage1, secondary, scored, mismatches = [], [], {}, []
    for (m, t), (p, preds) in stage1_files.items():
        rows = []
        for pred in preds:
            idx = int(pred['statement_idx']); record = records[(idx, t)]
            assert pred['task_type'] == t and pred['prompt'] == record['prompt']
            row = v3.score_record(record, pred.get('raw_response', ''), metadata[idx])
            row['statement_idx'] = idx; rows.append(row)
        scored[(m, t)] = rows
        agg = v3.aggregate_task(t, rows)
        if t in LIST_TASKS:
            metric = 'macro_F1'
            score = statistics.mean(r['set_metrics']['raw']['f1'] for r in rows)
            sec = statistics.mean(secondary_f1(t, r) for r in rows)
            ci = ('', '')
            if abs(sec-float(ref2[(m, t)]['v3_secondary_basic_score'])) > 1e-12:
                mismatches.append(('secondary', m, t, sec, ref2[(m, t)]['v3_secondary_basic_score']))
            secondary.append(dict(model=m, task=t, metric='macro_F1_lowercase_whitespace', n=len(rows), score=sec,
                                  strict_score=score, v2_score=float(ref2[(m, t)]['v2_secondary_basic_score']),
                                  delta_pp=100*(sec-float(ref2[(m, t)]['v2_secondary_basic_score']))))
        else:
            metric = 'accuracy_name_and_degree' if t == 'highest_node_degree' else 'accuracy'
            k = sum(bool(r['is_correct']) for r in rows); score = k/len(rows); ci = wilson(k, len(rows))
            assert abs(score-agg['strict_accuracy']) < 1e-12
        v2s = float(ref[(m, t)]['v2_score'])
        if abs(score-float(ref[(m, t)]['v3_score'])) > 1e-12:
            mismatches.append(('primary', m, t, score, ref[(m, t)]['v3_score']))
        stage1.append(dict(model=m, task=t, metric=metric, n=len(rows), score=score, ci_low=ci[0], ci_high=ci[1],
                           v2_score=v2s, delta_pp=100*(score-v2s)))
    if mismatches:
        sys.exit(f'STOP: v3 scores differ from per_task_v2_vs_v3.csv: {mismatches}')
    print(f'Cross-check: {len(stage1)} primary and {len(secondary)} secondary cells equal the v3 rescore CSVs exactly.')

    # ---- Majority-answer baselines (uses the test gold distribution).
    def gold_of(t, r):
        if t == 'highest_node_degree': return r['gold_degree']
        if t == 'relation_identification': return base.relation_key(r['gold_relation'])
        return r['gold_integer']
    baselines = []
    for t, label in (('node_number', 'node count'), ('edge_number', 'edge count'), ('node_degree', 'node degree'),
                     ('highest_node_degree', 'highest degree (degree only)'), ('relation_identification', 'relation')):
        golds = [gold_of(t, r) for r in scored[('qwen', t)]]
        for m in MODELS[1:]:  # same gold on every cohort
            assert [gold_of(t, r) for r in scored[(m, t)]] == golds
        value, k = Counter(golds).most_common(1)[0]
        lo, hi = wilson(k, len(golds))
        baselines.append(dict(task=t, description=label, baseline='majority-answer baseline (uses test gold distribution)',
                              majority_value=value, n=len(golds), accuracy=k/len(golds), ci_low=lo, ci_high=hi,
                              n_distinct_gold_values=len(set(golds))))
    # Degree-only accuracy for the highest-degree baseline comparison (diagnostic).
    hd_degree_only = {m: sum(r['degree_correct'] for r in scored[(m, 'highest_node_degree')])/100 for m in MODELS}

    llava_deg = scored[('llava', 'node_degree')]
    llava_three = sum(r['extracted_claim'] == 3 for r in llava_deg)
    llava_committed = sum(r['extraction_decision'] == 'COMMITTED' for r in llava_deg)
    gold_three = sum(r['gold_integer'] == 3 for r in llava_deg)

    # LLaVA relation statement 446 judgment call.
    r446 = next(r for r in scored[('llava', 'relation_identification')] if r['statement_idx'] == 446)
    assert r446['is_correct'] and r446['parser_rule'] == 'candidate_explicit_taxonomy'

    # ---- QA: recompute with parse_answer from raw responses; compare with stored values.
    qa_out, qa_checks = [], []
    stored_summary = {m: json.loads((ROOT/f'outputs/fullrun_consolidate_{m}_1/{m}/summary.json').read_text())
                      ['versions']['strict_64']['conditions'] for m in MODELS}
    qa_gold = {int(r['statement_idx']): r for r in read(QA_DATA)}
    for (m, c), p in qa_files.items():
        rows = read(p); k = 0; row_mismatch = 0; unparsed = 0
        for r in rows:
            assert r['gold_option'] == qa_gold[int(r['statement_idx'])]['answer']
            opt, tier = parse_answer(r['raw_response'], 4)  # as in the runners and fullrun_consolidate.py
            correct = opt == r['gold_option']
            k += correct; unparsed += opt == 'FAILED'
            stored_opt = 'FAILED' if r['predicted_option'] is None else r['predicted_option']  # runners store None
            row_mismatch += (opt != stored_opt) or (tier != r['parse_tier']) or (correct != r['is_correct'])
        stored_rows = sum(bool(r['is_correct']) for r in rows)
        if (m, c) in QA_CONTROLS:
            stored = json.loads((ROOT/QA_CONTROLS[(m, c)][1]).read_text())['accuracy']; stored_src = QA_CONTROLS[(m, c)][1]
        else:
            stored = stored_summary[m][c]['accuracy']; stored_src = f'outputs/fullrun_consolidate_{m}_1/{m}/summary.json'
        ok = row_mismatch == 0 and k == stored_rows and abs(k/len(rows)-stored) < 1e-12
        qa_checks.append(dict(model=m, condition=c, recomputed_correct=k, stored_row_correct=stored_rows, stored_accuracy=stored,
                              stored_source=stored_src, row_level_mismatches=row_mismatch, unparsed=unparsed, match=ok))
        lo, hi = wilson(k, len(rows))
        qa_out.append(dict(model=m, condition=c, n=len(rows), accuracy=k/len(rows), ci_low=lo, ci_high=hi))
    if not all(x['match'] for x in qa_checks):
        print('QA MISMATCH:', [x for x in qa_checks if not x['match']])

    write_csv('stage1_v3.csv', stage1)
    write_csv('stage1_v3_secondary_lists.csv', secondary)
    write_csv('qa_zero_shot.csv', qa_out)
    write_csv('baselines.csv', baselines)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', str(PARSER_DIR), str(QA_DIR)], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    prov = dict(
        generated_by=rel(Path(__file__)), generator_sha256=sha(__file__),
        git_commit=head, scorer_files_uncommitted_at_run=dirty,
        scorer=dict(path=rel(PARSER_DIR/'score_stage1_audit_v3.py'), sha256=V3_SHA, frozen_dependencies=freeze['dependency_sha256'],
                    dispatcher_sha256_now=sha(PARSER_DIR/'stage1_scoring.py'),
                    dispatcher_note='stage1_scoring.py differs from the freeze only by DEFAULT_EXTRACTOR=audit_v3; not imported by the v3 parser.'),
        qa_parser=dict(path=rel(QA_DIR/'prompt_common.py'), function='parse_answer', sha256=sha(QA_DIR/'prompt_common.py')),
        datasets=inputs, prediction_files=listing,
        cross_check=dict(reference=rel(AUDIT/'per_task_v2_vs_v3.csv'), reference_sha256=sha(AUDIT/'per_task_v2_vs_v3.csv'),
                         secondary_reference_sha256=sha(AUDIT/'per_task_v2_vs_v3_secondary_lists.csv'),
                         cells=len(stage1), secondary_cells=len(secondary), exact_match=True),
        qa_checks=qa_checks,
        regex_guard='Same no-op path-regex shortcut as scripts/rescore_stage1_audit_v3.py (proof and equivalence checks in outputs/parser_v3_rescore_2026-10-10/regex_guard_checks.json).',
        v2_scores_source='v2_score copied from the v3 rescore CSVs (computed there from the same raw responses).',
        highest_degree_degree_only_accuracy=hd_degree_only,
        llava_node_degree=dict(answers_3=llava_three, committed=llava_committed, n=len(llava_deg), gold_3=gold_three),
        llava_relation_446=dict(raw_response=r446['answer_span'], parser_rule=r446['parser_rule'], counted_correct=True,
                                effect_pp=1.0),
        wilson_z=Z)
    (OUT/'provenance.json').write_text(json.dumps(prov, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(baselines=baselines, hd_degree_only=hd_degree_only, llava_three=(llava_three, llava_committed, gold_three),
                          qa_checks_ok=all(x['match'] for x in qa_checks)), indent=1, default=str))


if __name__ == '__main__':
    main()
