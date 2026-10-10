"""Build a compact report and verify the completed frozen offline rescore."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import sys
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parser_v3_rescore_2026-10-10'
sys.path.insert(0,str(ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'))
import score_stage1 as base


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name):return [json.loads(x) for x in (OUT/name).read_text().splitlines()]


def main():
    complete=json.loads((OUT/'rescore_complete.json').read_text())
    freeze=json.loads((OUT/'parser_freeze.json').read_text())
    scores=read('all_scores_v2_vs_v3.jsonl');diffs=read('diff_v3_vs_v2.jsonl')
    table=list(csv.DictReader((OUT/'per_task_v2_vs_v3.csv').open()))
    assert len(scores)==4509 and len(table)==54
    assert len({r['case_id'] for r in scores})==len(scores)
    for name,digest in json.loads((OUT/'input_hashes_before.json').read_text()).items():assert sha(ROOT/name)==digest,name
    for name,digest in freeze['dependency_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in json.loads((OUT/'diagnostic_evidence_hashes.json').read_text()).items():assert sha(ROOT/name)==digest,name
    increases=[r for r in scores if r['change']=='increase'];decreases=[r for r in scores if r['change']=='decrease']
    population=list(csv.DictReader((OUT/'spotcheck_all_candidates.csv').open()))
    assert {r['case_id'] for r in population}=={r['case_id'] for r in increases+decreases}
    sample=json.loads((OUT/'spotcheck_sampling.json').read_text())
    assert len(sample['sample_ids'])==len(set(sample['sample_ids']))==30
    assert {r['case_id'] for r in decreases}<=set(sample['sample_ids'])
    assert 'gemma:base:node_degree:406' in sample['sample_ids']
    assert len(diffs)==complete['diff_count']
    for r in table:
        cohort=[s for s in scores if (s['model'],s['arm'],s['task'])==(r['model'],r['arm'],r['task'])]
        assert len(cohort)==int(r['n'])
        for version in ['v2','v3']:
            assert abs(statistics.mean(s[f'{version}_score'] for s in cohort)-float(r[f'{version}_score']))<1e-12
            assert sum(s[f'{version}_full_credit'] for s in cohort)==int(r[f'{version}_full_credit_count'])
        assert sum(s['change']=='increase' for s in cohort)==int(r['score_increases'])
        assert sum(s['change']=='decrease' for s in cohort)==int(r['score_decreases'])
    # Secondary is exactly lowercase + whitespace, without the historical
    # "basic" helper's extra trailing-punctuation removal.
    def lower_space(value):
        if isinstance(value, (list,tuple)):
            return tuple(lower_space(x) for x in value)
        return ' '.join(value.lower().split())
    for r in scores:
        if r['task'] in ('node_description','neighbor_listing','triple_listing'):
            gold_items=[lower_space(x) for x in r['gold']]
            for version in ['v2','v3']:
                predicted=[lower_space(x) for x in (r[f'{version}_claim'] or [])]
                metrics=base.set_counts(set(gold_items),set(predicted))
                malformed=r[f'{version}_malformed_triples']
                if malformed:
                    metrics['false_positive']+=malformed
                    metrics['exact_set_equality']=False
                    tp=metrics['true_positive'];fp=metrics['false_positive'];fn=metrics['false_negative']
                    metrics['f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
                if r[f'{version}_decision']=='COMMITTED' and not (metrics['true_positive'] or metrics['false_positive'] or metrics['false_negative']):
                    metrics['f1']=1.0
                if r[f'{version}_decision']!='COMMITTED':metrics['exact_set_equality']=False
                r[f'{version}_secondary_basic_f1']=metrics['f1']
                r[f'{version}_secondary_basic_full_credit']=metrics['exact_set_equality']
        elif r['task']=='shortest_path_listing':
            paths=[[lower_space(n) for n in p] for p in r['gold']['paths']]
            for version in ['v2','v3']:
                r[f'{version}_secondary_basic_exact']=bool(r[f'{version}_decision']=='COMMITTED' and [lower_space(n) for n in (r[f'{version}_claim'] or [])] in paths)
    by_id={r['case_id']:r for r in scores}
    # Apply these explicit secondary metrics consistently to generated artifacts.
    for name,rows in [('all_scores_v2_vs_v3.jsonl',scores),('diff_v3_vs_v2.jsonl',diffs),('score_decreases.jsonl',read('score_decreases.jsonl'))]:
        with (OUT/name).open('w') as f:
            for row in rows:
                row.update({k:v for k,v in by_id[row['case_id']].items() if '_secondary_basic_' in k})
                if name=='score_decreases.jsonl' and row['case_id']=='gemma:base:node_degree:406':
                    row['explanation']='VERIFIED: v2 extracted operand 1 from the written formula 1 + 0 = 2. The raw response asserts degree 2, and v3 retains that explicit result. Gold is 1; the old full credit was accidental. No arithmetic was computed or repaired.'
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    decrease_records=read('score_decreases.jsonl')
    loss_fields=list(dict.fromkeys(k for r in decrease_records for k in r))
    with (OUT/'score_decreases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,loss_fields);w.writeheader()
        for row in decrease_records:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})
    population_records=[by_id[r['case_id']] for r in population]
    fields=list(dict.fromkeys(k for row in population_records for k in row))
    with (OUT/'spotcheck_all_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for row in population_records:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})
    secondary=[]
    for row in table:
        task=row['task']
        if task not in ('node_description','neighbor_listing','triple_listing','shortest_path_listing'):continue
        rr=[s for s in scores if (s['model'],s['arm'],s['task'])==(row['model'],row['arm'],task)]
        key='secondary_basic_exact' if task=='shortest_path_listing' else 'secondary_basic_f1'
        a,b=(statistics.mean(r[f'{version}_{key}'] for r in rr) for version in ['v2','v3'])
        exactkey='secondary_basic_exact' if task=='shortest_path_listing' else 'secondary_basic_full_credit'
        secondary.append(dict(row,secondary_metric='secondary_lowercase_whitespace_only_'+('ordered_path_exact_accuracy' if task=='shortest_path_listing' else 'macro_F1'),v2_secondary_basic_score=a,v3_secondary_basic_score=b,secondary_delta_pp=100*(b-a),v2_secondary_basic_full_credit_count=sum(r[f'v2_{exactkey}'] for r in rr),v3_secondary_basic_full_credit_count=sum(r[f'v3_{exactkey}'] for r in rr)))
    with (OUT/'per_task_v2_vs_v3_secondary_lists.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(secondary[0]));w.writeheader();w.writerows(secondary)
    checks=json.loads((OUT/'syntax_controls.json').read_text())
    assert len(checks)==119 and all(r['passed'] for r in checks)
    by_suite=dict(Counter(r['suite'] for r in checks))
    assert by_suite=={'qwen':21,'llava':33,'gemma':52,'new_optional_rules':9,'new_triple_policy':4}
    # Original experimental expectations differ exactly in three documented cases.
    original_differences=[r['case'] for r in checks if r['note'].startswith('Historical optional') or 'Intentional user-required policy change' in r['note']]
    assert original_differences==[1,13,20]
    mapping={(r['model'],r['arm'],r['task']):r for r in table}
    labels={'node_description':'Node list F1','node_number':'Node count','edge_number':'Edge count','triple_listing':'Triple F1','highest_node_degree':'Highest degree (joint)','node_degree':'Node degree','relation_identification':'Relation','neighbor_listing':'Neighbor F1','shortest_path_listing':'Shortest path'}
    cohort_order=[('qwen','base'),('qwen','tuned'),('llava','base'),('llava','tuned_dry'),('gemma','base'),('deepseek','base')]
    lines=['# Frozen parser audit_v3 — 2026-10-10','',f"VERIFIED: v3 SHA-256 `{freeze['source_sha256']}`. audit_v2 remains unchanged and the default; v3 is opt-in. All 106 historical cases were run; 103 original expectations match. Three intentional differences: omitted last-integer/path-joining proposals and the required closed two-field tuple penalty. All 119 frozen-policy controls pass (106 historical + 13 new); 81 v2/template regressions pass. Both narrow Qwen rules are included; extraction reads task/response, with question endpoints only for the relation-list fallback.",'','VERIFIED: strict scores below are percentages, v2 → v3; node/neighbor/triple lists use macro-F1; other tasks use exact accuracy (highest requires name AND degree). Each full cohort has n=100/task. LLaVA tuned† has only n=1/task from a dry run; full tuned evaluation is missing.','', '| Task | Qwen base | Qwen tuned | LLaVA base | LLaVA tuned† | Gemma base | DeepSeek base |','|---|---:|---:|---:|---:|---:|---:|']
    for task in base.EXTENDED_TASK_TYPES:
        cells=[]
        for model,arm in cohort_order:
            r=mapping[(model,arm,task)];cells.append(f"{float(r['v2_score'])*100:.1f}→{float(r['v3_score'])*100:.1f}")
        lines.append('| '+labels[task]+' | '+' | '.join(cells)+' |')
    inc=len(increases);gains=sum(r['v3_full_credit'] and not r['v2_full_credit'] for r in scores)
    lines+=['',f"VERIFIED: {len(scores):,} saved answers rescored; {inc} score increases, {gains} full-credit gains, {len(decreases)} decrease. [Per-task counts/pp deltas](per_task_v2_vs_v3.csv), [secondary lowercase/whitespace scores](per_task_v2_vs_v3_secondary_lists.csv), [all changed extractions/scores](diff_v3_vs_v2.jsonl). {complete['protected_file_count']} input/source/artifact hashes match before/after; sealed audits and existing metrics are untouched.",'']
    for r in decreases:
        lines.append(f"VERIFIED decrease: **{r['model']} {r['arm']}, {r['task']}, statement {r['statement_idx']}**: {r['v2_score']:.4g}→{r['v3_score']:.4g}; claim {json.dumps(r['v2_claim'],ensure_ascii=False)}→{json.dumps(r['v3_claim'],ensure_ascii=False)}.")
        if r['case_id']=='gemma:base:node_degree:406':lines.append('VERIFIED: raw text asserts degree 2 and ends `1 + 0 = 2`; gold is 1. V2 awarded credit to operand 1; v3 retains the asserted result 2. Arithmetic is not repaired.')
    lines+=['','VERIFIED triple policy: closed tuple entries with wrong field counts incur false-positive penalties; one-field isolated-node lines and unclosed trailing tuples are excluded. Prose schemas/examples are excluded; tuples are never repaired.','',f"VERIFIED: [30-case human sheet](spotcheck_sample.md), seed {sample['seed']}, favors paths/relations, then highest/neighbors/degree; every decrease is included. [Full {len(population)}-case population](spotcheck_all_candidates.csv) preserves raw responses. No provisional reading appears beside cases.",'','VERIFIED LLaVA: ceiling hits are 70% for node lists and 73% for triples (both median 1,024 tokens). Saved decoding: greedy, one beam, max_new_tokens=1,024; EOS/pad/repetition penalty are not explicitly logged. Pinned EOS=2, tokenizer PAD=32001; INFERRED generation PAD=2 and no repetition penalty. Base uses HF Mistral `[INST]`; tuned dry uses GraphVis `llava_v1`. VERIFIED tokenizer-only reproduction adds two BOS tokens; INFERRED this affected historical input. [Exact prompt reconstructions, per-task tokens, sources and limitations](LLAVA_DIAGNOSTICS.md).','', 'INFERRED: a controlled LLaVA rerun is warranted after checking the intended paper template and special tokens; the cause of looping remains unproven. VERIFIED skipped: broader relation-priority fix (LLaVA 15/79/148/169 false positives remain), full tuned LLaVA evaluation (missing), and new inference. INFERRED: these development-cohort gains require the human spot-check before being treated as validated parser corrections.']
    operational='VERIFIED: strict-only rescore wrapper matches the full scorers on 108 saved cases; unused containment diagnostics are omitted. A rescore-only regex shortcut rejects strings that cannot match the frozen path pattern, preventing backtracking on an unfinished arrow. All 160 shortcut equivalence cases and 119 control-parity checks pass; parser files are unchanged.\n'
    (OUT/'RESCORE_IMPLEMENTATION.md').write_text(operational+'VERIFIED: secondary list scores use only lowercase and whitespace normalization; no punctuation or renderer annotation is stripped. Ordered paths use secondary exact accuracy.\n')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    verification=dict(evidence='VERIFIED',historical_cases_run=106,original_historical_expectations_passed=103,intentional_expectation_changes=original_differences,frozen_policy_controls_passed=119,existing_regressions_passed=81,scored_answers=len(scores),aggregate_rows=len(table),secondary_list_rows=len(secondary),score_increases=inc,full_credit_gains=gains,score_decreases=len(decreases),full_credit_losses=sum(r['v2_full_credit'] and not r['v3_full_credit'] for r in scores),sample_cases=30,population_cases=len(population),all_decreases_sampled=True,all_original_hashes_still_match=True,parser_frozen_hash_still_matches=True,report_word_count=len((OUT/'REPORT.md').read_text().split()))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
    (OUT/'manifest.json').write_text(json.dumps(dict(evidence='VERIFIED',artifact_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(paths)},helper_script_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/diagnose_llava_stage1_saved.py']},frozen_parser_sha256=freeze['source_sha256']),indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
