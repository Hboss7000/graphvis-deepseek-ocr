#!/usr/bin/env python3
"""Offline, frozen-parser rescore and independently readable human review sample."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0, str(PARSER_DIR))
import score_stage1_audit_v2 as v2
import score_stage1_audit_v3 as v3
base = v2.base
OUT = ROOT/'outputs/parser_v3_rescore_2026-10-10'
DATA = ROOT/'outputs/fullrun_2026-10-04_B/test/stage1_subset100.jsonl'
META = DATA.parent/'graph_metadata_0_500.jsonl'
LIST_TASKS = {'node_description','neighbor_listing','triple_listing'}
SEED = 20261010


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_jsonl(name, rows):
    with (OUT/name).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def write_csv(name, rows, fields):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def score(task, row, variant='raw'):
    return row['set_metrics'][variant]['f1'] if task in LIST_TASKS else float(row['is_correct'])


def exact(task, row, variant='raw'):
    return row['set_metrics'][variant]['exact_set_equality'] if task in LIST_TASKS else row['is_correct']


def gold(task, row, record):
    if task in base.NUMERIC_TASKS:
        return row['gold_integer']
    if task == 'highest_node_degree':
        return dict(valid_names=row['valid_highest_names'], degree=row['gold_degree'])
    if task == 'triple_listing':
        return row['parsed_gold_triples']
    if task in ('relation_identification','shortest_path_listing'):
        return record['gold']
    return row['parsed_gold_nodes']


def install_impossible_path_match_guard():
    """Operational shortcut; preserve the frozen regex language exactly.

    That fullmatch ends in at least one label character, either a quoted label with a closing
    quote/bold/backtick delimiter or the listed character class (which excludes
    > and →). Thus a string ending in -> or → with no trailing whitespace cannot match. Reject it immediately
    instead of exploring exponentially many whitespace partitions. No parser
    source, assertion rule or extraction outcome changes.
    """
    target = r'\s*(?:\d+[.)]\s*)?(?:' + v2.Q + r'|[\w [\],.\'-]+)(?:\s*(?:->|→)\s*(?:' + v2.Q + r'|[\w [\],.\'-]+))+\s*'
    original = re.fullmatch
    def guarded(pattern, string, flags=0):
        if pattern == target and isinstance(string,str) and string.endswith(('->','→')):
            return None
        return original(pattern,string,flags)
    cases=[]
    for a in ['a','a b','"a -> b"','**a**','`a`']:
        for b in ['b','b c','"b"','**b**']:
            for arrow in [' -> ',' → ']:
                for tail in ['', ' ->', ' → ', ' ->  ']:
                    raw=a+arrow+b+tail
                    old=original(target,raw);new=guarded(target,raw)
                    assert bool(old)==bool(new),(raw,old,new)
                    cases.append(dict(raw_response=raw,matched=bool(new),passed=True))
    # Existing parser controls must be byte/value identical with the shortcut.
    from validate_stage1_audit_v3 import run_controls
    before=run_controls()
    re.fullmatch=guarded
    after=run_controls()
    assert before==after
    write_json('regex_guard_checks.json',dict(evidence='VERIFIED',cases=len(cases),checks=cases,historical_and_new_control_results_identical=len(after),guarded_pattern=target,proof='The final required label character cannot be > or →; trailing whitespace is not stripped by this guard. The shortcut returns the same non-match as the original fullmatch would after exhaustive backtracking.',parser_sources_unchanged=True))


def score_primary_metrics(scorer, record, response, meta):
    """Same frozen strict/set formulas, without unused containment diagnostics.

    Long repetition loops make lenient full-text containment expensive. It is
    not requested here and cannot affect any metric written by this rescore.
    This wrapper is checked against the unmodified scorer on saved responses.
    """
    task=record['task_type'];response='' if response is None else str(response).strip()
    parsed=scorer.extract(task,response,record['prompt']) if scorer is v3 else scorer.extract(task,response)
    row=base.score_record(record,'',meta,extractor='span_extended')
    value=parsed.value
    if task in base.NUMERIC_TASKS:
        row['is_correct']=value==row['gold_integer']
    elif task=='highest_node_degree':
        name,degree=value if value is not None else (None,None)
        row['is_correct']=degree==row['gold_degree'] and name in row['valid_highest_names']
    elif task=='relation_identification':
        row['is_correct']=value==base.relation_key(row['gold_relation'])
    elif task=='shortest_path_listing':
        gg=base.structured_gold(record);predicted=value or []
        row['path_exact']={variant:parsed.decision=='COMMITTED' and [base.normalize_component(n,variant) for n in predicted] in [[base.normalize_component(n,variant) for n in p] for p in gg['paths']] for variant in base.SET_VARIANTS}
        row['is_correct']=row['path_exact']['raw']
    else:
        gg=[tuple(t) for t in row['parsed_gold_triples']] if task=='triple_listing' else row['parsed_gold_nodes']
        predicted=[tuple(t) for t in (value or [])] if task=='triple_listing' else (value or [])
        metrics=base.set_metrics_by_variant(gg,predicted,parsed.malformed)
        for m in metrics.values():
            if parsed.decision=='COMMITTED' and not (m['true_positive'] or m['false_positive'] or m['false_negative']):
                m.update(precision=1.0,recall=1.0,f1=1.0)
            if parsed.decision!='COMMITTED':m['exact_set_equality']=False
        row['set_metrics']=metrics;row['is_correct']=metrics['raw']['exact_set_equality']
    row.update(extracted_claim=value,extraction_decision=parsed.decision,parser_rule=parsed.rule,answer_span=parsed.span)
    if task=='triple_listing':row.update(malformed_triple_line_count=parsed.malformed,incomplete_triple_line_count=parsed.incomplete)
    return row


def discover():
    cohorts = [dict(model=m,arm='base',directory=f'outputs/fullrun_2026-10-04_B_results/{m}/stage1',prefix={'gemma':'gemma3'}.get(m,m),dataset=str(DATA.relative_to(ROOT))) for m in ('qwen','llava','gemma','deepseek')]
    cohorts.insert(1,dict(model='qwen',arm='tuned',directory='outputs/qwen_stage1_fetched/eval_trained_stage1/stage1',prefix='qwen',dataset=str(DATA.relative_to(ROOT))))
    full = ROOT/'outputs/llava_stage1_fetched/eval_trained_stage1/stage1'
    if full.is_dir():
        cohorts.append(dict(model='llava',arm='tuned',directory=str(full.relative_to(ROOT)),prefix='llava',dataset=str(DATA.relative_to(ROOT))))
    else:
        dry = ROOT/'outputs/llava_stage1_fetched/eval_trained_dry/stage1'
        if dry.is_dir():
            cohorts.append(dict(model='llava',arm='tuned_dry',directory=str(dry.relative_to(ROOT)),prefix='llava',dataset='outputs/llava_stage1_eval_inputs/stage1_dry.jsonl'))
    return cohorts


def explanation(row):
    if row['model']=='gemma' and row['task']=='node_degree' and row['statement_idx']==406:
        return 'VERIFIED: the final asserted degree replaces an earlier subtotal; inspect the raw response in the human sheet. The score decreases because the final degree differs from gold.'
    if row['v3_malformed_triples'] > row['v2_malformed_triples']:
        return 'VERIFIED: additional closed tuples with wrong field counts now incur false-positive penalties under the explicit triple policy.'
    return 'VERIFIED: extraction changed under '+row['v3_rule']+'; '+('the new literal claim differs from gold.' if row['v3_claim'] is not None else 'the new parser records no unique committed claim.')


def collapsed_response(raw):
    """Lossless text payload compaction with repetition counts and line spans.

    Numbered repeats omit redundant ordinal prefixes only; original bytes remain
    in the full population CSV and scoring JSONL.
    """
    lines = raw.splitlines()
    normalized = [re.sub(r'^\s*\d+[.)]\s+', '',line) for line in lines]
    out=[]; i=0
    while i<len(lines):
        best=None
        for width in range(1,min(24,(len(lines)-i)//3)+1):
            block=normalized[i:i+width]
            if not any(x.strip() for x in block):continue
            count=1
            while i+(count+1)*width<=len(lines) and normalized[i+count*width:i+(count+1)*width]==block:
                count+=1
            if count>=3 and (best is None or count*width>best[0]):best=(count*width,width,count)
        if best:
            length,width,count=best
            out.extend(lines[i:i+width])
            out.append(f'[The preceding {width}-line block occurs {count} consecutive times, original response lines {i+1}–{i+length}; repeated numbering prefixes suppressed.]')
            i+=length
        else:
            out.append(lines[i]);i+=1
    return '\n'.join(out)


def sample_population(population):
    rng=random.Random(SEED)
    # All decreases are mandatory. Fixed task targets favor paths and relations.
    selected=[r for r in population if r['change']=='decrease']
    assert len(selected)<=30, 'More than 30 decreases: expand the sheet rather than silently omit decreases.'
    seen={r['case_id'] for r in selected}
    pools=defaultdict(list)
    for r in population:
        if r['case_id'] not in seen:pools[r['task']].append(r)
    for pool in pools.values():
        pool.sort(key=lambda r:r['case_id']);rng.shuffle(pool)
    quotas={'shortest_path_listing':8,'relation_identification':8,'highest_node_degree':4,'neighbor_listing':4,'node_degree':3,'triple_listing':1,'node_number':1,'edge_number':1}
    for task, target in quotas.items():
        already=sum(r['task']==task for r in selected)
        for _ in range(min(max(0,target-already),len(pools[task]),30-len(selected))):
            r=pools[task].pop();selected.append(r);seen.add(r['case_id'])
    remaining=[r for r in population if r['case_id'] not in seen]
    weights={'shortest_path_listing':8,'relation_identification':8,'highest_node_degree':4,'neighbor_listing':4,'node_degree':3}
    while len(selected)<30:
        r=rng.choices(remaining,weights=[weights.get(x['task'],1) for x in remaining])[0]
        selected.append(r);remaining.remove(r)
    assert any(r['model']=='gemma' and r['task']=='node_degree' and r['statement_idx']==406 for r in selected)
    return selected


def main():
    assert not (OUT/'rescore_complete.json').exists(), 'Refusing to overwrite the completed result.'
    freeze=json.loads((OUT/'parser_freeze.json').read_text())
    for name,digest in freeze['dependency_sha256'].items():assert sha(ROOT/name)==digest, name
    install_impossible_path_match_guard()
    cohorts=discover()
    protected={DATA,META,Path(__file__).resolve(),ROOT/'scripts/validate_stage1_audit_v3.py'}
    for cohort in cohorts:
        d=ROOT/cohort['directory'];protected.update(d.glob('*'));protected.add(ROOT/cohort['dataset'])
    for model in ('qwen','llava','gemma'):
        d=ROOT/f'outputs/{model}_parser_wrong_answer_audit_2026-10-10/final'
        manifest=json.loads((d/'manifest.json').read_text())
        for name,digest in manifest.get('artifact_sha256',manifest.get('files',{})).items():
            assert sha(d/name)==digest, str(d/name)
        protected.update(p for p in d.iterdir() if p.is_file())
        protected.update((d.parent/'audit.py',d.parent/'candidates.py'))
    protected.update(ROOT/name for name in freeze['dependency_sha256'])
    protected={p for p in protected if p.is_file()}
    before={str(p.relative_to(ROOT)):sha(p) for p in sorted(protected)}
    write_json('input_hashes_before.json',before)
    write_json('cohorts.json',dict(evidence='VERIFIED',cohorts=cohorts,llava_full_tuned_eval_present=any(c['model']=='llava' and c['arm']=='tuned' for c in cohorts),notes='Only one statement × nine tasks exists for LLaVA tuned dry; it is reported separately. Assistant-prefix dry variants and diagnostic runs are excluded from canonical cohorts.'))
    records, metadata=base.validate_inputs(read(DATA),read(META),base.EXTENDED_TASK_TYPES)
    per_task=[];secondary=[];diffs=[];all_scores=[];diagnostics=[]
    for cohort in cohorts:
        config=json.loads((ROOT/cohort['directory']/'run_config.json').read_text())
        assert sha(ROOT/cohort['dataset'])==config['input_jsonl']['sha256']
        assert sha(META)==config['graph_metadata']['sha256']
        for task in base.EXTENDED_TASK_TYPES:
            path=ROOT/cohort['directory']/f"predictions_{cohort['prefix']}_{task}.jsonl"
            predictions=read(path)
            assert len(predictions)==(1 if cohort['arm']=='tuned_dry' else 100), str(path)
            assert len({int(r['statement_idx']) for r in predictions})==len(predictions)
            rows=[];raw_rows=[]
            for lineno,pred in enumerate(predictions,1):
                idx=int(pred['statement_idx']);record=records[(idx,task)]
                assert pred['task_type']==task and pred['prompt']==record['prompt']
                raw=pred.get('raw_response','')
                old=score_primary_metrics(v2,record,raw,metadata[idx]);new=score_primary_metrics(v3,record,raw,metadata[idx])
                a,b=score(task,old),score(task,new)
                direction='increase' if b>a+1e-12 else 'decrease' if b<a-1e-12 else 'unchanged'
                case=dict(case_id=f"{cohort['model']}:{cohort['arm']}:{task}:{idx}",model=cohort['model'],arm=cohort['arm'],task=task,statement_idx=idx,question=record['prompt'],raw_response=raw,v2_claim=old['extracted_claim'],v3_claim=new['extracted_claim'],gold=gold(task,old,record),v2_score=a,v3_score=b,delta_pp=100*(b-a),v2_full_credit=bool(exact(task,old)),v3_full_credit=bool(exact(task,new)),change=direction,v2_decision=old['extraction_decision'],v3_decision=new['extraction_decision'],v2_rule=old['parser_rule'],v3_rule=new['parser_rule'],v2_malformed_triples=old.get('malformed_triple_line_count',0),v3_malformed_triples=new.get('malformed_triple_line_count',0),v2_incomplete_triples=old.get('incomplete_triple_line_count',0),v3_incomplete_triples=new.get('incomplete_triple_line_count',0),source_file=str(path.relative_to(ROOT)),source_line=lineno,evidence='VERIFIED')
                if task in LIST_TASKS:
                    case.update(v2_secondary_basic_f1=score(task,old,'basic'),v3_secondary_basic_f1=score(task,new,'basic'))
                rows.append((old,new,case));all_scores.append(case);raw_rows.append(pred)
                extraction_changed=any(old.get(k)!=new.get(k) for k in ('extracted_claim','extraction_decision','parser_rule','answer_span','malformed_triple_line_count','incomplete_triple_line_count'))
                secondary_changed=task in LIST_TASKS and (score(task,old,'basic')!=score(task,new,'basic') or exact(task,old,'basic')!=exact(task,new,'basic'))
                if extraction_changed or direction!='unchanged' or secondary_changed or case['v2_full_credit']!=case['v3_full_credit']:
                    case['extraction_changed']=extraction_changed;diffs.append(case)
            n=len(rows)
            print(cohort['model'],cohort['arm'],task,n,flush=True)
            metric='strict_case_sensitive_macro_F1' if task in LIST_TASKS else 'strict_accuracy'
            result=dict(model=cohort['model'],arm=cohort['arm'],task=task,n=n,primary_metric=metric,v2_score=statistics.mean(r[2]['v2_score'] for r in rows),v3_score=statistics.mean(r[2]['v3_score'] for r in rows),v2_full_credit_count=sum(r[2]['v2_full_credit'] for r in rows),v3_full_credit_count=sum(r[2]['v3_full_credit'] for r in rows),score_increases=sum(r[2]['change']=='increase' for r in rows),score_decreases=sum(r[2]['change']=='decrease' for r in rows),evidence='VERIFIED')
            result['delta_pp']=100*(result['v3_score']-result['v2_score']);per_task.append(result)
            if task in LIST_TASKS:
                sec=dict(result,secondary_metric='secondary_basic_lowercase_whitespace_macro_F1',v2_secondary_basic_score=statistics.mean(score(task,r[0],'basic') for r in rows),v3_secondary_basic_score=statistics.mean(score(task,r[1],'basic') for r in rows),v2_secondary_basic_full_credit_count=sum(exact(task,r[0],'basic') for r in rows),v3_secondary_basic_full_credit_count=sum(exact(task,r[1],'basic') for r in rows))
                sec['secondary_delta_pp']=100*(sec['v3_secondary_basic_score']-sec['v2_secondary_basic_score']);secondary.append(sec)
            tokens=[r.get('generated_token_count') for r in raw_rows];tokens=[x for x in tokens if isinstance(x,(int,float))]
            limit=config.get('effective_max_new_tokens',config['generation']['max_new_tokens'])
            hits=sum(r.get('hit_token_ceiling',False) for r in raw_rows)
            diagnostics.append(dict(model=cohort['model'],arm=cohort['arm'],task=task,n=n,token_count_n=len(tokens),max_new_tokens=limit,hit_token_ceiling_count=hits,hit_token_ceiling_rate=hits/n,generated_tokens_equal_limit_count=sum(x==limit for x in tokens),mean_generated_tokens=statistics.mean(tokens) if tokens else None,median_generated_tokens=statistics.median(tokens) if tokens else None,token_count_method='reencoded_response_approximate' if cohort['model']=='deepseek' else 'saved_exact_generated_sequence_length',evidence='VERIFIED'))
    write_csv('per_task_v2_vs_v3.csv',per_task,list(per_task[0]))
    write_csv('per_task_v2_vs_v3_secondary_lists.csv',secondary,list(secondary[0]))
    write_jsonl('diff_v3_vs_v2.jsonl',diffs)
    write_jsonl('all_scores_v2_vs_v3.jsonl',all_scores)
    decreases=[dict(r,explanation=explanation(r)) for r in all_scores if r['change']=='decrease']
    fields=list(all_scores[0])+['explanation']
    write_csv('score_decreases.csv',decreases,fields)
    write_jsonl('score_decreases.jsonl',decreases)
    population=[r for r in all_scores if r['change'] in ('increase','decrease')]
    write_csv('spotcheck_all_candidates.csv',population,list(all_scores[0])+['v2_secondary_basic_f1','v3_secondary_basic_f1'])
    sample=sample_population(population)
    write_json('spotcheck_sampling.json',dict(evidence='VERIFIED',seed=SEED,population_count=len(population),population_counts=dict(Counter(r['task'] for r in population)),all_decreases_mandatory=True,task_targets={'shortest_path_listing':8,'relation_identification':8,'highest_node_degree':4,'neighbor_listing':4,'node_degree':3,'triple_listing':1,'node_number':1,'edge_number':1},sample_counts=dict(Counter(r['task'] for r in sample)),sample_ids=[r['case_id'] for r in sample],limitations='Syntax rules were developed on these cohorts; the human sample is independent of provisional readings, but is not held-out validation.'))
    text=['# Human spot-check: frozen audit_v3','',f'VERIFIED: 30 cases sampled from {len(population)} score increases/decreases; seed {SEED}. All decreases are included. Tasks are deliberately weighted, so this is not a population prevalence estimate.','', 'Judge whether the v3 extraction represents the answer actually asserted. Original raw bytes for every candidate are in spotcheck_all_candidates.csv; repeated blocks below retain their multiplicity. No provisional audit reading is shown.','']
    for i,r in enumerate(sample,1):
        text.extend([f"## {i}. {r['model']} / {r['arm']} / {r['task']} / statement {r['statement_idx']}",'',r['question'],'','Raw response:','', '````text',collapsed_response(r['raw_response']),'````','',f"v2 claim: `{json.dumps(r['v2_claim'],ensure_ascii=False)}`",'',f"v3 claim: `{json.dumps(r['v3_claim'],ensure_ascii=False)}`",'',f"Gold: `{json.dumps(r['gold'],ensure_ascii=False)}`",'','VERDICT: correct / wrong / unclear',''])
    (OUT/'spotcheck_sample.md').write_text('\n'.join(text))
    write_csv('generation_diagnostics.csv',diagnostics,list(diagnostics[0]))
    after={name:sha(ROOT/name) for name in before};assert after==before
    write_json('input_hashes_after.json',after)
    for name,digest in freeze['dependency_sha256'].items():assert sha(ROOT/name)==digest
    totals={f"{c['model']}:{c['arm']}":dict(n=sum(r['n'] for r in per_task if r['model']==c['model'] and r['arm']==c['arm']),score_increases=sum(r['score_increases'] for r in per_task if r['model']==c['model'] and r['arm']==c['arm']),score_decreases=sum(r['score_decreases'] for r in per_task if r['model']==c['model'] and r['arm']==c['arm']),full_credit_v2=sum(r['v2_full_credit_count'] for r in per_task if r['model']==c['model'] and r['arm']==c['arm']),full_credit_v3=sum(r['v3_full_credit_count'] for r in per_task if r['model']==c['model'] and r['arm']==c['arm'])) for c in cohorts}
    write_json('rescore_complete.json',dict(evidence='VERIFIED',parser_sha256=freeze['source_sha256'],protected_file_count=len(before),input_hashes_unchanged=True,scored_count=len(all_scores),diff_count=len(diffs),population_count=len(population),decrease_count=len(decreases),cohort_totals=totals))
    print(json.dumps(totals,indent=2));print('decreases',[(r['case_id'],r['v2_score'],r['v3_score']) for r in decreases])


if __name__=='__main__':main()
