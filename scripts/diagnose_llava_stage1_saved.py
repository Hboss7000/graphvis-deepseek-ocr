"""Read-only reconstruction of saved LLaVA generation settings; no model inference."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import jinja2

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parser_v3_rescore_2026-10-10'
E=OUT/'evidence'
sys.path.insert(0,str(ROOT/'scripts'))
from llava_stage1_prompt import format_stage1_prompt as stage1_format_prompt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    config=json.loads((ROOT/'outputs/fullrun_2026-10-04_B_results/llava/stage1/run_config.json').read_text())
    tok=json.loads((E/'llava_tokenizer_config.json').read_text())
    check=json.loads((E/'tokenizer_configuration_check.json').read_text())
    template=jinja2.Environment().from_string(tok['chat_template'])
    # Read only the literal Conversation assignments; never execute downloaded code.
    tree=ast.parse((E/'graphvis_llava_conversation.py').read_text())
    definitions={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and any(isinstance(t,ast.Name) and t.id in ('conv_llava_v1','conv_mistral_instruct') for t in n.targets):
            name=n.targets[0].id
            definitions[name]={k.arg:ast.literal_eval(k.value) for k in n.value.keywords if k.arg in ('system','roles','sep','sep2','version')}
    prompts=[]
    for arm,directory in [('base','outputs/fullrun_2026-10-04_B_results/llava/stage1'),('tuned_dry','outputs/llava_stage1_fetched/eval_trained_dry/stage1')]:
        for p in sorted((ROOT/directory).glob('predictions_llava_*.jsonl')):
            for line in p.open():
                r=json.loads(line);body='<image>\n'+r['prompt']
                hf=template.render(messages=[dict(role='user',content=body)],bos_token=tok['bos_token'],eos_token=tok['eos_token'],add_generation_prompt=True)
                conv=definitions['conv_llava_v1'];graphvis=conv['system']+conv['sep']+conv['roles'][0]+': '+body+conv['sep']+conv['roles'][1]+':'
                mistral='[INST] '+body+' [/INST]'
                actual=hf if arm=='base' else stage1_format_prompt(None,body,'llava_v1')
                if arm=='tuned_dry':assert actual==graphvis
                prompts.append(dict(arm=arm,task=r['task_type'],statement_idx=r['statement_idx'],question=r['prompt'],user_body=body,reconstructed_prompt=actual,hf_mistral_prompt=hf,graphvis_mistral_prompt=mistral,graphvis_llava_v1_prompt=graphvis,base_matches_graphvis_mistral_after_removing_explicit_bos=hf.removeprefix('<s>')==mistral,evidence='INFERRED historical runtime string; VERIFIED deterministic reconstruction from pinned template and saved question'))
    with (OUT/'llava_formatted_prompts.jsonl').open('w') as f:
        for r in prompts:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    sample=next(r for r in prompts if r['arm']=='base' and r['task']=='node_number')
    details=dict(evidence='VERIFIED unless explicitly marked INFERRED',run_config_generation=config['generation'],generation_kwargs_from_current_code=dict(do_sample=False,num_beams=1,max_new_tokens=1024),tokenizer=dict(eos_token_id=check['tokenizer_eos_token_id'],pad_token_id=check['tokenizer_pad_token_id'],bos_token_id=1),pinned_generation_config=check['generation_config'],effective_eos_pad=dict(evidence='INFERRED: not logged at historical generate(); expected from pinned defaults and checked Transformers 5.16.1 fallback',eos_token_id=2,pad_token_id=2,explanation='No explicit EOS/pad kwargs; generation config and model text config pad are unset. generate() falls back to EOS 2 for pad. Tokenizer PAD 32001 is a separate setting.'),repetition_penalty=dict(evidence='VERIFIED unset in saved run config and pinned generation config; INFERRED historical effective setting',configured_value=None,effective_behavior='No repetition penalty (equivalent to 1.0).'),example=sample,graphvis_revision=json.loads((E/'graphvis_tree.json').read_text())['sha'],graphvis_templates=definitions,comparison=dict(evidence='VERIFIED',base_matches_mistral_instruct_framing=True,base_matches_released_predict_llava_v1=False,tuned_dry_matches_released_predict_llava_v1=True,repository_stage1_evaluator_found=False),double_bos=dict(evidence='VERIFIED tokenizer-only reproduction; INFERRED historical input tensor since input IDs were not saved',default_prefix_ids=check['token_ids_default_special_tokens'][:4],without_special_tokens_prefix_ids=check['token_ids_without_special_tokens'][:4],observed_in_reconstruction=check['token_ids_default_special_tokens'][:2]==[1,1],explanation='HF rendered prompt already includes BOS. prepare_inputs() leaves add_special_tokens at its tokenizer default, producing a second BOS in the checked tokenizer-only reproduction.'),historical_source_match=dict(llava_common_sha256=sha(ROOT/'scripts/llava_common.py'),matches_saved_fullrun_hash=sha(ROOT/'scripts/llava_common.py')==config['fullrun']['code_files']['scripts/llava_common.py'],current_runner_matches_historical_hash=sha(ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py')==config['fullrun']['code_files']['experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py']),limitations='The saved run does not log input token IDs or effective generation_config. Current runner has changed since the base run; its exact historic generate() body is not available. No full tuned evaluation exists; nine tuned dry answers cannot establish a tuned/base comparison.',verdict=dict(evidence='INFERRED',rerun_recommended=True,reason='Severe token-ceiling behavior plus a template difference from the released GraphVis predict path justify a controlled rerun. Verify the intended paper conversation mode and special-token handling first. These checks do not prove the causal source of repetition or recover the paper Table 4 evaluation settings.'))
    (OUT/'llava_generation_settings.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    rev=details['graphvis_revision']
    hfurl='https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/2424fdd47412fccc66d91719126b420e9fbd7065/tokenizer_config.json'
    graphurl=f'https://github.com/yihedeng9/GraphVis/blob/{rev}/predict.py'
    text=f'''# LLaVA saved-generation diagnostics

VERIFIED: saved decoding kwargs are `do_sample=False`, `num_beams=1`, `max_new_tokens=1024`. The saved config does not explicitly set repetition_penalty, EOS or pad. Pinned generation config EOS is 2, with pad and repetition_penalty unset. Tokenizer EOS=2, PAD=32001. Checked local Transformers 5.16.1 uses EOS as the generation pad fallback. INFERRED historical effective settings: EOS=2, generation PAD=2, no repetition penalty (equivalent to 1.0); the runtime generation configuration was not saved. The current runner differs from its historical source hash, so runtime values are not certified.

INFERRED historical prompt reconstruction (VERIFIED pinned template plus saved question; all 909 reconstructions in llava_formatted_prompts.jsonl):

```text
{sample['reconstructed_prompt']}
```

VERIFIED: the [pinned HF template]({hfurl}) frames the prompt with Mistral `[INST]` and explicit BOS. GraphVis `mistral_instruct` has matching instruction framing (BOS supplied through tokenization). However, its [released predict.py]({graphurl}) hard-codes `llava_v1`, which adds a system message and USER/ASSISTANT delimiters. Base uses the former; tuned dry uses the latter. No Stage 1 evaluator exists in the checked public repository tree; matching the paper's Table 4 runtime remains INFERRED/unresolved.

VERIFIED: tokenizer-only reconstruction with normal special-token handling gives prefix IDs {check['token_ids_default_special_tokens'][:4]}, versus {check['token_ids_without_special_tokens'][:4]} with special tokens disabled. The saved-hash-matching llava_common.py does not disable special tokens. INFERRED: the base run likely received two BOS tokens. Historical input IDs were not recorded. No model weights or inference were used for this check.

VERIFIED: generation_diagnostics.csv contains ceiling-hit counts/rates and mean/median token counts for every task and available arm. Exact sequence counts are saved for Qwen, LLaVA and Gemma; DeepSeek counts are approximate re-encodings. Only one statement per task exists for the tuned LLaVA dry run; full tuned metrics are missing.

INFERRED: a controlled LLaVA rerun is warranted after selecting the intended paper template and verifying BOS/EOS/pad handling. These observations do not prove why the model loops. No inference or decoding settings were changed here.
'''
    (OUT/'LLAVA_DIAGNOSTICS.md').write_text(text)
    sources=json.loads((E/'sources.json').read_text());sources.pop('graphvis_model_vqa.py',None)
    sources.update({'graphvis_predict.py':f'https://raw.githubusercontent.com/yihedeng9/GraphVis/{rev}/predict.py','graphvis_llava_conversation.py':f'https://raw.githubusercontent.com/yihedeng9/GraphVis/{rev}/llava/conversation.py','llava_tokenizer.json':hfurl.replace('/blob/','/resolve/').replace('tokenizer_config.json','tokenizer.json')})
    (E/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    evidence={str(p.relative_to(ROOT)):sha(p) for p in E.rglob('*') if p.is_file()}
    for p in [ROOT/'scripts/llava_common.py',ROOT/'scripts/llava_stage1_prompt.py',ROOT/'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py',ROOT/'.venv_llava_cpu/lib/python3.12/site-packages/transformers/generation/utils.py',ROOT/'.venv_llava_cpu/lib/python3.12/site-packages/transformers/generation/configuration_utils.py',ROOT/'.venv_llava_cpu/lib/python3.12/site-packages/transformers/processing_utils.py']:
        evidence[str(p.relative_to(ROOT))]=sha(p)
    (OUT/'diagnostic_evidence_hashes.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(prompt_count=len(prompts),double_bos=details['double_bos'],comparison=details['comparison']),indent=2))


if __name__=='__main__':main()
