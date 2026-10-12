#!/usr/bin/env python3
"""LLaVA branch only: strict checkpoints, calibrated assets, sequential loads."""
import argparse
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from synth_mcq_common import (ROOT, OBQA, DEFAULT_DATASET, LETTERS, PARSER_PROTOCOL,
    load_graphs, dataset_stats, solver_check, questions_for, question_type, evaluation_items,
    expand_conditions, permuted_choices, evidence_prompt, read_predictions, row_key,
    parse_answer, logit_scores, sha256_file, write_json)
from synth_mcq_llava_common import (STYLE, VARIANT, CHECKPOINT_SHA256, PAIRING_SHA256,
    verify_assets, validate_gate, letter_variants, score_letters, smoke_report,
    ANSWER_PREFIX, VERIFIED_LETTER_IDS, PREFIX_GATE_PROTOCOL, MIN_PREFIX_ROWS,
    FALLBACK_PROTOCOL, token_letter, generated_prefix_length, prefix_generated_letter, prefix_scores, fallback_fields)
from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION, configure_processor, prepare_inputs, image_diagnostics
from run_zero_shot_llava import format_qa_prompt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-path', type=Path); p.add_argument('--graphs', type=Path)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--image-audit', type=Path)
    p.add_argument('--pairing', type=Path)
    p.add_argument('--expected-render-checksum')
    p.add_argument('--models', default='base,tuned')
    p.add_argument('--conditions', default='image,text,none,wrong_graph')
    p.add_argument('--wrong-graph-modality', choices=('image', 'text', 'both'), default='both')
    p.add_argument('--prompt-style', choices=('obqa',), default='obqa')
    p.add_argument('--prompt-styles', choices=('obqa',), default='obqa')
    p.add_argument('--render-style', choices=('training',), default='training')
    p.add_argument('--render-fontsize', type=int, choices=(42,), default=42)
    p.add_argument('--max-new-tokens', type=int, choices=(64,), default=64)
    p.add_argument('--n-perms', type=int, choices=range(1, 5), default=4)
    p.add_argument('--seed', type=int, choices=(0,), default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--preflight', action='store_true', help='Check assets, full checkpoint and exact training versions; no GPU inference')
    p.add_argument('--allow-failed-legibility', action='store_true')
    p.add_argument('--smoke-report', action='store_true')
    args = p.parse_args(argv)
    if args.graphs and args.dataset_path and args.graphs.resolve() != args.dataset_path.resolve():
        p.error('--graphs and --dataset-path must resolve to the same path')
    args.dataset_path = (args.dataset_path or args.graphs or DEFAULT_DATASET).resolve()
    args.graphs = args.dataset_path; args.backend = 'llava'
    args.assets = args.assets.resolve()
    args.pairing = args.pairing or args.assets / 'wrong_graph_pairing.json'
    args.models = args.models.split(',')
    if not args.models or len(set(args.models)) != len(args.models) or not set(args.models) <= {'base', 'tuned'}:
        p.error('Invalid --models')
    try: args.conditions = expand_conditions(args.conditions.split(','), args.wrong_graph_modality)
    except ValueError as error: p.error(str(error))
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    if not args.out and not (args.validate_only or args.preflight): p.error('--out is required')
    if 'tuned' in args.models and not args.adapter and not args.validate_only: p.error('--adapter is required')
    args.model_id, args.revision, args.prompt_template = DEFAULT_MODEL_ID, DEFAULT_REVISION, 'llava_v1'
    args.prompt_styles = ['obqa']
    return args


def strict_artifact(path):
    from llava_adapter import adapter_provenance
    path = Path(path)
    absent = [name for name in ('checkpoint_manifest.json', 'run_config.json', 'training_state.pt',
              'projector.pt', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors') if not (path / name).is_file()]
    if absent:
        raise ValueError('STOP: full LLaVA checkpoint missing: ' + ', '.join(absent)
                         + '. The sparse laptop copy is insufficient; no loose loader or reconstructed manifest is allowed.')
    provenance = adapter_provenance(path, DEFAULT_MODEL_ID, DEFAULT_REVISION)
    if provenance['sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Full checkpoint adapter/projector composite SHA differs from the recorded trained run')
    config = json.loads((path / 'run_config.json').read_text())
    if (config.get('lora_r'), config.get('lora_alpha'), config.get('template'), config.get('projector_trained'), config.get('vision_tower_frozen')) != (128, 256, 'llava_v1', True, True):
        raise ValueError('Checkpoint training recipe differs')
    if config.get('versions') != TRAINING_VERSIONS:
        raise ValueError('Checkpoint training versions differ from the recorded run')
    return provenance


def projector_digest(model):
    import hashlib
    import torch
    from llava_stage1_training import projector
    h = hashlib.sha256()
    for name, value in sorted(projector(model).state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        h.update(name.encode() + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_sanity(model, name, checkpoint=None, pristine_projector=None):
    lora = [n for n, _ in model.named_parameters() if 'lora_' in n]
    peft = [n for n, module in model.named_modules() if type(module).__module__.startswith('peft') or 'lora_' in n]
    if name == 'base':
        if lora or peft or hasattr(model, 'peft_config'): raise AssertionError('Pristine base has PEFT/LoRA modules')
        return dict(pristine_base=True, lora_tensors=0, peft_modules=0, projector_sha256=projector_digest(model))
    import torch
    from safetensors import safe_open
    from llava_stage1_training import projector
    with safe_open(str(Path(checkpoint) / 'adapter/adapter_model.safetensors'), framework='pt', device='cpu') as handle:
        count = sum('lora_' in key for key in handle.keys())
    if count != 448 or len(lora) != count: raise AssertionError('Expected 448 restored LLaVA LoRA tensors')
    restored = torch.load(Path(checkpoint) / 'projector.pt', map_location='cpu', weights_only=True)
    current = projector(model).state_dict()
    if set(restored) != set(current) or any(not torch.equal(current[n].cpu(), restored[n].to(dtype=current[n].dtype)) for n in current):
        raise AssertionError('Loaded projector differs from checkpoint tensors')
    loaded_digest = projector_digest(model)
    if loaded_digest == pristine_projector: raise AssertionError('Tuned projector did not differ from pristine base')
    return dict(pristine_base=False, lora_tensors=count, projector_tensors=len(current),
                projector_sha256=loaded_digest, pristine_projector_sha256=pristine_projector,
                projector_differs_from_base=True, peft_modules=len(peft))


def forced_choice_probe(model, inputs, tokenizer, variants, torch):
    """One teacher-forced forward pass, preserving every original multimodal input."""
    prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError('Empty LLaVA answer-prefix tokenization')
    probe_inputs = dict(inputs)
    prefix = torch.tensor([prefix_ids], device=inputs['input_ids'].device, dtype=inputs['input_ids'].dtype)
    probe_inputs['input_ids'] = torch.cat((inputs['input_ids'], prefix), dim=-1)
    probe_inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(prefix)), dim=-1)
    with torch.inference_mode():
        output = model(**probe_inputs, use_cache=False, logits_to_keep=1)
    logits = output.logits[0, -1].float().cpu()
    return dict(prefix_probe=ANSWER_PREFIX, prefix_token_ids=prefix_ids, prefix_letter_token_ids=variants,
                prefix_letter_logits=score_letters(logits, variants),
                prefix_letter_variant_logits={l: [float(logits[token]) for token in variants[l]] for l in LETTERS})


def infer_one(model, processor, body, modality, image_path, args, torch):
    from PIL import Image
    image = None
    if modality == 'image':
        with Image.open(image_path) as opened: image = opened.convert('RGB')
    prompt = format_qa_prompt(processor, body, modality, args)
    inputs = prepare_inputs(processor, prompt, image)
    budget = image_diagnostics(processor, inputs, image.size) if image is not None else None
    input_length = int(inputs['input_ids'].shape[-1])
    inputs = inputs.to(model.device)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=64,
                                return_dict_in_generate=True, output_logits=True)
    torch.cuda.synchronize()
    probe_started = perf_counter()
    prefix_result = forced_choice_probe(model, inputs, processor.tokenizer, args.letter_ids, torch)
    torch.cuda.synchronize()
    prefix_result['prefix_probe_elapsed_seconds'] = perf_counter() - probe_started
    first = output.logits[0][0].float().cpu()
    scores = score_letters(first, args.letter_ids)
    probs = torch.tensor([scores[l] for l in LETTERS]).softmax(0).tolist()
    tail = output.sequences[0, input_length:]
    token_ids = tail.tolist()
    prefix_length = generated_prefix_length(token_ids, processor.tokenizer)
    raw = processor.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
    response = processor.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    letter, tier = parse_answer(response, 4)
    torch.cuda.synchronize()
    result = dict(raw_response=raw, response=response, predicted_letter=None if letter == 'FAILED' else letter,
        parse_ok=letter != 'FAILED', parse_tier=tier, parser_protocol=PARSER_PROTOCOL,
        first_generated_letter=token_letter(token_ids[0], args.letter_ids) if token_ids else None,
        generated_token_ids=token_ids,
        generated_prefix_token_count=prefix_length,
        prefix_generated_letter=prefix_generated_letter(response, token_ids, prefix_result['prefix_token_ids'], args.letter_ids, prefix_length),
        first_letter_token_ids=args.letter_ids, first_letter_logits=scores, first_letter_probs=dict(zip(LETTERS, probs)),
        first_letter_variant_logits={l: [float(first[token]) for token in args.letter_ids[l]] for l in LETTERS},
        probability_protocol='softmax over per-letter max raw logit of bare and SentencePiece variants',
        generated_token_count=int(tail.shape[0]), hit_token_ceiling=int(tail.shape[0]) >= 64,
        input_tokens=input_length, vision_tokens=budget['realised_image_tokens'] if budget else 0,
        item_elapsed_seconds=perf_counter() - started, peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        **prefix_result)
    if budget: result['image_budget'] = budget
    return result


def run_identity(args, checksum):
    return dict(protocol='synth_mcq_llava_v2', backend='llava', model_id=DEFAULT_MODEL_ID, model_revision=DEFAULT_REVISION,
        dataset_path=str(args.dataset_path), dataset_sha256=sha256_file(args.dataset_path),
        graphs_sha256=checksum, render_manifest_sha256=sha256_file(args.assets / 'render_manifest.json'),
        pairing_sha256=sha256_file(args.pairing), image_audit_sha256=sha256_file(args.image_audit) if args.image_audit else None,
        allow_failed_legibility=args.allow_failed_legibility, render_variant=VARIANT, render_settings=STYLE,
        prompt_style='obqa', prompt_template='llava_v1', n_perms=args.n_perms, seed=0,
        max_new_tokens=64, parser_protocol=PARSER_PROTOCOL, batch_size=1, dtype='bfloat16',
        generation=dict(do_sample=False, num_beams=1),
        prefix_probe=dict(prefix=ANSWER_PREFIX, letter_token_ids=VERIFIED_LETTER_IDS,
                          scoring='max over both verified variants', gate_protocol=PREFIX_GATE_PROTOCOL, minimum_n=MIN_PREFIX_ROWS),
        secondary_parser_protocol=FALLBACK_PROTOCOL,
        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__),
            ROOT / 'scripts/synth_mcq_llava_common.py', ROOT / 'scripts/synth_mcq_common.py',
            ROOT / 'scripts/llava_common.py', ROOT / 'scripts/llava_adapter.py', ROOT / 'scripts/llava_stage1_prompt.py',
            OBQA / 'run_zero_shot_llava.py', OBQA / 'prompt_common.py']})


def validate_resume(rows, graphs, pairing, args):
    from analyze_synth_mcq import validate_rows
    if rows:
        validate_rows(rows)
    by_id = {g['id']: g for g in graphs}
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen: raise ValueError('Duplicate resumed row')
        seen.add(key)
        if (row.get('backend') != 'llava' or row.get('render_variant') != VARIANT or row['model'] not in ('base', 'tuned')
            or row['condition'] not in ('image', 'text', 'none', 'wrong_image', 'wrong_text') or row['perm'] not in range(args.n_perms)
            or row['model_revision'] != DEFAULT_REVISION or row['prompt_style'] != 'obqa'
            or row['render_fontsize'] != 42 or row['render_edge_fontsize'] != 34):
            raise ValueError('Invalid resumed row identity')
        graph = by_id[row['graph_id']]
        question = next(q for q in questions_for(graph) if q['id'] == row['question_id'])
        choices = permuted_choices(graph, row['perm'], question)
        shown = pairing['mapping'][graph['id']] if row['condition'].startswith('wrong_') else graph['id']
        predicted = choices.get(row['predicted_letter'])
        if (row['choices'] != choices or row['correct_node'] != question['correct_node'] or row['shown_graph_id'] != shown
            or row['predicted_node'] != predicted or row['is_correct'] != (predicted == question['correct_node'])):
            raise ValueError('Invalid resumed answer/evidence mapping')
    return seen


TRAINING_VERSIONS = dict(torch='2.8.0+cu128', transformers='5.16.1', peft='0.21.2', accelerate='1.15.0')


def version_preflight():
    import torch, transformers, peft, accelerate
    versions = {name: module.__version__ for name, module in [('torch', torch), ('transformers', transformers), ('peft', peft), ('accelerate', accelerate)]}
    if versions != TRAINING_VERSIONS: raise ValueError('LLaVA venv differs from training run: ' + json.dumps(versions))
    return versions


def cached_model_preflight():
    """Verify the pinned local weights and native processor without loading weights."""
    from huggingface_hub import hf_hub_download
    from transformers import LlavaNextProcessor, LlavaNextConfig
    index_path = hf_hub_download(DEFAULT_MODEL_ID, 'model.safetensors.index.json',
                                 revision=DEFAULT_REVISION, local_files_only=True)
    index = json.loads(Path(index_path).read_text())
    for shard in sorted(set(index['weight_map'].values())):
        hf_hub_download(DEFAULT_MODEL_ID, shard, revision=DEFAULT_REVISION, local_files_only=True)
    processor = LlavaNextProcessor.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True)
    configure_processor(processor, LlavaNextConfig.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True))
    return dict(letter_token_ids=letter_variants(processor.tokenizer, verify_pin=True),
                prefix_probe=ANSWER_PREFIX, prefix_token_ids=processor.tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False),
                image_grid_pinpoints=processor.image_processor.image_grid_pinpoints,
                weight_shards=len(set(index['weight_map'].values())))


def main(argv=None):
    args = parse_args(argv)
    graphs = load_graphs(args.dataset_path)
    solver = solver_check(graphs)
    manifest, checksum = verify_assets(args.dataset_path, args.assets, args.pairing, args.expected_render_checksum)
    pairing = json.loads(args.pairing.read_text())
    by_id = {g['id']: g for g in graphs}
    if set(pairing['mapping']) != set(by_id) or set(pairing['mapping'].values()) != set(by_id):
        raise ValueError('Pairing must be a complete derangement')
    for gid, shown in pairing['mapping'].items():
        golds = {q['correct_node'] for q in questions_for(by_id[gid])}
        people = {n['id'] for n in by_id[shown]['nodes'] if n['kind'] == 'person_name'}
        if gid == shown or golds & people: raise ValueError('Pairing has fixed point or gold-name collision')
    images_requested = bool(set(args.conditions) & {'image', 'wrong_image'})
    gate_info = validate_gate(args.image_audit, manifest, args.assets, args.allow_failed_legibility, images_requested)
    selected = graphs[:args.limit] if args.limit else graphs
    items = evaluation_items(selected, args.models, args.conditions, ['obqa'], args.n_perms)
    print('VALIDATED ' + json.dumps(dataset_stats(graphs)), flush=True)
    print('INDEPENDENT SOLVER ' + json.dumps({k: v for k, v in solver.items() if k != 'questions'}), flush=True)
    print('WRONG GRAPH PAIRING sha256=' + sha256_file(args.pairing) + ' ' + json.dumps(pairing), flush=True)
    print('LEGIBILITY ' + json.dumps(gate_info), flush=True)
    print(f'EXPECTED ROWS: {len(items)}', flush=True)
    if args.validate_only: return
    artifact = strict_artifact(args.adapter) if 'tuned' in args.models else None
    if args.preflight:
        versions = version_preflight()
        cached = cached_model_preflight()
        if args.image_audit and cached['image_grid_pinpoints'] != json.loads(args.image_audit.read_text())['image_grid_pinpoints']:
            raise ValueError('Cached native processor grids differ from the CPU audit')
        print('PREFLIGHT PASS ' + json.dumps(dict(versions=versions, adapter=artifact,
              cached_model=cached, graphs_sha256=checksum)), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'predictions.jsonl'; config_path = args.out / 'config.json'
    if path.exists() and path.stat().st_size and not args.resume: raise FileExistsError('Use --resume or a fresh OUT')
    rows = read_predictions(path) if args.resume else []
    done = validate_resume(rows, graphs, pairing, args)
    identity = run_identity(args, checksum)
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config['identity'] != identity: raise ValueError('Incompatible LLaVA resume identity')
        previous_artifact = config.get('adapter')
        if artifact and previous_artifact and previous_artifact['sha256'] != artifact['sha256']:
            raise ValueError('Resumed checkpoint differs')
        if any(r['model'] == 'tuned' for r in rows) and not previous_artifact:
            raise ValueError('Tuned predictions lack checkpoint provenance')
        if artifact:
            config['adapter'] = artifact
    elif rows: raise ValueError('Predictions exist without config.json')
    else:
        config = dict(identity=identity, backend='llava', dataset_path=str(args.dataset_path), dataset_sha256=identity['dataset_sha256'],
            render_variant=VARIANT, render_settings=STYLE, render_fontsize=42, render_edge_fontsize=34,
            prompt_style='obqa', prompt_template='llava_v1', pairing_sha256=PAIRING_SHA256, adapter=artifact,
            legibility=gate_info, wrong_graph_pairing=pairing, dataset_stats=dataset_stats(graphs), model_loads={},
            qwen_image_comparability='Images differ from Qwen; compare base/tuned within LLaVA using identical assets')
        write_json(config_path, config)
    write_json(args.out / 'solver_check.json', solver)
    for filename in ('render_manifest.json', 'wrong_graph_pairing.json'):
        src = args.assets / filename if filename == 'render_manifest.json' else args.pairing
        dest = args.out / filename
        if dest.resolve() != src.resolve(): dest.write_bytes(src.read_bytes())
    link = args.out / 'graphs'
    if not link.exists(): link.symlink_to(args.assets / 'graphs', target_is_directory=True)
    elif not link.is_symlink() or link.resolve() != (args.assets / 'graphs').resolve(): raise ValueError('OUT/graphs must link to accepted assets')
    pending = [key for key in items if key not in done]
    first_key = ('obqa', 'image' if 'image' in args.conditions else args.conditions[0], selected[0]['id'], questions_for(selected[0])[0]['id'], 0)
    probe = {r['model']: r['first_letter_logits'] for r in rows if (r['prompt_style'], r['condition'], r['graph_id'], r['question_id'], r['perm']) == first_key}
    def record_probe_sanity():
        if set(probe) != {'base', 'tuned'}:
            return
        difference = max(abs(probe['base'][l] - probe['tuned'][l]) for l in LETTERS)
        config['first_position_sanity'] = dict(base=probe['base'], tuned=probe['tuned'],
            prompt_style=first_key[0], condition=first_key[1], graph_id=first_key[2],
            question_id=first_key[3], perm=first_key[4], differ=difference > 1e-6, max_abs_difference=difference)
        write_json(config_path, config)
        print('BASE/TUNED LOGIT SANITY ' + json.dumps(config['first_position_sanity']), flush=True)
        if difference <= 1e-6:
            raise AssertionError('Base/tuned first-position letter logits did not differ')
    # Re-check persisted probes too; a resume cannot bypass an earlier failure.
    record_probe_sanity()
    if pending:
        import torch, transformers
        config['versions'] = version_preflight()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA/bf16 required for inference; use CPU validate-only')
        from transformers import LlavaNextProcessor, LlavaNextConfig, LlavaNextForConditionalGeneration
        transformers.set_seed(0)
        for name in ('base', 'tuned'):
            if name not in args.models or not any(key[1] == name for key in pending): continue
            model = processor = None
            try:
                started = perf_counter()
                processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.revision, local_files_only=True)
                configure_processor(processor, LlavaNextConfig.from_pretrained(args.model_id, revision=args.revision, local_files_only=True))
                args.letter_ids = letter_variants(processor.tokenizer, verify_pin=True)
                # Exact existing QA model-loading API/settings; no alternative model loader.
                model = LlavaNextForConditionalGeneration.from_pretrained(args.model_id, revision=args.revision,
                            torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True).eval()
                pristine = projector_digest(model)
                provenance = None
                if name == 'tuned':
                    from llava_adapter import load_adapter
                    model, provenance = load_adapter(model, args.adapter, args.model_id, args.revision)
                sanity = load_sanity(model, name, args.adapter if name == 'tuned' else None, pristine)
                info = dict(sanity=sanity, adapter=provenance, loading_seconds=perf_counter() - started,
                            torch_version=torch.__version__, transformers_version=transformers.__version__)
                config['model_loads'][name] = info
                pinpoints = [list(pair) for pair in processor.image_processor.image_grid_pinpoints]
                if args.image_audit:
                    audited = json.loads(args.image_audit.read_text())
                    if pinpoints != audited['image_grid_pinpoints']: raise ValueError('Processor grids differ from CPU audit')
                    config['chosen_grid_by_graph'] = {r['graph_id']: r['selected_grid_pinpoint'] for r in audited['measurements']}
                config.update(letter_token_ids=args.letter_ids, processor_image_grid_pinpoints=pinpoints,
                              prefix_probe=dict(prefix=ANSWER_PREFIX,
                                  token_ids=processor.tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False),
                                  letter_token_ids=args.letter_ids, scoring='max over both verified variants'),
                              processor_version=transformers.__version__, processor_class=type(processor.image_processor).__name__)
                write_json(config_path, config)
                print(f'LOAD SANITY [{name}] ' + json.dumps(info), flush=True)
                with path.open('a') as output:
                    for style, _, condition, gid, qid, perm in [key for key in pending if key[1] == name]:
                        graph = by_id[gid]; question = next(q for q in questions_for(graph) if q['id'] == qid)
                        shown_id = pairing['mapping'][gid] if condition.startswith('wrong_') else gid
                        choices = permuted_choices(graph, perm, question)
                        body, modality = evidence_prompt(manifest, graph, question, perm, condition, style, by_id[shown_id])
                        result = infer_one(model, processor, body, modality, args.assets / 'graphs' / f'{shown_id}.png', args, torch)
                        predicted = choices.get(result['predicted_letter'])
                        row = dict(backend='llava', model=name, model_revision=DEFAULT_REVISION, condition=condition,
                            graph_id=gid, question_id=qid, question_ids=[q['id'] for q in questions_for(graph)], questions_per_graph=2,
                            dataset=graph['dataset'], question_type=question_type(question), prompt_style=style, perm=perm,
                            render_style='training', render_variant=VARIANT, render_fontsize=42, render_edge_fontsize=34,
                            shown_graph_id=shown_id, wrong_graph_modality=modality if condition.startswith('wrong_') else None,
                            choices=choices, correct_node=question['correct_node'], correct_letter=next(l for l in LETTERS if choices[l] == question['correct_node']),
                            predicted_node=predicted, is_correct=predicted == question['correct_node'],
                            **result, **logit_scores(result['first_letter_logits'], choices, question['correct_node']),
                            **prefix_scores(result['prefix_letter_logits'], choices, question['correct_node']),
                            **fallback_fields(result['response'], choices, question['correct_node']))
                        if modality == 'image' and args.image_audit:
                            expected = next(r for r in audited['measurements'] if r['graph_id'] == shown_id)
                            if row['vision_tokens'] != expected['realised_image_tokens'] or row['image_budget']['selected_grid_pinpoint'] != expected['selected_grid_pinpoint']:
                                raise ValueError('Actual vision grid/tokens differ from CPU audit')
                        output.write(json.dumps(row, ensure_ascii=False) + '\n'); output.flush(); rows.append(row)
                        print(f'PROGRESS rows={len(rows)} {name}/{condition}/{gid}/{qid}/{perm} shown={shown_id} parse={row["parse_tier"]} correct={row["is_correct"]} logit={row["logit_argmax_is_correct"]} vision_tokens={row["vision_tokens"]} seconds={row["item_elapsed_seconds"]:.3f}', flush=True)
                        if (style, condition, gid, qid, perm) == first_key:
                            probe[name] = result['first_letter_logits']
                            record_probe_sanity()
            finally:
                del model, processor
                gc.collect(); torch.cuda.empty_cache()
                print(f'GPU MEMORY RELEASED [{name}] allocated_bytes={torch.cuda.memory_allocated()}', flush=True)
    values = [r['vision_tokens'] for r in rows if r['condition'] in ('image', 'wrong_image')]
    config['vision_token_summary'] = dict(n_image_rows=len(values), min=min(values) if values else None,
                                        median=median(values) if values else None, max=max(values) if values else None)
    print('REALISED VISION TOKENS ' + json.dumps(config['vision_token_summary']), flush=True)
    if args.smoke_report:
        report = smoke_report(rows, sum(v['loading_seconds'] for v in config['model_loads'].values()))
        report.update(identity=identity, adapter_sha256=artifact['sha256'] if artifact else None,
                      first_position_sanity=config.get('first_position_sanity'))
        report['passed'] = report['passed'] and bool(config.get('first_position_sanity', {}).get('differ'))
        config['smoke_report'] = report; write_json(args.out / 'smoke_report.json', report)
        print('SMOKE REPORT ' + json.dumps(report), flush=True)
        if not report['passed']:
            write_json(config_path, config)
            raise ValueError('STOP: smoke incomplete, base/tuned sanity failed, or pooled prefix agreement below 100% / qualifying n below minimum; do not launch the next step')
    write_json(config_path, config)


if __name__ == '__main__':
    main()
