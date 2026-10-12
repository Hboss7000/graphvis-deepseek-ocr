#!/usr/bin/env python3
"""Render locally, then evaluate pristine Qwen and its complete Stage 1 checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

from synth_mcq_common import (ROOT, OBQA, MODEL_ID, REVISION, MIN_PIXELS, MAX_PIXELS,
    LETTERS, load_graphs, dataset_stats, render_graphs, verify_renders, permuted_choices,
    read_predictions, row_key, parse_answer, sha256_file, write_json,
    styled_prompt_body, logit_scores, PARSER_PROTOCOL, DEFAULT_DATASET,
    questions_for, question_type, solver_check, wrong_graph_pairing, expand_conditions,
    evaluation_items, evidence_prompt)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, help='Defaults to the name-grounded 100-question file')
    parser.add_argument('--graphs', type=Path, help='Backward-compatible alias of --dataset-path')
    parser.add_argument('--adapter', type=Path, help='Full verified checkpoint directory, containing adapter/ and projector.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', default='base,tuned')
    parser.add_argument('--conditions', help='Default new grid: image,text,none,wrong_graph')
    parser.add_argument('--wrong-graph-modality', choices=('image', 'text', 'both'), default='both')
    parser.add_argument('--prompt-style', choices=('obqa', 'needed'), default='obqa')
    parser.add_argument('--prompt-styles', help='Comma-separated obqa,needed; defaults to obqa')
    parser.add_argument('--render-style', choices=('training', 'compact'), default='training')
    parser.add_argument('--render-fontsize', type=int,
                        help='Node font size recorded/verified against renders; use separate OUT for a variant')
    parser.add_argument('--n-perms', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--image-audit', type=Path, help='Passing legibility_gate.json required before new-dataset GPU inference')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--mlflow-uri', help='Optional file: store; reuse the Stage 1 TrainingLogger and qwen-stage1 experiment')
    args = parser.parse_args(argv)
    if args.dataset_path is not None and args.graphs is not None and args.dataset_path.resolve() != args.graphs.resolve():
        parser.error('--dataset-path and --graphs must resolve to the same path')
    args.dataset_path = (args.dataset_path or args.graphs or DEFAULT_DATASET).resolve()
    args.graphs = args.dataset_path
    args.prompt_styles = (args.prompt_styles or args.prompt_style).split(',')
    if len(set(args.prompt_styles)) != len(args.prompt_styles) or not set(args.prompt_styles) <= {'obqa', 'needed'}:
        parser.error('Invalid --prompt-styles')
    if '--prompt-style' in (argv if argv is not None else sys.argv[1:]) and args.prompt_style not in args.prompt_styles:
        parser.error('--prompt-style conflicts with --prompt-styles')
    args.prompt_style = args.prompt_styles[0]
    args.models = args.models.split(',')
    args.conditions_default = args.conditions is None
    try:
        args.conditions = expand_conditions((args.conditions or 'image,text,none,wrong_graph').split(','), args.wrong_graph_modality)
    except ValueError as error: parser.error(str(error))
    for values, allowed, flag in [(args.models, {'base', 'tuned'}, '--models')]:
        if not values or len(values) != len(set(values)) or not set(values) <= allowed:
            parser.error(f'Invalid {flag}')
    if not 1 <= args.n_perms <= 4:
        parser.error('--n-perms must be 1..4 (distinct cyclic rotations)')
    if not 1 <= args.max_new_tokens <= 64:
        parser.error('--max-new-tokens must be 1..64')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.render_fontsize is not None and args.render_fontsize <= 0:
        parser.error('--render-fontsize must be positive')
    if args.render_fontsize is None:
        args.render_fontsize = 30 if args.render_style == 'training' else 18
    if 'tuned' in args.models and not args.adapter and not (args.render_only or args.validate_only):
        parser.error('--adapter is required for tuned inference')
    if args.mlflow_uri and not args.mlflow_uri.startswith('file:'):
        parser.error('--mlflow-uri must use a local file: store')
    # Settings consumed by the existing Qwen load/prepare helpers.
    args.model_id, args.revision = MODEL_ID, REVISION
    args.min_pixels, args.max_pixels = MIN_PIXELS, MAX_PIXELS
    return args


def load_sanity(model, model_name, checkpoint=None):
    """No PEFT import or adapter helper call is made in the base branch."""
    lora_names = [name for name, _ in model.named_parameters() if 'lora_' in name]
    peft_modules = [name for name, module in model.named_modules()
                    if type(module).__module__.startswith('peft') or 'lora_' in name]
    if model_name == 'base':
        if lora_names or peft_modules or hasattr(model, 'peft_config'):
            raise AssertionError('Base model contains PEFT/LoRA modules')
        return dict(pristine_base=True, lora_tensors=0, peft_modules=0)
    import torch
    from safetensors import safe_open
    from llava_stage1_training import projector
    with safe_open(str(checkpoint / 'adapter' / 'adapter_model.safetensors'), framework='pt', device='cpu') as handle:
        restored_lora = sum('lora_' in key for key in handle.keys())
    merger_state = torch.load(checkpoint / 'projector.pt', map_location='cpu', weights_only=True)
    restored_mergers = sum(torch.is_tensor(value) for value in merger_state.values())
    loaded_mergers = len(projector(model).state_dict())
    if not restored_lora or not restored_mergers:
        raise AssertionError('Tuned checkpoint must restore both LoRA and merger tensors')
    if restored_lora != len(lora_names) or restored_mergers != loaded_mergers:
        raise AssertionError('Checkpoint/model restored tensor counts differ')
    return dict(pristine_base=False, lora_tensors=restored_lora, merger_tensors=restored_mergers,
                peft_modules=len(peft_modules), merger_groups=4)


def infer_one(model, processor, body, condition, image_path, args, torch):
    from PIL import Image
    from run_zero_shot_qwen import format_qwen_prompt, prepare_inputs
    image = None
    if condition == 'image':
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
    prompt = format_qwen_prompt(processor, body, 'image' if condition == 'image' else 'text')
    if '<|im_start|>system' in prompt:
        raise AssertionError('Unexpected default system prompt')
    inputs = prepare_inputs(processor, prompt, image, args).to('cuda')
    input_length = int(inputs['input_ids'].shape[1])
    letter_ids = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in letter_ids) or len({ids[0] for ids in letter_ids}) != 4:
        raise AssertionError('A/B/C/D must each be a distinct single token')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, num_beams=1,
            max_new_tokens=args.max_new_tokens, return_dict_in_generate=True, output_logits=True)
    # Raw first-position logits, before any generation logit processors.
    logits = generated.logits[0][0, [ids[0] for ids in letter_ids]].float()
    probs = logits.softmax(dim=-1)
    tail = generated.sequences[:, input_length:]
    raw = processor.batch_decode(tail, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)[0].strip()
    response = raw.removesuffix('<|im_end|>').strip()
    letter, tier = parse_answer(response, n_choices=4)
    torch.cuda.synchronize()
    result = dict(raw_response=raw, response=response,
                  predicted_letter=None if letter == 'FAILED' else letter,
                  parse_ok=letter != 'FAILED', parse_tier=tier, parser_protocol=PARSER_PROTOCOL,
                  first_letter_token_ids=dict(zip(LETTERS, [ids[0] for ids in letter_ids])),
                  first_letter_logits=dict(zip(LETTERS, logits.cpu().tolist())),
                  first_letter_probs=dict(zip(LETTERS, probs.cpu().tolist())),
                  probability_protocol='softmax over bare A/B/C/D raw first-position logits; conditional on these four tokens',
                  generated_token_count=int(tail.shape[1]),
                  hit_token_ceiling=int(tail.shape[1]) >= args.max_new_tokens,
                  input_tokens=input_length,
                  vision_tokens=int((inputs['input_ids'] == processor.image_token_id).sum().item()),
                  item_elapsed_seconds=perf_counter() - started,
                  peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    if condition == 'image':
        grid = inputs['image_grid_thw'][0].cpu().tolist()
        patch = processor.image_processor.patch_size
        grid_tokens = int(grid[0] * grid[1] * grid[2]) // processor.image_processor.merge_size ** 2
        if grid_tokens != result['vision_tokens']:
            raise AssertionError('Realised image-token count differs from processed image grid')
        result['image_budget'] = dict(source_size=list(image.size), grid_thw=grid,
                                     resolved_size=[int(grid[2] * patch), int(grid[1] * patch)],
                                     realised_vision_tokens=result['vision_tokens'])
    return result


def run_identity(args):
    files = [Path(__file__), ROOT / 'scripts/synth_mcq_common.py', OBQA / 'prompt_common.py',
             OBQA / 'run_zero_shot_qwen.py', ROOT / 'scripts/generate_graphvis_datasets.py',
             ROOT / 'scripts/llava_adapter.py']
    return dict(protocol='synth_mcq_v4', dataset_path=str(args.dataset_path),
                dataset_sha256=sha256_file(args.dataset_path), input_sha256=sha256_file(args.graphs),
                prompt_styles=args.prompt_styles, wrong_graph_modality=args.wrong_graph_modality,
                image_audit_sha256=sha256_file(args.image_audit) if args.image_audit else None,
                render_style=args.render_style,
                prompt_style=args.prompt_style, render_fontsize=args.render_fontsize, parser_protocol=PARSER_PROTOCOL,
                model_id=MODEL_ID, model_revision=REVISION, seed=args.seed,
                n_perms=args.n_perms, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                max_new_tokens=args.max_new_tokens, batch_size=1, dtype='bfloat16',
                generation=dict(do_sample=False, num_beams=1),
                triple_order='sha256(seed:graph_id)-seeded shuffle, shared by DOT and text',
                model_loading='two sequential pristine loads; tuned restores LoRA and all four mergers',
                code_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in files})


def print_vision_summary(rows, config, config_path):
    """Summarize realised counts from image rows, including resumed rows."""
    values = [r['vision_tokens'] for r in rows if r['condition'] in ('image', 'wrong_image')]
    summary = dict(n_image_rows=len(values), n_images=len({r.get('shown_graph_id', r['graph_id']) for r in rows if r['condition'] in ('image', 'wrong_image')}),
                   min=min(values) if values else None, median=median(values) if values else None,
                   max=max(values) if values else None)
    config['vision_token_summary'] = summary
    write_json(config_path, config)
    print('REALISED VISION TOKENS (image rows, includes resume): ' + json.dumps(summary), flush=True)


def validate_image_gate(args, manifest_sha256):
    if args.image_audit is None:
        raise ValueError('Name-grounded inference requires --image-audit legibility_gate.json from the CPU Qwen audit')
    gate = json.loads(args.image_audit.read_text())
    expected = dict(dataset_sha256=sha256_file(args.dataset_path), render_manifest_sha256=manifest_sha256,
                    min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, n_images=50,
                    render_style=args.render_style, node_fontsize=args.render_fontsize,
                    edge_fontsize=24 if args.render_style == 'training' else 14, threshold_px=10)
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError('Image-audit gate differs from dataset/renders/style/fonts/pixel budget')
    if gate.get('passed') is not True or gate.get('nodes_below_10px') != 0 or gate.get('edges_below_10px') != 0:
        raise ValueError('STOP: image audit failed the 10 px gate; report results and wait for the user decision')


def main(argv=None):
    args = parse_args(argv)
    graphs = load_graphs(args.graphs)
    stats = dataset_stats(graphs)
    grounded = graphs[0]['dataset'] == 'name_grounded_50'
    if not grounded and args.conditions_default:
        args.conditions = ['image', 'text', 'none']
    if not grounded and any(c.startswith('wrong_') for c in args.conditions):
        raise ValueError('Wrong-graph controls require the name-grounded dataset')
    solver = solver_check(graphs)
    pairing = wrong_graph_pairing(graphs, args.seed) if grounded else None
    print('VALIDATED ' + json.dumps(stats, sort_keys=True), flush=True)
    print('INDEPENDENT SOLVER ' + json.dumps({k: v for k, v in solver.items() if k != 'questions'}), flush=True)
    if pairing: print('WRONG GRAPH DERANGEMENT ' + json.dumps(pairing, sort_keys=True), flush=True)
    selected = graphs[:args.limit] if args.limit else graphs
    items = evaluation_items(selected, args.models, args.conditions, args.prompt_styles, args.n_perms)
    expected = len(items)
    print(f'EXPECTED ROWS: {len(selected)} graphs / {sum(len(questions_for(g)) for g in selected)} questions x {args.n_perms} perms x {len(args.models)} models x {len(args.conditions)} conditions x {len(args.prompt_styles)} styles = {expected}', flush=True)
    if args.validate_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'solver_check.json', solver)
    if args.render_only:
        render_graphs(selected, args.graphs, args.out, args.seed, args.render_fontsize, args.render_style)
        return
    manifest = verify_renders(graphs if pairing else selected, args.graphs, args.out, args.seed, args.render_fontsize, args.render_style)
    for condition, body in manifest['graphs'][selected[0]['id']]['prompt_bodies']['0'].items():
        body = styled_prompt_body(body, condition, args.prompt_style)
        print(f'PROMPT PREVIEW [{args.prompt_style}/{condition}]\n{body}\nEND PROMPT', flush=True)
    path = args.out / 'predictions.jsonl'
    if path.exists() and path.stat().st_size and not args.resume:
        raise FileExistsError('predictions.jsonl exists; use --resume or a new OUT')
    rows = read_predictions(path) if args.resume else []
    graphs_by_id = {g['id']: g for g in graphs}
    questions_by_id = {q['id']: q for g in graphs for q in questions_for(g)}
    for row in rows:
        graph = graphs_by_id.get(row['graph_id'])
        if (graph is None or row['model'] not in ('base', 'tuned')
                or row.get('prompt_style') not in args.prompt_styles
                or row.get('render_fontsize') != args.render_fontsize
                or row.get('render_style') != args.render_style
                or row['condition'] not in ('image', 'text', 'none', 'wrong_image', 'wrong_text')
                or row.get('question_id') not in {q['id'] for q in questions_for(graph)}
                or type(row['perm']) is not int or row['perm'] not in range(args.n_perms)):
            raise ValueError(f'Existing prediction has invalid identity: {row_key(row)}')
        question = questions_by_id[row['question_id']]
        choices = permuted_choices(graph, row['perm'], question)
        shown = pairing['mapping'][graph['id']] if row['condition'].startswith('wrong_') and pairing else graph['id']
        if row.get('shown_graph_id', graph['id']) != shown:
            raise ValueError('Existing prediction has invalid shown graph')
        predicted_node = choices.get(row['predicted_letter'])
        if (row['choices'] != choices or row['correct_node'] != question['correct_node']
                or row['predicted_node'] != predicted_node
                or row['is_correct'] != (predicted_node == question['correct_node'])):
            raise ValueError(f'Existing prediction has invalid mapping/score: {row_key(row)}')
    done = {row_key(row) for row in rows}
    config_path = args.out / 'config.json'
    identity = run_identity(args)
    identity['render_manifest_sha256'] = sha256_file(args.out / 'render_manifest.json')
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config['identity'] != identity:
            raise ValueError('Incompatible config.json: dataset, renders, code or experimental settings changed')
    elif rows:
        raise ValueError('Predictions exist without config.json; cannot safely resume')
    else:
        config = dict(identity=identity, prompt_style=args.prompt_style, render_fontsize=args.render_fontsize,
                      render_style=args.render_style, prompt_styles=args.prompt_styles,
                      dataset_path=str(args.dataset_path), dataset_sha256=identity['dataset_sha256'],
                      dataset_stats=stats, solver_check=solver, wrong_graph_pairing=pairing, model_loads={})
        write_json(config_path, config)
    if 'tuned' in args.models and any(row['model'] == 'tuned' for row in rows):
        from llava_adapter import adapter_provenance
        provenance = adapter_provenance(args.adapter, args.model_id, args.revision)
        previous = config['model_loads'].get('tuned', {}).get('adapter')
        if not previous or previous['sha256'] != provenance['sha256']:
            raise ValueError('Tuned checkpoint differs from existing predictions')
    pending = [key for key in items if key not in done]
    print(f'Rows pending in this invocation: {len(pending)} / {expected}', flush=True)
    if not pending:
        print_vision_summary(rows, config, config_path)
        return
    if grounded:
        validate_image_gate(args, identity['render_manifest_sha256'])
    import torch
    import transformers
    if transformers.__version__ != '5.16.1':
        raise RuntimeError(f'Require repo-pinned transformers==5.16.1, got {transformers.__version__}; no silent version changes')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting bf16 is required; CPU inference is not permitted')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from run_zero_shot_qwen import load_model, load_processor
    transformers.set_seed(args.seed)
    logger = None
    if args.mlflow_uri:
        from llava_training_logging import TrainingLogger
        logger = TrainingLogger(args.out / 'logging', identity, args.mlflow_uri,
                                resume_step=len(rows) if args.resume else None)
    first_key = (args.prompt_styles[0], selected[0]['id'], questions_for(selected[0])[0]['id'], args.conditions[0], 0)
    probe = {r['model']: r['first_letter_logits'] for r in rows
             if (r['prompt_style'], r['graph_id'], r['question_id'], r['condition'], r['perm']) == first_key}
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row['prompt_style'], row['model'], row['condition']].append(row)
    try:
        # Always put the pristine run first, regardless of CLI ordering.
        for name in ('base', 'tuned'):
            if name not in args.models or not any(key[1] == name for key in pending):
                continue
            model = processor = None
            try:
                started = perf_counter()
                processor, budget_api = load_processor(AutoProcessor, args)
                model, dtype_argument, dtype = load_model(Qwen3VLForConditionalGeneration, torch, args)
                provenance = None
                if name == 'tuned':
                    from llava_adapter import load_adapter, adapter_provenance
                    # Validate identity BEFORE appending resumed tuned predictions.
                    provenance = adapter_provenance(args.adapter, args.model_id, args.revision)
                    previous = config['model_loads'].get(name, {}).get('adapter')
                    if previous and previous['sha256'] != provenance['sha256']:
                        raise ValueError('Tuned checkpoint differs from existing predictions')
                    model, provenance = load_adapter(model, args.adapter, args.model_id, args.revision)
                sanity = load_sanity(model, name, args.adapter if name == 'tuned' else None)
                load_info = dict(sanity=sanity, adapter=provenance, dtype=dtype,
                                 dtype_argument=dtype_argument, processor_budget_api=budget_api,
                                 loading_seconds=perf_counter() - started,
                                 transformers_version=transformers.__version__,
                                 torch_version=torch.__version__)
                config['model_loads'][name] = load_info
                write_json(config_path, config)
                print(f'LOAD SANITY [{name}] ' + json.dumps(load_info, sort_keys=True), flush=True)
                with path.open('a', encoding='utf-8') as output:
                    for style, _, condition, gid, qid, permutation in [key for key in items if key[1] == name]:
                        key = style, name, condition, gid, qid, permutation
                        if key in done:
                            continue
                        graph, question = graphs_by_id[gid], questions_by_id[qid]
                        shown_id = pairing['mapping'][gid] if condition.startswith('wrong_') else gid
                        shown_graph = graphs_by_id[shown_id]
                        choices = permuted_choices(graph, permutation, question)
                        body, modality = evidence_prompt(manifest, graph, question, permutation, condition, style, shown_graph)
                        result = infer_one(model, processor, body, modality,
                            args.out / 'graphs' / f'{shown_id}.png', args, torch)
                        predicted = choices.get(result['predicted_letter'])
                        row = dict(model=name, condition=condition, graph_id=gid, prompt_style=style,
                            render_fontsize=args.render_fontsize, render_style=args.render_style,
                            question_id=qid, question_ids=[q['id'] for q in questions_for(graph)],
                            questions_per_graph=len(questions_for(graph)), dataset=graph['dataset'],
                            question_type=question_type(question), shown_graph_id=shown_id,
                            wrong_graph_modality=modality if condition.startswith('wrong_') else None,
                            reasoning_hops=question.get('reasoning_hops'), query_style=question.get('query_style'),
                            perm=permutation, permutation_original_letters=[LETTERS[(i + permutation) % 4] for i in range(4)],
                            choices=choices, correct_node=question['correct_node'],
                            correct_letter=next(l for l in LETTERS if choices[l] == question['correct_node']),
                            predicted_node=predicted, is_correct=predicted == question['correct_node'], model_revision=REVISION,
                            **logit_scores(result['first_letter_logits'], choices, question['correct_node']), **result)
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                        done.add(key); rows.append(row)
                        by_cell[style, name, condition].append(row)
                        print(f'PROGRESS rows={len(done)} requested={expected} {style}/{name}/{condition}/{gid}/{qid}/perm{permutation} shown={shown_id} pred={result["predicted_letter"]} parse_tier={result["parse_tier"]} correct={row["is_correct"]} logit_argmax_correct={row["logit_argmax_is_correct"]} vision_tokens={row["vision_tokens"]} seconds={result["item_elapsed_seconds"]:.3f} peak_VRAM={result["peak_memory_allocated_bytes"]}', flush=True)
                        if (style, gid, qid, condition, permutation) == first_key:
                            probe[name] = result['first_letter_logits']
                            if set(probe) == {'base', 'tuned'}:
                                differences = [abs(probe['base'][l] - probe['tuned'][l]) for l in LETTERS]
                                sanity_probe = dict(graph_id=gid, question_id=qid, condition=condition, perm=permutation,
                                                    base=probe['base'], tuned=probe['tuned'],
                                                    differ=any(d > 1e-6 for d in differences), max_abs_difference=max(differences))
                                config['first_position_sanity'] = sanity_probe
                                write_json(config_path, config)
                                print('BASE/TUNED LOGIT SANITY ' + json.dumps(sanity_probe), flush=True)
                        if logger:
                            cell = by_cell[style, name, condition]
                            graph_means, logit_graph_means = defaultdict(list), defaultdict(list)
                            for r in cell:
                                graph_means[r['graph_id']].append(float(r['is_correct']))
                                logit_graph_means[r['graph_id']].append(float(r['logit_argmax_is_correct']))
                            accuracy = sum(sum(v) / len(v) for v in graph_means.values()) / len(graph_means)
                            logit_accuracy = sum(sum(v) / len(v) for v in logit_graph_means.values()) / len(logit_graph_means)
                            logger.log('evaluation', len(done), {
                                f'{style}_{name}_{condition}_accuracy': accuracy,
                                f'{style}_{name}_{condition}_logit_argmax_accuracy': logit_accuracy,
                                f'{style}_{name}_{condition}_parse_failure_rate': sum(not r['parse_ok'] for r in cell) / len(cell),
                                'peak_vram_bytes': row['peak_memory_allocated_bytes'], 'step_seconds': row['item_elapsed_seconds']})
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                print(f'GPU MEMORY RELEASED [{name}]: allocated_bytes={torch.cuda.memory_allocated()}', flush=True)
                print_vision_summary(rows, config, config_path)
    except BaseException:
        if logger:
            logger.close('FAILED')
        raise
    else:
        if logger:
            logger.close()
    print(f'DONE: requested rows={expected}; matching rows={sum(key in done for key in items)}', flush=True)


if __name__ == '__main__':
    main()
