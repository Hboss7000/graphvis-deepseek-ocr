#!/usr/bin/env python3
"""Compare cap-height proxies or actual pinned Qwen CPU preprocessing for both sets."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from synth_mcq_common import (ROOT, MODEL_ID, REVISION, MIN_PIXELS, MAX_PIXELS,
                             load_graphs, graph_metadata, sha256_file, write_json,
                             render_style, verify_renders)


def measure_font_references(out):
    """Measure each font's own H ink bounds on this rendering host, independently."""
    from io import BytesIO
    import graphviz
    from PIL import Image
    from image_audit import reference_cap_height
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    references = {}
    for name in ('training', 'compact', 'obqa_stage2'):
        style = render_style(style='compact' if name == 'compact' else 'training')
        references[name] = {}
        for kind, family in [('node', 'Helvetica-Bold'), ('edge', 'Helvetica')]:
            size, dpi = style[f'{kind}_fontsize'], style['dpi']
            dot = graphviz.Digraph(format='png')
            dot.attr('graph', dpi=str(dpi), pad='0', margin='0', bgcolor='white')
            dot.attr('node', shape='plaintext', margin='0', fontname=family, fontsize=str(size))
            dot.node('reference', 'H')
            png = dot.pipe()
            glyph_path = out / f'{name}_{kind}_H.png'
            glyph_path.write_bytes(png)
            with Image.open(BytesIO(png)) as image:
                bounds = image.convert('L').point(lambda value: 255 if value < 128 else 0).getbbox()
            if bounds is None:
                raise RuntimeError('Reference glyph contains no measurable ink')
            cap = bounds[3] - bounds[1]
            if kind == 'node' and cap != reference_cap_height(size, dpi):
                raise AssertionError('Node glyph differs from unchanged image_audit measurement')
            references[name][kind] = dict(fontname=family, fontsize=size, dpi=dpi,
                reference_cap_height_px=cap, glyph='H', threshold=128, bounds=list(bounds),
                glyph_png=str(glyph_path.resolve()), glyph_sha256=sha256_file(glyph_path))
    write_json(out / 'font_references.json', references)
    print('MEASURED FONT REFERENCES ' + json.dumps(references), flush=True)
    return references


def audit_comparison(graphs_path, renders, obqa_input, obqa_metadata, obqa_image_root, out,
                     obqa_node_fontsize=30, obqa_dpi=200):
    from image_audit import audit_split
    renders, out = Path(renders), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = load_graphs(graphs_path)
    synthetic_metadata = []
    for idx, graph in enumerate(graphs):
        meta = graph_metadata(graph, graph['triples'])[2]
        synthetic_metadata.append(dict(statement_idx=idx, image=f"graphs/{graph['id']}.png", **meta))
    synth_meta_path = out / 'synthetic_graph_metadata.jsonl'
    synth_meta_path.write_text(''.join(json.dumps(r) + '\n' for r in synthetic_metadata))
    stage2 = [json.loads(line) for line in Path(obqa_input).read_text().splitlines() if line.strip()]
    metadata = [json.loads(line) for line in Path(obqa_metadata).read_text().splitlines() if line.strip()]
    ids = {r['statement_idx']: r for r in metadata}
    if len(ids) != len(metadata) or len({r['statement_idx'] for r in stage2}) != len(stage2):
        raise ValueError('Duplicate OBQA metadata/input identities')
    # Audit exactly the Stage 2 images, without inferring its population from a directory glob.
    matched = []
    for row in stage2:
        meta = ids[row['statement_idx']]
        if meta['image'] != row['image']:
            raise ValueError('OBQA Stage 2 input/metadata image mismatch')
        matched.append(meta)
    matched_path = out / 'obqa_stage2_graph_metadata.jsonl'
    matched_path.write_text(''.join(json.dumps(r) + '\n' for r in matched))
    style = json.loads((renders / 'render_manifest.json').read_text())['style']
    reports = {
        'synthetic': audit_split(synth_meta_path, renders, node_fontsize=style['node_fontsize'], dpi=style['dpi']),
        'obqa_stage2': audit_split(matched_path, obqa_image_root, node_fontsize=obqa_node_fontsize, dpi=obqa_dpi)}
    summaries = []
    for name, report in reports.items():
        write_json(out / f'{name}_image_audit.json', report)
        for method, column in [('fit_896', 'estimated_label_cap_px_896'),
                               ('llava_anyres', 'estimated_label_cap_px_llava_anyres')]:
            values = [row[column] for row in report['images']]
            summary = dict(dataset=name, method=method, n_images=len(values),
                           node_fontsize=report['node_fontsize'], dpi=report['dpi'],
                           reference_cap_height_px=report['reference_cap_height_px'],
                           min=min(values), median=median(values), max=max(values),
                           below_10px=sum(value < 10 for value in values))
            summaries.append(summary)
            print(f"CAP HEIGHT {name}/{method}: n={len(values)} min/median/max={summary['min']:.3f}/{summary['median']:.3f}/{summary['max']:.3f} px; below10={summary['below_10px']}")
    with (out / 'cap_height_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    write_json(out / 'comparison_config.json', dict(
        method='Unmodified image_audit.audit_split: measured Helvetica-Bold H at each dataset render font/DPI; graph_scale=1',
        synthetic_node_fontsize=style['node_fontsize'], synthetic_dpi=style['dpi'],
        obqa_node_fontsize=obqa_node_fontsize, obqa_dpi=obqa_dpi,
        interpretation='896 fit and LLaVA anyres are geometric proxy estimates, not Qwen realised token/resizing measurements or measured readability',
        source_sha256={str(p): sha256_file(p) for p in map(Path, [graphs_path, obqa_input, obqa_metadata])},
        image_audit_source_sha256=sha256_file(Path(__file__).with_name('image_audit.py')),
        renders=str(renders.resolve()), obqa_image_root=str(Path(obqa_image_root).resolve())))
    return summaries


def qwen_image_measurement(processor, image, prompt_body, args, reference_cap, edge_reference_cap=None):
    """Execute real CPU image preprocessing and count the expanded native vision IDs."""
    from run_zero_shot_qwen import format_qwen_prompt, prepare_inputs
    inputs = prepare_inputs(processor, format_qwen_prompt(processor, prompt_body, 'image'), image, args)
    if any(value.device.type != 'cpu' for value in inputs.values() if hasattr(value, 'device')):
        raise AssertionError('Audit must use CPU tensors only')
    grid = [int(n) for n in inputs['image_grid_thw'][0].tolist()]
    patch, merge = processor.image_processor.patch_size, processor.image_processor.merge_size
    tokens = int((inputs['input_ids'] == processor.image_token_id).sum().item())
    if tokens != grid[0] * grid[1] * grid[2] // merge ** 2:
        raise AssertionError('Expanded vision tokens differ from processed grid')
    width, height = image.size
    resized_width, resized_height = grid[2] * patch, grid[1] * patch
    # Patch rounding can produce slightly different x/y scales. Capitals scale vertically.
    row = dict(width=width, height=height, resized_width=resized_width, resized_height=resized_height,
                image_grid_thw=grid, realised_image_tokens=tokens,
                scale_x=resized_width / width, scale_y=resized_height / height,
                estimated_label_cap_px_qwen=reference_cap * resized_height / height)
    row['estimated_node_cap_px_qwen'] = row['estimated_label_cap_px_qwen']
    if edge_reference_cap is not None:
        row['estimated_edge_cap_px_qwen'] = edge_reference_cap * resized_height / height
    return row


def audit_qwen_processor(graphs_path, renders, obqa_input, obqa_image_root, out, processor_assets,
                         font_references, obqa_qwen_config, compact_renders=None):
    import torch
    import transformers
    from transformers import AutoProcessor
    from PIL import Image
    from run_zero_shot_qwen import load_processor
    if transformers.__version__ != '5.16.1':
        raise ValueError('Require repo-pinned transformers==5.16.1')
    torch.set_num_threads(1)
    renders, out = Path(renders), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(obqa_qwen_config).read_text())
    if (baseline['model_id'], baseline['model_revision'], baseline['min_pixels'], baseline['max_pixels']) != (
            MODEL_ID, REVISION, MIN_PIXELS, MAX_PIXELS):
        raise ValueError('OBQA Qwen config differs from pinned model/pixel settings')
    args = SimpleNamespace(model_id=str(processor_assets), revision=REVISION,
                           min_pixels=baseline['min_pixels'], max_pixels=baseline['max_pixels'],
                           local_files_only=True)
    processor, budget_api = load_processor(AutoProcessor, args)
    graphs = load_graphs(graphs_path)
    records = [json.loads(x) for x in Path(obqa_input).read_text().splitlines() if x.strip()] if obqa_input else []
    references = json.loads(Path(font_references).read_text())
    flags = baseline['source_image_generation']['render']['flags']
    obqa_font = int(flags[flags.index('--node-fontsize') + 1])
    obqa_dpi = int(flags[flags.index('--dpi') + 1]) if '--dpi' in flags else 200
    obqa_edge_font = int(flags[flags.index('--edge-fontsize') + 1])
    styles = {'obqa_stage2': dict(node_fontsize=obqa_font, edge_fontsize=obqa_edge_font, dpi=obqa_dpi)}
    reference_names = {'obqa_stage2': 'obqa_stage2'}
    summaries, all_rows = [], []
    datasets, manifest_hashes, layouts = {}, {}, {}
    for variant, folder in [('training', renders), ('compact', compact_renders)]:
        if folder is None:
            continue
        folder = Path(folder)
        raw_manifest = json.loads((folder / 'render_manifest.json').read_text())
        manifest = verify_renders(graphs, graphs_path, folder, raw_manifest['seed'],
                                  raw_manifest['style']['node_fontsize'], variant)
        name = f'synthetic_{variant}'
        styles[name], reference_names[name] = manifest['style'], variant
        layouts[name] = {g['id']: dict(engine=manifest['style']['engine'],
                           rankdir_selected=manifest['graphs'][g['id']]['rankdir_selected']) for g in graphs}
        manifest_hashes[str(folder / 'render_manifest.json')] = sha256_file(folder / 'render_manifest.json')
        datasets[name] = [(g['id'], folder / 'graphs' / f"{g['id']}.png",
                          manifest['graphs'][g['id']]['prompt_bodies']['0']['image']) for g in graphs]
    if records:
        datasets['obqa_stage2'] = [(r['statement_idx'], Path(obqa_image_root) / r['image'], '<image>\n' + r['prompt'])
                                  for r in records]
    for name, items in datasets.items():
        rows = []
        refs = references[reference_names[name]]
        for kind, family in [('node', 'Helvetica-Bold'), ('edge', 'Helvetica')]:
            if (refs[kind]['fontsize'], refs[kind]['dpi'], refs[kind]['fontname']) != (
                    styles[name][f'{kind}_fontsize'], styles[name]['dpi'], family):
                raise ValueError(f'{name}/{kind} reference font/DPI differs from render provenance')
        reference, edge_reference = [refs[k]['reference_cap_height_px'] for k in ('node', 'edge')]
        with (out / f'{name}_qwen_processor.jsonl').open('w') as output:
            for number, (identity, path, body) in enumerate(items, 1):
                with Image.open(path) as opened:
                    image = opened.convert('RGB')
                measured = qwen_image_measurement(processor, image, body, args, reference, edge_reference)
                row = dict(dataset=name, render_style=reference_names[name], image_id=identity, image=str(path),
                           node_fontsize=styles[name]['node_fontsize'], edge_fontsize=styles[name]['edge_fontsize'],
                           dpi=styles[name]['dpi'], reference_node_cap_height_px=reference,
                           reference_edge_cap_height_px=edge_reference,
                           **layouts.get(name, {}).get(identity, dict(engine='dot', rankdir_selected=None)), **measured)
                rows.append(row)
                output.write(json.dumps(row) + '\n'); output.flush()
                if name.startswith('synthetic') or number % 25 == 0 or number == len(items):
                    print(f"QWEN CPU {name} {number}/{len(items)} {identity}: {measured['resized_width']}x{measured['resized_height']} tokens={measured['realised_image_tokens']} node_cap={measured['estimated_node_cap_px_qwen']:.3f}px edge_cap={measured['estimated_edge_cap_px_qwen']:.3f}px", flush=True)
        all_rows.extend(rows)
        with (out / f'{name}_qwen_processor.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
        tokens = [r['realised_image_tokens'] for r in rows]
        for kind in ('node', 'edge'):
            caps = [r[f'estimated_{kind}_cap_px_qwen'] for r in rows]
            summary = dict(dataset=name, label_kind=kind, fontname=refs[kind]['fontname'],
                           fontsize=refs[kind]['fontsize'], reference_cap_height_px=refs[kind]['reference_cap_height_px'],
                           n_images=len(rows), dpi=styles[name]['dpi'], min=min(caps), median=median(caps), max=max(caps),
                           below_10px=sum(v < 10 for v in caps), fraction_below_10px=sum(v < 10 for v in caps) / len(caps),
                           image_tokens_min=min(tokens), image_tokens_median=median(tokens), image_tokens_max=max(tokens))
            summaries.append(summary)
            print('QWEN CAP HEIGHT SUMMARY ' + json.dumps(summary), flush=True)
    if records:
        first = next(r for r in all_rows if r['dataset'] == 'obqa_stage2')
        expected = baseline['first_image_budget']
        if first['image_grid_thw'] != expected['image_grid_thw'] or first['realised_image_tokens'] != expected['image_tokens']:
            raise ValueError('First OBQA realised processor grid/tokens differ from earlier run')
    with (out / 'qwen_cap_height_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    write_json(out / 'qwen_processor_audit_config.json', dict(
        model_id=MODEL_ID, model_revision=REVISION, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        processor_assets=str(Path(processor_assets).resolve()), processor_class=type(processor.image_processor).__name__,
        processor_budget_api=budget_api, transformers_version=transformers.__version__, torch_version=torch.__version__,
        method='Actual CPU processor preprocessing; independently measured Helvetica-Bold node and Helvetica edge H glyphs times realised vertical resize scale',
        no_model_loaded=True, first_obqa_matches_recorded_grid_and_1224_tokens=True if records else None,
        reference_files_sha256={str(p): sha256_file(p) for p in map(Path, [font_references, obqa_qwen_config, graphs_path] + ([obqa_input] if obqa_input else []))},
        render_manifest_sha256=manifest_hashes,
        summaries=summaries))
    primary = [r for r in all_rows if r['dataset'] == 'synthetic_training']
    gate = dict(dataset_sha256=sha256_file(graphs_path),
                render_manifest_sha256=sha256_file(renders / 'render_manifest.json'),
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, render_style='training',
                node_fontsize=styles['synthetic_training']['node_fontsize'],
                edge_fontsize=styles['synthetic_training']['edge_fontsize'],
                n_images=len(primary), threshold_px=10,
                nodes_below_10px=sum(r['estimated_node_cap_px_qwen'] < 10 for r in primary),
                edges_below_10px=sum(r['estimated_edge_cap_px_qwen'] < 10 for r in primary))
    gate['passed'] = bool(primary) and gate['nodes_below_10px'] == gate['edges_below_10px'] == 0
    write_json(out / 'legibility_gate.json', gate)
    print('LEGIBILITY GATE ' + json.dumps(gate), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs', type=Path)
    parser.add_argument('--dataset-path', type=Path)
    parser.add_argument('--renders', type=Path)
    parser.add_argument('--compact-renders', type=Path)
    parser.add_argument('--obqa-input', type=Path)
    parser.add_argument('--obqa-metadata', type=Path)
    parser.add_argument('--obqa-image-root', type=Path)
    parser.add_argument('--obqa-node-fontsize', type=int, default=30,
                        help='Existing fullrun_2026-10-04_B renders use 30pt; supply the real source setting')
    parser.add_argument('--obqa-dpi', type=int, default=200)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qwen-processor-assets', type=Path, help='Switch to actual pinned Qwen CPU preprocessing')
    parser.add_argument('--measure-fonts-only', action='store_true')
    parser.add_argument('--font-references', type=Path)
    parser.add_argument('--synthetic-only', action='store_true', help='Audit only supplied synthetic renders; no OBQA images')
    parser.add_argument('--obqa-qwen-config', type=Path,
                        default=ROOT / 'outputs/fullrun_2026-10-04_B_results/qwen/qa_image/run_config.json')
    args = parser.parse_args()
    if args.measure_fonts_only:
        measure_font_references(args.out)
        return
    if args.graphs and args.dataset_path and args.graphs.resolve() != args.dataset_path.resolve():
        parser.error('--graphs and --dataset-path must resolve to the same path')
    args.graphs = args.dataset_path or args.graphs
    for name in (('graphs', 'renders') if args.synthetic_only else ('graphs', 'renders', 'obqa_input', 'obqa_image_root')):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required for an image audit')
    if args.qwen_processor_assets:
        if not args.font_references:
            parser.error('Qwen audit requires --font-references from --measure-fonts-only on the rendering host')
        audit_qwen_processor(args.graphs, args.renders, args.obqa_input, args.obqa_image_root, args.out,
                             args.qwen_processor_assets, args.font_references, args.obqa_qwen_config, args.compact_renders)
        if load_graphs(args.graphs)[0]['dataset'] == 'name_grounded_50':
            gate = json.loads((args.out / 'legibility_gate.json').read_text())
            if not gate['passed']:
                print('STOP: name-grounded image audit failed the 10 px gate. Fonts/sizes unchanged; decision belongs to the user.', flush=True)
                raise SystemExit(2)
    else:
        if args.synthetic_only:
            parser.error('--synthetic-only requires --qwen-processor-assets')
        if not args.obqa_metadata:
            parser.error('--obqa-metadata is required for the geometric proxy audit')
        audit_comparison(args.graphs, args.renders, args.obqa_input, args.obqa_metadata, args.obqa_image_root, args.out,
                         args.obqa_node_fontsize, args.obqa_dpi)


if __name__ == '__main__':
    main()
