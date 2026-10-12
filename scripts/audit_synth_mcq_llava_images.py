#!/usr/bin/env python3
"""CPU-only blocking LLaVA audit of existing, checksum-pinned graph renders.

Observe the pinned processor's own anyres resize before padding, rather than
estimating the grid content size. No model weights are loaded or graphs rendered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

from synth_mcq_common import ROOT, DEFAULT_DATASET, load_graphs, sha256_file, write_json
from llava_common import (DEFAULT_MODEL_ID, DEFAULT_REVISION, configure_processor,
                          image_diagnostics, prepare_inputs)
from llava_stage1_prompt import format_llava_prompt

DATASET_SHA256 = '9e0742bc624e1fb96a97e543fb6d2283cd3932ded03f9bfc084596dd541a9da2'
GRAPHS_SHA256 = '8e311840a4765884b648bef45aa31b760a3f70b19f0eab6e4b39b5b0507e48c7'


def graph_checksum(folder):
    """Exactly sha256sum G*.png G*.dot G*.txt | sha256sum, with C ordering."""
    folder = Path(folder)
    names = [p.name for suffix in ('png', 'dot', 'txt') for p in sorted(folder.glob(f'G*.{suffix}'))]
    expected = [f'G{i:02d}.{suffix}' for suffix in ('png', 'dot', 'txt') for i in range(1, 51)]
    if names != expected:
        raise ValueError('Expected exactly G01-G50 PNG/DOT/TXT files')
    listing = ''.join(f'{sha256_file(folder / name)}  {name}\n' for name in names)
    return hashlib.sha256(listing.encode()).hexdigest()


def summary(values):
    if not values:
        raise ValueError('Cannot summarize an empty image set')
    return dict(min=min(values), median=median(values), max=max(values),
                below_10px=sum(value < 10 for value in values), n_images=len(values))


def measure_image(processor, image, body, references):
    """Intercept the actual resize return shape while running normal preprocessing."""
    owner = processor.image_processor
    observed = []
    original = owner._resize_for_patching
    had_instance_method = '_resize_for_patching' in vars(owner)

    def observe(*args, **kwargs):
        resized = original(*args, **kwargs)
        observed.append([int(n) for n in resized.shape[-2:]])
        return resized

    owner._resize_for_patching = observe
    try:
        prompt = format_llava_prompt(processor, body, 'image', 'llava_v1')
        inputs = prepare_inputs(processor, prompt, image)
    finally:
        if had_instance_method:
            owner._resize_for_patching = original
        else:
            del owner._resize_for_patching
    if len(observed) != 1:
        raise AssertionError(f'Expected one anyres content resize, got {observed}')
    if any(value.device.type != 'cpu' for value in inputs.values() if hasattr(value, 'device')):
        raise AssertionError('Audit must run on CPU')
    diagnostics = image_diagnostics(processor, inputs, image.size)
    resized_height, resized_width = observed[0]
    # The existing processor resizes the separate base view directly to its
    # configured square size. Measure its vertical scale separately from anyres.
    base_height = owner.size.height or owner.size.shortest_edge
    base_width = owner.size.width or owner.size.shortest_edge
    row = dict(**diagnostics, resized_width=resized_width, resized_height=resized_height,
               base_width=int(base_width), base_height=int(base_height))
    for kind in ('node', 'edge'):
        cap = references[kind]['reference_cap_height_px']
        row[f'{kind}_cap_px_grid'] = cap * resized_height / image.height
        row[f'{kind}_cap_px_base'] = cap * base_height / image.height
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    assets = ROOT / 'outputs/synth_mcq_name_grounded_2026-10-10'
    parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--renders', type=Path, default=assets)
    parser.add_argument('--font-references', type=Path, default=assets / 'audit/font_references/font_references.json')
    parser.add_argument('--qwen-config', type=Path, default=ROOT / 'outputs/synth_mcq_name_grounded_run_2026-10-10/full/config.json')
    parser.add_argument('--pairing', type=Path, default=assets / 'wrong_graph_pairing.json')
    parser.add_argument('--training-root', type=Path, default=ROOT / 'outputs/llava_stage1_training_2026-10-06')
    parser.add_argument('--training-sample', type=int, default=50,
                        help='First N distinct training images in statement-index order; descriptive reference')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.training_sample < 1:
        parser.error('--training-sample must be positive')
    if sha256_file(args.dataset_path) != DATASET_SHA256:
        raise ValueError('Dataset checksum differs from the fixed Qwen experiment')
    graphs = load_graphs(args.dataset_path)
    checksum = graph_checksum(args.renders / 'graphs')
    if checksum != GRAPHS_SHA256:
        raise ValueError('Graph checksum differs; do not re-render')
    pairing = json.loads(args.pairing.read_text())
    qwen = json.loads(args.qwen_config.read_text())
    if pairing != qwen['wrong_graph_pairing']:
        raise ValueError('Pairing differs from the actual Qwen run')
    references = json.loads(args.font_references.read_text())['training']
    from PIL import Image
    for kind, size, family in [('node', 30, 'Helvetica-Bold'), ('edge', 24, 'Helvetica')]:
        ref = references[kind]
        if (ref['fontsize'], ref['dpi'], ref['fontname']) != (size, 200, family):
            raise ValueError('Font references differ from training-style renders')
        glyph_path = Path(ref['glyph_png'])
        if sha256_file(glyph_path) != ref['glyph_sha256']:
            raise ValueError('Measured reference glyph hash differs')
        with Image.open(glyph_path) as glyph:
            bounds = glyph.convert('L').point(lambda value: 255 if value < 128 else 0).getbbox()
        if not bounds or bounds[3] - bounds[1] != ref['reference_cap_height_px']:
            raise ValueError('Reference glyph ink measurement differs')
    manifest = json.loads((args.renders / 'render_manifest.json').read_text())
    train_manifest = json.loads((args.training_root / 'data_manifest.json').read_text())
    meta_path = args.training_root / 'train/graph_metadata_training.jsonl'
    if sha256_file(meta_path) != train_manifest['files_sha256'][str(meta_path.relative_to(args.training_root))]:
        raise ValueError('Training metadata differs from the recorded training input')
    metadata = sorted([json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()],
                      key=lambda row: row['statement_idx'])
    if len({row['image'] for row in metadata}) != len(metadata):
        raise ValueError('Training reference must contain distinct images')
    selected = metadata[:args.training_sample]
    if len(selected) != args.training_sample:
        raise ValueError('Training sample exceeds available images')
    items = [('synthetic', g['id'], args.renders / 'graphs' / f"{g['id']}.png",
              manifest['graphs'][g['id']]['prompt_bodies']['0']['image']) for g in graphs]
    for row in selected:
        path = args.training_root / row['image']
        if sha256_file(path) != train_manifest['files_sha256'][row['image']]:
            raise ValueError('Training image differs from the recorded training input')
        items.append(('stage1_training_sample', str(row['statement_idx']), path, '<image>\nDescribe this graph.'))
    import torch
    import transformers
    from transformers import LlavaNextConfig, LlavaNextProcessor
    if transformers.__version__ != '5.16.1':
        raise ValueError('Require pinned transformers==5.16.1; no version changes')
    torch.set_num_threads(1)
    processor = LlavaNextProcessor.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True)
    model_config = LlavaNextConfig.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True)
    expansion = configure_processor(processor, model_config)
    tokens = {letter: {variant: processor.tokenizer.convert_tokens_to_ids(variant)
                       for variant in (letter, '▁' + letter)} for letter in 'ABCD'}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, gid, path, body in items:
        with Image.open(path) as source:
            image = source.convert('RGB')
        row = dict(dataset=dataset, graph_id=gid, image=str(path), image_sha256=sha256_file(path),
                   **measure_image(processor, image, body, references))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (args.out / 'images.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out / 'images.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summaries = {dataset: {f'{kind}_{view}': summary([row[f'{kind}_cap_px_{view}'] for row in rows if row['dataset'] == dataset])
                           for kind in ('node', 'edge') for view in ('grid', 'base')}
                 for dataset in ('synthetic', 'stage1_training_sample')}
    synthetic = [row for row in rows if row['dataset'] == 'synthetic']
    gate = dict(backend='llava', model_id=DEFAULT_MODEL_ID, model_revision=DEFAULT_REVISION,
                dataset_sha256=DATASET_SHA256, graphs_sha256=checksum, pairing_sha256=sha256_file(args.pairing),
                render_manifest_sha256=sha256_file(args.renders / 'render_manifest.json'),
                image_grid_pinpoints=processor.image_processor.image_grid_pinpoints,
                threshold_px=10, n_images=len(synthetic),
                nodes_below_10px=summaries['synthetic']['node_grid']['below_10px'],
                edges_below_10px=summaries['synthetic']['edge_grid']['below_10px'],
                failing_graph_count=sum(row['node_cap_px_grid'] < 10 or row['edge_cap_px_grid'] < 10 for row in synthetic))
    gate['passed'] = gate['failing_graph_count'] == 0
    write_json(args.out / 'legibility_gate.json', gate)
    write_json(args.out / 'audit_config.json', dict(gate=gate, summaries=summaries, letter_token_variants=tokens,
        no_model_loaded=True, transformers_version=transformers.__version__, torch_version=torch.__version__,
        processor_class=type(processor.image_processor).__name__, processor_expansion=expansion,
        method='Measured own H glyphs times observed processor anyres content-height / source-height; base thumbnail information only',
        training_sample_selection='First N distinct training images in ascending statement-index order; descriptive only',
        training_sample_indices=[row['statement_idx'] for row in selected],
        font_references_sha256=sha256_file(args.font_references),
        training_manifest_sha256=sha256_file(args.training_root / 'data_manifest.json')))
    lines = ['# LLaVA synthetic audit (CPU only)', '', f"Gate: {'PASS' if gate['passed'] else 'FAIL'}. Nodes below 10 px: {gate['nodes_below_10px']}/50. Edges: {gate['edges_below_10px']}/50.",
             '', '| Dataset | Font/view | Min px | Median px | Max px | Below 10 px |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for dataset, measurements in summaries.items():
        for kind, stats in measurements.items():
            lines.append(f"| {dataset} | {kind} | {stats['min']:.3f} | {stats['median']:.3f} | {stats['max']:.3f} | {stats['below_10px']}/{stats['n_images']} |")
    lines += ['', 'Grid is height × width; resize is unpadded content width × height. Base view is information only.', '',
              '| Graph | Grid | Resized W×H | Tokens | Node cap px | Edge cap px | Base node px | Base edge px |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in synthetic:
        grid = row['selected_grid_pinpoint']
        lines.append(f"| {row['graph_id']} | {grid['height']}×{grid['width']} | {row['resized_width']}×{row['resized_height']} | {row['realised_image_tokens']} | {row['node_cap_px_grid']:.3f} | {row['edge_cap_px_grid']:.3f} | {row['node_cap_px_base']:.3f} | {row['edge_cap_px_base']:.3f} |")
    (args.out / 'image_audit.md').write_text('\n'.join(lines) + '\n')
    print('SUMMARIES ' + json.dumps(summaries), flush=True)
    print('LEGIBILITY GATE ' + json.dumps(gate), flush=True)
    if not gate['passed']:
        print('STOP: fixed LLaVA renders fail the 10 px gate; report and await the user decision. Renders unchanged.', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
