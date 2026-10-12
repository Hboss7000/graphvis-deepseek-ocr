#!/usr/bin/env python3
"""CPU layout/font/shape sweep; original renders and inference remain untouched.

Separate environments: render/png/report in the Graphviz venv;
score/audit in the pinned Transformers CPU venv. Every candidate uses all graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from io import BytesIO
import json
from pathlib import Path
from statistics import median
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

from synth_mcq_common import (ROOT, DEFAULT_DATASET, load_graphs, shuffled_triples,
                             graph_metadata, format_kg_block, sha256_file, write_json)
from audit_synth_mcq_llava_images import DATASET_SHA256, GRAPHS_SHA256, graph_checksum, measure_image

FONTS = ((30, 24), (42, 34), (60, 48), (84, 68))
LAYOUTS = (
    ('dot_lr_standard', 'dot', 'LR', .5, .4),
    ('dot_tb_standard', 'dot', 'TB', .5, .4),
    ('dot_lr_tight', 'dot', 'LR', .2, .15),
    ('dot_tb_tight', 'dot', 'TB', .2, .15),
    ('neato', 'neato', 'LR', .2, .15),
    ('fdp', 'fdp', 'LR', .2, .15),
)
SHAPES = (('natural', None), ('landscape', 1 / 3), ('portrait', 3))


def configurations():
    return [dict(name=f'{layout}_{shape}_n{node}_e{edge}', layout=layout, engine=engine,
                 rankdir=direction, nodesep=nodesep, ranksep=ranksep, shape=shape,
                 graph_ratio=ratio, node_fontsize=node, edge_fontsize=edge, dpi=200)
            for node, edge in FONTS for layout, engine, direction, nodesep, ranksep in LAYOUTS
            for shape, ratio in SHAPES]


def svg_geometry(svg, graph):
    root = ET.fromstring(svg)
    groups = list(root.iter('{http://www.w3.org/2000/svg}g'))
    nodes = [g for g in groups if g.get('class') == 'node']
    edges = [g for g in groups if g.get('class') == 'edge']
    if len(nodes) != len(graph['nodes']) or len(edges) != len(graph['triples']):
        raise ValueError('Layout omitted nodes or triples')
    if any(not list(g.iter('{http://www.w3.org/2000/svg}text')) for g in nodes + edges):
        raise ValueError('Layout omitted a node or relation label')
    # Graphviz at dpi=200 writes SVG outer dimensions equal to its PNG pixel
    # dimensions. Finalist PNGs independently assert this exact equality.
    return dict(width=int(root.attrib['width'].removesuffix('pt')),
                height=int(root.attrib['height'].removesuffix('pt')),
                n_nodes=len(nodes), n_edges=len(edges))


def capture_dot(graph, config):
    """Reuse unchanged production renderer; intercept only its final output call."""
    import graphviz
    import generate_graphvis_datasets as renderer
    sources = []

    class CapturedDigraph(graphviz.Digraph):
        def render(self, filename=None, *args, **kwargs):
            if self.strict:
                raise AssertionError('Strict DOT loses parallel triples')
            self.attr('graph', start='13')  # Fixed seed for force-directed engines.
            sources.append(self.source)
            return str(filename) + '.png'

    triples = shuffled_triples(graph, 0)
    nodes, rendered, meta = graph_metadata(graph, triples)
    with patch.object(renderer, 'graphviz', SimpleNamespace(Digraph=CapturedDigraph)):
        renderer.render_graph(Path(graph['id']), nodes, rendered, '', config['engine'], False,
            dpi=config['dpi'], node_fontsize=config['node_fontsize'], edge_fontsize=config['edge_fontsize'],
            nodesep=config['nodesep'], ranksep=config['ranksep'],
            graph_ratio=str(config['graph_ratio']) if config['graph_ratio'] is not None else None,
            rankdir=config['rankdir'], node_style_mode='uniform', node_fill='#ADD8E6')
    if len(sources) != 1 or 'cluster_no_evidence' in sources[0]:
        raise AssertionError('Unexpected rendering/answer cluster')
    return sources[0], format_kg_block(meta, preserve_order=True) + '\n'


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render_layouts(args):
    import graphviz
    from PIL import Image
    graphs = load_graphs(args.dataset_path)
    if sha256_file(args.dataset_path) != DATASET_SHA256 or graph_checksum(args.original / 'graphs') != GRAPHS_SHA256:
        raise ValueError('Fixed source dataset/original render checksum mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    refs = {}
    for node, edge in FONTS:
        name = f'n{node}_e{edge}'
        refs[name] = {}
        for kind, family, size in [('node', 'Helvetica-Bold', node), ('edge', 'Helvetica', edge)]:
            dot = graphviz.Digraph(format='png')
            dot.attr('graph', dpi='200', pad='0', margin='0', bgcolor='white')
            dot.attr('node', shape='plaintext', margin='0', fontname=family, fontsize=str(size))
            dot.node('H', 'H')
            png = dot.pipe()
            folder = args.out / 'font_references'; folder.mkdir(exist_ok=True)
            path = folder / f'{name}_{kind}.png'; path.write_bytes(png)
            with Image.open(BytesIO(png)) as image:
                bounds = image.convert('L').point(lambda x: 255 if x < 128 else 0).getbbox()
            refs[name][kind] = dict(fontname=family, fontsize=size, dpi=200,
                reference_cap_height_px=bounds[3] - bounds[1], glyph_sha256=sha256_file(path))
    write_json(args.out / 'font_references.json', refs)
    configs = configurations()
    engine_probes = {engine: subprocess.run(['dot', f'-K{engine}', '-Tsvg'], input=b'digraph {}',
                       capture_output=True, timeout=15) for engine in sorted({c['engine'] for c in configs})}
    unavailable = {engine: result.stderr.decode().strip() for engine, result in engine_probes.items() if result.returncode}
    write_json(args.out / 'sweep_config.json', dict(dataset_path=str(args.dataset_path.resolve()),
        dataset_sha256=DATASET_SHA256, original_graphs_sha256=GRAPHS_SHA256,
        configurations=configs, n_graphs=len(graphs), n_configurations=len(configs), unavailable_engines=unavailable,
        graphviz_version=subprocess.check_output(['dot', '-V'], stderr=subprocess.STDOUT).decode().strip(),
        source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in
                      [Path(__file__), ROOT / 'scripts/generate_graphvis_datasets.py']},
        changes='Font sizes; layout engine/orientation/spacing; natural or numeric Graphviz aspect ratio. No pruning, wrapping, label hiding, or graph-size scaling.'))
    jobs = []
    for config in configs:
        if config['engine'] in unavailable:
            continue
        folder = args.out / 'candidates' / config['name']; folder.mkdir(parents=True)
        for graph in graphs:
            source, text = capture_dot(graph, config)
            (folder / f"{graph['id']}.dot").write_text(source)
            (folder / f"{graph['id']}.txt").write_text(text)
            jobs.append((config, graph, folder, source))

    def work(job):
        config, graph, folder, source = job
        result = subprocess.run(['dot', f"-K{config['engine']}", '-Tsvg'], input=source.encode(),
                                capture_output=True, timeout=45, check=True)
        svg = result.stdout.decode()
        geom = svg_geometry(svg, graph)
        (folder / f"{graph['id']}.svg").write_text(svg)
        return dict(configuration=config['name'], graph_id=graph['id'], **geom,
                    graphviz_warnings=result.stderr.decode().strip())

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(work, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(configuration=job[0]['name'], graph_id=job[1]['id'], error=str(error)))
            completed = len(rows) + len(failures)
            if completed % 100 == 0 or completed == len(jobs):
                print(f'CPU SVG layouts {completed}/{len(jobs)}, failures={len(failures)}', flush=True)
    rows.sort(key=lambda r: (r['configuration'], r['graph_id']))
    write_csv(args.out / 'geometry.csv', rows)
    write_json(args.out / 'layout_failures.json', failures)


def load_processor():
    import torch, transformers
    from transformers import LlavaNextConfig, LlavaNextProcessor
    from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION, configure_processor
    if transformers.__version__ != '5.16.1':
        raise ValueError('Require pinned transformers==5.16.1')
    torch.set_num_threads(1)
    processor = LlavaNextProcessor.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True)
    config = LlavaNextConfig.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True)
    configure_processor(processor, config)
    return processor


def score(args):
    from transformers.image_processing_utils import get_patch_output_size, select_best_resolution
    from transformers.image_utils import ChannelDimension
    processor = load_processor()
    config = json.loads((args.out / 'sweep_config.json').read_text())
    configs = {c['name']: c for c in config['configurations']}
    refs = json.loads((args.out / 'font_references.json').read_text())
    reference = json.loads((args.training_audit / 'audit_config.json').read_text())['summaries']['stage1_training_sample']
    targets = dict(node=reference['node_grid']['median'], edge=reference['edge_grid']['median'])
    rows = []
    for geom in csv.DictReader((args.out / 'geometry.csv').open()):
        cfg = configs[geom['configuration']]
        width, height = int(geom['width']), int(geom['height'])
        grid = select_best_resolution((height, width), processor.image_processor.image_grid_pinpoints)
        virtual_image = SimpleNamespace(shape=(3, height, width))
        rh, rw = get_patch_output_size(virtual_image, grid, ChannelDimension.FIRST)
        fonts = refs[f"n{cfg['node_fontsize']}_e{cfg['edge_fontsize']}"]
        row = dict(geom)
        row.update(width=width, height=height, resized_width=rw, resized_height=rh,
                   grid_height=grid[0], grid_width=grid[1])
        for kind in ('node', 'edge'):
            row[f'{kind}_cap_px'] = fonts[kind]['reference_cap_height_px'] * rh / height
        rows.append(row)
    summaries = []
    for name, cfg in configs.items():
        group = [r for r in rows if r['configuration'] == name]
        if len(group) != 50:
            continue
        result = dict(**cfg, n_graphs=len(group))
        for kind in ('node', 'edge'):
            values = [r[f'{kind}_cap_px'] for r in group]
            result.update({f'{kind}_min': min(values), f'{kind}_median': median(values),
                           f'{kind}_max': max(values), f'{kind}_below_10px': sum(v < 10 for v in values)})
        result['graphs_at_both_training_medians'] = sum(r['node_cap_px'] >= targets['node'] and r['edge_cap_px'] >= targets['edge'] for r in group)
        result['both_medians_reached'] = result['node_median'] >= targets['node'] and result['edge_median'] >= targets['edge']
        result['minimum_target_ratio'] = min(result[f'{kind}_min'] / targets[kind] for kind in ('node', 'edge'))
        result['median_target_ratio'] = min(result[f'{kind}_median'] / targets[kind] for kind in ('node', 'edge'))
        result['grid_distribution'] = dict(Counter(f"{r['grid_height']}x{r['grid_width']}" for r in group))
        summaries.append(result)
    summaries.sort(key=lambda r: (-r['graphs_at_both_training_medians'], -r['minimum_target_ratio'], -r['median_target_ratio'], r['node_fontsize'], r['name']))
    # Audit one best candidate from each engine/orientation family, including a
    # dot layout even if force-directed geometry ranks higher.
    matches = [row for row in summaries if row['both_medians_reached']]
    closest = min(matches, key=lambda row: (max(row[f'{k}_median'] / targets[k] for k in ('node', 'edge')),
                                           row['node_fontsize'], row['name'])) if matches else None
    finalists, seen = ([closest['name']] if closest else []), set()
    for row in summaries:
        family = (row['engine'], row['rankdir'] if row['engine'] == 'dot' else None)
        if family not in seen:
            if row['name'] not in finalists:
                finalists.append(row['name'])
            seen.add(family)
    write_csv(args.out / 'screened_images.csv', rows)
    write_csv(args.out / 'screening_comparison.csv', summaries)
    write_json(args.out / 'screening.json', dict(targets=targets, summaries=summaries, finalists=finalists,
        processor_image_grid_pinpoints=processor.image_processor.image_grid_pinpoints,
        method='All 50 graphs per candidate; exact SVG/PNG dimensions and processor-owned get_patch_output_size. Finalists require full CPU preprocessing.'))
    print(json.dumps(dict(targets=targets, finalists=finalists, top=summaries[:8]), indent=2), flush=True)


def render_pngs(args):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 250_000_000
    screening = json.loads((args.out / 'screening.json').read_text())
    configs = {c['name']: c for c in json.loads((args.out / 'sweep_config.json').read_text())['configurations']}
    geometry = {(r['configuration'], r['graph_id']): r for r in csv.DictReader((args.out / 'geometry.csv').open())}
    for name in screening['finalists']:
        folder = args.out / 'candidates' / name
        for i in range(1, 51):
            gid = f'G{i:02d}'
            output = subprocess.run(['dot', f"-K{configs[name]['engine']}", '-Tpng'],
                                    input=(folder / f'{gid}.dot').read_bytes(), capture_output=True, timeout=60, check=True)
            path = folder / f'{gid}.png'; path.write_bytes(output.stdout)
            with Image.open(path) as image:
                expected = geometry[name, gid]
                if image.size != (int(expected['width']), int(expected['height'])):
                    raise AssertionError('PNG differs from projected SVG pixel dimensions')
            if i % 10 == 0:
                print(f'CPU PNG {name}: {i}/50', flush=True)


def audit(args):
    import torch
    import transformers
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 250_000_000
    processor = load_processor()
    screening = json.loads((args.out / 'screening.json').read_text())
    configs = {c['name']: c for c in json.loads((args.out / 'sweep_config.json').read_text())['configurations']}
    refs = json.loads((args.out / 'font_references.json').read_text())
    projected = {(r['configuration'], r['graph_id']): r for r in csv.DictReader((args.out / 'screened_images.csv').open())}
    manifest = json.loads((args.original / 'render_manifest.json').read_text())
    rows = []
    for name in screening['finalists']:
        cfg = configs[name]
        fonts = refs[f"n{cfg['node_fontsize']}_e{cfg['edge_fontsize']}"]
        folder = args.out / 'candidates' / name
        for i in range(1, 51):
            gid = f'G{i:02d}'
            path = folder / f'{gid}.png'
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            measured = measure_image(processor, image, manifest['graphs'][gid]['prompt_bodies']['0']['image'], fonts)
            projection = projected[name, gid]
            if (measured['resized_width'], measured['resized_height']) != (int(projection['resized_width']), int(projection['resized_height'])):
                raise AssertionError('Real processor resize differs from screening calculation')
            row = dict(configuration=name, graph_id=gid, image_sha256=sha256_file(path), **measured)
            rows.append(row)
            if i in (1, 25, 50):
                # Save the real normalized/reconstructed tile grid for visual review.
                from llava_common import prepare_inputs
                from llava_stage1_prompt import format_llava_prompt
                prompt = format_llava_prompt(processor, manifest['graphs'][gid]['prompt_bodies']['0']['image'], 'image', 'llava_v1')
                values = prepare_inputs(processor, prompt, image)['pixel_values'][0]
                mean = np.asarray(processor.image_processor.image_mean).reshape(3, 1, 1)
                std = np.asarray(processor.image_processor.image_std).reshape(3, 1, 1)
                tiles = ((values.cpu().numpy() * std + mean) * 255).round().clip(0, 255).astype('uint8')
                gh, gw = measured['selected_grid_pinpoint']['height'], measured['selected_grid_pinpoint']['width']
                size = tiles.shape[-1]
                grid = np.zeros((gh, gw, 3), dtype='uint8')
                for index, tile in enumerate(tiles[1:]):
                    y, x = divmod(index, gw // size)
                    grid[y * size:(y + 1) * size, x * size:(x + 1) * size] = tile.transpose(1, 2, 0)
                Image.fromarray(grid).save(folder / f'{gid}_processor_grid.png')
                Image.fromarray(tiles[0].transpose(1, 2, 0)).save(folder / f'{gid}_processor_base.png')
            if i % 10 == 0:
                print(f'CPU PROCESSOR {name}: {i}/50', flush=True)
        del image
    write_csv(args.out / 'audited_images.csv', rows)
    write_json(args.out / 'audited_images.json', rows)
    results = []
    for name in screening['finalists']:
        group = [r for r in rows if r['configuration'] == name]
        result = dict(**configs[name], n_images=len(group))
        for kind in ('node', 'edge'):
            caps = [r[f'{kind}_cap_px_grid'] for r in group]
            result.update({f'{kind}_min': min(caps), f'{kind}_median': median(caps), f'{kind}_max': max(caps),
                           f'{kind}_below_10px': sum(v < 10 for v in caps)})
        result['graphs_at_both_training_medians'] = sum(all(r[f'{kind}_cap_px_grid'] >= screening['targets'][kind] for kind in ('node', 'edge')) for r in group)
        values = [r['realised_image_tokens'] for r in group]
        result.update(tokens_min=min(values), tokens_median=median(values), tokens_max=max(values))
        result['both_medians_reached'] = all(result[f'{kind}_median'] >= screening['targets'][kind] for kind in ('node', 'edge'))
        results.append(result)
    write_csv(args.out / 'audited_comparison.csv', results)
    original_after = graph_checksum(args.original / 'graphs')
    if original_after != GRAPHS_SHA256:
        raise AssertionError('Original Qwen renders changed during the sweep')
    from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION
    write_json(args.out / 'audit_summary.json', dict(targets=screening['targets'], results=results,
        no_model_loaded=True, original_graphs_sha256_after=original_after,
        model_id=DEFAULT_MODEL_ID, model_revision=DEFAULT_REVISION,
        processor_class=type(processor.image_processor).__name__,
        processor_image_grid_pinpoints=processor.image_processor.image_grid_pinpoints,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__, audit_script_sha256=sha256_file(Path(__file__))))
    print(json.dumps(results, indent=2), flush=True)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit_result = json.loads((args.out / 'audit_summary.json').read_text())
    screening = json.loads((args.out / 'screening.json').read_text())
    baseline = json.loads((args.training_audit / 'audit_config.json').read_text())
    sweep_settings = json.loads((args.out / 'sweep_config.json').read_text())
    n_completed = len(screening['summaries'])
    unavailable = sweep_settings.get('unavailable_engines', {})
    engine_note = ('Unavailable engines: ' + ', '.join(sorted(unavailable)) + '. '
                   if unavailable else '')
    targets = audit_result['targets']
    all_measured = json.loads((args.out / 'audited_images.json').read_text())
    for result in audit_result['results']:
        folder = args.out / 'candidates' / result['name']
        image_files = {}
        for i in range(1, 51):
            gid = f'G{i:02d}'
            original = args.original / 'graphs' / gid
            if (folder / f'{gid}.txt').read_bytes() != original.with_suffix('.txt').read_bytes():
                raise AssertionError('Candidate triple text differs from the original Qwen evidence')
            edge_lines = lambda source: [line.strip() for line in source.splitlines() if ' -> ' in line]
            if edge_lines((folder / f'{gid}.dot').read_text()) != edge_lines(original.with_suffix('.dot').read_text()):
                raise AssertionError('Candidate DOT triple order differs from the original Qwen evidence')
            image_files[gid] = {suffix: sha256_file(folder / f'{gid}.{suffix}') for suffix in ('png', 'dot', 'txt', 'svg')}
        write_json(folder / 'variant_manifest.json', dict(
            variant=result, dataset_sha256=DATASET_SHA256, original_graphs_sha256=GRAPHS_SHA256,
            seed=0, files_sha256=image_files, same_triples_and_order_as_qwen=True,
            processor_audit=[r for r in all_measured if r['configuration'] == result['name']],
            audit_summary_sha256=sha256_file(args.out / 'audit_summary.json'),
            base_and_tuned_must_use_same_images=True))
    lines = ['# LLaVA synthetic font/layout/shape sweep (CPU only)', '',
        f'{n_completed} complete configurations were evaluated on all 50 graphs ({n_completed * 50} layouts). '
        f'{len(sweep_settings["configurations"]) - n_completed} requested configurations did not complete. '
        + engine_note + 'No dependencies were changed.', '',
        'Each configuration keeps all 18 nodes and all 40–49 supplied triples, '
        'including parallel edges, in the original seeded order. No disconnected-answer cluster. '
        'Fonts are Helvetica-Bold nodes / Helvetica edges, 200 DPI, light-blue fill. '
        'The production renderer and original Qwen renders are unchanged.', '',
        'Screening used exact Graphviz SVG pixel dimensions and the pinned LLaVA processor-owned resize calculation. '
        'Finalists were rasterized without image-size caps and processed through the actual CPU processor; '
        'PNG dimensions and actual content resizes matched every screening projection.', '',
        '| Render | Node min / median / max px | Nodes <10 px | Edge min / median / max px | Edges <10 px | Both ≥ training medians | Tokens min / median / max |',
        '| --- | --- | ---: | --- | ---: | ---: | --- |']
    for name, stats in [('Original synthetic', baseline['summaries']['synthetic']),
                        ('Stage 1 sample (50 images)', baseline['summaries']['stage1_training_sample'])]:
        node, edge = stats['node_grid'], stats['edge_grid']
        lines.append(f"| {name} | {node['min']:.3f} / {node['median']:.3f} / {node['max']:.3f} | {node['below_10px']}/50 | {edge['min']:.3f} / {edge['median']:.3f} / {edge['max']:.3f} | {edge['below_10px']}/50 | — | — |")
    for row in audit_result['results']:
        lines.append(f"| {row['name']} | {row['node_min']:.3f} / {row['node_median']:.3f} / {row['node_max']:.3f} | {row['node_below_10px']}/50 | {row['edge_min']:.3f} / {row['edge_median']:.3f} / {row['edge_max']:.3f} | {row['edge_below_10px']}/50 | {row['graphs_at_both_training_medians']}/50 | {row['tokens_min']} / {row['tokens_median']} / {row['tokens_max']} |")
    lines += ['', 'The closest candidate reaching both medians is the first finalist. '
        'Candidates are ranked over the complete 50-graph cohort, without question answers or model predictions. '
        'The Stage 1 reference remains a descriptive 50-image sample, not a full-training-set audit.', '',
        'Natural shape performed best. Explicit Graphviz ratios of 1/3 (wide) and 3 (tall) '
        'stretch the layout; the extra canvas outweighs anyres grid gains. Font increases '
        'produce diminishing returns because the source canvas grows too. Tight spacing helps.', '',
        'Cap heights are measured reference-glyph estimates, not OCR/readability scores. '
        'Processed tile-grid previews for G01, G25, G50 are stored beside each finalist. '
        'Visual review of G01/G25/G50 found remaining edge-label crowding, including '
        'overlapping at-location labels in G50. No systematic overlap/readability score was measured. '
        'This sweep does not approve '
        'a GPU job or bypass the previous 10 px gate.', '',
        'A future LLaVA base/tuned comparison must use the same chosen variant for both loads '
        'and both image controls. Qwen-versus-LLaVA image comparisons would then include '
        'the render change as an additional difference. No inference defaults were changed.', '',
        'Data: screening_comparison.csv (all successful configurations), screened_images.csv, '
        'audited_comparison.csv, audited_images.csv; candidate PNG/DOT/TXT/SVG files under candidates/.', '',
        '![Font effect](font_effect.png)', '']
    (args.out / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind, ax in zip(('node', 'edge'), axes):
        for layout in ('dot_lr_standard', 'dot_lr_tight', 'dot_tb_standard', 'dot_tb_tight'):
            rows = sorted([row for row in screening['summaries'] if row['layout'] == layout and row['shape'] == 'natural'], key=lambda row: row['node_fontsize'])
            ax.plot([row['node_fontsize'] for row in rows], [row[f'{kind}_median'] for row in rows], marker='o', label=layout.replace('dot_', ''))
        ax.axhline(targets[kind], linestyle='--', color='black', label='Stage 1 sample median')
        ax.set(xlabel='Node / edge font size (pt)', ylabel=f'Median {kind} cap height after LLaVA (px)',
               xticks=[n for n, _ in FONTS], xticklabels=[f'{n}/{e}' for n, e in FONTS])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / 'font_effect.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('render', 'score', 'png', 'audit', 'report'), required=True)
    parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--original', type=Path, default=ROOT / 'outputs/synth_mcq_name_grounded_2026-10-10')
    parser.add_argument('--training-audit', type=Path, default=ROOT / 'outputs/synth_mcq_name_grounded_llava_audit_2026-10-10')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    {'render': render_layouts, 'score': score, 'png': render_pngs, 'audit': audit, 'report': report}[args.phase](args)


if __name__ == '__main__':
    main()
