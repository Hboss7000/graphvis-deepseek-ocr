#!/usr/bin/env python3
"""Render and audit the accepted LLaVA variant locally, CPU only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median
import subprocess
import xml.etree.ElementTree as ET

from synth_mcq_common import ROOT, DEFAULT_DATASET, load_graphs, sha256_file, write_json
from synth_mcq_llava_common import STYLE, VARIANT, check_triple_order, range_failures
from audit_synth_mcq_llava_images import graph_checksum, DATASET_SHA256, GRAPHS_SHA256, measure_image
from sweep_synth_mcq_llava_legibility import capture_dot, svg_geometry, load_processor


def verify_svg_labels(svg, graph):
    """Check node identities and the full multiset of labelled endpoints, not just counts."""
    from prompt_common import RELATION_TEXT
    ns = '{http://www.w3.org/2000/svg}'
    groups = list(ET.fromstring(svg).iter(ns + 'g'))
    normalize = lambda text: ''.join(text.split())
    label = lambda group: normalize(' '.join(''.join(t.itertext()) for t in group.iter(ns + 'text')))
    nodes = {group.find(ns + 'title').text: label(group) for group in groups if group.get('class') == 'node'}
    expected_nodes = {str(i): normalize(n['id'].replace('_', ' ')) for i, n in enumerate(graph['nodes'])}
    if nodes != expected_nodes:
        raise ValueError(f'{graph["id"]}: SVG node identities/labels differ from dataset')
    indices = {n['id']: str(i) for i, n in enumerate(graph['nodes'])}
    edges = Counter((group.find(ns + 'title').text, label(group)) for group in groups if group.get('class') == 'edge')
    expected = Counter((indices[t['source']] + '->' + indices[t['target']],
                        normalize(RELATION_TEXT.get(t['relation'], t['relation']))) for t in graph['triples'])
    if edges != expected:
        raise ValueError(f'{graph["id"]}: SVG triples/relation labels differ from dataset')
    return svg_geometry(svg, graph)


def edge_label_boxes(layout):
    """Conservative text-box proxy from Graphviz draw ops; not an ink overlap test."""
    boxes = []
    for edge in layout['edges']:
        size = None
        for op in edge.get('_ldraw_', []):
            if op['op'] == 'F':
                size = op['size']
            elif op['op'] == 'T':
                x, y = op['pt']; width = op['width']
                left = x if op['align'] == 'l' else x - width if op['align'] == 'r' else x - width / 2
                boxes.append(dict(edge_id=edge['_gvid'], text=op['text'],
                    bounds=[left, y - .25 * size, left + width, y + .8 * size]))
    pairs = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            x0, y0, x1, y1 = a['bounds']; u0, v0, u1, v1 = b['bounds']
            if min(x1, u1) > max(x0, u0) and min(y1, v1) > max(y0, v0):
                pairs.append(dict(edge_ids=[a['edge_id'], b['edge_id']], labels=[a['text'], b['text']]))
    return pairs


def render(args):
    if sha256_file(args.dataset_path) != DATASET_SHA256 or graph_checksum(args.original / 'graphs') != GRAPHS_SHA256:
        raise ValueError('Original dataset/render checksums differ')
    args.out.mkdir(parents=True, exist_ok=False)
    folder = args.out / 'graphs'; folder.mkdir()
    graphs = load_graphs(args.dataset_path)
    original = json.loads((args.original / 'render_manifest.json').read_text())
    manifest = dict(protocol='synth_mcq_llava_renders_v1', variant=VARIANT, style=STYLE,
                    dataset_path=str(args.dataset_path.resolve()), dataset_sha256=DATASET_SHA256,
                    original_graphs_sha256=GRAPHS_SHA256, seed=0, graphs={})
    from PIL import Image
    config = dict(name=VARIANT, **STYLE)
    for graph in graphs:
        gid = graph['id']; source, text = capture_dot(graph, config)
        (folder / f'{gid}.dot').write_text(source)
        (folder / f'{gid}.txt').write_text(text)
        for fmt in ('png', 'svg', 'json'):
            result = subprocess.run(['dot', '-T' + fmt], input=source.encode(), capture_output=True, check=True, timeout=60)
            (folder / f'{gid}.{fmt}').write_bytes(result.stdout)
        geometry = verify_svg_labels((folder / f'{gid}.svg').read_text(), graph)
        with Image.open(folder / f'{gid}.png') as image:
            if image.size != (geometry['width'], geometry['height']):
                raise ValueError('SVG/PNG dimensions differ')
        pairs = edge_label_boxes(json.loads((folder / f'{gid}.json').read_text()))
        entry = dict(original['graphs'][gid])
        entry.pop('exceeds_max_pixels', None)  # Qwen's pixel-budget field is inapplicable here.
        entry.update(**geometry, pixels=geometry['width'] * geometry['height'],
                     orientation=dict(engine_selected='dot', rankdir_selected='LR', method='fixed accepted LLaVA variant'),
                     triples=original['graphs'][gid]['triples'], rankdir_selected='LR',
                     edge_count=len(graph['triples']), files_sha256={s: sha256_file(folder / f'{gid}.{s}') for s in ('png', 'dot', 'txt')},
                     edge_label_overlap=dict(flagged=bool(pairs) or gid == 'G50', proxy_pairs=pairs,
                         visual_confirmation=True if gid == 'G50' else None,
                         method='Conservative Graphviz text-box intersections; supplemented by prior visual G50 confirmation',
                         limits='Approximate font-height boxes can flag whitespace intersections. No systematic ink/OCR/arrow/label-node collision test. Sweep visual review covered G01/G25/G50 only.'))
        manifest['graphs'][gid] = entry
        print(f'RENDER {gid} {geometry["width"]}x{geometry["height"]}; all {len(graph["nodes"])} nodes / {len(graph["triples"])} triples/labels verified', flush=True)
    manifest['triple_order_check'] = check_triple_order(args.original / 'graphs', folder, [g['id'] for g in graphs])
    manifest['graphs_sha256'] = graph_checksum(folder)
    write_json(args.out / 'render_manifest.json', manifest)
    (args.out / 'graphs.sha256').write_text(manifest['graphs_sha256'] + '\n')
    print('COMBINED RENDER CHECKSUM ' + manifest['graphs_sha256'], flush=True)
    print('TRIPLE ORDER ' + json.dumps(manifest['triple_order_check']), flush=True)


def audit(args):
    from PIL import Image
    from synth_mcq_llava_common import verify_assets
    manifest, checksum = verify_assets(args.dataset_path, args.out, args.pairing)
    processor = load_processor()
    first = json.loads((args.reference_audit / 'audit_config.json').read_text())
    refs = json.loads((args.sweep / 'font_references.json').read_text())['n42_e34']
    train_refs = json.loads((args.original / 'audit/font_references/font_references.json').read_text())['training']
    reference_rows = [r for r in [json.loads(line) for line in (args.reference_audit / 'images.jsonl').read_text().splitlines()]
                      if r['dataset'] == 'stage1_training_sample']
    indices = first['training_sample_indices']
    if [int(r['graph_id']) for r in reference_rows] != indices:
        raise ValueError('Reference list differs from first audit')
    recomputed = []
    for row in reference_rows:
        path = Path(row['image'])
        if sha256_file(path) != row['image_sha256']:
            raise ValueError('Fixed Stage 1 reference image hash differs')
        with Image.open(path) as image:
            measured = measure_image(processor, image.convert('RGB'), '<image>\nDescribe this graph.', train_refs)
        recomputed.append(dict(graph_id=row['graph_id'], **measured))
    ranges = {kind: dict(min=min(r[f'{kind}_cap_px_grid'] for r in recomputed),
                         max=max(r[f'{kind}_cap_px_grid'] for r in recomputed)) for kind in ('node', 'edge')}
    original_ranges = {kind: {key: first['summaries']['stage1_training_sample'][kind + '_grid'][key] for key in ('min', 'max')} for kind in ('node', 'edge')}
    if ranges != original_ranges:
        raise ValueError('Recomputed reference range differs from the first audit')
    rows = []
    for gid, entry in manifest['graphs'].items():
        with Image.open(args.out / 'graphs' / f'{gid}.png') as image:
            measured = measure_image(processor, image.convert('RGB'), entry['prompt_bodies']['0']['image'], refs)
        row = dict(graph_id=gid, **measured, edge_labels_overlap_flagged=entry['edge_label_overlap']['flagged'],
                   overlap_proxy_pairs=len(entry['edge_label_overlap']['proxy_pairs']))
        rows.append(row)
        print(f'CPU LLAVA {gid} tokens={measured["realised_image_tokens"]} node={measured["node_cap_px_grid"]:.3f} edge={measured["edge_cap_px_grid"]:.3f}', flush=True)
    folder = args.out / 'audit'; folder.mkdir(exist_ok=True)
    for filename, data in [('images.json', rows), ('reference_images.json', recomputed)]:
        write_json(folder / filename, data)
    with (folder / 'images.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    failures = range_failures(rows, ranges)
    gate = dict(backend='llava', criterion='fixed_stage1_reference_min_max', model_revision=first['gate']['model_revision'],
        dataset_sha256=DATASET_SHA256, graphs_sha256=checksum,
        render_manifest_sha256=sha256_file(args.out / 'render_manifest.json'), n_images=50,
        training_reference_indices=indices, training_reference_audit_sha256=sha256_file(args.reference_audit / 'audit_config.json'),
        ranges=ranges, measurements=rows, failures=failures, failing_graph_count=len(failures), passed=not failures,
        image_grid_pinpoints=processor.image_processor.image_grid_pinpoints,
        old_10px_counts_information_only={kind: sum(r[f'{kind}_cap_px_grid'] < 10 for r in rows) for kind in ('node', 'edge')})
    gate.update(processor_version=__import__('transformers').__version__,
                processor_class=type(processor.image_processor).__name__,
                font_references=refs, training_font_references=train_refs)
    write_json(folder / 'legibility_gate.json', gate)
    lines = ['# Accepted LLaVA render audit (CPU only)', '', f'Training-reference range gate: {"PASS" if not failures else "FAIL"}; outside range: {len(failures)}/50.', '',
             '| Labels | Min / median / max px | Outside reference range | Below 10 px (information only) |', '| --- | --- | ---: | ---: |']
    for kind in ('node', 'edge'):
        values = [r[f'{kind}_cap_px_grid'] for r in rows]
        lines.append(f'| {kind} | {min(values):.3f} / {median(values):.3f} / {max(values):.3f} | {sum(not ranges[kind]["min"] <= v <= ranges[kind]["max"] for v in values)}/50 | {sum(v < 10 for v in values)}/50 |')
    lines += ['', 'Fixed Stage 1 training reference (the original 50 images, re-measured):', '',
              '| Labels | Min / median / max px | Below 10 px (information only) |', '| --- | --- | ---: |']
    for kind in ('node', 'edge'):
        values = [r[f'{kind}_cap_px_grid'] for r in recomputed]
        lines.append(f'| {kind} | {min(values):.3f} / {median(values):.3f} / {max(values):.3f} | {sum(v < 10 for v in values)}/50 |')
    tokens = [r['realised_image_tokens'] for r in rows]
    lines += ['', f'Synthetic image tokens min / median / max: {min(tokens)} / {median(tokens):g} / {max(tokens)}.',
              'All 50 selected grids are 672×672; the resized content dimensions below exclude padding. Native CPU processor: transformers 5.16.1, pinned LLaVA revision 2424fdd47412fccc66d91719126b420e9fbd7065.',
              'Own-font H glyphs: 42 pt Helvetica-Bold nodes = 84 px; 34 pt Helvetica edges = 68 px at 200 DPI. Training references use 30/24 pt glyphs (60/48 px). Edge glyphs are measured independently.',
              'Measurements for both grid-resized and base-thumbnail views are in images.csv and reference_images.json. The reference indices and source-image checksums are recorded in legibility_gate.json and reference_images.json.']
    flagged = [r['graph_id'] for r in rows if r['edge_labels_overlap_flagged']]
    lines += ['', 'Edge-label overlap proxy flags: ' + ', '.join(flagged) + '.',
        'Method: conservative Graphviz text-box intersections, supplemented by the sweep’s visual G50 confirmation. Text boxes include whitespace; no systematic ink/OCR test. Arrow crossings and label-node collisions are not measured. Overlap is not a gate.', '',
        '| Graph | Grid H×W | Resized W×H | Tokens | Node cap px | Edge cap px | Overlap flag |', '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in rows:
        grid = r['selected_grid_pinpoint']
        lines.append(f'| {r["graph_id"]} | {grid["height"]}×{grid["width"]} | {r["resized_width"]}×{r["resized_height"]} | {r["realised_image_tokens"]} | {r["node_cap_px_grid"]:.3f} | {r["edge_cap_px_grid"]:.3f} | {"flagged" if r["edge_labels_overlap_flagged"] else "not flagged by proxy"} |')
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')
    print('LEGIBILITY GATE ' + json.dumps({k: v for k, v in gate.items() if k != 'measurements'}), flush=True)
    if failures:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('render', 'audit'), required=True)
    parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--original', type=Path, default=ROOT / 'outputs/synth_mcq_name_grounded_2026-10-10')
    parser.add_argument('--sweep', type=Path, default=ROOT / 'outputs/synth_mcq_llava_legibility_sweep_2026-10-10')
    parser.add_argument('--reference-audit', type=Path, default=ROOT / 'outputs/synth_mcq_name_grounded_llava_audit_2026-10-10')
    parser.add_argument('--pairing', type=Path, default=ROOT / 'outputs/synth_mcq_name_grounded_2026-10-10/wrong_graph_pairing.json')
    args = parser.parse_args()
    {'render': render, 'audit': audit}[args.phase](args)


if __name__ == '__main__':
    main()
