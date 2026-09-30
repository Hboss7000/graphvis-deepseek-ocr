#!/usr/bin/env python3
"""Estimate cap height from a measured Graphviz reference glyph, not point math.

Render a borderless capital H with Helvetica-Bold at the configured node font
size/DPI; threshold the reference PNG and measure the glyph's ink bounding box.
Scale that measured height by 896/max(image width, image height). This models
aspect-preserving fit into 896x896, not pan-and-scan crops or OCR accuracy.
Assumes the same Graphviz/font installation and no extra whole-graph scaling.
If Graphviz `size` shrinks the graph, supply --graph-scale or treat the estimate
as an upper bound; a canvas cap alone does not guarantee legibility.
"""
import argparse
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import graphviz
from PIL import Image

from pruning_stats import distribution, load_split


@lru_cache(maxsize=None)
def reference_cap_height(node_fontsize=18, dpi=200):
    dot = graphviz.Digraph(format='png')
    dot.attr('graph', dpi=str(dpi), pad='0', margin='0', bgcolor='white')
    dot.attr('node', shape='plaintext', margin='0', fontname='Helvetica-Bold', fontsize=str(node_fontsize))
    dot.node('reference', 'H')
    with Image.open(BytesIO(dot.pipe())) as image:
        mask = image.convert('L').point(lambda value: 255 if value < 128 else 0)
        bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError('Reference glyph contained no measurable ink')
    return bounds[3] - bounds[1]


def audit_split(split, image_root=None, node_fontsize=18, dpi=200, min_label_px=10, graph_scale=1.0):
    split = Path(split)
    metadata = load_split(split)
    base = split if split.is_dir() else split.parent
    image_root = Path(image_root) if image_root else base.parent
    reference = reference_cap_height(node_fontsize, dpi)
    rows = []
    for idx, meta in sorted(metadata.items()):
        image_path = image_root / meta['image']
        with Image.open(image_path) as image:
            width, height = image.size
        cap = reference * graph_scale * 896 / max(width, height)
        rows.append({'statement_idx': idx, 'image': str(image_path), 'width': width, 'height': height,
                     'aspect_ratio': width / height, 'estimated_label_cap_px_896': cap,
                     'node_count': len(meta['visible_nodes']), 'edge_count': len(meta['edges'])})
    return {'method': __doc__, 'node_fontsize': node_fontsize, 'dpi': dpi,
            'reference_cap_height_px': reference, 'graph_scale': graph_scale,
            'min_label_px': min_label_px, 'images': rows,
            'summary': {'image_count': len(rows),
                        'scaled_label_height': distribution(r['estimated_label_cap_px_896'] for r in rows),
                        'below_threshold': sum(r['estimated_label_cap_px_896'] < min_label_px for r in rows),
                        'aspect_ratio': distribution(r['aspect_ratio'] for r in rows)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--node-fontsize', type=int, default=18)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--min-label-px', type=float, default=10)
    parser.add_argument('--graph-scale', type=float, default=1.0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_split(args.split, args.image_root, args.node_fontsize, args.dpi, args.min_label_px, args.graph_scale)
    text = json.dumps(report, indent=2) + '\n'
    print(text, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == '__main__':
    main()
