"""CPU-only data, prompts, rendering and row identities for synthetic MCQ evaluation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OBQA = ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'
sys.path.insert(0, str(OBQA))
from prompt_common import (format_kg_block, parse_answer, render_prompt, sha256_file,
                           IMAGE_REFERENCE_SENTENCE, KG_TEXT_REFERENCE_SENTENCE, PARSER_PROTOCOL)

LETTERS = 'ABCD'
MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
MIN_PIXELS = 262144
MAX_PIXELS = 1310720
DEFAULT_DATASET = ROOT / 'reproduction/inputs/synthetic_graphs.json'
CONDITIONS = ('image', 'text', 'none', 'wrong_image', 'wrong_text')
COMPACT_RENDER_STYLE = dict(engine='dot', hide_relatedto_labels=False,
                    reveal_correct_answer=False, dpi=200, disconnected_rows=3,
                    node_fontsize=18, edge_fontsize=14, nodesep=.5, ranksep=.7,
                    rankdir='LR', auto_orient='off', graph_size=None, graph_ratio=None,
                    wrap_labels=0, edge_label_style='plain', node_style_mode='legacy', node_fill=None)
RENDER_STYLE = {**COMPACT_RENDER_STYLE, 'node_fontsize': 30, 'edge_fontsize': 24,
                'ranksep': .4, 'rankdir': 'TB', 'auto_orient': 'llava',
                'node_style_mode': 'uniform', 'node_fill': '#ADD8E6'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def triple_tuple(triple):
    values = tuple(triple[k] for k in ('source', 'relation', 'target'))
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError('Triple fields must be nonempty strings')
    return values


def questions_for(graph):
    # The compatibility question is never an additional evaluation item.
    return graph['questions'] if 'questions' in graph else [graph['question']]


def question_type(question):
    return '+'.join(sorted(t['relation'] for t in question['supporting_triples']
                           if t['relation'] != 'createdby'))


def validate_question(graph, question, question_ids):
    q, nodes = question, {n['id'] for n in graph['nodes']}
    if not isinstance(q['id'], str) or not q['id'] or q['id'] in question_ids:
        raise ValueError('Invalid/duplicate question id')
    question_ids.add(q['id'])
    if not isinstance(q['text'], str) or not q['text'].strip():
        raise ValueError('Empty question text')
    choices = q['choices']
    if set(choices) != set(LETTERS):
        raise ValueError('Require exactly choices A/B/C/D')
    if any(not isinstance(n, str) or n not in nodes for n in choices.values()):
        raise ValueError('Choice absent from node ids')
    if len(set(choices.values())) != 4:
        raise ValueError('Choices must be distinct')
    if q['correct_answer'] not in LETTERS or choices[q['correct_answer']] != q['correct_node']:
        raise ValueError('Correct letter/node mismatch')
    triples = {triple_tuple(t) for t in graph['triples']}
    if not q['supporting_triples'] or any(triple_tuple(t) not in triples for t in q['supporting_triples']):
        raise ValueError('Supporting triple absent from graph')
    if graph['dataset'] == 'name_grounded_50':
        people = {n['id'] for n in graph['nodes'] if n['kind'] == 'person_name'}
        if set(choices.values()) != people:
            raise ValueError('Choices must be exactly the four person_name nodes')
        support = q['supporting_triples']
        if len(support) != 3 or sum(t['relation'] == 'createdby' for t in support) != 1:
            raise ValueError('Require one createdby and two clue supporting triples')
    if graph['dataset'] == 'expanded_25':
        if type(q['reasoning_hops']) is not int or q['reasoning_hops'] <= 0:
            raise ValueError('Invalid reasoning_hops')
        if not isinstance(q['query_style'], str) or not q['query_style']:
            raise ValueError('Invalid query_style')


def load_graphs(path):
    """Validate the entire file before applying a smoke-test limit."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Top-level JSON must be an object')
    if not isinstance(payload.get('metadata'), dict) or not isinstance(payload.get('graphs'), list):
        raise ValueError('Require top-level metadata object and graphs list')
    graphs = payload['graphs']
    if not graphs:
        raise ValueError('Empty graphs list')
    graph_ids, question_ids = set(), set()
    for graph in graphs:
        if not isinstance(graph, dict):
            raise ValueError('Every graph must be an object')
        try:
            gid = graph['id']
            if not isinstance(gid, str) or not re.fullmatch(r'G[0-9]+', gid) or gid in graph_ids:
                raise ValueError('Invalid/duplicate graph id')
            graph_ids.add(gid)
            if graph['dataset'] not in ('original_25', 'expanded_25', 'name_grounded_50'):
                raise ValueError('Unknown dataset')
            if not isinstance(graph['synthetic'], bool):
                raise ValueError('synthetic must be boolean')
            for field in ('original_graph_id', 'title'):
                if not isinstance(graph[field], str) and not (field == 'original_graph_id'
                        and graph['dataset'] == 'name_grounded_50' and graph[field] is None):
                    raise ValueError(f'{field} must be a string')
            node_ids = [node['id'] for node in graph['nodes']]
            if not node_ids or any(not isinstance(n, str) or not n for n in node_ids):
                raise ValueError('Node ids must be nonempty strings')
            if len(node_ids) != len(set(node_ids)):
                raise ValueError('Duplicate node ids')
            if any(not isinstance(node['kind'], str) for node in graph['nodes']):
                raise ValueError('Node kind must be a string')
            nodes = set(node_ids)
            triples = [triple_tuple(t) for t in graph['triples']]
            if not triples or len(triples) != len(set(triples)):
                raise ValueError('Empty/duplicate triples')
            for source, relation, target in triples:
                if source not in nodes or target not in nodes:
                    raise ValueError('Triple endpoint absent from nodes')
                if source == target:
                    raise ValueError('Self-loop')
                if not re.fullmatch(r'[a-z]+', relation):
                    raise ValueError('Relation must be lowercase and unspaced')
            if graph['dataset'] == 'name_grounded_50':
                if len(node_ids) != 18 or Counter(n['kind'] for n in graph['nodes']) != {
                        'person_name': 4, 'concept': 14}:
                    raise ValueError('Require 18 nodes: 4 person_name and 14 concept')
                if {n for s, r, t in triples for n in (s, t)} != nodes:
                    raise ValueError('Every node must appear in at least one triple')
                if not isinstance(graph['questions'], list) or len(graph['questions']) != 2:
                    raise ValueError('Require exactly two questions per graph')
                if graph['question'] != graph['questions'][0]:
                    raise ValueError('Compatibility question must equal questions[0]')
            for q in questions_for(graph):
                validate_question(graph, q, question_ids)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Graph {graph.get('id', '?')}: {error}") from error
    if any(g['dataset'] == 'name_grounded_50' for g in graphs):
        if len(graphs) != 50 or len(question_ids) != 100 or any(g['dataset'] != 'name_grounded_50' for g in graphs):
            raise ValueError('Name-grounded dataset requires exactly 50 graphs and 100 questions')
    return graphs


def solver_check(graphs):
    """Solve clues independently over full triples, requiring a shared created artifact."""
    results = []
    for graph in graphs:
        if graph['dataset'] != 'name_grounded_50':
            continue
        people = {n['id'] for n in graph['nodes'] if n['kind'] == 'person_name'}
        triples = {triple_tuple(t) for t in graph['triples']}
        for q in questions_for(graph):
            artifact, _, person = next(triple_tuple(t) for t in q['supporting_triples'] if t['relation'] == 'createdby')
            clues = [t for t in q['supporting_triples'] if t['relation'] != 'createdby']
            if person != q['correct_node'] or any(c['source'] not in (artifact, person) for c in clues):
                raise ValueError(f"Invalid clue sources/gold creator: {q['id']}")
            singles, conjunction = [set(), set()], set()
            for candidate in people:
                for created, relation, creator in triples:
                    if relation != 'createdby' or creator != candidate:
                        continue
                    matches = [(created if c['source'] == artifact else candidate,
                                c['relation'], c['target']) in triples for c in clues]
                    for i, match in enumerate(matches):
                        if match: singles[i].add(candidate)
                    if all(matches): conjunction.add(candidate)
            result = dict(graph_id=graph['id'], question_id=q['id'], question_type=question_type(q),
                          clue_1_matches=sorted(singles[0]), clue_2_matches=sorted(singles[1]),
                          conjunction_matches=sorted(conjunction), correct_node=q['correct_node'])
            if tuple(map(len, singles)) != (2, 2) or conjunction != {q['correct_node']}:
                raise ValueError('Independent solver mismatch: ' + json.dumps(result))
            results.append(result)
    return dict(n_questions=len(results), passed=len(results), mismatches=0,
                clue_count_distribution=dict(Counter(f"{len(r['clue_1_matches'])},{len(r['clue_2_matches'])},{len(r['conjunction_matches'])}" for r in results)),
                questions=results)


def wrong_graph_pairing(graphs, seed, max_draws=10_000_000):
    """Seeded permutation rejection; first prove a collision-free matching exists."""
    ids = sorted(g['id'] for g in graphs)
    people = {g['id']: {n['id'] for n in g['nodes'] if n['kind'] == 'person_name'} for g in graphs}
    gold = {g['id']: {q['correct_node'] for q in questions_for(g)} for g in graphs}
    allowed = {a: {b for b in ids if b != a and not gold[a] & people[b]} for a in ids}
    matched = {}
    def augment(source, visited):
        for target in sorted(allowed[source]):
            if target in visited: continue
            visited.add(target)
            if target not in matched or augment(matched[target], visited):
                matched[target] = source
                return True
        return False
    if not all(augment(source, set()) for source in ids):
        raise ValueError('No valid wrong-graph derangement exists without gold-name collisions')
    rng = random.Random(seed)
    for redraws in range(max_draws):
        shown = ids.copy(); rng.shuffle(shown)
        if all(b in allowed[a] for a, b in zip(ids, shown)):
            return dict(seed=seed, redraws=redraws, draws=redraws + 1,
                        mapping=dict(zip(ids, shown)), gold_name_collisions=0,
                        algorithm='sorted graph IDs; random.Random(seed).shuffle; reject fixed points or any source gold in shown person nodes')
    raise ValueError(f'Valid matching exists but random derangement sampling exhausted {max_draws} draws')


def expand_conditions(conditions, modality):
    result = []
    for condition in conditions:
        if condition == 'wrong_graph':
            result.extend(['wrong_image', 'wrong_text'] if modality == 'both' else [f'wrong_{modality}'])
        else: result.append(condition)
    if not result or len(result) != len(set(result)) or not set(result) <= set(CONDITIONS):
        raise ValueError('Invalid or duplicate conditions')
    return result


def evaluation_items(graphs, models, conditions, prompt_styles, n_perms):
    return [(style, model, condition, g['id'], q['id'], p)
            for style in prompt_styles for model in models for g in graphs
            for q in questions_for(g) for condition in conditions for p in range(n_perms)]


def dataset_stats(graphs):
    pairs = [Counter((t['source'], t['target']) for t in g['triples']) for g in graphs]
    counts = [len(g['triples']) for g in graphs]
    return dict(graphs=len(graphs), questions=sum(len(questions_for(g)) for g in graphs), triples=sum(counts),
                nodes_per_graph=dict(Counter(len(g['nodes']) for g in graphs)),
                triples_min=min(counts), triples_max=max(counts),
                answer_letters=dict(sorted(Counter(q['correct_answer'] for g in graphs for q in questions_for(g)).items())),
                parallel_node_pairs=sum(sum(n > 1 for n in p.values()) for p in pairs),
                graphs_with_parallel_edges=sum(any(n > 1 for n in p.values()) for p in pairs))


def shuffled_triples(graph, seed):
    # Stable across Python processes; never use Python's randomized hash().
    graph_seed = int.from_bytes(hashlib.sha256(f"{seed}:{graph['id']}".encode()).digest(), 'big')
    triples = list(graph['triples'])
    random.Random(graph_seed).shuffle(triples)
    return triples


def graph_metadata(graph, triples):
    nodes = {i: dict(name=n['id'], in_question=False, in_choices=set())
             for i, n in enumerate(graph['nodes'])}
    cid = {node['name']: i for i, node in nodes.items()}
    edges = [(cid[t['source']], t['relation'], cid[t['target']]) for t in triples]
    rendered = dict(connected_nodes=list(nodes), visible_nodes=list(nodes), edges=edges,
                    disconnected_answers=[])
    meta = dict(visible_nodes=[dict(cid=i, name=n['name'], connected=True) for i, n in nodes.items()],
                edges=[dict(source_cid=s, relation=r, target_cid=t) for s, r, t in edges])
    return nodes, rendered, meta


def permuted_choices(graph, permutation, question=None):
    if permutation not in range(4):
        raise ValueError('Cyclic permutation must be 0..3')
    original = (question if question is not None else graph['question'])['choices']
    return {letter: original[LETTERS[(i + permutation) % 4]] for i, letter in enumerate(LETTERS)}


def styled_prompt_body(body, condition, prompt_style='obqa'):
    """Replace only the inherited reference sentence; question/choices stay byte-identical."""
    if prompt_style not in ('obqa', 'needed'):
        raise ValueError('Unknown prompt_style')
    if prompt_style == 'obqa' or condition == 'none':
        return body
    prefix = ('<image>\n' + IMAGE_REFERENCE_SENTENCE if condition == 'image'
              else KG_TEXT_REFERENCE_SENTENCE if condition == 'text' else None)
    if prefix is None or not body.startswith(prefix):
        raise ValueError('Expected inherited graph-reference sentence')
    needed = prefix.replace('which may or may not be useful',
                            'which contains the information needed to answer the question')
    return needed + body[len(prefix):]


def logit_scores(logits, choices, correct_node):
    if set(logits) != set(LETTERS) or any(not math.isfinite(float(logits[l])) for l in LETTERS):
        raise ValueError('Require four finite first-position A/B/C/D logits')
    # Deterministic A/B/C/D order for ties, matching tensor argmax.
    letter = max(LETTERS, key=lambda l: float(logits[l]))
    node = choices[letter]
    return dict(logit_argmax_letter=letter, logit_argmax_node=node,
                logit_argmax_is_correct=node == correct_node)


def prompt_body(graph, choices, condition, triples, prompt_style='obqa', question=None):
    from generate_graphvis_datasets import build_stage2_record
    q = question if question is not None else graph['question']
    statement = dict(answerKey=next(l for l in LETTERS if choices[l] == q['correct_node']),
                     question=dict(stem=q['text'],
                     choices=[dict(label=l, text=choices[l]) for l in LETTERS]))
    template = build_stage2_record(Path('unused.png'), 'synthetic', 0, statement)['prompt']
    mapping = dict(image='image', text='kg_text', none='text_noref')
    body = render_prompt(template, mapping[condition],
                         kg_block=format_kg_block(graph_metadata(graph, triples)[2], preserve_order=True)
                         if condition == 'text' else None)
    return styled_prompt_body(body, condition, prompt_style)


def row_key(row):
    return (row.get('prompt_style', 'obqa'), row['model'], row['condition'], row['graph_id'],
            row.get('question_id', row['graph_id']), row['perm'])


def evidence_prompt(manifest, graph, question, permutation, condition, style, shown_graph):
    entry = manifest['graphs'][graph['id']]
    bodies = entry.get('question_prompt_bodies', {}).get(question['id'], entry['prompt_bodies'])[str(permutation)]
    modality = 'image' if condition in ('image', 'wrong_image') else 'text' if condition in ('text', 'wrong_text') else 'none'
    body = bodies[modality]
    if condition == 'wrong_text':
        block = format_kg_block(graph_metadata(shown_graph, manifest['graphs'][shown_graph['id']]['triples'])[2], preserve_order=True)
        body = KG_TEXT_REFERENCE_SENTENCE + '\n' + block + '\n\n' + bodies['none']
    return styled_prompt_body(body, modality, style), modality


def read_predictions(path):
    path = Path(path)
    if not path.exists():
        return []
    rows, seen = [], set()
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                key = row_key(row)
                if key in seen:
                    raise ValueError('Duplicate prediction key')
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}; repair incomplete rows explicitly before resume') from error
            seen.add(key)
            rows.append(row)
    return rows


def render_style(render_fontsize=None, *, style='training'):
    if style not in ('training', 'compact'):
        raise ValueError('Unknown render_style')
    if render_fontsize is not None and (type(render_fontsize) is not int or render_fontsize <= 0):
        raise ValueError('render_fontsize must be a positive integer')
    settings = dict(RENDER_STYLE if style == 'training' else COMPACT_RENDER_STYLE)
    if render_fontsize is not None:
        settings['node_fontsize'] = render_fontsize
    return settings


def render_identity(source, seed, render_fontsize=None, render_style_name='training'):
    return dict(input_sha256=sha256_file(source), seed=seed, render_style=render_style_name,
                style=render_style(render_fontsize, style=render_style_name),
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                code_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in
                             [ROOT / 'scripts/generate_graphvis_datasets.py', OBQA / 'prompt_common.py']})


def render_graphs(graphs, source, out, seed=0, render_fontsize=None, render_style_name='training'):
    """Call the unchanged Graphviz renderer and capture its exact DOT source."""
    import graphviz
    from PIL import Image, ImageDraw
    from types import SimpleNamespace
    from unittest.mock import patch
    import generate_graphvis_datasets as renderer
    out = Path(out)
    folder = out / 'graphs'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(**render_identity(source, seed, render_fontsize, render_style_name), graphs={})
    manifest_path = out / 'render_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k) != manifest[k] for k in manifest if k != 'graphs'):
            raise ValueError('Existing render manifest uses different data/seed/style/budget; choose a new OUT')

    class CapturedDigraph(graphviz.Digraph):
        def render(self, filename=None, *args, **kwargs):
            if self.strict:
                raise AssertionError('Strict DOT would collapse parallel edges')
            edges = [line for line in self.body if ' -> ' in line]
            if len(edges) != expected_edges:
                raise AssertionError(f'{filename}: rendered {len(edges)} edges, expected {expected_edges}')
            captured_sources[Path(filename).name] = self.source
            Path(str(filename) + '.dot').write_text(self.source, encoding='utf-8')
            return super().render(filename, *args, **kwargs)

    for graph in graphs:
        gid = graph['id']
        triples = shuffled_triples(graph, seed)
        nodes, rendered, meta = graph_metadata(graph, triples)
        expected_edges = len(triples)
        captured_sources = {}
        # Module-local replacement only; the production renderer itself is unchanged.
        with patch.object(renderer, 'graphviz', SimpleNamespace(Digraph=CapturedDigraph)):
            _, orientation = renderer.render_selected_orientation(
                folder / gid, nodes, rendered, '', **manifest['style'])
        selected = orientation['rankdir_selected'] if orientation else manifest['style']['rankdir']
        source_key = selected.lower() if manifest['style']['auto_orient'] != 'off' else gid
        (folder / f'{gid}.dot').write_text(captured_sources[source_key], encoding='utf-8')
        text_path = folder / f'{gid}.txt'
        text_path.write_text(format_kg_block(meta, preserve_order=True) + '\n', encoding='utf-8')
        with Image.open(folder / f'{gid}.png') as image:
            width, height = image.size
        oversized = width * height > MAX_PIXELS
        entry = dict(width=width, height=height, pixels=width * height,
                     exceeds_max_pixels=oversized, edge_count=expected_edges,
                     triples=triples, rankdir_selected=selected, orientation=orientation,
                     question_ids=[q['id'] for q in questions_for(graph)],
                     question_prompt_bodies={q['id']: {str(p): {c: prompt_body(graph, permuted_choices(graph, p, q), c, triples, question=q)
                                             for c in ('image', 'text', 'none')} for p in range(4)} for q in questions_for(graph)},
                     prompt_bodies={str(p): {c: prompt_body(graph, permuted_choices(graph, p), c, triples)
                                             for c in ('image', 'text', 'none')} for p in range(4)},
                     files_sha256={suffix: sha256_file(folder / f'{gid}.{suffix}')
                                   for suffix in ('png', 'dot', 'txt')})
        manifest['graphs'][gid] = entry
        print(f'RENDER {gid}: {width}x{height}; edges={expected_edges}; max_pixels={MAX_PIXELS}', flush=True)
        if oversized:
            print(f'WARNING {gid}: image exceeds Qwen pixel budget and will be downscaled; inspect edge-label legibility.', flush=True)
    write_json(manifest_path, manifest)
    # Preview thumbnails only: inference always uses the original full-size PNGs.
    sheet = Image.new('RGB', (1600, 1000), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, graph in enumerate(graphs[:4]):
        with Image.open(folder / f"{graph['id']}.png") as opened:
            preview = opened.convert('RGB')
        preview.thumbnail((780, 450))
        x, y = (i % 2) * 800, (i // 2) * 500
        entry = manifest['graphs'][graph['id']]
        draw.text((x + 10, y + 10), f"{graph['id']} | {entry['width']}x{entry['height']} | preview only", fill='black')
        sheet.paste(preview, (x + (800 - preview.width) // 2, y + 35))
    sheet.save(out / 'contact_sheet.png')
    return manifest


def verify_renders(graphs, source, out, seed, render_fontsize=None, render_style_name='training'):
    manifest = json.loads((out / 'render_manifest.json').read_text())
    for key, expected in render_identity(source, seed, render_fontsize, render_style_name).items():
        if manifest.get(key) != expected:
            raise ValueError(f'Render manifest mismatch: {key}')
    for graph in graphs:
        gid = graph['id']
        entry = manifest['graphs'][gid]
        if entry['triples'] != shuffled_triples(graph, seed) or entry['edge_count'] != len(graph['triples']):
            raise ValueError(f'Render order/edge mismatch: {gid}')
        for suffix, digest in entry['files_sha256'].items():
            if sha256_file(out / 'graphs' / f'{gid}.{suffix}') != digest:
                raise ValueError(f'Render file hash mismatch: {gid}.{suffix}')
    return manifest
