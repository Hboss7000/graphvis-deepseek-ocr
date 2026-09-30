#!/usr/bin/env python3
"""Generate GraphVis-style OBQA image datasets.

This script intentionally uses the current working interpretation for OBQA:
QA-GNN provides one graph per (question, answer choice), so we merge the four
choice graphs into one question-level visual graph by taking their union.
GraphVis does not specify this aggregation step directly.
"""

import argparse
import json
import math
import pickle
import random
from collections import defaultdict, deque
from itertools import combinations
import re
from pathlib import Path

import graphviz


RELATIONS = [
    'antonym', 'atlocation', 'capableof', 'causes', 'createdby',
    'isa', 'desires', 'hassubevent', 'partof', 'hascontext',
    'hasproperty', 'madeof', 'notcapableof', 'notdesires', 'receivesaction',
    'relatedto', 'usedfor',
]

RELATION_TEXT = {
    'antonym': 'antonym',
    'atlocation': 'at location',
    'capableof': 'capable of',
    'causes': 'causes',
    'createdby': 'created by',
    'isa': 'is a',
    'desires': 'desires',
    'hassubevent': 'has subevent',
    'partof': 'part of',
    'hascontext': 'has context',
    'hasproperty': 'has property',
    'madeof': 'made of',
    'notcapableof': 'not capable of',
    'notdesires': 'not desires',
    'receivesaction': 'receives action',
    'relatedto': 'related to',
    'usedfor': 'used for',
}

RELATION_PRIORITY = {
    'isa': 0, 'partof': 0, 'madeof': 0, 'usedfor': 0, 'capableof': 0,
    'causes': 1, 'hassubevent': 1, 'createdby': 1, 'receivesaction': 1,
    'atlocation': 1, 'hasproperty': 1,
    'desires': 2, 'notdesires': 2, 'notcapableof': 2, 'hascontext': 2,
    'antonym': 3,
    'relatedto': 4,
}

NODE_DESCRIPTION_PROMPTS = [
    'List all nodes of the graph shown in the image.',
    'Provide the names of all nodes displayed in the graph image.',
    'Can you name all the nodes shown in the graph image?',
    'Identify all the vertices in the diagram of the graph provided.',
    'Detail all the vertices from the graph depicted in the image.',
]

HIGHEST_DEGREE_PROMPTS = [
    'Name one of the node with the highest degree in the graph. And what is its degree?',
    'Identify one of the node that has the most connections in the graph and specify its degree.',
    'Can you tell me which node (name one) has the highest degree in this graph and what that degree is?',
    'Provide the name and degree of the node with the most connections in the graph.',
    'Which node in the graph has the greatest number of connections, and what is that total?',
]

NODE_DEGREE_PROMPTS = [
    'What is the degree of the node with the name "{node}"?',
    'What is the degree of the node labeled "{node}"?',
    'Can you tell me the degree of the node named "{node}"?',
    'What is the total number of connections that the node "{node}" has?',
    'How many connections does the node "{node}" have?',
]

NODE_NUMBER_PROMPTS = [
    'How many nodes are there in the graph?',
    'What is the total number of nodes in the graph?',
    'Can you tell me how many nodes are in the graph?',
    'What is the total number of vertices in the graph?',
    'How many vertices are there in the graph?',
]

EDGE_NUMBER_PROMPTS = [
    'How many edges are there in the graph?',
    'What is the total number of edges in the graph?',
    'Can you tell me how many edges are in the graph?',
    'What is the total number of connections in the graph?',
    'How many connections are there in the graph?',
]

TRIPLE_LISTING_PROMPTS = [
    'List all the triples in the graph.',
    'Provide all the triples in the graph.',
    'Can you list all the triples in the graph?',
    'Detail all the triples in the graph.',
    'Enumerate all the triples in the graph.',
]


RELATEDTO_KEEP_RATE = 0.2
MAX_SHORTEST_PATHS = 32

RELATION_IDENTIFICATION_PROMPTS = [
    'What relation connects the nodes "{node_a}" and "{node_b}"?',
    'What is the relationship between "{node_a}" and "{node_b}" in the graph?',
    'Which relation links "{node_a}" to "{node_b}"?',
    'Name the relation on the edge between "{node_a}" and "{node_b}".',
    'How are the nodes "{node_a}" and "{node_b}" related in the graph?',
]
NEIGHBOR_LISTING_PROMPTS = [
    'List all nodes directly connected to "{node}".',
    'Which nodes share an edge with "{node}"?',
    'Name every neighbor of the node "{node}".',
    'Provide all nodes adjacent to "{node}" in the graph.',
    'What nodes is "{node}" connected to?',
]
SHORTEST_PATH_PROMPTS = [
    'List the nodes on the shortest path from "{node_a}" to "{node_b}".',
    'Give the sequence of nodes connecting "{node_a}" to "{node_b}" by the shortest route.',
    'What is the shortest chain of nodes from "{node_a}" to "{node_b}"?',
    'Trace the shortest path between "{node_a}" and "{node_b}", naming each node.',
    'Provide the shortest sequence of nodes linking "{node_a}" and "{node_b}".',
]


def adjacency(graph):
    """cid -> set of neighbour cids, over pruned edges only (undirected)."""
    adj = defaultdict(set)
    for src, _rel, tgt in graph['edges']:
        adj[src].add(tgt)
        adj[tgt].add(src)
    return adj


def _too_similar(label_a, label_b, threshold=2):
    """Compare Levenshtein distance after stripping choice-letter suffixes."""
    a, b = (re.sub(r'\s*\[[A-Z](?:,[A-Z])*\]$', '', label)
            for label in (label_a, label_b))
    if abs(len(a) - len(b)) > threshold:
        return False
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1] <= threshold


def all_shortest_paths(adj, source, target):
    """Return sorted shortest cid paths, capped at MAX_SHORTEST_PATHS.

    Callers conservatively mark truncation when the cap is reached, including
    when there are exactly that many paths.
    """
    dist = {source: 0}
    parents = defaultdict(set)
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if target in dist and dist[node] >= dist[target]:
            continue
        for neighbor in sorted(adj.get(node, ())):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
            if dist[neighbor] == dist[node] + 1:
                parents[neighbor].add(node)
    if target not in dist:
        return []
    paths = []

    def backtrack(node, reverse_path):
        if len(paths) >= MAX_SHORTEST_PATHS:
            return
        if node == source:
            paths.append(list(reversed(reverse_path)))
            return
        for parent in sorted(parents[node]):
            backtrack(parent, reverse_path + [parent])
            if len(paths) >= MAX_SHORTEST_PATHS:
                break

    backtrack(target, [target])
    return sorted(paths)


def build_stage1_extended_tasks(merged_nodes, graph, adj, rng, hide_relatedto_labels):
    """Build eligible tasks using only the visible, pruned graph."""
    labels = {cid: label_for_node(merged_nodes[cid]['name'])
              for cid in sorted(graph['visible_nodes'])}
    records = []
    edges = [edge for edge in sorted(graph['edges'])
             if not _too_similar(labels[edge[0]], labels[edge[2]])
             and not (hide_relatedto_labels and edge[1] == 'relatedto')]
    if any(rel != 'relatedto' for _, rel, _ in edges):
        edges = [edge for edge in edges
                 if edge[1] != 'relatedto' or rng.random() < RELATEDTO_KEEP_RATE]
    if edges:
        src, rel, tgt = rng.choice(edges)
        a, b = labels[src], labels[tgt]
        records.append({
            'task_type': 'relation_identification',
            'prompt': rng.choice(RELATION_IDENTIFICATION_PROMPTS).format(node_a=a, node_b=b),
            'answer': f'The relation between "{a}" and "{b}" is "{RELATION_TEXT[rel]}".',
            'gold': {'relation': rel, 'relation_text': RELATION_TEXT[rel],
                     'node_a': a, 'node_b': b},
        })
    candidates = sorted(cid for cid in graph['connected_nodes'] if adj[cid])
    if candidates:
        chosen = rng.choice(candidates)
        name = labels[chosen]
        neighbor_labels = sorted(labels[cid] for cid in adj[chosen])
        records.append({
            'task_type': 'neighbor_listing',
            'prompt': rng.choice(NEIGHBOR_LISTING_PROMPTS).format(node=name),
            'answer': f'The node "{name}" is directly connected to: {", ".join(neighbor_labels)}.',
            'gold': {'node': name, 'neighbors': neighbor_labels, 'degree': len(neighbor_labels)},
        })
    pairs = list(combinations(sorted(graph['connected_nodes']), 2))
    rng.shuffle(pairs)
    for src, tgt in pairs:
        a, b = labels[src], labels[tgt]
        if tgt in adj[src] or _too_similar(a, b):
            continue
        paths = all_shortest_paths(adj, src, tgt)
        if not paths:
            continue
        path_labels = sorted([labels[cid] for cid in path] for path in paths)
        records.append({
            'task_type': 'shortest_path_listing',
            'prompt': rng.choice(SHORTEST_PATH_PROMPTS).format(node_a=a, node_b=b),
            'answer': f'The shortest path from "{a}" to "{b}" is: {" -> ".join(path_labels[0])}.',
            'gold': {'node_a': a, 'node_b': b, 'distance': len(paths[0]) - 1,
                     'paths': path_labels, 'paths_truncated': len(paths) >= MAX_SHORTEST_PATHS},
        })
        break
    return records


def load_jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def label_for_node(name):
    """Return the visible concept label without answer-choice markers."""
    return name.replace('_', ' ')


def merge_choice_graphs(statement_idx, graph_entries, statement, id2concept, n_choices=4):
    correct_label = statement['answerKey']
    choices = statement['question']['choices']
    merged_nodes = {}
    merged_edges = set()

    for choice_idx in range(n_choices):
        adj_idx = statement_idx * n_choices + choice_idx
        entry = graph_entries[adj_idx]
        choice_label = choices[choice_idx]['label']
        concepts = entry['concepts']
        qmask = entry['qmask']
        amask = entry['amask']
        adj = entry['adj']
        n_nodes = len(concepts)

        for i, cid in enumerate(concepts):
            cid = int(cid)
            if cid not in merged_nodes:
                merged_nodes[cid] = {
                    'name': id2concept[cid],
                    'in_question': False,
                    'in_choices': set(),
                }
            if qmask[i]:
                merged_nodes[cid]['in_question'] = True
            if amask[i]:
                merged_nodes[cid]['in_choices'].add(choice_label)

        adj_coo = adj.tocoo()
        for row, col, val in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            if val == 0:
                continue
            rel_idx = int(row // n_nodes)
            src_idx = int(row % n_nodes)
            tgt_idx = int(col)
            merged_edges.add((int(concepts[src_idx]), RELATIONS[rel_idx], int(concepts[tgt_idx])))

    return merged_nodes, merged_edges, correct_label


def prune_graph(merged_nodes, merged_edges, max_nodes, max_edges, max_degree, *,
                bridge_rule='qa-bridge', lifelines=True):
    if bridge_rule not in ('qa-bridge', 'core-neighbor', 'any', 'none'):
        raise ValueError(f'Unknown bridge rule: {bridge_rule}')
    node_budget = max_nodes or None
    edge_budget = max_edges or None
    neighbors = defaultdict(set)
    for src, _rel, tgt in merged_edges:
        neighbors[src].add(tgt)
        neighbors[tgt].add(src)

    q_cids = {cid for cid, node in merged_nodes.items() if node['in_question']}
    a_cids = {cid for cid, node in merged_nodes.items() if node['in_choices']}

    core = set(q_cids) | set(a_cids)
    core_truncated = node_budget is not None and len(core) > node_budget
    if core_truncated:
        # Question and answer nodes alone exceed the budget: keep all question
        # nodes first, then fill with the best-connected answer nodes.
        ranked_q = sorted(q_cids, key=lambda cid: (-len(neighbors[cid]), cid))
        ranked_a = sorted(a_cids, key=lambda cid: (-len(neighbors[cid]), cid))
        keep = set()
        for cid in ranked_q + ranked_a:
            if len(keep) >= node_budget:
                break
            keep.add(cid)
        print(
            f'[prune_graph] core question+answer nodes ({len(core)}) exceed '
            f'max_nodes={max_nodes}; truncated to {len(keep)}'
        )
    else:
        keep = set(core)

    bridges = []
    for cid in sorted(merged_nodes):
        if cid in keep:
            continue
        touches_q = any(q in neighbors[cid] for q in q_cids)
        touches_a = any(a in neighbors[cid] for a in a_cids)
        if (bridge_rule == 'any'
                or (bridge_rule == 'qa-bridge' and touches_q and touches_a)
                or (bridge_rule == 'core-neighbor' and (touches_q or touches_a))):
            bridges.append(cid)

    if bridge_rule == 'any':
        bridges.sort(key=lambda cid: (-len(neighbors[cid] & core), -len(neighbors[cid]), cid))
    else:
        bridges.sort(key=lambda cid: (-len(neighbors[cid] & core), cid))
    if node_budget is not None:
        bridges = bridges[:max(0, node_budget - len(keep))]
    keep.update(bridges)

    # Stable input order also resolves equal-priority relation/direction ties.
    edges = sorted((src, rel, tgt) for src, rel, tgt in merged_edges if src in keep and tgt in keep)

    best_for_pair = {}
    for src, rel, tgt in edges:
        pair = (min(src, tgt), max(src, tgt))
        prio = RELATION_PRIORITY.get(rel, 5)
        if pair not in best_for_pair or prio < best_for_pair[pair][0]:
            best_for_pair[pair] = (prio, src, rel, tgt)
    edges = [(src, rel, tgt) for _prio, src, rel, tgt in best_for_pair.values()]

    nodes_in_edges = {node for edge in edges for node in (edge[0], edge[2])}
    lifeline_edges = set()
    for cid in (q_cids | a_cids) & nodes_in_edges:
        candidates = [
            (RELATION_PRIORITY.get(rel, 5), (src, rel, tgt))
            for src, rel, tgt in edges
            if src == cid or tgt == cid
        ]
        if candidates:
            candidates.sort()
            lifeline_edges.add(candidates[0][1])

    degree = defaultdict(int)
    for src, _rel, tgt in edges:
        degree[src] += 1
        degree[tgt] += 1

    if max_degree > 0 and any(count > max_degree for count in degree.values()):
        def edge_rank(edge):
            src, rel, tgt = edge
            touches_qa = src in q_cids or src in a_cids or tgt in q_cids or tgt in a_cids
            return (RELATION_PRIORITY.get(rel, 5), 0 if touches_qa else 1, edge)

        edges.sort(key=edge_rank)
        kept_edges = []
        deg = defaultdict(int)
        for edge in edges:
            if lifelines and edge in lifeline_edges:
                kept_edges.append(edge)
                deg[edge[0]] += 1
                deg[edge[2]] += 1
        for edge in edges:
            if lifelines and edge in lifeline_edges:
                continue
            src, _rel, tgt = edge
            if deg[src] >= max_degree or deg[tgt] >= max_degree:
                continue
            if not lifelines and src == tgt and deg[src] + 2 > max_degree:
                continue
            kept_edges.append(edge)
            deg[src] += 1
            deg[tgt] += 1
        edges = kept_edges

    edges_before_truncation = len(edges)
    if edge_budget is not None and len(edges) > edge_budget:
        edges.sort(key=lambda edge: (0 if edge in lifeline_edges else 1, RELATION_PRIORITY.get(edge[1], 5), edge))
        edges = edges[:edge_budget]
        print(f'[prune_graph] max_edges truncation: {edges_before_truncation} -> {len(edges)} edges')

    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2]))

    connected = set()
    for src, _rel, tgt in edges:
        connected.add(src)
        connected.add(tgt)

    disconnected_answers = sorted(cid for cid in a_cids if cid not in connected)
    disconnected_questions = sorted(cid for cid in q_cids if cid not in connected)
    connected_keep = {cid for cid in keep if cid in connected}
    visible_nodes = sorted(connected_keep | set(disconnected_answers))

    return {
        'connected_nodes': sorted(connected_keep),
        'visible_nodes': visible_nodes,
        'edges': edges,
        'disconnected_answers': disconnected_answers,
        'disconnected_questions': disconnected_questions,
        'q_cids': sorted(q_cids),
        'a_cids': sorted(a_cids),
        'pruning': {'max_nodes': max_nodes, 'max_edges': max_edges, 'max_degree': max_degree,
                    'bridge_rule': bridge_rule, 'lifelines': lifelines,
                    'core_size': len(core), 'core_truncated': core_truncated,
                    'bridges_added': len(bridges),
                    'edges_before_truncation': edges_before_truncation, 'edges_after': len(edges)},
    }


def node_style(cid, merged_nodes, correct_label, reveal_correct_answer=False):
    info = merged_nodes[cid]
    if info['in_question']:
        return label_for_node(info['name']), '#ADD8E6', '1.5'
    if info['in_choices']:
        is_correct = reveal_correct_answer and correct_label in info['in_choices']
        fill = '#90EE90' if is_correct else '#E0E0E0'
        penwidth = '3' if is_correct else '1.5'
        return label_for_node(info['name']), fill, penwidth
    return label_for_node(info['name']), 'white', '1.5'


def render_graph(
    image_stem, merged_nodes, graph, correct_label, engine, hide_relatedto_labels,
    reveal_correct_answer=False, dpi=200, disconnected_rows=3,
    node_fontsize=18, edge_fontsize=14, nodesep=0.5, ranksep=0.7,
    graph_size=None, graph_ratio=None, rankdir="LR",
):
    dot = graphviz.Digraph(format='png', engine=engine)
    graph_attrs = {
        'overlap': 'false',
        'splines': 'true',
        'dpi': str(dpi),
        'bgcolor': 'white',
        'pad': '0.3',
        'nodesep': str(nodesep),
        'ranksep': str(ranksep),
    }
    if engine == 'dot':
        graph_attrs['rankdir'] = rankdir
    if graph_size is not None:
        graph_attrs['size'] = graph_size
    if graph_ratio is not None:
        graph_attrs['ratio'] = graph_ratio
    dot.attr('graph', **graph_attrs)
    dot.attr(
        'node',
        shape='box',
        style='rounded,filled',
        fontname='Helvetica-Bold',
        fontsize=str(node_fontsize),
        margin='0.2,0.1',
        penwidth='1.5',
        fillcolor='white',
    )
    dot.attr('edge', fontname='Helvetica', fontsize=str(edge_fontsize), arrowsize='0.8', penwidth='1.2')

    for cid in graph['connected_nodes']:
        label, fill, penwidth = node_style(cid, merged_nodes, correct_label, reveal_correct_answer)
        dot.node(str(cid), label=label, fillcolor=fill, penwidth=penwidth)

    if graph['disconnected_answers']:
        with dot.subgraph(name='cluster_no_evidence') as sub:
            sub.attr(
                label='no connections found in KG',
                fontsize='14',
                fontname='Helvetica',
                style='dashed',
                color='gray50',
            )
            disconnected = graph['disconnected_answers']
            # Wrap into a grid: each column is a rank (same-rank nodes stack
            # vertically under rankdir=LR), columns chained left-to-right via
            # an invisible edge between one anchor node per column.
            columns = [
                disconnected[i:i + disconnected_rows]
                for i in range(0, len(disconnected), disconnected_rows)
            ]
            prev_anchor = None
            for column in columns:
                with sub.subgraph() as col:
                    col.attr(rank='same')
                    for cid in column:
                        label, fill, penwidth = node_style(cid, merged_nodes, correct_label, reveal_correct_answer)
                        col.node(str(cid), label=label, fillcolor=fill, penwidth=penwidth, style='rounded,filled,dashed')
                anchor = column[0]
                if prev_anchor is not None:
                    sub.edge(str(prev_anchor), str(anchor), style='invis')
                prev_anchor = anchor

    for src, rel, tgt in graph['edges']:
        if rel == 'relatedto' and hide_relatedto_labels:
            dot.edge(str(src), str(tgt), color='gray60', penwidth='1.0')
        else:
            dot.edge(str(src), str(tgt), label=RELATION_TEXT.get(rel, rel), penwidth='1.3')

    dot.render(str(image_stem), cleanup=True)
    return image_stem.with_suffix('.png')


def visible_node_names(merged_nodes, graph):
    return [label_for_node(merged_nodes[cid]['name']) for cid in graph['visible_nodes']]


def edge_degree(graph):
    degree = defaultdict(int)
    for src, _rel, tgt in graph['edges']:
        degree[src] += 1
        degree[tgt] += 1
    return degree


def triple_text(merged_nodes, edge):
    src, rel, tgt = edge
    src_label = label_for_node(merged_nodes[src]['name'])
    tgt_label = label_for_node(merged_nodes[tgt]['name'])
    return f'({src_label}, {RELATION_TEXT.get(rel, rel)}, {tgt_label})'


def build_stage1_records(
    image_path, split, statement_idx, merged_nodes, graph, rng, tasks_per_graph,
    stage1_task_set="paper", stage1_balance="pool", hide_relatedto_labels=False,
):
    nodes = visible_node_names(merged_nodes, graph)
    edges = graph['edges']
    degree = edge_degree(graph)
    connected_nodes = graph['connected_nodes']
    max_degree = max((degree[cid] for cid in connected_nodes), default=0)
    highest_nodes = [cid for cid in connected_nodes if degree[cid] == max_degree]
    chosen_highest = rng.choice(highest_nodes) if highest_nodes else None

    candidates = [
        {
            'task_type': 'node_description',
            'prompt': rng.choice(NODE_DESCRIPTION_PROMPTS),
            'answer': 'The image depicts the following nodes: ' + ', '.join(nodes) + '.',
        },
        {
            'task_type': 'node_number',
            'prompt': rng.choice(NODE_NUMBER_PROMPTS),
            'answer': f'There are {len(nodes)} nodes in the graph.',
        },
        {
            'task_type': 'edge_number',
            'prompt': rng.choice(EDGE_NUMBER_PROMPTS),
            'answer': f'There are {len(edges)} edges in the graph.',
        },
        {
            'task_type': 'triple_listing',
            'prompt': rng.choice(TRIPLE_LISTING_PROMPTS),
            'answer': 'The triples in the graph are listed as: ' + ', '.join(triple_text(merged_nodes, edge) for edge in edges) + '.',
        },
    ]

    if chosen_highest is not None:
        name = label_for_node(merged_nodes[chosen_highest]['name'])
        candidates.append({
            'task_type': 'highest_node_degree',
            'prompt': rng.choice(HIGHEST_DEGREE_PROMPTS),
            'answer': f'One node with the highest degree is "{name}" with a degree of {max_degree}.',
        })

    node_degree_options = [cid for cid in graph['visible_nodes']]
    if node_degree_options:
        chosen = rng.choice(node_degree_options)
        name = label_for_node(merged_nodes[chosen]['name'])
        candidates.append({
            'task_type': 'node_degree',
            'prompt': rng.choice(NODE_DEGREE_PROMPTS).format(node=name),
            'answer': f'The degree of the node "{name}" is {degree[chosen]}.',
        })

    if stage1_task_set == "extended":
        candidates.extend(build_stage1_extended_tasks(
            merged_nodes, graph, adjacency(graph), rng, hide_relatedto_labels
        ))

    if stage1_balance == "pool" and tasks_per_graph is not None and tasks_per_graph < len(candidates):
        # Preserve the historical paper pool order and RNG draws exactly.
        if stage1_task_set == "extended":
            candidates = sorted(candidates, key=lambda item: item["task_type"])
        candidates = rng.sample(candidates, tasks_per_graph)

    records = []
    for item in candidates:
        records.append({
            'image': str(image_path),
            'split': split,
            'statement_idx': statement_idx,
            'task_type': item['task_type'],
            'prompt': item['prompt'],
            'answer': item['answer'],
            **({'gold': item['gold']} if 'gold' in item else {}),
            'source': 'graphvis_stage1_clean_union_of_four',
        })
    return records


def build_stage2_record(image_path, split, statement_idx, statement):
    choices = statement['question']['choices']
    choice_text = '\n'.join(f'{choice["label"]}. {choice["text"]}' for choice in choices)
    prompt = (
        'The image represents a knowledge graph relevant to the question, which may or may not be useful. '
        f'Question: {statement["question"]["stem"]}\nChoices:\n{choice_text}\n'
        "Answer with the correct option's letter."
    )
    return {
        'image': str(image_path),
        'split': split,
        'statement_idx': statement_idx,
        'task_type': 'obqa_answer',
        'prompt': prompt,
        'answer': statement['answerKey'],
        'source': 'graphvis_stage2_clean_union_of_four',
    }


def graph_metadata(statement_idx, statement, merged_nodes, graph, image_path):
    return {
        'statement_idx': statement_idx,
        'question': statement['question']['stem'],
        'answerKey': statement['answerKey'],
        'choices': statement['question']['choices'],
        'image': str(image_path),
        'visible_nodes': [
            {
                'cid': cid,
                'label': label_for_node(merged_nodes[cid]['name']),
                'name': merged_nodes[cid]['name'],
                'in_question': merged_nodes[cid]['in_question'],
                'in_choices': sorted(merged_nodes[cid]['in_choices']),
                'connected': cid in graph['connected_nodes'],
            }
            for cid in graph['visible_nodes']
        ],
        'edges': [
            {
                'source_cid': src,
                'relation': rel,
                'target_cid': tgt,
                'source_label': label_for_node(merged_nodes[src]['name']),
                'target_label': label_for_node(merged_nodes[tgt]['name']),
            }
            for src, rel, tgt in graph['edges']
        ],
        'disconnected_answer_cids': graph['disconnected_answers'],
        'disconnected_question_cids': graph['disconnected_questions'],
        **({'pruning': dict(graph['pruning'])} if 'pruning' in graph else {}),
        'aggregation_note': 'Union of four QA-GNN answer-choice graphs; this is an implementation interpretation, not directly specified by GraphVis.',
    }


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--data-root', type=Path, default=Path('data_preprocessed_release'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/graphvis_obqa'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--tasks-per-graph', type=int, default=None)
    parser.add_argument('--stage1-task-set', choices=['paper', 'extended'], default='paper')
    parser.add_argument('--stage1-balance', choices=['pool', 'per-task'], default='pool',
                        help='per-task emits every eligible task, ignoring tasks-per-graph.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--max-nodes', type=int, default=18, help='Node cap; 0 disables it.')
    parser.add_argument('--max-edges', type=int, default=60, help='Edge cap; 0 disables it.')
    parser.add_argument('--max-degree', type=int, default=0, help='Degree cap; 0 disables it.')
    parser.add_argument('--bridge-rule', choices=['qa-bridge', 'core-neighbor', 'any', 'none'],
                        default='qa-bridge', help='Eligibility rule for filler nodes.')
    parser.add_argument('--no-lifelines', dest='lifelines', action='store_false',
                        help='Enforce a hard degree cap by disabling lifeline exemptions.')
    parser.add_argument('--engine', default='dot', help='Graphviz layout engine. Local install currently supports dot.')
    parser.add_argument('--node-fontsize', type=int, default=18)
    parser.add_argument('--edge-fontsize', type=int, default=14)
    parser.add_argument('--nodesep', type=float, default=0.5)
    parser.add_argument('--ranksep', type=float, default=0.7)
    parser.add_argument('--graph-size')
    parser.add_argument('--graph-ratio')
    parser.add_argument('--rankdir', choices=['LR', 'TB'], default='LR')
    parser.add_argument('--dpi', type=int, default=200, help='Rendered PNG resolution.')
    parser.add_argument(
        '--disconnected-rows', type=int, default=3,
        help='Number of rows to wrap the "no connections found in KG" nodes into.',
    )
    parser.add_argument(
        '--hide-relatedto-labels',
        action='store_true',
        help='Render relatedto edges unlabeled. Leave this off for Stage 1 triple-listing data.',
    )
    parser.add_argument(
        '--reveal-correct-answer',
        action='store_true',
        help=(
            'Debug only: highlight the correct-answer node in green with a bold border. '
            'Leaks the answer visually -- must stay OFF (default) for any Stage 2 training data.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_degree < 0 or args.max_nodes < 0 or args.max_edges < 0:
        raise ValueError('Require max_degree >= 0, max_nodes >= 0 and max_edges >= 0')
    rng = random.Random(args.seed)

    cpnet_dir = args.data_root / 'cpnet'
    obqa_dir = args.data_root / 'obqa'
    concept_path = cpnet_dir / 'concept.txt'
    graph_path = obqa_dir / 'graph' / f'{args.split}.graph.adj.pk'
    statement_path = obqa_dir / 'statement' / f'{args.split}.statement.jsonl'

    with concept_path.open('r', encoding='utf-8') as f:
        id2concept = [line.strip() for line in f]
    with graph_path.open('rb') as f:
        graph_entries = pickle.load(f)
    statements = load_jsonl(statement_path)

    end = min(len(statements), args.start + args.limit)
    selected = range(args.start, end)

    split_out = args.out_dir / args.split
    image_dir = split_out / 'images'
    graph_dir = split_out / 'graphs'
    # Never replace an existing split or any of its selected image/metadata files.
    suffix = f'{args.start}_{end}'
    planned = [split_out / f'{prefix}_{suffix}.jsonl' for prefix in
               ('stage1_graph_comprehension', 'stage2_obqa', 'graph_metadata')]
    planned += [image_dir / f'q{idx:05d}_clean.png' for idx in selected]
    planned += [graph_dir / f'q{idx:05d}.jsonl' for idx in selected]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f'Preserve existing splits; use a new --out-dir. First collision: {existing[0]}')
    image_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    stage1_records = []
    stage2_records = []
    metadata_records = []

    for statement_idx in selected:
        statement = statements[statement_idx]
        merged_nodes, merged_edges, correct_label = merge_choice_graphs(
            statement_idx, graph_entries, statement, id2concept
        )
        graph = prune_graph(merged_nodes, merged_edges, args.max_nodes, args.max_edges, args.max_degree,
                            bridge_rule=args.bridge_rule, lifelines=args.lifelines)
        image_stem = image_dir / f'q{statement_idx:05d}_clean'
        image_path = render_graph(
            image_stem, merged_nodes, graph, correct_label, args.engine, args.hide_relatedto_labels,
            args.reveal_correct_answer, args.dpi, args.disconnected_rows,
            args.node_fontsize, args.edge_fontsize, args.nodesep, args.ranksep,
            args.graph_size, args.graph_ratio, args.rankdir,
        )
        rel_image_path = image_path.relative_to(args.out_dir)

        stage1_records.extend(
            build_stage1_records(
                rel_image_path, args.split, statement_idx, merged_nodes, graph, rng, args.tasks_per_graph,
                args.stage1_task_set, args.stage1_balance, args.hide_relatedto_labels,
            )
        )
        stage2_records.append(build_stage2_record(rel_image_path, args.split, statement_idx, statement))
        metadata = graph_metadata(statement_idx, statement, merged_nodes, graph, rel_image_path)
        metadata_records.append(metadata)
        write_jsonl(graph_dir / f'q{statement_idx:05d}.jsonl', [metadata])

    suffix = f'{args.start}_{end}'
    write_jsonl(split_out / f'stage1_graph_comprehension_{suffix}.jsonl', stage1_records)
    write_jsonl(split_out / f'stage2_obqa_{suffix}.jsonl', stage2_records)
    write_jsonl(split_out / f'graph_metadata_{suffix}.jsonl', metadata_records)

    print(f'Wrote {len(stage1_records)} Stage 1 records')
    print(f'Wrote {len(stage2_records)} Stage 2 records')
    print(f'Images: {image_dir}')
    print(f'JSONL: {split_out}')


if __name__ == '__main__':
    main()
