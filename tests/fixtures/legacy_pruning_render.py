"""Frozen pre-change pruning/render functions for compatibility checks."""
import graphviz
from collections import defaultdict
from generate_graphvis_datasets import RELATION_PRIORITY, RELATION_TEXT, node_style

def prune_graph(merged_nodes, merged_edges, max_nodes, max_edges, max_degree):
    neighbors = defaultdict(set)
    for src, _rel, tgt in merged_edges:
        neighbors[src].add(tgt)
        neighbors[tgt].add(src)

    q_cids = {cid for cid, node in merged_nodes.items() if node['in_question']}
    a_cids = {cid for cid, node in merged_nodes.items() if node['in_choices']}

    core = set(q_cids) | set(a_cids)
    if len(core) > max_nodes:
        # Question and answer nodes alone exceed the budget: keep all question
        # nodes first, then fill with the best-connected answer nodes.
        ranked_q = sorted(q_cids, key=lambda cid: -len(neighbors[cid]))
        ranked_a = sorted(a_cids, key=lambda cid: -len(neighbors[cid]))
        keep = set()
        for cid in ranked_q + ranked_a:
            if len(keep) >= max_nodes:
                break
            keep.add(cid)
        print(
            f'[prune_graph] core question+answer nodes ({len(core)}) exceed '
            f'max_nodes={max_nodes}; truncated to {len(keep)}'
        )
    else:
        keep = set(core)

    bridges = []
    for cid in merged_nodes:
        if cid in keep:
            continue
        touches_q = any(q in neighbors[cid] for q in q_cids)
        touches_a = any(a in neighbors[cid] for a in a_cids)
        if touches_q and touches_a:
            bridges.append(cid)

    bridges.sort(key=lambda cid: len(neighbors[cid] & (q_cids | a_cids)), reverse=True)
    keep |= set(bridges[:max(0, max_nodes - len(keep))])

    edges = [(src, rel, tgt) for src, rel, tgt in merged_edges if src in keep and tgt in keep]

    best_for_pair = {}
    for src, rel, tgt in edges:
        pair = (min(src, tgt), max(src, tgt))
        prio = RELATION_PRIORITY.get(rel, 5)
        if pair not in best_for_pair or prio < best_for_pair[pair][0]:
            best_for_pair[pair] = (prio, src, rel, tgt)
    edges = [(src, rel, tgt) for _prio, src, rel, tgt in best_for_pair.values()]

    nodes_in_edges = {node for edge in edges for node in (edge[0], edge[2])}
    lifelines = set()
    for cid in (q_cids | a_cids) & nodes_in_edges:
        candidates = [
            (RELATION_PRIORITY.get(rel, 5), (src, rel, tgt))
            for src, rel, tgt in edges
            if src == cid or tgt == cid
        ]
        if candidates:
            candidates.sort(key=lambda item: item[0])
            lifelines.add(candidates[0][1])

    degree = defaultdict(int)
    for src, _rel, tgt in edges:
        degree[src] += 1
        degree[tgt] += 1

    if any(count > max_degree for count in degree.values()):
        def edge_rank(edge):
            src, rel, tgt = edge
            touches_qa = src in q_cids or src in a_cids or tgt in q_cids or tgt in a_cids
            return (RELATION_PRIORITY.get(rel, 5), 0 if touches_qa else 1)

        edges.sort(key=edge_rank)
        kept_edges = []
        deg = defaultdict(int)
        for edge in edges:
            if edge in lifelines:
                kept_edges.append(edge)
                deg[edge[0]] += 1
                deg[edge[2]] += 1
        for edge in edges:
            if edge in lifelines:
                continue
            src, _rel, tgt = edge
            if deg[src] >= max_degree or deg[tgt] >= max_degree:
                continue
            kept_edges.append(edge)
            deg[src] += 1
            deg[tgt] += 1
        edges = kept_edges

    if len(edges) > max_edges:
        edges.sort(key=lambda edge: (0 if edge in lifelines else 1, RELATION_PRIORITY.get(edge[1], 5)))
        edges = edges[:max_edges]

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
    }

def render_graph(
    image_stem, merged_nodes, graph, correct_label, engine, hide_relatedto_labels,
    reveal_correct_answer=False, dpi=200, disconnected_rows=3,
):
    dot = graphviz.Digraph(format='png', engine=engine)
    graph_attrs = {
        'overlap': 'false',
        'splines': 'true',
        'dpi': str(dpi),
        'bgcolor': 'white',
        'pad': '0.3',
        'nodesep': '0.5',
        'ranksep': '0.7',
    }
    if engine == 'dot':
        graph_attrs['rankdir'] = 'LR'
    dot.attr('graph', **graph_attrs)
    dot.attr(
        'node',
        shape='box',
        style='rounded,filled',
        fontname='Helvetica-Bold',
        fontsize='18',
        margin='0.2,0.1',
        penwidth='1.5',
        fillcolor='white',
    )
    dot.attr('edge', fontname='Helvetica', fontsize='14', arrowsize='0.8', penwidth='1.2')

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
