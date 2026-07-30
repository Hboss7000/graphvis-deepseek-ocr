import json
import graphviz
import sys
import pickle
from collections import defaultdict

# Usage: python3 render_question.py <statement_idx>
# Renders all 4 OBQA graphs (one per answer choice) for the given question.

STMT_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_CHOICES = 4  # OBQA

# === Load shared resources once ===
with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]

relations = [
    'antonym', 'atlocation', 'capableof', 'causes', 'createdby',
    'isa', 'desires', 'hassubevent', 'partof', 'hascontext',
    'hasproperty', 'madeof', 'notcapableof', 'notdesires', 'receivesaction',
    'relatedto', 'usedfor',
]

with open('train.graph.adj.pk', 'rb') as f:
    data = pickle.load(f)

with open('train.statement.jsonl', 'r') as f:
    statements = [json.loads(line) for line in f]

stmt = statements[STMT_IDX]
correct_label = stmt['answerKey']
question = stmt['question']['stem']
choices = stmt['question']['choices']

print(f'=== Statement {STMT_IDX} ===')
print(f'Q: {question}')
for i, c in enumerate(choices):
    mark = ' (CORRECT)' if c['label'] == correct_label else ''
    print(f'  [{c["label"]}] {c["text"]}{mark}')
print()

# === Per-choice rendering ===
role_colors = {
    'question': 'lightblue',
    'answer': 'lightyellow',
    'other': 'white',
}

for choice_idx in range(N_CHOICES):
    adj_idx = STMT_IDX * N_CHOICES + choice_idx
    entry = data[adj_idx]
    choice = choices[choice_idx]
    is_correct = (choice['label'] == correct_label)

    # Decode entry to nodes + links
    concepts = entry['concepts']
    qmask = entry['qmask']
    amask = entry['amask']
    adj = entry['adj']
    n_nodes = len(concepts)
    node_names = [id2concept[cid] for cid in concepts]

    nodes = []
    for i, (cid, name) in enumerate(zip(concepts, node_names)):
        role = 'question' if qmask[i] else ('answer' if amask[i] else 'other')
        nodes.append({'id': i, 'cid': int(cid), 'name': name, 'role': role})

    adj_coo = adj.tocoo()
    links = []
    for row, col, val in zip(adj_coo.row, adj_coo.col, adj_coo.data):
        if val == 0:
            continue
        r = int(row // n_nodes)
        i = int(row % n_nodes)
        j = int(col)
        links.append({'source': i, 'target': j, 'relation': relations[r]})

    # Filter to bridges
    neighbors = defaultdict(set)
    for link in links:
        neighbors[link['source']].add(link['target'])
        neighbors[link['target']].add(link['source'])

    q_ids = {n['id'] for n in nodes if n['role'] == 'question'}
    a_ids = {n['id'] for n in nodes if n['role'] == 'answer'}

    keep = set(q_ids) | set(a_ids)
    for nid in range(len(nodes)):
        if nid in keep:
            continue
        touches_q = any(q in neighbors[nid] for q in q_ids)
        touches_a = any(a in neighbors[nid] for a in a_ids)
        if touches_q and touches_a:
            keep.add(nid)

    filtered_nodes = [n for n in nodes if n['id'] in keep]
    filtered_links = [
        l for l in links
        if l['source'] in keep and l['target'] in keep
    ]

    # Render
    g = graphviz.Digraph(format='png', engine='sfdp')

    title = f'[{choice["label"]}] "{choice["text"]}"'
    if is_correct:
        title += '   ★ CORRECT ★'
    title += f'\\n(adj entry {adj_idx}: {len(filtered_nodes)} nodes, {len(filtered_links)} edges)'

    g.attr('graph',
           overlap='false', splines='true', dpi='200',
           bgcolor='white', pad='0.5',
           nodesep='0.6', ranksep='0.8',
           label=title, labelloc='t', fontsize='18', fontname='Helvetica')
    g.attr('node',
           shape='ellipse', fontname='Helvetica', fontsize='14',
           width='1.2', height='0.6', margin='0.15,0.08')
    g.attr('edge',
           fontname='Helvetica', fontsize='11', arrowsize='0.7')

    for node in filtered_nodes:
        label = node['name'].replace('_', ' ')
        g.node(
            str(node['id']),
            label=label,
            style='filled',
            fillcolor=role_colors.get(node['role'], 'white'),
        )

    for link in filtered_links:
        g.edge(
            str(link['source']),
            str(link['target']),
            label=link['relation'],
        )

    out_name = f'q{STMT_IDX}_choice{choice_idx}_{choice["label"]}'
    g.render(out_name, cleanup=True)
    marker = ' [CORRECT]' if is_correct else ''
    print(f'  Choice {choice["label"]} "{choice["text"]}"{marker}: '
          f'{len(filtered_nodes)} nodes, {len(filtered_links)} edges → {out_name}.png')
