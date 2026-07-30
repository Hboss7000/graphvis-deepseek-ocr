import json
import graphviz
import sys
import pickle
from collections import defaultdict

# Usage: python3 render_question_unified.py <statement_idx>
# Renders ONE image showing the union of all 4 answer-choice subgraphs.

STMT_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_CHOICES = 4  # OBQA

# === Shared resources ===
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

print(f'Q: {question}')
for c in choices:
    mark = ' (CORRECT)' if c['label'] == correct_label else ''
    print(f'  [{c["label"]}] {c["text"]}{mark}')
print()

# === Merge all 4 entries into a single graph ===
# Strategy:
# - Concept ids are global (they reference concept.txt), so we deduplicate by cid.
# - Each merged node tracks which choice(s) it appeared in (for color coding).
# - Edges are deduplicated by (source_cid, relation, target_cid).

# Map cid -> merged node info
merged_nodes = {}  # cid -> {'name', 'in_question', 'in_choices': set of choice labels}
merged_edges = set()  # set of (src_cid, relation, tgt_cid)

for choice_idx in range(N_CHOICES):
    adj_idx = STMT_IDX * N_CHOICES + choice_idx
    entry = data[adj_idx]
    choice_label = choices[choice_idx]['label']

    concepts = entry['concepts']
    qmask = entry['qmask']
    amask = entry['amask']
    adj = entry['adj']
    n_nodes = len(concepts)

    # Register nodes
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

    # Register edges (dedup by global cid triple)
    adj_coo = adj.tocoo()
    for row, col, val in zip(adj_coo.row, adj_coo.col, adj_coo.data):
        if val == 0:
            continue
        r = int(row // n_nodes)
        i = int(row % n_nodes)
        j = int(col)
        src_cid = int(concepts[i])
        tgt_cid = int(concepts[j])
        merged_edges.add((src_cid, relations[r], tgt_cid))

print(f'Merged graph: {len(merged_nodes)} unique nodes, {len(merged_edges)} unique edges')

# === Filter: keep Q nodes, A nodes, and bridges (any node connected to a Q AND to any A) ===
# Build undirected adjacency over cids
neighbors = defaultdict(set)
for src_cid, rel, tgt_cid in merged_edges:
    neighbors[src_cid].add(tgt_cid)
    neighbors[tgt_cid].add(src_cid)

q_cids = {cid for cid, info in merged_nodes.items() if info['in_question']}
a_cids = {cid for cid, info in merged_nodes.items() if info['in_choices']}

keep = set(q_cids) | set(a_cids)
for cid in list(merged_nodes.keys()):
    if cid in keep:
        continue
    touches_q = any(q in neighbors[cid] for q in q_cids)
    touches_a = any(a in neighbors[cid] for a in a_cids)
    if touches_q and touches_a:
        keep.add(cid)

filtered_edges = [
    (s, r, t) for (s, r, t) in merged_edges
    if s in keep and t in keep
]
print(f'After filtering: {len(keep)} nodes, {len(filtered_edges)} edges')

# === Render ===
g = graphviz.Digraph(format='png', engine='dot')

title = f'Q{STMT_IDX}: {question}'
g.attr('graph',
       overlap='false', splines='true', dpi='200',
       bgcolor='white', pad='0.5',
       nodesep='0.6', ranksep='0.8',
       label=title, labelloc='t',
       fontsize='20', fontname='Helvetica')
g.attr('node',
       shape='ellipse', fontname='Helvetica', fontsize='12',
       width='1.0', height='0.5', margin='0.12,0.06')
g.attr('edge',
       fontname='Helvetica', fontsize='10', arrowsize='0.7')

# Color scheme: distinct color per answer choice; question nodes blue;
# nodes that appear in multiple choices get a "shared" color.
choice_colors = {
    'A': '#FFD9B3',  # peach
    'B': '#D9F2D9',  # mint
    'C': '#E6D9F2',  # lilac
    'D': '#FFFFB3',  # pale yellow
}
correct_color = '#90EE90'  # bright green for correct answer

for cid in keep:
    info = merged_nodes[cid]
    label = info['name'].replace('_', ' ')

    if info['in_question']:
        fill = 'lightblue'
    elif info['in_choices']:
        if len(info['in_choices']) == 1:
            label_letter = next(iter(info['in_choices']))
            if label_letter == correct_label:
                fill = correct_color
            else:
                fill = choice_colors[label_letter]
            label = f'{label}\n[{label_letter}]'
        else:
            fill = '#E0E0E0'  # gray = appears in multiple answer choices
            label = f'{label}\n[{",".join(sorted(info["in_choices"]))}]'
    else:
        fill = 'white'

    g.node(str(cid), label=label, style='filled', fillcolor=fill)

for src_cid, rel, tgt_cid in filtered_edges:
    g.edge(str(src_cid), str(tgt_cid), label=rel)

out_name = f'q{STMT_IDX}_unified'
g.render(out_name, cleanup=True)
print(f'Wrote {out_name}.png')
