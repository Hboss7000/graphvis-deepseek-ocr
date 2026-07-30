import json
import graphviz
import sys
import pickle
from collections import defaultdict

# ============================================================
# render_clean.py
# Usage: python3 render_clean.py <statement_idx>
#
# Renders ONE unified image per OBQA question (union of all 4
# answer-choice subgraphs), optimized for DeepSeek-OCR:
#   - node/edge caps so text survives downscaling to 640px
#   - per-node degree cap to avoid unreadable hubs
#   - parallel edges deduplicated (most informative relation wins)
#   - 'relatedto' drawn unlabeled/gray (de-emphasized)
#   - Q/A nodes protected by "lifeline" edges (never orphaned by pruning)
#   - genuinely disconnected answer concepts shown in a dashed
#     "no connections found in KG" cluster instead of floating
# ============================================================

STMT_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_CHOICES = 4  # OBQA

# === Tuning knobs ===
MAX_NODES = 18
MAX_EDGES = 30
MAX_DEGREE = 5
RELATION_PRIORITY = {   # lower = more informative = kept first
    'isa': 0, 'partof': 0, 'madeof': 0, 'usedfor': 0, 'capableof': 0,
    'causes': 1, 'hassubevent': 1, 'createdby': 1, 'receivesaction': 1,
    'atlocation': 1, 'hasproperty': 1,
    'desires': 2, 'notdesires': 2, 'notcapableof': 2, 'hascontext': 2,
    'antonym': 3,
    'relatedto': 4,
}

# === Load shared resources ===
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

# === Merge the 4 per-choice entries (union, dedup by global cid) ===
merged_nodes = {}   # cid -> {'name', 'in_question', 'in_choices'}
merged_edges = set()

for choice_idx in range(N_CHOICES):
    adj_idx = STMT_IDX * N_CHOICES + choice_idx
    entry = data[adj_idx]
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
        r = int(row // n_nodes)
        i = int(row % n_nodes)
        j = int(col)
        merged_edges.add((int(concepts[i]), relations[r], int(concepts[j])))

print(f'Merged: {len(merged_nodes)} nodes, {len(merged_edges)} edges')

# === Step 1: bridge filter ===
neighbors = defaultdict(set)
for s, r, t in merged_edges:
    neighbors[s].add(t)
    neighbors[t].add(s)

q_cids = {c for c, n in merged_nodes.items() if n['in_question']}
a_cids = {c for c, n in merged_nodes.items() if n['in_choices']}

keep = set(q_cids) | set(a_cids)
bridges = []
for cid in merged_nodes:
    if cid in keep:
        continue
    tq = any(q in neighbors[cid] for q in q_cids)
    ta = any(a in neighbors[cid] for a in a_cids)
    if tq and ta:
        bridges.append(cid)

# === Step 2: cap node count — keep best-connected bridges ===
def bridge_score(cid):
    return len(neighbors[cid] & (q_cids | a_cids))

bridges.sort(key=bridge_score, reverse=True)
budget = max(0, MAX_NODES - len(keep))
if len(bridges) > budget:
    print(f'Pruned {len(bridges) - budget} of {len(bridges)} bridge nodes (cap {MAX_NODES})')
keep |= set(bridges[:budget])

# === Step 3a: edges among kept nodes only ===
edges = [(s, r, t) for (s, r, t) in merged_edges if s in keep and t in keep]

# === Step 3b: deduplicate node pairs (best relation wins) ===
best_for_pair = {}
for s, r, t in edges:
    pair = (min(s, t), max(s, t))
    prio = RELATION_PRIORITY.get(r, 5)
    if pair not in best_for_pair or prio < best_for_pair[pair][0]:
        best_for_pair[pair] = (prio, s, r, t)
edges = [(s, r, t) for (_, s, r, t) in best_for_pair.values()]

# === Step 3c: lifeline edges — each Q/A node's best edge is untouchable ===
nodes_in_edges = {n for e in edges for n in (e[0], e[2])}
qa_in_graph = (q_cids | a_cids) & nodes_in_edges
lifelines = set()
for n in qa_in_graph:
    candidates = [(RELATION_PRIORITY.get(r, 5), (s, r, t))
                  for (s, r, t) in edges if s == n or t == n]
    if candidates:
        candidates.sort(key=lambda x: x[0])
        lifelines.add(candidates[0][1])

# === Step 3d: per-node degree cap (lifelines exempt) ===
degree = defaultdict(int)
for s, r, t in edges:
    degree[s] += 1
    degree[t] += 1

if any(d > MAX_DEGREE for d in degree.values()):
    def edge_rank(e):
        s, r, t = e
        touches_qa = (s in q_cids or s in a_cids or t in q_cids or t in a_cids)
        return (RELATION_PRIORITY.get(r, 5), 0 if touches_qa else 1)
    edges.sort(key=edge_rank)

    kept_edges = []
    deg = defaultdict(int)
    for e in edges:                      # lifelines first, unconditionally
        if e in lifelines:
            kept_edges.append(e)
            deg[e[0]] += 1
            deg[e[2]] += 1
    for e in edges:
        if e in lifelines:
            continue
        s, r, t = e
        if deg[s] >= MAX_DEGREE or deg[t] >= MAX_DEGREE:
            continue
        kept_edges.append(e)
        deg[s] += 1
        deg[t] += 1
    print(f'Degree cap: {len(edges)} -> {len(kept_edges)} edges')
    edges = kept_edges

# === Step 3e: global edge cap (lifelines protected) ===
if len(edges) > MAX_EDGES:
    edges.sort(key=lambda e: (0 if e in lifelines else 1,
                              RELATION_PRIORITY.get(e[1], 5)))
    print(f'Global cap: {len(edges)} -> {MAX_EDGES} edges')
    edges = edges[:MAX_EDGES]

# === Step 3f: identify connectivity; disconnected answers go to cluster ===
connected = set()
for s, r, t in edges:
    connected.add(s)
    connected.add(t)

disconnected_a = sorted(c for c in a_cids if c not in connected)
disconnected_q = sorted(c for c in q_cids if c not in connected)
if disconnected_a:
    names = [merged_nodes[c]['name'] for c in disconnected_a]
    print(f'Answer concepts with no KG connection (drawn in cluster): {names}')
if disconnected_q:
    names = [merged_nodes[c]['name'] for c in disconnected_q]
    print(f'WARNING: question concepts with no KG connection (not drawn): {names}')

keep = {c for c in keep if c in connected}
print(f'Final: {len(keep)} connected nodes, {len(edges)} edges, '
      f'{len(disconnected_a)} disconnected answer concepts')

# === Step 4: render ===
engine = 'dot' if len(keep) <= 20 else 'fdp'
print(f'Engine: {engine}')

g = graphviz.Digraph(format='png', engine=engine)

graph_attrs = {
    'overlap': 'false',
    'splines': 'true',
    'dpi': '150',
    'bgcolor': 'white',
    'pad': '0.3',
    'nodesep': '0.5',
    'ranksep': '0.7',
    'size': '8,8',         # cap canvas: text must survive 640px downscale
    'ratio': 'compress',
}
if engine == 'dot':
    graph_attrs['rankdir'] = 'LR'
g.attr('graph', **graph_attrs)

g.attr('node',
       shape='box', style='rounded,filled',
       fontname='Helvetica-Bold', fontsize='18',
       margin='0.2,0.1', penwidth='1.5',
       fillcolor='white')
g.attr('edge',
       fontname='Helvetica', fontsize='14',
       arrowsize='0.8', penwidth='1.2')

choice_colors = {
    'A': '#FFD9B3',   # peach
    'B': '#D9E8F5',   # pale blue-gray (NOT green — avoid confusion w/ correct)
    'C': '#E6D9F2',   # lilac
    'D': '#FFF2B3',   # pale yellow
}
correct_color = '#90EE90'
QUESTION_COLOR = '#ADD8E6'

def node_style(cid):
    """Return (label, fillcolor, penwidth) for a concept node."""
    info = merged_nodes[cid]
    label = info['name'].replace('_', ' ')
    if info['in_question']:
        return label, QUESTION_COLOR, '1.5'
    if info['in_choices']:
        letters = sorted(info['in_choices'])
        tag = ','.join(letters)
        is_correct = correct_label in info['in_choices']
        if len(letters) == 1:
            fill = correct_color if is_correct else choice_colors[letters[0]]
        else:
            fill = correct_color if is_correct else '#E0E0E0'
        pw = '3' if is_correct else '1.5'
        return f'{label} [{tag}]', fill, pw
    return label, 'white', '1.5'

# Connected nodes
for cid in keep:
    label, fill, pw = node_style(cid)
    g.node(str(cid), label=label, fillcolor=fill, penwidth=pw)

# Disconnected answer concepts: dashed cluster
if disconnected_a:
    with g.subgraph(name='cluster_no_evidence') as sub:
        sub.attr(label='no connections found in KG',
                 fontsize='14', fontname='Helvetica',
                 style='dashed', color='gray50')
        for cid in disconnected_a:
            label, fill, pw = node_style(cid)
            sub.node(str(cid), label=label, fillcolor=fill,
                     penwidth=pw, style='rounded,filled,dashed')

# Edges: 'relatedto' de-emphasized (gray, unlabeled), others labeled
for s, r, t in edges:
    if r == 'relatedto':
        g.edge(str(s), str(t), color='gray60', penwidth='1.0')
    else:
        g.edge(str(s), str(t), label=r, penwidth='1.3')

out_name = f'q{STMT_IDX}_clean'
g.render(out_name, cleanup=True)
print(f'Wrote {out_name}.png')
