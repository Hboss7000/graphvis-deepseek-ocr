import pickle
from collections import defaultdict

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

entry = data[0]
concepts = entry['concepts']
adj = entry['adj']
n_nodes = len(concepts)
node_names = [id2concept[cid] for cid in concepts]

# Group edges by relation index, show 2 examples per relation
adj_coo = adj.tocoo()
by_rel = defaultdict(list)
for row, col, val in zip(adj_coo.row, adj_coo.col, adj_coo.data):
    if val == 0:
        continue
    r = row // n_nodes
    i = row % n_nodes
    j = col
    by_rel[r].append((i, j))

print(f'{len(by_rel)} relation indices have edges in this entry')
print()
for r in sorted(by_rel.keys()):
    label = relations[r] if r < len(relations) else f'??? r={r}'
    print(f'Relation {r} ({label}): {len(by_rel[r])} edges. Examples:')
    for i, j in by_rel[r][:2]:
        print(f'    {node_names[i]} --> {node_names[j]}')
    print()
