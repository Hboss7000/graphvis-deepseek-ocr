import pickle
import json
import sys

ENTRY_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

entry = data[ENTRY_IDX]
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

graph_obj = {'nodes': nodes, 'links': links}

out_path = f'entry_{ENTRY_IDX}.jsonl'
with open(out_path, 'w') as f:
    f.write(json.dumps(graph_obj) + '\n')

print(f'Entry {ENTRY_IDX}: {len(nodes)} nodes, {len(links)} links → {out_path}')
