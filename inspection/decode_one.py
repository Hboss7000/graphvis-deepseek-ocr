import pickle

# Load the concept lookup table
with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]
print(f'Loaded {len(id2concept)} concepts from concept.txt')

# The 17 merged relations from KagNet/QA-GNN/GraphVis
# (Order matters — this is the convention used in adj)
relations = [
    'antonym', 'atlocation', 'capableof', 'causes', 'createdby',
    'isa', 'desires', 'hassubevent', 'partof', 'hascontext',
    'hasproperty', 'madeof', 'notcapableof', 'notdesires', 'receivesaction',
    'relatedto', 'usedfor',
]
print(f'{len(relations)} relation types')
print()

# Load the pickle
with open('train.graph.adj.pk', 'rb') as f:
    data = pickle.load(f)

entry = data[0]
concepts = entry['concepts']
qmask = entry['qmask']
amask = entry['amask']
adj = entry['adj']

print(f'concepts: {concepts}')
print(f'qmask: {qmask}')
print(f'amask: {amask}')
print(f'adj: {adj}')


n_nodes = len(concepts)
n_rels = adj.shape[0] // n_nodes
print(f'Entry 0: {n_nodes} nodes, {n_rels} relation types')
print()

# Map concept ids to strings
node_names = [id2concept[cid] for cid in concepts]

# Show which nodes came from the question vs the answer
print('Question concepts (qmask=True):')
for i, name in enumerate(node_names):
    if qmask[i]:
        print(f'  [{i}] {name}')
print()
print('Answer concepts (amask=True):')
for i, name in enumerate(node_names):
    if amask[i]:
        print(f'  [{i}] {name}')
print()

# Decode the adjacency to triples
# adj is (n_rels * n_nodes, n_nodes); adj[r*n_nodes + i, j] = 1 means edge i --r--> j
adj_coo = adj.tocoo()
triples = []
for row, col, val in zip(adj_coo.row, adj_coo.col, adj_coo.data):
    if val == 0:
        continue
    r = row // n_nodes
    i = row % n_nodes
    j = col
    triples.append((i, r, j))

print(f'Total edges: {len(triples)}')
print()
print('First 20 edges:')
for i, r, j in triples[:20]:
    print(f'  {node_names[i]} --[{relations[r]}]--> {node_names[j]}')
