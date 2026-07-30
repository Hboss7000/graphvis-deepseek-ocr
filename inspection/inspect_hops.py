import pickle
import networkx as nx
from collections import Counter

ENTRY_IDX = 19  # telescope/mailing_tube as starting point

with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]

with open('conceptnet.en.pruned.graph', 'rb') as f:
    kg = pickle.load(f)

# Force undirected view for path lengths (relations are bidirectional in spirit)
if kg.is_directed():
    kg = kg.to_undirected(as_view=False)

with open('train.graph.adj.pk', 'rb') as f:
    data = pickle.load(f)

entry = data[ENTRY_IDX]
concepts = [int(c) for c in entry['concepts']]
qmask = entry['qmask']
amask = entry['amask']

q_cids = [c for c, m in zip(concepts, qmask) if m]
a_cids = [c for c, m in zip(concepts, amask) if m]
print(f'Entry {ENTRY_IDX}:')
print(f'  Q nodes ({len(q_cids)}): {[id2concept[c] for c in q_cids]}')
print(f'  A nodes ({len(a_cids)}): {[id2concept[c] for c in a_cids]}')
print(f'  Other nodes: {len(concepts) - len(q_cids) - len(a_cids)}')
print()

def shortest_to_set(node, target_set, cutoff=4):
    """Shortest path length in the pruned KG from `node` to any node in
    `target_set`. Returns None if no path within `cutoff` hops."""
    if node in target_set:
        return 0
    if node not in kg:
        return None
    # BFS limited by cutoff
    lengths = nx.single_source_shortest_path_length(kg, node, cutoff=cutoff)
    hits = [d for n, d in lengths.items() if n in target_set]
    return min(hits) if hits else None

# Distance distribution of each node to nearest Q and nearest A
q_set = set(q_cids)
a_set = set(a_cids)
sum_dist_dist = Counter()
non_qa = [c for c in concepts if c not in q_set and c not in a_set]

print(f'Computing path lengths in pruned KG for {len(non_qa)} non-Q/A nodes...')
print('(This may take a moment for large entries.)')

unusual = []
for cid in non_qa:
    dq = shortest_to_set(cid, q_set, cutoff=4)
    da = shortest_to_set(cid, a_set, cutoff=4)
    if dq is None or da is None:
        sum_dist_dist['unreachable'] += 1
        unusual.append((cid, dq, da))
    else:
        sum_dist_dist[dq + da] += 1
        if dq + da > 2:
            unusual.append((cid, dq, da))

print()
print('Distribution of (dist-to-Q + dist-to-A) for non-Q/A nodes:')
for k in sorted(sum_dist_dist.keys(), key=lambda x: (isinstance(x, str), x)):
    print(f'  total {k}: {sum_dist_dist[k]} nodes')

if unusual:
    print()
    print('Nodes that violate the 2-hop expectation:')
    for cid, dq, da in unusual[:20]:
        print(f'  {id2concept[cid]:30s}  dQ={dq}  dA={da}')
    if len(unusual) > 20:
        print(f'  ... and {len(unusual) - 20} more')
