import pickle
import networkx as nx

with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]
concept2id = {c: i for i, c in enumerate(id2concept)}

with open('conceptnet.en.pruned.graph', 'rb') as f:
    kg = pickle.load(f)

print(f'Graph type: {type(kg).__name__}')
print(f'Directed: {kg.is_directed()}, Multigraph: {kg.is_multigraph()}')
print()

START = 'mailing_tube'
start_cid = concept2id[START]

if start_cid not in kg:
    print(f'{START} (cid {start_cid}) is not a node in the pruned graph.')
    raise SystemExit

# 1-hop neighbors (treat edges as undirected for exploration)
g_undir = kg.to_undirected(as_view=True) if kg.is_directed() else kg
neighbors_1hop = set(g_undir.neighbors(start_cid))
print(f'1-hop neighbors of {START} ({len(neighbors_1hop)}):')
for nb in neighbors_1hop:
    print(f'  {id2concept[nb]} (cid {nb})')
print()

# 2-hop neighbors (reachable in exactly <=2 hops, excluding self and 1-hop)
neighbors_2hop = set()
for nb in neighbors_1hop:
    neighbors_2hop.update(g_undir.neighbors(nb))
neighbors_2hop -= {start_cid}
neighbors_2hop -= neighbors_1hop

print(f'2-hop-only neighbors of {START} ({len(neighbors_2hop)}):')
for nb in sorted(neighbors_2hop, key=lambda x: id2concept[x])[:50]:
    print(f'  {id2concept[nb]}')
if len(neighbors_2hop) > 50:
    print(f'  ... and {len(neighbors_2hop) - 50} more')
print()

# Show the actual edges (with relations) on the 1-hop ring
print(f'Edges incident to {START}:')
def fmt_edge(u, v, key=None, data=None):
    u_name = id2concept[u]
    v_name = id2concept[v]
    if data is None:
        return f'  {u_name} -- {v_name}'
    # Edge data — relations are stored under various conventions across
    # KagNet/QA-GNN releases. Print everything so we can see what we have.
    return f'  {u_name} -- {v_name}   {dict(data)}'

if kg.is_multigraph():
    for u, v, key, data in kg.edges(start_cid, keys=True, data=True):
        print(fmt_edge(u, v, key, data))
    # And the reverse direction if directed
    if kg.is_directed():
        for u, v, key, data in kg.in_edges(start_cid, keys=True, data=True):
            print(fmt_edge(u, v, key, data))
else:
    for u, v, data in kg.edges(start_cid, data=True):
        print(fmt_edge(u, v, data=data))
    if kg.is_directed():
        for u, v, data in kg.in_edges(start_cid, data=True):
            print(fmt_edge(u, v, data=data))
