import pickle
import graphviz
import sys
import networkx as nx

CONCEPT = sys.argv[1] if len(sys.argv) > 1 else 'telescope'
HOPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAX_NODES = 40  # cap so the image stays readable

with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]
concept2id = {c: i for i, c in enumerate(id2concept)}

with open('conceptnet.en.pruned.graph', 'rb') as f:
    kg = pickle.load(f)

# The pruned graph uses the original 42 ConceptNet relation ids, not the merged 17.
# We don't have a full mapping handy, so just label edges with the raw rel id.

start_cid = concept2id.get(CONCEPT)
if start_cid is None or start_cid not in kg:
    print(f'{CONCEPT} not found in pruned graph')
    raise SystemExit

# BFS to collect nodes within HOPS
g_undir = kg.to_undirected(as_view=True) if kg.is_directed() else kg
lengths = nx.single_source_shortest_path_length(g_undir, start_cid, cutoff=HOPS)
nbhd = set(lengths.keys())

# If too big, keep the closest ones
if len(nbhd) > MAX_NODES:
    closest = sorted(lengths.items(), key=lambda x: x[1])[:MAX_NODES]
    nbhd = {n for n, _ in closest}
    print(f'Neighborhood has {len(lengths)} nodes; rendering closest {MAX_NODES}')
else:
    print(f'Neighborhood: {len(nbhd)} nodes within {HOPS} hops')

# Collect edges inside the neighborhood
edges = []
for u, v, data in kg.edges(data=True):
    if u in nbhd and v in nbhd:
        edges.append((u, v, data.get('rel', '?')))

print(f'Edges in neighborhood: {len(edges)}')

# Render
g = graphviz.Digraph(format='png', engine='sfdp')
g.attr('graph', overlap='false', splines='true', dpi='150',
       bgcolor='white', size='12,12', ratio='compress')
g.attr('node', shape='box', style='rounded,filled', fillcolor='white',
       fontname='Helvetica', fontsize='14', margin='0.15,0.08')
g.attr('edge', fontname='Helvetica', fontsize='9', arrowsize='0.6')

for cid in nbhd:
    label = id2concept[cid].replace('_', ' ')
    fill = '#90EE90' if cid == start_cid else 'white'
    pw = '3' if cid == start_cid else '1'
    g.node(str(cid), label=label, fillcolor=fill, penwidth=pw)

for u, v, rel in edges:
    g.edge(str(u), str(v), label=str(rel), color='gray50')

out_name = f'neighborhood_{CONCEPT}_{HOPS}hop'
g.render(out_name, cleanup=True)
print(f'Wrote {out_name}.png')
