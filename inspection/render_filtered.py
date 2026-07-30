import json
import graphviz
from collections import defaultdict, deque

with open('test_single.jsonl', 'r') as f:
    graph_obj = json.loads(f.readline())

nodes = graph_obj['nodes']
links = graph_obj['links']
print(f'Before filtering: {len(nodes)} nodes, {len(links)} links')

# === Filter step ===
# Keep: question nodes, answer nodes, and any node within 1 hop of either.
# This mirrors what the GraphVis paper effectively renders.

qa_node_ids = {n['id'] for n in nodes if n['role'] in ('question', 'answer')}

# Build adjacency (undirected) for BFS
neighbors = defaultdict(set)
for link in links:
    neighbors[link['source']].add(link['target'])
    neighbors[link['target']].add(link['source'])

# 1-hop expansion from QA nodes
keep = set(qa_node_ids)
for nid in qa_node_ids:
    keep.update(neighbors[nid])

# Filter nodes and links
filtered_nodes = [n for n in nodes if n['id'] in keep]
filtered_links = [
    l for l in links
    if l['source'] in keep and l['target'] in keep
]

print(f'After filtering:  {len(filtered_nodes)} nodes, {len(filtered_links)} links')

# === Render ===
g = graphviz.Digraph(format='png', engine='sfdp')
g.attr('graph', overlap='false', splines='true')
g.attr('node', shape='ellipse', fontsize='10')
g.attr('edge', fontsize='8')

role_colors = {
    'question': 'lightblue',
    'answer': 'lightyellow',
    'other': 'white',
}
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

g.render('test_filtered', cleanup=True)
print('Wrote test_filtered.png')
