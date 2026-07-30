import json
import graphviz
import sys
from collections import defaultdict

ENTRY_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
in_path = f'entry_{ENTRY_IDX}.jsonl'
out_name = f'entry_{ENTRY_IDX}_render'

with open(in_path, 'r') as f:
    graph_obj = json.loads(f.readline())

nodes = graph_obj['nodes']
links = graph_obj['links']
print(f'Before filtering: {len(nodes)} nodes, {len(links)} links')

neighbors = defaultdict(set)
for link in links:
    neighbors[link['source']].add(link['target'])
    neighbors[link['target']].add(link['source'])

q_ids = {n['id'] for n in nodes if n['role'] == 'question'}
a_ids = {n['id'] for n in nodes if n['role'] == 'answer'}
print(f'Question nodes: {len(q_ids)}, Answer nodes: {len(a_ids)}')

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
print(f'After filtering:  {len(filtered_nodes)} nodes, {len(filtered_links)} links')

g = graphviz.Digraph(format='png', engine='sfdp')
# Render at higher DPI (default is 96, bump to 200-300)
g.attr('graph',
       overlap='false',
       splines='true',
       dpi='200',
       bgcolor='white',
       pad='0.5',         # margin around the graph
       nodesep='0.6',     # min space between sibling nodes
       ranksep='0.8')     # min space between ranks/layers

g.attr('node',
       shape='ellipse',
       fontname='Helvetica',
       fontsize='14',     # bigger text inside nodes
       width='1.2',
       height='0.6',
       margin='0.15,0.08')  # padding inside the ellipse

g.attr('edge',
       fontname='Helvetica',
       fontsize='11',     # bigger edge labels
       arrowsize='0.7')

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

g.render(out_name, cleanup=True)
print(f'Wrote {out_name}.png')
