import json
import graphviz

# Load the single entry we just wrote
with open('test_single.jsonl', 'r') as f:
    graph_obj = json.loads(f.readline())

nodes = graph_obj['nodes']
links = graph_obj['links']
print(f'Rendering {len(nodes)} nodes, {len(links)} links')

# Build the Graphviz digraph
g = graphviz.Digraph(format='png')
g.attr('graph', rankdir='LR', overlap='false', splines='true')
g.attr('node', shape='ellipse', fontsize='10')
g.attr('edge', fontsize='8')

# Add nodes — color by role so question/answer concepts stand out
role_colors = {
    'question': 'lightblue',
    'answer': 'lightyellow',
    'other': 'white',
}
for node in nodes:
    label = node['name'].replace('_', ' ')
    g.node(
        str(node['id']),
        label=label,
        style='filled',
        fillcolor=role_colors.get(node['role'], 'white'),
    )

# Add edges
for link in links:
    g.edge(
        str(link['source']),
        str(link['target']),
        label=link['relation'],
    )

# Render
g.render('test_single', cleanup=True)
print('Wrote test_single.png')
