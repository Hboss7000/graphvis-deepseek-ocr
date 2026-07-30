import graphviz 

g = graphviz.Digraph()
g.edge('hello', 'world')
g.render('test_graph', format='png', cleanup=True)

print('OK - check test_graph.png')


