import pickle
import networkx as nx

with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]
concept2id = {c: i for i, c in enumerate(id2concept)}

# nx.read_gpickle was removed in NetworkX 3.x — plain pickle works
with open('conceptnet.en.pruned.graph', 'rb') as f:
    kg = pickle.load(f)

for name in ['mailing_tube', 'mailing', 'candle', 'tube', 'straw', 'telescope']:
    cid = concept2id.get(name)
    if cid is None:
        print(f'{name}: NOT IN VOCAB')
    elif cid in kg:
        print(f'{name}: degree {kg.degree(cid)} in pruned ConceptNet')
    else:
        print(f'{name}: in vocab but NOT in pruned graph')
