import pickle

with open('concept.txt', 'r') as f:
    id2concept = [line.strip() for line in f]

with open('train.graph.adj.pk', 'rb') as f:
    data = pickle.load(f)

# Look at first 20 entries — print question and answer concepts
for idx in range(0, 20):
    entry = data[idx]
    q_names = [id2concept[c] for c, m in zip(entry['concepts'], entry['qmask']) if m]
    a_names = [id2concept[c] for c, m in zip(entry['concepts'], entry['amask']) if m]
    print(f'[{idx}] Q={q_names}  A={a_names}')
