import pickle

# Load the pickle
with open('train.graph.adj.pk', 'rb') as f:
    data = pickle.load(f)

print('=== top level ===')
print('type:', type(data).__name__)
print('length:', len(data))
print()

# Look at the very first entry
entry = data[0]
print('=== entry[0] ===')
print('type:', type(entry).__name__)

if isinstance(entry, dict):
    print('keys:', list(entry.keys()))
    print()
    for key, value in entry.items():
        t = type(value).__name__
        if hasattr(value, 'shape'):
            print(f'  {key}: {t}, shape={value.shape}, dtype={value.dtype}')
        elif hasattr(value, '__len__'):
            print(f'  {key}: {t}, len={len(value)}')
        else:
            print(f'  {key}: {t}, value={value}')
else:
    # Might be a tuple instead of a dict
    print('not a dict, contents:')
    for i, item in enumerate(entry):
        t = type(item).__name__
        if hasattr(item, 'shape'):
            print(f'  [{i}]: {t}, shape={item.shape}')
        elif hasattr(item, '__len__'):
            print(f'  [{i}]: {t}, len={len(item)}')
        else:
            print(f'  [{i}]: {t}, value={item}')
