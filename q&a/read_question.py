import json
import sys

ADJ_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_CHOICES = 4  # OBQA

stmt_idx = ADJ_IDX // N_CHOICES
choice_idx = ADJ_IDX % N_CHOICES

with open('train-fact.statement.jsonl', 'r') as f:
    statements = [json.loads(line) for line in f]

stmt = statements[stmt_idx]
print(f'Adjacency entry {ADJ_IDX} → statement {stmt_idx}, choice index {choice_idx}')
print()
print(f'Q: {stmt["question"]["stem"]}')
print()
print('Choices:')
for i, c in enumerate(stmt['question']['choices']):
    is_this_one = ' <-- THIS ENTRY' if i == choice_idx else ''
    is_correct = ' (CORRECT)' if c['label'] == stmt.get('answerKey') else ''
    print(f'  {c["label"]}. {c["text"]}{is_correct}{is_this_one}')
