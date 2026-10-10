"""Run all historical syntax cases without importing mutable audit modules."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT / 'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts'
sys.path.insert(0, str(PARSER_DIR))
import score_stage1_audit_v2 as v2
import score_stage1_audit_v3 as v3


def canonical(task, value):
    if value is None:
        return None
    if task in ('node_description', 'neighbor_listing'):
        return sorted(set(value))
    if task == 'triple_listing':
        return sorted(set(tuple(t) for t in value))
    return value


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def assigned_list(node, name):
    return next(ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def run_controls():
    rows = []
    def case(suite, task, raw, expected, question='', malformed=None, historical_expected=None, note=''):
        parsed = v3.extract(task, raw, question)
        passed = canonical(task, parsed.value) == canonical(task, expected) and (malformed is None or parsed.malformed == malformed)
        rows.append(dict(suite=suite, case=len(rows)+1, task=task, raw_response=raw, question=question,
                         expected=expected, actual=parsed.value, malformed_expected=malformed,
                         malformed_actual=parsed.malformed, passed=passed, historical_expected=historical_expected,
                         note=note, evidence='VERIFIED'))
    qfile = ROOT/'outputs/qwen_parser_wrong_answer_audit_2026-10-10/audit.py'
    q = function(qfile, 'controls')
    for task, raw, question, expected, experiment in assigned_list(q, 'checks'):
        historical = expected
        note = ''
        if experiment == 'last_integer' and expected is not None:
            expected = v2.extract(task, raw).value
            note = 'Historical optional last-integer proposal intentionally omitted; frozen-policy expectation is audit_v2.'
        if experiment == 'overlapping_path' and canonical(task, expected) != canonical(task, v2.extract(task, raw).value):
            expected = v2.extract(task, raw).value
            note = 'Historical optional path-joining proposal intentionally omitted; no joining/solving paths.'
        case('qwen', task, raw, expected, question, historical_expected=historical, note=note)
    for raw, malformed in assigned_list(q, 'triples'):
        # Historical tests checked malformed counts only, not extracted claims.
        parsed = v3.extract('triple_listing', raw)
        historical_malformed = malformed
        note = 'Historical malformed-count assertion only.'
        if raw == 'The triples are: (hub, related to, leaf), (isolated, no connections found in KG).':
            malformed = 1
            note += ' Intentional user-required policy change: this closed two-field tuple is penalised; historical proposal excluded it.'
        case('qwen', 'triple_listing', raw, parsed.value, malformed=malformed, note=note)
        rows[-1]['historical_malformed_expected'] = historical_malformed
    for model in ('llava', 'gemma'):
        path = ROOT/f'outputs/{model}_parser_wrong_answer_audit_2026-10-10/audit.py'
        node = function(path, 'control_checks')
        for task, raw, expected in assigned_list(node, 'cases'):
            case(model, task, raw, expected)
        loop = next(n for n in node.body if isinstance(n, ast.For) and isinstance(n.iter, ast.List))
        for raw, expected, malformed in ast.literal_eval(loop.iter):
            case(model, 'triple_listing', raw, expected, malformed=malformed)
        if model == 'gemma':
            extra = assigned_list(function(path, 'additional_controls'), 'cases')
            for task, raw, expected in extra:
                case(model, task, raw, expected, malformed=1 if (task, raw, expected) == extra[-1] else None)
    new = [
        ('highest_node_degree','- "Water": 8 ← Highest\nFinal Answer:\nThe node **`', ['Water',8], ''),
        ('highest_node_degree','- "water": 8 ← Highest\nI was wrong.\nFinal Answer:\nThe node **`', None, ''),
        ('highest_node_degree','- "water": 8 ← Highest\nFinal Answer: "rock", degree 2', ['rock',2], ''),
        ('relation_identification','The following relations for "earth": antonym to "water", part of "current".', 'antonym', 'Which relation links "earth" to "water"?'),
        ('relation_identification','Perhaps the following relations for "earth": antonym to "water".', None, 'Which relation links "earth" to "water"?'),
        ('relation_identification','The following relations for "earth": antonym to "water".\nI was wrong.', None, 'Which relation links "earth" to "water"?'),
        ('relation_identification','The following relations for "earth": antonym to "water".', None, 'Which relation links earth to water?'),
        ('relation_identification','The following relations for "earth": antonym to "water".', None, 'Which relation links "water" to "earth"?'),
        ('node_number','The following relations for "earth": antonym to "water".', None, 'Which relation links "earth" to "water"?'),
    ]
    for task, raw, expected, question in new:
        case('new_optional_rules', task, raw, expected, question)
    for raw, expected, malformed in [
        ('1. (a, related to, b)\n2. (isolated)', [['a','related to','b']], 0),
        ('1. (a, related to, b)\n2. (x, relation)', [['a','related to','b']], 1),
        ('1. (a, related to, b)\n2. (x, relation, y, z)', [['a','related to','b']], 1),
        ('1. (a, related to, b)\n2. (x, relation, y', [['a','related to','b']], 0),
    ]:
        case('new_triple_policy', 'triple_listing', raw, expected, malformed=malformed)
    return rows


if __name__ == '__main__':
    rows = run_controls()
    out = ROOT/'outputs/parser_v3_rescore_2026-10-10'
    out.mkdir(exist_ok=True)
    (out/'syntax_controls.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    failures = [r for r in rows if not r['passed']]
    print(json.dumps(dict(total=len(rows), passed=len(rows)-len(failures), failures=failures), ensure_ascii=False, indent=2))
    if failures:
        sys.exit(1)
