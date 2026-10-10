#!/usr/bin/env python3
"""Opt-in template_v1 extraction; reuse the frozen Stage 1 scoring formulas.

Parsing depends on task and response only, never gold-member lookup. Explicit
task templates take precedence. Fallback selection follows span_extended,
with literal list labels retained instead of stopword/relation-name filtering.
The original scorer and generation/evaluation defaults are not modified.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_stage1 as base

VERSION = 'template_v1'
TASK_TYPES = base.EXTENDED_TASK_TYPES
QUOTED = r'(?:"[^"\n]*"|“[^”\n]*”)'


def scope(text: str) -> str:
    clean = base.strip_markdown(text)
    markers = list(base.ANSWER_MARKER_RE.finditer(clean))
    return clean[markers[-1].end():].lstrip(' \t\r\n:,') if markers else clean


def unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and (value[0], value[-1]) in {('"', '"'), ('“', '”')}:
        return value[1:-1]
    return value


def list_fields(payload: str) -> list[str]:
    """Separate fields outside quoted labels and [A,B] renderer annotations."""
    parts, start, depth, closing = [], 0, 0, None
    for index, char in enumerate(payload):
        if closing:
            if char == closing:
                closing = None
        elif char in {'"', '“'}:
            closing = '”' if char == '“' else '"'
        elif char == '[':
            depth += 1
        elif char == ']' and depth:
            depth -= 1
        elif char in ',;\n' and not depth:
            parts.append(payload[start:index])
            start = index + 1
    parts.append(payload[start:])
    return parts


def label_list(payload: str) -> list[str]:
    """Consume list syntax, preserving commas/conjunctions inside quoted names.

Unquoted labels use comma/semicolon/newline separators. An unquoted 'and'
inside a label is preserved; quoted labels may be joined by 'and' or 'or'.
No stopword/relation-name filtering or gold lookup removes asserted labels.
"""
    payload = payload.strip().removesuffix('.').strip()
    quoted = list(re.finditer(QUOTED, payload))
    residue = re.sub(QUOTED, '', payload)
    if quoted and re.fullmatch(r'[\s,;]*(?:(?:and|or)[\s,;]*)*', residue, re.I):
        return [unquote(m.group()) for m in quoted if unquote(m.group())]
    items = []
    for part in list_fields(payload):
        part = re.sub(r'^\s*(?:[-•]|\d+[.)])\s*', '', part).strip()
        value = unquote(part)
        if value:
            items.append(value)
    return items


def last_match(pattern: str, text: str):
    matches = list(re.finditer(pattern, text, re.I | re.M))
    return matches[-1] if matches else None


def list_fallback(text: str, *, neighbors=False) -> list[str]:
    """Keep span_extended list selection, retaining every literal node label."""
    payload = base.structured_answer_span(text)
    payload = re.sub(r':\s*(?=[-•]\s+|\d+[.)]\s+)', ':\n', payload)
    bullets = [m.group(1) for line in payload.splitlines() if (m := base.LIST_ITEM_RE.match(line))]
    if bullets:
        return [item for bullet in bullets for item in base.parse_node_items(bullet)]
    last_line = payload.splitlines()[-1] if payload.splitlines() else payload
    if len(base.split_outside_annotations(last_line, ',')) > 1:
        payload = last_line
    parser = base.parse_neighbor_items if neighbors else base.parse_node_items
    items = parser(payload)
    if neighbors and len(items) == 1:
        assertion = re.search(r'\b(?:is|are)\s*:?\s*([^\n]+)$', payload, re.I)
        if assertion:
            items = base.parse_node_items(assertion[1])
    return items


def triple_items(text: str) -> tuple[list[tuple[str, str, str]], int]:
    """Retain the old malformed-triple policy, without filtering literal names."""
    clean = base.strip_markdown(re.sub(
        r'`([^`\n]+)`\s+`([^`\n]+)`\s+`([^`\n]+)`',
        lambda m: '(' + ', '.join(m.groups()) + ')', text))
    markers = list(re.finditer(r'(?im)^\s*Answer\s*:', clean))
    payload = clean[markers[-1].end():] if markers else clean
    triples, malformed = [], 0
    for line in payload.splitlines() or [payload]:
        if not line.strip():
            continue
        groups = base.PARENTHESIZED_RE.findall(line)
        valid = []
        for group in groups:
            components = [unquote(c.strip()) for c in list_fields(group)]
            if len(components) == 3 and all(components):
                valid.append(tuple(components))
            elif not markers and ',' in group:
                malformed += int(any(not c for c in components) or len(components) >= 3 or
                    (len(components) == 2 and base.relation_key(components[1]) in base.RELATION_VOCABULARY))
        triples.extend(valid)
        if markers:
            residue = base.PARENTHESIZED_RE.sub('', line)
            residue = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', residue)
            residue = re.sub(r'[\s,;:.]+', '', residue)
            malformed += int(len(groups) != 1 or len(valid) != 1 or bool(residue))
    return triples, malformed


def parse_template(task: str, response: str) -> tuple[str, object] | None:
    """Return an explicit answer or None to request the compatibility fallback."""
    if task not in TASK_TYPES:
        raise ValueError(f'Unknown Stage 1 task: {task}')
    text = scope(response)
    if task in {'node_number', 'edge_number'}:
        noun = r'(?:nodes?|vertices?)' if task == 'node_number' else r'(?:edges?|connections?)'
        match = last_match(r'\b(?:there\s+are|the\s+graph\s+(?:has|contains))\s+([-+]?\d+)\s+' + noun + r'\b', text)
        return ('count_template', int(match[1])) if match else None
    if task == 'node_degree':
        match = last_match(r'\b(?:the\s+)?degree\s+of\s+(?:the\s+)?node\s+' + QUOTED + r'\s+is\s+([-+]?\d+)\b', text)
        return ('degree_template', int(match[1])) if match else None
    if task == 'highest_node_degree':
        match = last_match(
            r'\b(?:node\s+with\s+(?:the\s+)?highest\s+degree|highest[- ]degree\s+node)\s+is\s+'
            r'(?P<name>' + QUOTED + r'|[^\n,;]+?)\s*,?\s+with\s+(?:a\s+)?degree\s+of\s+(?P<degree>[-+]?\d+)\b', text)
        return ('highest_template', (unquote(match['name']), int(match['degree']))) if match else None
    if task == 'node_description':
        match = last_match(r'\bthe\s+image\s+depicts\s+the\s+following\s+nodes\s*:\s*(?P<payload>[^\n]+)', text)
        return ('node_list_template', label_list(match['payload'])) if match else None
    if task == 'triple_listing':
        match = last_match(r'\bthe\s+triples\s+in\s+the\s+graph\s+are\s+listed\s+as\s*:', text)
        if match:
            return 'triple_list_template', triple_items(text[match.end():])
        return None
    if task == 'relation_identification':
        match = last_match(
            r'\b(?:relation(?:ship)?)(?:\s+on\s+the\s+edge)?\s+between\s+' + QUOTED + r'\s+and\s+' + QUOTED +
            r'\s+is\s+(?P<relation>' + QUOTED + r'|[^\n.!?]+)', text)
        if match:
            relation = base.relation_key(unquote(match['relation']))
            if relation in base.RELATION_VOCABULARY or re.fullmatch(QUOTED, match['relation']):
                return 'relation_template', relation if relation in base.RELATION_VOCABULARY else None
        return None
    if task == 'shortest_path_listing':
        match = last_match(r'\b(?:one\s+|the\s+)?shortest\s+path\s+from\s+' + QUOTED + r'\s+to\s+' + QUOTED +
                          r'\s+is\s*:\s*(?P<path>[^\n]+)', text)
        if match:
            payload = match['path'].strip().removesuffix('.').strip()
            parsed = ([unquote(part) for part in re.split(r'\s*(?:->|→)\s*', payload)]
                      if '->' in payload or '→' in payload else label_list(payload))
            return 'path_template', parsed
        return None
    # Neighbor answers: consume a task-specific assertion before splitting its
    # payload, so query node and degree are never predicted neighbors.
    patterns = (
        r'\b(?:its\s+)?neighbou?rs?\s+(?:is|are)\s*:?\s*(?P<payload>[^\n]+)',
        r'\b(?:is|are)\s+(?:directly\s+)?connected\s+to\s*:?\s*(?:the\s+node\s+|node\s+)?(?P<payload>[^\n]+)',
        r'\b(?:one\s+)?node\s+that\s+shares?\s+an?\s+edge\s+with\s+' + QUOTED + r'\s+is\s*(?P<payload>[^\n]+)',
        r'\b(?:only\s+)?(?:node\s+)?adjacent\s+(?:node\s+)?to\s+' + QUOTED + r'\s+is\s*:?\s*(?P<payload>[^\n]+)',
    )
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    # Multi-line reasoning/bullet lists keep span_extended's established list
    # selection. New assertions must be a complete one-line answer statement.
    for match in sorted(matches, key=lambda m: m.start(), reverse=True):
        payload = match['payload'].strip()
        # Separate explanatory sentences outside quoted labels. Names may
        # themselves contain periods (e.g. "St. Louis").
        closing, cut = None, len(payload)
        for index, char in enumerate(payload):
            if closing:
                if char == closing:
                    closing = None
            elif char in {'"', '“'}:
                closing = '”' if char == '“' else '"'
            elif char in '.!?' and (index + 1 == len(payload) or payload[index + 1].isspace()):
                cut = index
                break
        payload = payload[:cut].strip()
        if not payload or payload.startswith(('-', '•')) or '\n' in match.group():
            continue
        if payload.endswith(':') or re.match(r'^(?:the\s+)?(?:following\b|listed\b|shown\b|(?:only\s+)?(?:one|\d+)\s+(?:other\s+)?nodes?\b)', payload, re.I):
            continue
        if re.fullmatch(r'(?:none|no\s+(?:adjacent\s+)?nodes?|no\s+neighbou?rs?)', payload, re.I):
            return 'neighbor_template', []
        # Do not convert ordinary baseline prose explanations into node lists.
        # Templates with bare labels must be headed by an explicit neighbor cue
        # or colon; connected-to narrative prose uses quoted payloads only.
        quoted_residue = re.sub(QUOTED, '', payload)
        if re.search(QUOTED, payload) and not re.fullmatch(r'[\s,;]*(?:(?:and|or)[\s,;]*)*', quoted_residue, re.I):
            continue
        if not re.search(QUOTED, payload) and 'connected' in match.group().lower() and ':' not in match.group():
            continue
        return 'neighbor_template', label_list(payload)
    if re.fullmatch(r'\s*(?:the\s+)?degree\s+of\s+(?:the\s+)?node\s+' + QUOTED + r'\s+is\s+[-+]?\d+\s*\.?\s*', text, re.I):
        return 'degree_only_no_neighbors', []
    return None


def score_record(record: dict, response: str, meta: dict, extractor: str = VERSION) -> dict:
    if extractor != VERSION:
        raise ValueError(f'This scorer only supports {VERSION}')
    response = '' if response is None else str(response).strip()
    task = record['task_type']
    row = base.score_record(record, response, meta, extractor='span_extended')
    parsed = parse_template(task, response)
    rule = 'span_extended_fallback'
    if parsed is None and task in {'node_description', 'neighbor_listing'}:
        parsed = ('list_syntax_fallback', list_fallback(response, neighbors=task == 'neighbor_listing'))
    if parsed is None and task == 'triple_listing':
        parsed = ('triple_syntax_fallback', triple_items(response))
    if parsed is not None:
        rule, value = parsed
        if task in base.NUMERIC_TASKS:
            row.update(parsed_integer=value, signed_error=value - row['gold_integer'], is_correct=value == row['gold_integer'])
        elif task == 'highest_node_degree':
            name, degree = value
            row.update(parsed_name=name, parsed_degree=degree, degree_correct=degree == row['gold_degree'],
                       signed_error=degree - row['gold_degree'])
            for variant in base.SET_VARIANTS:
                row[f'name_correct_{variant}'] = base.normalize_component(name, variant) in {
                    base.normalize_component(n, variant) for n in row['valid_highest_names']}
                row[f'both_correct_{variant}'] = row['degree_correct'] and row[f'name_correct_{variant}']
            row['is_correct'] = row['both_correct_raw']
        elif task == 'relation_identification':
            correct = value == base.relation_key(row['gold_relation'])
            row.update(parsed_relation=value, relation_correct={v: correct for v in base.SET_VARIANTS}, is_correct=correct)
        elif task == 'shortest_path_listing':
            row.update(base.score_new_task(record, ' -> '.join(value)))
            keys = [base.normalize_component(n, 'annotation_stripped') for n in value]
            row.update(parsed_path=value, degenerate_output=len(keys) != len(set(keys)))
        else:
            if task == 'triple_listing':
                predicted, malformed = value
                gold = [tuple(t) for t in row['parsed_gold_triples']]
                row.update(parsed_predicted_triples=[list(t) for t in predicted], malformed_triple_line_count=malformed)
            else:
                predicted, malformed, gold = value, 0, row['parsed_gold_nodes']
                row['parsed_predicted_nodes'] = predicted
            row['set_metrics'] = base.set_metrics_by_variant(gold, predicted, malformed)
            row['is_correct'] = row['set_metrics']['raw']['exact_set_equality']
    row.update(extractor=VERSION, scorer_version=VERSION, parser_rule=rule)
    return row


def aggregate_task(task: str, rows: list[dict], extractor: str = VERSION) -> dict:
    if extractor != VERSION:
        raise ValueError(f'This scorer only supports {VERSION}')
    return base.aggregate_task(task, rows, extractor='span_extended')


def score_files(args):
    """Validate original prediction keys, then replace every derived score."""
    if getattr(args, 'extractor', VERSION) != VERSION:
        raise ValueError(f'This scorer only supports {VERSION}')
    old_args = copy.copy(args)
    old_args.extractor = 'span_extended'
    old_args.task_set = getattr(args, 'task_set', 'extended')
    metrics, old_rows = base.score_files(old_args)
    records, metadata = base.validate_inputs(base.read_jsonl(args.input_jsonl), base.read_jsonl(args.graph_metadata),
                                             base.TASK_SETS[getattr(args, 'task_set', 'extended')])
    rows = {task: [{**r, **score_record(records[(int(r['statement_idx']), task)], r.get('raw_response', ''),
                                      metadata[int(r['statement_idx'])])} for r in task_rows]
            for task, task_rows in old_rows.items()}
    metrics['tasks'] = {task: aggregate_task(task, task_rows) for task, task_rows in rows.items()}
    metrics['scorer_version'] = VERSION
    metrics['scoring_notes'].update(extractor=VERSION,
        template_precedence='Last explicit task template within the final answer block, else whole response. No gold-member selection. Fallback selection: span_extended; literal list/tuple labels retained.',
        highest_node_degree_extraction='Pair asserted name and degree; consume quoted name before its degree suffix. Ties and scoring formulas unchanged.',
        numeric_parse='Explicit count/quoted-node degree template first, then span_extended.',
        literal_list_labels='No stopword or relation-name filtering of node-list/tuple assertions, in either explicit templates or fallback lists.',
        neighbor_extraction='Extract neighbor assertion payload before list splitting; quoted labels preserve commas/and. Degree-only replies assert no neighbors.')
    return metrics, rows
