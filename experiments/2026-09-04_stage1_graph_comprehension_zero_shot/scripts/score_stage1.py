#!/usr/bin/env python3
"""Score Stage 1 graph-comprehension predictions without fuzzy matching."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PAPER_TASK_TYPES = (
    "node_description",
    "node_number",
    "edge_number",
    "triple_listing",
    "highest_node_degree",
    "node_degree",
)
NEW_TASK_TYPES = ("relation_identification", "neighbor_listing", "shortest_path_listing")
EXTENDED_TASK_TYPES = PAPER_TASK_TYPES + NEW_TASK_TYPES
TASK_SETS = {"paper": PAPER_TASK_TYPES, "extended": EXTENDED_TASK_TYPES}
TASK_TYPES = PAPER_TASK_TYPES  # Backward-compatible imports and default scoring.
NUMERIC_TASKS = {"node_number", "edge_number", "node_degree"}
SET_VARIANTS = ("raw", "basic", "annotation_stripped")

INTEGER_RE = re.compile(r"[-+]?\d+")
PARENTHESIZED_RE = re.compile(r"\(([^()]*)\)")
ANSWER_ANNOTATION_RE = re.compile(
    r"\s*\[\s*[A-D](?:\s*,\s*[A-D])*\s*\]\s*$", re.IGNORECASE
)
TRAILING_PUNCTUATION_RE = re.compile(r"[\s.!?;:]+$")
NODE_GOLD_PREFIX = "The image depicts the following nodes:"
TRIPLE_GOLD_PREFIX = "The triples in the graph are listed as:"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_component(value: str, variant: str) -> str:
    value = value.strip()
    if variant == "raw":
        return value
    if variant not in {"basic", "annotation_stripped"}:
        raise ValueError(f"Unknown normalization variant: {variant}")
    value = " ".join(value.lower().split())
    value = TRAILING_PUNCTUATION_RE.sub("", value)
    if variant == "annotation_stripped":
        value = ANSWER_ANNOTATION_RE.sub("", value)
        value = TRAILING_PUNCTUATION_RE.sub("", value).strip()
    return value


def _strip_payload_sentence(text: str) -> str:
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def split_outside_annotations(payload: str, separators: str = ",;\n") -> list[str]:
    """Split a list while preserving commas inside renderer labels such as [B,C]."""
    parts = []
    start = 0
    square_depth = 0
    for index, char in enumerate(payload):
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth:
            square_depth -= 1
        elif char in separators and square_depth == 0:
            parts.append(payload[start:index])
            start = index + 1
    parts.append(payload[start:])
    return parts


def parse_node_items(text: str, *, gold: bool = False) -> list[str]:
    """Parse the comma-separated node payload; sentence punctuation is syntax."""
    payload = text.strip()
    if gold and payload.startswith(NODE_GOLD_PREFIX):
        payload = payload[len(NODE_GOLD_PREFIX) :]
    else:
        prefix_patterns = (
            r"^.*?following\s+(?:nodes|vertices)\s*:\s*",
            r"^.*?(?:nodes|vertices)\s+(?:are|include)\s*:\s*",
            r"^.*?(?:nodes|vertices)\s+(?:are|include)\s+",
        )
        for pattern in prefix_patterns:
            replaced = re.sub(pattern, "", payload, count=1, flags=re.IGNORECASE | re.DOTALL)
            if replaced != payload:
                payload = replaced
                break
        if ":" in payload and payload.lower().split(":", 1)[0].strip().endswith(
            ("nodes", "vertices")
        ):
            payload = payload.split(":", 1)[1]
    payload = _strip_payload_sentence(payload)
    parts = split_outside_annotations(payload)
    items: list[str] = []
    for part in parts:
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip()
        item = item.strip('"\'')
        if item:
            items.append(item)
    return items


def parse_triples(text: str) -> list[tuple[str, str, str]]:
    triples = []
    for payload in PARENTHESIZED_RE.findall(text):
        components = [component.strip() for component in split_outside_annotations(payload, ",")]
        if len(components) == 3 and all(components):
            triples.append(tuple(components))
    return triples


def normalized_set(
    items: Iterable[str] | Iterable[tuple[str, str, str]], variant: str
) -> set[str] | set[tuple[str, str, str]]:
    normalized = set()
    for item in items:
        if isinstance(item, tuple):
            normalized.add(tuple(normalize_component(value, variant) for value in item))
        else:
            normalized.add(normalize_component(item, variant))
    return normalized


def set_counts(gold: set, predicted: set) -> dict[str, int | float | bool]:
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    # An empty prediction makes no positive assertion. Its precision is 0.0,
    # including the empty-gold case; exact-set equality remains a separate metric.
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else float(not predicted)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "exact_set_equality": gold == predicted,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def set_metrics_by_variant(
    gold_items: list, predicted_items: list, malformed_false_positives: int = 0
) -> dict[str, dict]:
    output = {
        variant: set_counts(
            normalized_set(gold_items, variant),
            normalized_set(predicted_items, variant),
        )
        for variant in SET_VARIANTS
    }
    if malformed_false_positives:
        for metrics in output.values():
            metrics["false_positive"] += malformed_false_positives
            metrics["exact_set_equality"] = False
            denominator = metrics["true_positive"] + metrics["false_positive"]
            metrics["precision"] = metrics["true_positive"] / denominator
            precision, recall = metrics["precision"], metrics["recall"]
            metrics["f1"] = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
    return output


def first_integer(text: str) -> int | None:
    match = INTEGER_RE.search(text)
    return int(match.group()) if match else None


def gold_integer(task: str, text: str) -> int:
    patterns = {
        "node_number": r"\bThere are\s+(\d+)\s+nodes?\b",
        "edge_number": r"\bThere are\s+(\d+)\s+edges?\b",
        "node_degree": r"\bdegree\s+of\s+the\s+node\s+.+?\s+is\s+(\d+)\b",
        "highest_node_degree": r"\bwith\s+a\s+degree\s+of\s+(\d+)\b",
    }
    match = re.search(patterns[task], text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse {task} gold integer: {text!r}")
    return int(match.group(1))


def response_degree(text: str) -> int | None:
    patterns = (
        r"\bdegree(?:\s+of|\s+is|\s*=|\s*:)?\s+([-+]?\d+)\b",
        r"\b([-+]?\d+)\s+(?:connections?|edges?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return first_integer(text)


def metadata_node_labels(meta: dict) -> dict[int, str]:
    return {
        int(node["cid"]): str(node.get("label", node["name"])).replace("_", " ")
        for node in meta["visible_nodes"]
    }


def valid_highest_nodes(meta: dict) -> tuple[list[str], int]:
    labels = metadata_node_labels(meta)
    degrees = Counter({cid: 0 for cid in labels})
    for edge in meta["edges"]:
        source = int(edge["source_cid"])
        target = int(edge["target_cid"])
        if source not in labels or target not in labels:
            raise ValueError(
                f"Metadata edge references an absent visible node for statement_idx="
                f"{meta.get('statement_idx')}"
            )
        degrees[source] += 1
        degrees[target] += 1
    if not degrees:
        raise ValueError(f"No visible nodes for statement_idx={meta.get('statement_idx')}")
    maximum = max(degrees.values())
    return sorted(labels[cid] for cid, degree in degrees.items() if degree == maximum), maximum


def _phrase_present(response: str, phrase: str, variant: str) -> bool:
    response_norm = normalize_component(response, variant)
    phrase_norm = normalize_component(phrase, variant)
    if not phrase_norm:
        return False
    return re.search(
        rf"(?<![\w]){re.escape(phrase_norm)}(?![\w])", response_norm, re.IGNORECASE
    ) is not None


def parse_highest_name(response: str) -> str | None:
    quoted = re.findall(r'["“]([^"”]+)["”]', response)
    if quoted:
        return quoted[0].strip()
    patterns = (
        r"(?:node\s+)?(.+?)\s+with\s+(?:a\s+)?degree\s+of\s+[-+]?\d+",
        r"(?:node\s+)?(.+?)\s+(?:has|is at)\s+(?:a\s+)?(?:maximum\s+)?degree\s+[-+]?\d+",
    )
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" \t\n:,-.\"")
            candidate = re.sub(
                r"^(?:the\s+)?(?:answer\s+is\s+|node(?:\s+is)?\s+)",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            return candidate or None
    return None


def _component_recall(
    gold_triples: list[tuple[str, str, str]], response: str, variant: str
) -> dict[str, int | float]:
    gold_nodes = {
        normalize_component(node, variant)
        for source, _, target in gold_triples
        for node in (source, target)
    }
    found = sum(_phrase_present(response, node, variant) for node in gold_nodes)
    return {
        "gold_node_count": len(gold_nodes),
        "found_gold_nodes": found,
        "recall": found / len(gold_nodes) if gold_nodes else 1.0,
    }


def require_task(task: str, task_types=TASK_TYPES) -> None:
    if task not in task_types:
        raise ValueError(
            f"Task {task!r} is not enabled by the selected task set; "
            "use --task-set extended for the nine-task input."
        )


def structured_gold(record: dict) -> dict:
    gold = record.get("gold")
    if not isinstance(gold, dict):
        raise ValueError(f"Task {record['task_type']} requires a structured gold dict in the input")
    required = {
        "relation_identification": {"relation", "relation_text", "node_a", "node_b"},
        "neighbor_listing": {"node", "neighbors", "degree"},
        "shortest_path_listing": {"node_a", "node_b", "distance", "paths", "paths_truncated"},
    }[record["task_type"]]
    if required - gold.keys():
        raise ValueError(f"Task {record['task_type']} lacks gold fields: {sorted(required - gold.keys())}")
    if record['task_type'] == 'shortest_path_listing' and not gold['paths']:
        raise ValueError('Shortest-path gold must contain at least one path')
    return gold


def parse_relation(response: str) -> str:
    """Extract one asserted relation, not arbitrary relation mentions in prose."""
    payload = response.strip()
    # The generator's sentence quotes both endpoints. Consume them before
    # searching for 'is', which could itself appear inside a concept label.
    template = re.search(
        r'\b(?:relation|relationship)\s+between\s+"[^"\n]*"\s+and\s+"[^"\n]*"\s+is\s+',
        payload, re.IGNORECASE,
    )
    generic = re.search(
        r'\b(?:relation(?:ship)?|edge label|answer)\b[^\n]*?(?:\bis\s+|:\s*)', payload, re.IGNORECASE,
    )
    match = template or generic
    if match:
        payload = payload[match.end():]
    elif re.match(r'^(?:relation(?:ship)?|edge label|answer)\s*:', payload, re.IGNORECASE):
        payload = payload.split(':', 1)[1]
    payload = _strip_payload_sentence(payload.strip()).strip('"\'`')
    # A leading quoted assertion can be followed by an explanation sentence.
    quoted = re.match(r'^["\'`]([^"\'`]+)["\'`](?:[.!]\s+.*|[.!]?)$',
                      response[match.end():].strip() if match else response.strip(), re.DOTALL)
    if quoted:
        payload = quoted.group(1)
    return payload.strip()


def parse_neighbor_items(response: str) -> list[str]:
    """Remove only neighbour-specific lead-ins, then reuse node-list parsing."""
    payload = re.sub(r'^.*?\b(?:directly\s+)?connected\s+to\s*:?\s*', '', response,
                     count=1, flags=re.IGNORECASE | re.DOTALL)
    if payload == response:
        payload = re.sub(r'^.*?\b(?:neighbors?|neighbours?|adjacent nodes)\b.*?(?:\bare\s*:?[ \t]*|:\s*)',
                         '', response, count=1, flags=re.IGNORECASE | re.DOTALL)
    return parse_node_items(payload)


def parse_path_items(response: str) -> list[str]:
    """Parse an ordered arrow/comma/numbered sequence, preserving [A,B] labels."""
    payload = re.sub(r'^.*?\b(?:path|sequence|route|chain)\b.*?(?:\bis\s*:?[ \t]*|:\s*)',
                     '', response, count=1, flags=re.IGNORECASE | re.DOTALL)
    payload = re.sub(r'(?:^|(?<=\s))\d+[.)]\s+', '\n', payload)
    payload = payload.replace('->', '\n').replace('→', '\n')
    return parse_node_items(payload)


def score_new_task(record: dict, response: str) -> dict:
    task, gold = record['task_type'], structured_gold(record)
    if task == 'relation_identification':
        parsed = parse_relation(response)
        def relation_key(value, variant):
            return re.sub(r'\s+', '', normalize_component(value, variant).lower())
        correct = {variant: relation_key(parsed, variant) in {
            relation_key(gold['relation'], variant), relation_key(gold['relation_text'], variant)
        } for variant in SET_VARIANTS}
        return {'gold_relation': gold['relation'], 'parsed_relation': parsed,
                'relation_correct': correct, 'is_correct': correct['raw']}
    if task == 'neighbor_listing':
        parsed = parse_neighbor_items(response)
        metrics = set_metrics_by_variant(gold['neighbors'], parsed)
        return {'parsed_gold_nodes': gold['neighbors'], 'parsed_predicted_nodes': parsed,
                'gold_degree': gold['degree'], 'set_metrics': metrics,
                # Kept only for the shared failure-case selector; primary score is F1.
                'is_correct': metrics['raw']['f1'] == 1.0}
    parsed = parse_path_items(response)
    exact, partial = {}, {}
    for variant in SET_VARIANTS:
        predicted = [normalize_component(node, variant) for node in parsed]
        paths = [[normalize_component(node, variant) for node in path] for path in gold['paths']]
        exact[variant] = predicted in paths
        closest = max((set_counts(set(path), set(predicted)) for path in paths),
                      key=lambda metrics: metrics['f1'])
        partial[variant] = {'hop_count_correct': bool(parsed) and len(parsed) - 1 == gold['distance'],
                            'node_overlap_f1': closest['f1']}
    return {'parsed_path': parsed, 'gold_distance': gold['distance'],
            'parsed_hop_count': len(parsed) - 1 if parsed else None,
            'path_exact': exact, 'path_partial': partial,
            'paths_truncated': gold['paths_truncated'], 'is_correct': exact['raw']}


def score_record_legacy(record: dict, response: str, meta: dict) -> dict:
    """Return JSON-safe parsed fields and correctness flags for one response."""
    task = record["task_type"]
    gold = str(record["answer"])
    response = "" if response is None else str(response).strip()
    if task in NEW_TASK_TYPES:
        return score_new_task(record, response)
    if task in NUMERIC_TASKS:
        gold_value = gold_integer(task, gold)
        predicted_integer = first_integer(response)
        signed_error = (
            predicted_integer - gold_value if predicted_integer is not None else None
        )
        return {
            "gold_integer": gold_value,
            "parsed_integer": predicted_integer,
            "signed_error": signed_error,
            "is_correct": predicted_integer == gold_value,
        }

    if task == "highest_node_degree":
        gold_degree = gold_integer(task, gold)
        valid_names, metadata_degree = valid_highest_nodes(meta)
        if gold_degree != metadata_degree:
            raise ValueError(
                f"Gold/metadata maximum-degree mismatch at statement_idx="
                f"{record['statement_idx']}: gold={gold_degree}, metadata={metadata_degree}"
            )
        predicted_degree = response_degree(response)
        parsed_name = parse_highest_name(response)
        result = {
            "gold_degree": gold_degree,
            "parsed_degree": predicted_degree,
            "degree_correct": predicted_degree == gold_degree,
            "parsed_name": parsed_name,
            "valid_highest_names": valid_names,
        }
        for variant in SET_VARIANTS:
            valid = {normalize_component(name, variant) for name in valid_names}
            parsed_correct = (
                parsed_name is not None
                and normalize_component(parsed_name, variant) in valid
            )
            mentioned_valid = [
                name for name in valid_names if _phrase_present(response, name, variant)
            ]
            # Accept any maximum-degree node named in the response. This is
            # deliberately tie-aware and avoids penalising harmless prose that
            # the diagnostic single-name parser cannot isolate.
            name_correct = parsed_correct or bool(mentioned_valid)
            result[f"mentioned_valid_names_{variant}"] = mentioned_valid
            result[f"name_correct_{variant}"] = name_correct
            result[f"both_correct_{variant}"] = bool(
                result["degree_correct"] and name_correct
            )
        result["is_correct"] = result["both_correct_raw"]
        return result

    if task == "node_description":
        gold_items = parse_node_items(gold, gold=True)
        predicted_items = parse_node_items(response)
        metrics = set_metrics_by_variant(gold_items, predicted_items)
        return {
            "parsed_gold_nodes": gold_items,
            "parsed_predicted_nodes": predicted_items,
            "set_metrics": metrics,
            "is_correct": metrics["raw"]["exact_set_equality"],
        }

    if task == "triple_listing":
        gold_triples = parse_triples(gold)
        predicted_triples = parse_triples(response)
        if not gold_triples:
            raise ValueError(f"Could not parse triple gold: {gold!r}")
        metrics = set_metrics_by_variant(gold_triples, predicted_triples)
        return {
            "parsed_gold_triples": [list(triple) for triple in gold_triples],
            "parsed_predicted_triples": [list(triple) for triple in predicted_triples],
            "set_metrics": metrics,
            "gold_node_component_recall": {
                variant: _component_recall(gold_triples, response, variant)
                for variant in SET_VARIANTS
            },
            "is_correct": metrics["raw"]["exact_set_equality"],
        }
    raise ValueError(f"Unknown task_type: {task}")


def _mean(values: list[float]) -> float | None:
    # Match the left-to-right float summation used by the completed runs'
    # Python interpreter (built-in sum changed in Python 3.12).
    total = 0
    for value in values:
        total += value
    return total / len(values) if values else None


def _counter_json(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def aggregate_set_metrics(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for variant in SET_VARIANTS:
        entries = [row["set_metrics"][variant] for row in rows]
        tp = sum(int(entry["true_positive"]) for entry in entries)
        fp = sum(int(entry["false_positive"]) for entry in entries)
        fn = sum(int(entry["false_negative"]) for entry in entries)
        micro_precision = tp / (tp + fp) if tp + fp else 1.0
        micro_recall = tp / (tp + fn) if tp + fn else 1.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        output[variant] = {
            "exact_set_equality_count": sum(
                bool(entry["exact_set_equality"]) for entry in entries
            ),
            "exact_set_equality": _mean(
                [float(entry["exact_set_equality"]) for entry in entries]
            ),
            "macro_precision": _mean([float(entry["precision"]) for entry in entries]),
            "macro_recall": _mean([float(entry["recall"]) for entry in entries]),
            "macro_f1": _mean([float(entry["f1"]) for entry in entries]),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "micro_true_positive": tp,
            "micro_false_positive": fp,
            "micro_false_negative": fn,
        }
    return output


def aggregate_new_task(task: str, rows: list[dict]) -> dict:
    if task == 'relation_identification':
        def accuracy(group):
            return {variant: {'accuracy': _mean([float(row['relation_correct'][variant]) for row in group]),
                              'correct_count': sum(row['relation_correct'][variant] for row in group)}
                    for variant in SET_VARIANTS}
        relations = sorted({row['gold_relation'] for row in rows})
        return {**accuracy(rows), 'per_relation': {
            rel: {'record_count': sum(row['gold_relation'] == rel for row in rows),
                  **accuracy([row for row in rows if row['gold_relation'] == rel])}
            for rel in relations}}
    if task == 'neighbor_listing':
        def averages(group):
            return {variant: {f'mean_{name}': _mean([float(row['set_metrics'][variant][name]) for row in group])
                              for name in ('precision', 'recall', 'f1')}
                    for variant in SET_VARIANTS}
        degrees = sorted({row['gold_degree'] for row in rows})
        return {**averages(rows), 'by_degree': {
            str(degree): {'record_count': sum(row['gold_degree'] == degree for row in rows),
                          **averages([row for row in rows if row['gold_degree'] == degree])}
            for degree in degrees}}
    return {'paths_truncated_count': sum(bool(row['paths_truncated']) for row in rows),
            **{variant: {
                'path_exact': _mean([float(row['path_exact'][variant]) for row in rows]),
                'hop_count_correct': _mean([float(row['path_partial'][variant]['hop_count_correct']) for row in rows]),
                'mean_overlap_f1': _mean([float(row['path_partial'][variant]['node_overlap_f1']) for row in rows]),
            } for variant in SET_VARIANTS}}


def aggregate_task_legacy(task: str, rows: list[dict]) -> dict:
    ceiling_count = sum(bool(row.get("hit_token_ceiling")) for row in rows)
    metrics: dict = {
        "record_count": len(rows),
        "hit_token_ceiling_count": ceiling_count,
        "hit_token_ceiling_fraction": ceiling_count / len(rows) if rows else None,
        "token_ceiling_interpretation": (
            "A high ceiling-hit fraction invalidates this task score as a pure "
            "graph-comprehension measurement."
        ),
    }
    if task in NEW_TASK_TYPES:
        metrics.update(aggregate_new_task(task, rows))
        return metrics
    if task in NUMERIC_TASKS:
        parsed = [row for row in rows if row["parsed_integer"] is not None]
        errors = [int(row["signed_error"]) for row in parsed]
        metrics.update(
            {
                "parsed_count": len(parsed),
                "unparsed_count": len(rows) - len(parsed),
                "exact_correct_count": sum(bool(row["is_correct"]) for row in rows),
                "exact_accuracy": _mean([float(row["is_correct"]) for row in rows]),
                "mean_absolute_error": _mean([abs(error) for error in errors]),
                "signed_error_distribution": _counter_json(errors),
                "gold_value_distribution": _counter_json(
                    int(row["gold_integer"]) for row in rows
                ),
            }
        )
        return metrics

    if task == "highest_node_degree":
        metrics.update(
            {
                "parsed_degree_count": sum(row["parsed_degree"] is not None for row in rows),
                "parsed_name_count": sum(row["parsed_name"] is not None for row in rows),
                "degree_correct_count": sum(bool(row["degree_correct"]) for row in rows),
                "degree_accuracy": _mean([float(row["degree_correct"]) for row in rows]),
                "gold_degree_distribution": _counter_json(
                    int(row["gold_degree"]) for row in rows
                ),
            }
        )
        for variant in SET_VARIANTS:
            metrics[variant] = {
                "name_correct_count": sum(
                    bool(row[f"name_correct_{variant}"]) for row in rows
                ),
                "name_accuracy": _mean(
                    [float(row[f"name_correct_{variant}"]) for row in rows]
                ),
                "both_correct_count": sum(
                    bool(row[f"both_correct_{variant}"]) for row in rows
                ),
                "both_accuracy": _mean(
                    [float(row[f"both_correct_{variant}"]) for row in rows]
                ),
            }
        cap_count = sum(int(row["gold_degree"]) == 5 for row in rows)
        metrics["pruning_cap_5_count"] = cap_count
        metrics["pruning_cap_5_fraction"] = cap_count / len(rows) if rows else None
        metrics["constant_degree_5_baseline_accuracy"] = (
            cap_count / len(rows) if rows else None
        )
        metrics["valid_highest_name_count_distribution"] = _counter_json(
            len(row["valid_highest_names"]) for row in rows
        )
        metrics["confound_warning"] = (
            "The maximum-degree gold is saturated at the max_degree=5 pruning cap; "
            "degree accuracy may reward a constant-output strategy."
        )
        return metrics

    metrics["set_metrics"] = aggregate_set_metrics(rows)
    if task == "triple_listing":
        component = {}
        for variant in SET_VARIANTS:
            entries = [row["gold_node_component_recall"][variant] for row in rows]
            found = sum(int(entry["found_gold_nodes"]) for entry in entries)
            gold_count = sum(int(entry["gold_node_count"]) for entry in entries)
            component[variant] = {
                "macro_recall": _mean([float(entry["recall"]) for entry in entries]),
                "micro_recall": found / gold_count if gold_count else 1.0,
                "found_gold_nodes": found,
                "gold_node_count": gold_count,
            }
        metrics["gold_node_component_recall"] = component
    return metrics


# The 17 ConceptNet relations rendered by generate_graphvis_datasets.py.
# Kept local so offline scoring requires only the Python standard library.
RELATION_VOCABULARY = (
    'antonym', 'atlocation', 'capableof', 'causes', 'createdby', 'isa',
    'desires', 'hassubevent', 'partof', 'hascontext', 'hasproperty', 'madeof',
    'notcapableof', 'notdesires', 'receivesaction', 'relatedto', 'usedfor',
)
ANSWER_MARKER_RE = re.compile(
    r'\b(?:final\s+answer\b\s*(?:is\b)?|answer\s*:|therefore\b|'
    r'in\s+conclusion\b|so\s+the\b|the\s+answer\s+is\b)', re.IGNORECASE,
)
ANSWER_COLON_LINE_RE = re.compile(r'^\s*Answer\s*:', re.IGNORECASE)
OCR_STYLE_DUMP_RE = re.compile(
    r'<\s*/?\s*(?:table|tr|td|th|html)\b|<\|(?:ref|det)\|>|```(?:html|xml)',
    re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+(.+)$')
STOPWORDS = frozenset('a an the this that these those is are was were be been being '
                     'and or but to of in on at by for with from it its they them '
                     'as here there following node nodes neighbor neighbors '
                     'neighbour neighbours answer none no'.split())


def strip_markdown(text: str) -> str:
    # Preserve star bullets before removing emphasis.
    text = re.sub(r'(?m)^(\s*)\*\s+', r'\1- ', text)
    return re.sub(r'[*`]', '', text)


def answer_span(text: str) -> str:
    """Take the last answer marker, else last nonempty line, else whole text."""
    text = strip_markdown(text)
    markers = list(ANSWER_MARKER_RE.finditer(text))
    if markers:
        return text[markers[-1].end():].lstrip(' \t\r\n:,')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if len(lines) > 1 else text


def structured_answer_span(text: str) -> str:
    """Retain explicit lists/tuples even when followed by explanatory prose.

    Numeric and relation parsing use answer_span directly. Sets and paths must
    retain preceding list members when the generic helper selects a last line.
    With no answer marker, the first-to-last list-item region is the answer
    structure, including grouped lists separated by headings.
    """
    span = answer_span(text)
    clean = strip_markdown(text)
    if ANSWER_MARKER_RE.search(clean):
        return span
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    indices = [i for i, line in enumerate(lines) if LIST_ITEM_RE.match(line) or parse_triples(line)]
    if indices:
        return '\n'.join(lines[indices[0]:indices[-1] + 1])
    return span


EXPLICIT_NUMERIC_PATTERNS = {
    'node_number': (
        r'\b(?:there\s+are|the\s+graph\s+(?:has|contains))\s+([-+]?\d+)\s+(?:nodes?|vertices?)\b',
        r'\b(?:the\s+)?(?:total\s+)?(?:number|count)\s+of\s+(?:nodes?|vertices?)'
        r'(?:\s+in\s+(?:the\s+)?graph)?\s+(?:is|=|:)\s*([-+]?\d+)\b',
        r'\btotal(?:\s+(?:nodes?|vertices?))?\s*(?::|=|is)\s*'
        r'(?:[-+]?\d+\s*(?:\+\s*[-+]?\d+\s*)*=\s*)?([-+]?\d+)\b',
    ),
    'edge_number': (
        r'\b(?:there\s+are|the\s+graph\s+(?:has|contains))\s+([-+]?\d+)\s+(?:edges?|connections?)\b',
        r'\b(?:the\s+)?(?:total\s+)?(?:number|count)\s+of\s+(?:edges?|connections?)'
        r'(?:\s+in\s+(?:the\s+)?graph)?\s+(?:is|=|:)\s*([-+]?\d+)\b',
        r'\btotal(?:\s+(?:edges?|connections?))?\s*(?::|=|is)\s*'
        r'(?:[-+]?\d+\s*(?:\+\s*[-+]?\d+\s*)*=\s*)?([-+]?\d+)\b',
    ),
    'node_degree': (
        r'\bdegree(?:\s+of\s+(?:the\s+)?(?:node\s+)?[^\n,;:.]+)?\s+(?:is|=|:|of)\s*([-+]?\d+)\b',
        r'\bhas\s+([-+]?\d+)\s+(?:connections?|edges?|neighbou?rs?)\b',
    ),
}


def explicit_numeric_value(text: str, task: str) -> int | None:
    matches = [
        (match.start(), int(match.group(1)))
        for pattern in EXPLICIT_NUMERIC_PATTERNS.get(task, ())
        for match in re.finditer(pattern, strip_markdown(text), re.IGNORECASE)
    ]
    return max(matches, default=(None, None), key=lambda item: item[0])[1]


def span_integer(text: str, task: str | None = None) -> int | None:
    clean = strip_markdown(text)
    span = answer_span(text)
    if ANSWER_MARKER_RE.search(clean):
        values = INTEGER_RE.findall(span)
        if values:
            return int(values[-1])
        values = INTEGER_RE.findall(clean)
        return int(values[-1]) if values else None
    if task:
        explicit = explicit_numeric_value(text, task)
        if explicit is not None:
            return explicit
    values = INTEGER_RE.findall(span)
    if values:
        return int(values[-1])
    values = INTEGER_RE.findall(text)
    return int(values[-1]) if values else None


def span_degree(text: str) -> int | None:
    span = answer_span(text)
    compact = compact_highest_answer(span)
    if compact:
        return compact[1]
    pattern = (
        r'\b(?:degree(?:\s+of|\s+is|\s*=|\s*:)?|with\s+a\s+degree(?:\s+of)?|'
        r'has\s+(?:a\s+)?degree(?:\s+of)?)\s+([-+]?\d+)\b'
    )
    matches = list(re.finditer(pattern, span, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    if not ANSWER_MARKER_RE.search(strip_markdown(text)):
        explicit = list(re.finditer(
            pattern + r'|\bhas\s+([-+]?\d+)\s+(?:connections?|edges?|neighbou?rs?)\b',
            strip_markdown(text), re.IGNORECASE,
        ))
        if explicit:
            match = explicit[-1]
            return int(match.group(1) or match.group(2))
    try:
        return gold_integer('highest_node_degree', text)
    except ValueError:
        # Bare degrees are safe, unlike a numeric node name in an assertion.
        return int(span.strip()) if re.fullmatch(r'[-+]?\d+', span.strip()) else None


def compact_highest_answer(text: str) -> tuple[str, int] | None:
    parts = [part.strip() for part in split_outside_annotations(text.strip(), ',')]
    if len(parts) != 2 or not re.fullmatch(r'[-+]?\d+', parts[1]):
        return None
    name = parts[0].strip(' "\'“”')
    return (name, int(parts[1])) if name else None


def explicit_highest_name(text: str) -> str | None:
    clean = strip_markdown(text)
    name = r'(?P<name>"[^"\n]+"|“[^”\n]+”|[^\n,;:.]+?)'
    patterns = (
        rf'\b(?:node\s+with\s+(?:the\s+)?highest\s+degree|highest[- ]degree\s+node|'
        rf'node\s+with\s+(?:the\s+)?(?:most|greatest\s+number\s+of)\s+connections?)\s+'
        rf'(?:is|:)\s*(?:node\s+)?{name}(?=\s*(?:[,.;\n]|$))',
        rf'(?m)^\s*(?:the\s+)?(?:node\s+)?{name}\s+(?:has|with)\s+(?:a\s+)?'
        rf'(?:maximum\s+|highest\s+)degree(?:\s+of|\s+is|\s*=|\s*:)?\s+[-+]?\d+\b',
    )
    matches = [
        (match.start(), match.group('name'))
        for pattern in patterns
        for match in re.finditer(pattern, clean, re.IGNORECASE)
    ]
    if not matches:
        return None
    candidate = max(matches, key=lambda item: item[0])[1]
    return candidate.strip(' \t\n:,-.\"\'“”') or None


def span_highest_name(text: str) -> str | None:
    span = answer_span(text)
    compact = compact_highest_answer(span)
    if compact:
        return compact[0]
    if ANSWER_MARKER_RE.search(strip_markdown(text)):
        return parse_highest_name(span)
    return explicit_highest_name(text) or parse_highest_name(span)


def relation_key(value: str) -> str:
    return re.sub('[^a-z]', '', value.lower())


RELATION_PATTERNS = {
    key: re.compile(r'(?<![a-z])' + r'[^a-z]*'.join(key) + r'(?![a-z])', re.IGNORECASE)
    for key in RELATION_VOCABULARY
}


def relation_mentions(text: str) -> list[tuple[int, int, str]]:
    matches = [(m.start(), m.end(), key) for key, pattern in RELATION_PATTERNS.items()
               for m in pattern.finditer(text)]
    # 'not capable of' must not also match its contained 'capable of'.
    return sorted((start, end, key) for start, end, key in matches
                  if not any(a <= start and end <= b and (a, b) != (start, end)
                             for a, b, _ in matches))


def span_relation(text: str) -> str | None:
    matches = relation_mentions(answer_span(text))
    return max(matches, key=lambda match: (match[1], match[0]))[2] if matches else None


def valid_node_candidate(value: str) -> bool:
    words = set(re.findall(r'[a-z]+', value.lower()))
    return bool(value.strip()) and relation_key(value) not in RELATION_VOCABULARY and (
        not words or not words <= STOPWORDS)


def span_node_items(text: str, *, neighbors: bool = False) -> list[str]:
    payload = structured_answer_span(text)
    # A bullet can follow a colon on the same line as a prose introduction.
    payload = re.sub(r':\s*(?=[-•]\s+|\d+[.)]\s+)', ':\n', payload)
    bullets = [match.group(1) for line in payload.splitlines()
               if (match := LIST_ITEM_RE.match(line))]
    if bullets:
        items = [item for bullet in bullets for item in parse_node_items(bullet)]
    else:
        last_line = payload.splitlines()[-1] if payload.splitlines() else payload
        if len(split_outside_annotations(last_line, ',')) > 1:
            payload = last_line
        parser = parse_neighbor_items if neighbors else parse_node_items
        items = parser(payload)
        if neighbors and len(items) == 1:
            # Singular assertions: ... adjacent node to "looking" is: "look".
            assertion = re.search(r'\b(?:is|are)\s*:?\s*([^\n]+)$', payload, re.IGNORECASE)
            if assertion:
                items = parse_node_items(assertion.group(1))
    return [item for item in items if valid_node_candidate(item)]


def scored_triples(text: str) -> tuple[list[tuple[str, str, str]], int]:
    """Return valid triples and the count of malformed triple assertions.

    Inside a final ``Answer:`` block every nonblank line must consist of exactly
    one valid triple. Outside a block, malformed counts apply per parenthesized
    group only when the group contains a comma and has wrong arity or an empty
    component. Other commentary is ignored.
    """
    # Models also emit three separately quoted components on each list line.
    clean = strip_markdown(re.sub(
        r'`([^`\n]+)`\s+`([^`\n]+)`\s+`([^`\n]+)`',
        lambda m: '(' + ', '.join(m.groups()) + ')', text,
    ))
    marker = list(re.finditer(r'(?im)^\s*Answer\s*:', clean))
    payload = clean[marker[-1].end():] if marker else clean
    triples: list[tuple[str, str, str]] = []
    malformed = 0
    for raw_line in payload.splitlines() or [payload]:
        line = raw_line.strip()
        if not line:
            continue
        groups = PARENTHESIZED_RE.findall(line)
        structurally_valid = []
        for group in groups:
            components = [part.strip() for part in split_outside_annotations(group, ',')]
            if len(components) == 3 and all(components):
                structurally_valid.append(tuple(components))
            elif not marker and ',' in group:
                # A comma in natural-language commentary is not enough by
                # itself. Empty slots, 3+ fields, or a 2-field prefix whose
                # second field is a known relation make this a triple attempt.
                looks_like_attempt = (
                    any(not component for component in components)
                    or len(components) >= 3
                    or (len(components) == 2
                        and relation_key(components[1]) in RELATION_VOCABULARY)
                )
                malformed += int(looks_like_attempt)
        parsed = [triple for triple in structurally_valid
                  if valid_node_candidate(triple[0]) and valid_node_candidate(triple[2])]
        triples.extend(parsed)
        if not marker:
            continue
        # In a constrained block, syntax is line-based: exactly one complete
        # parenthesized triple and no prose or second triple on that line.
        residue = PARENTHESIZED_RE.sub("", line)
        residue = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', residue)
        residue = re.sub(r'[\s,;:.]+', '', residue)
        if len(groups) != 1 or len(structurally_valid) != 1 or residue:
            malformed += 1
    return triples, malformed


def span_triples(text: str) -> list[tuple[str, str, str]]:
    return scored_triples(text)[0]


def answer_format_diagnostics(task: str, text: str, answer_format: str = 'constrained') -> dict:
    """Describe final-format compliance without changing answer extraction.

    OCR/HTML detection is diagnostic only. It deliberately does not add a
    model-specific parser or alter the scored response.
    """
    clean = strip_markdown('' if text is None else str(text)).strip()
    markers = list(re.finditer(r'(?im)^\s*Answer\s*:', clean))
    marker_found = bool(markers)
    payload = clean[markers[-1].end():].lstrip(' \t') if markers else ''
    parsed = False
    compliant = False
    if marker_found:
        tail = clean[markers[-1].start():].strip()
        if task in NUMERIC_TASKS:
            parsed = re.fullmatch(r'[-+]?\d+', payload.strip()) is not None
            compliant = re.fullmatch(r'Answer\s*:\s*[-+]?\d+', tail, re.IGNORECASE) is not None
        elif task == 'highest_node_degree':
            parsed = compact_highest_answer(payload.strip()) is not None
            compliant = '\n' not in tail and parsed
        elif task == 'node_description':
            parsed = bool(span_node_items('Answer: ' + payload))
            compliant = '\n' not in tail and bool(payload.strip())
        elif task == 'triple_listing':
            lines = [line.strip() for line in payload.splitlines() if line.strip()]
            parsed = bool(lines) and all(len(parse_triples(line)) == 1 for line in lines)
            compliant = (
                re.match(r'(?i)^Answer\s*:\s*(?:\n|$)', tail) is not None
                and parsed
                and all(re.fullmatch(r'\([^()]+\)', line) is not None for line in lines)
            )
        else:
            parsed = bool(payload.strip())
    return {
        'answer_format': answer_format,
        'final_answer_marker_found': marker_found,
        'answer_marker_extraction_success': marker_found and parsed,
        'final_format_compliant': compliant if answer_format == 'constrained' else None,
        'ocr_style_dump': bool(OCR_STYLE_DUMP_RE.search(clean)),
    }


def has_repetition_loop(text: str) -> bool:
    """Conservative, model-agnostic repetition-loop diagnostic."""
    clean = strip_markdown('' if text is None else str(text)).lower()
    lines = [re.sub(r'^\s*\d+[.)]\s*', '', line).strip()
             for line in clean.splitlines() if line.strip()]
    if any(count >= 3 and len(line.split()) >= 3 for line, count in Counter(lines).items()):
        return True
    tokens = re.findall(r"[\w']+|[^\w\s]", clean)
    if len(tokens) < 36:
        return False
    grams = Counter(tuple(tokens[i:i + 12]) for i in range(len(tokens) - 11))
    return max(grams.values(), default=0) >= 3


def classify_ceiling_response(task: str, text: str) -> str:
    diagnostics = answer_format_diagnostics(task, text, 'constrained')
    if (diagnostics['final_answer_marker_found']
            and not diagnostics['final_format_compliant']):
        return 'answer_block_too_long'
    if has_repetition_loop(text):
        return 'repetition_loop'
    return 'progressing_reasoning'


def span_path(text: str, source: str, target: str) -> tuple[list[str], bool]:
    span = structured_answer_span(text)
    key = lambda node: normalize_component(node, 'annotation_stripped')

    def chains(payload):
        output = []
        # Each sentence/line contains a separate trace or candidate path.
        for line in re.split(r'\n|;|(?<=[.!?])\s+', payload):
            if '->' in line or '→' in line or len(split_outside_annotations(line, ',')) > 1:
                line = re.sub(r'^\s*\d+[.)]\s*', '', line)
                if ':' in line:
                    line = line.rsplit(':', 1)[-1]
                parsed = parse_path_items(line)
                # Commas in later commentary must not erase an arrow chain
                # (and its cycle flag). Accept traced comma lines only when
                # they form a complete chain; bare comma answers fall through
                # to parse_path_items(span) below.
                if ('->' in line or '→' in line or parsed and
                        key(parsed[0]) == key(source) and key(parsed[-1]) == key(target)):
                    output.append(parsed)
        return output

    candidates = chains(span)
    # Path-specific exception: look through all traces for a complete chain.
    all_chains = chains(strip_markdown(text))
    complete = [chain for chain in all_chains if chain and
                key(chain[0]) == key(source) and key(chain[-1]) == key(target)]
    # A final numbered sequence may supersede earlier arrow traces.
    numbered = parse_path_items(span) if not candidates else []
    if numbered and key(numbered[0]) == key(source) and key(numbered[-1]) == key(target):
        complete.append(numbered)
    parsed = (complete[-1] if complete else candidates[-1] if candidates
              else all_chains[-1] if all_chains else parse_path_items(span))
    parsed = [node for node in parsed if relation_key(node) not in RELATION_VOCABULARY]
    keys = [key(node) for node in parsed]
    return parsed, len(keys) != len(set(keys))


def integer_containment(text: str, value: int) -> bool:
    return any(int(match.group()) == value for match in
               re.finditer(r'(?<![\w.+-])[-+]?\d+(?!\w|[.,]\d)', text))


def member_pattern(member: str, variant: str) -> str:
    return r'(?<!\w)' + re.escape(normalize_component(member, variant)) + r'(?!\w)'


def member_containment(text: str, members: Iterable, variant: str) -> float:
    normalized = normalize_component(strip_markdown(text), variant)
    members = normalized_set(members, variant)
    def present(member):
        components = member if isinstance(member, tuple) else (member,)
        return all(re.search(member_pattern(component, variant), normalized) is not None
                   for component in components)
    return sum(present(member) for member in members) / len(members) if members else 1.0


def path_containment(text: str, paths: list[list[str]], variant: str) -> bool:
    normalized = normalize_component(strip_markdown(text), variant)
    for path in paths:
        offset = 0
        for node in path:
            match = re.search(member_pattern(node, variant), normalized[offset:])
            if not match:
                break
            offset += match.end()
        else:
            return True
    return False


def score_record(record: dict, response: str, meta: dict, extractor: str = 'span') -> dict:
    """Score one best extraction; containment is a separate diagnostic bound."""
    if extractor == 'legacy':
        return score_record_legacy(record, response, meta)
    if extractor != 'span':
        raise ValueError(f'Unknown extractor: {extractor}')
    task = record['task_type']
    require_task(task, EXTENDED_TASK_TYPES)
    response = '' if response is None else str(response).strip()
    span = answer_span(response)
    # Reuse the unchanged scoring formulas after replacing their parsed input.
    if task in NUMERIC_TASKS:
        value = span_integer(response, task)
        row = score_record_legacy(record, str(value) if value is not None else '', meta)
        containment = integer_containment(response, row['gold_integer'])
    elif task == 'highest_node_degree':
        row = score_record_legacy(record, span, meta)
        value = span_degree(response)
        parsed_name = span_highest_name(response)
        row.update(parsed_name=parsed_name, parsed_degree=value,
                   degree_correct=value == row['gold_degree'],
                   signed_error=value - row['gold_degree'] if value is not None else None)
        # A strict answer commits to one name; full-response mentions stay lenient.
        for variant in SET_VARIANTS:
            row[f'name_correct_{variant}'] = row['parsed_name'] is not None and any(
                normalize_component(row['parsed_name'], variant) == normalize_component(name, variant)
                for name in row['valid_highest_names'])
            row[f'both_correct_{variant}'] = row['degree_correct'] and row[f'name_correct_{variant}']
        row['is_correct'] = row['both_correct_raw']
        containment = integer_containment(response, row['gold_degree'])
    elif task == 'relation_identification':
        gold = structured_gold(record)
        parsed = span_relation(response)
        correct = parsed == relation_key(gold['relation'])
        row = {'gold_relation': gold['relation'], 'parsed_relation': parsed,
               'relation_correct': {variant: correct for variant in SET_VARIANTS}, 'is_correct': correct}
        # Containment also counts negated/candidate mentions; only strict
        # extraction resolves overlapping positive and negative labels.
        containment = bool(RELATION_PATTERNS[relation_key(gold['relation'])].search(response))
    elif task == 'shortest_path_listing':
        gold = structured_gold(record)
        parsed, degenerate = span_path(response, gold['node_a'], gold['node_b'])
        row = score_new_task(record, ' -> '.join(parsed))
        row['degenerate_output'] = degenerate
        by_variant = {variant: path_containment(response, gold['paths'], variant) for variant in SET_VARIANTS}
        row['lenient_containment_by_variant'] = by_variant
        containment = by_variant['raw']
    else:
        if task == 'triple_listing':
            gold_items = parse_triples(str(record['answer']))
            if not gold_items:
                raise ValueError(f'Could not parse triple gold: {record["answer"]!r}')
            predicted, malformed = scored_triples(response)
            row = {'parsed_gold_triples': [list(t) for t in gold_items],
                   'parsed_predicted_triples': [list(t) for t in predicted],
                   'malformed_triple_line_count': malformed,
                   'gold_node_component_recall': {v: _component_recall(gold_items, response, v) for v in SET_VARIANTS}}
        else:
            gold = structured_gold(record) if task == 'neighbor_listing' else None
            gold_items = gold['neighbors'] if gold else parse_node_items(str(record['answer']), gold=True)
            predicted = span_node_items(response, neighbors=task == 'neighbor_listing')
            row = {'parsed_gold_nodes': gold_items, 'parsed_predicted_nodes': predicted}
            if gold:
                row['gold_degree'] = gold['degree']
        row['set_metrics'] = set_metrics_by_variant(
            gold_items, predicted,
            malformed if task == 'triple_listing' else 0,
        )
        row['is_correct'] = row['set_metrics']['raw']['exact_set_equality']
        by_variant = {variant: member_containment(response, gold_items, variant) for variant in SET_VARIANTS}
        row['lenient_containment_by_variant'] = by_variant
        containment = by_variant['raw']
    row.update(extractor='span', answer_span=span, lenient_containment=containment)
    return row


def aggregate_task(task: str, rows: list[dict], extractor: str = 'span') -> dict:
    metrics = aggregate_task_legacy(task, rows)
    if extractor == 'legacy':
        return metrics
    if extractor != 'span':
        raise ValueError(f'Unknown extractor: {extractor}')
    metrics['lenient_containment'] = _mean([float(row['lenient_containment']) for row in rows])
    if task in NUMERIC_TASKS:
        metrics['strict_accuracy'] = metrics['exact_accuracy']
    elif task == 'highest_node_degree':
        metrics['strict_accuracy'] = metrics['degree_accuracy']
        metrics['strict_joint_accuracy'] = metrics['raw']['both_accuracy']
        errors = [row['signed_error'] for row in rows if row['signed_error'] is not None]
        metrics['mean_absolute_error'] = _mean([abs(error) for error in errors])
        metrics['signed_error_distribution'] = _counter_json(errors)
    elif task == 'relation_identification':
        metrics['strict_accuracy'] = metrics['raw']['accuracy']
        excluded = [row for row in rows if relation_key(row['gold_relation']) != 'relatedto']
        metrics['relatedto_excluded'] = {
            'record_count': len(excluded),
            'strict_accuracy': _mean([float(row['is_correct']) for row in excluded]),
            'lenient_containment': _mean([float(row['lenient_containment']) for row in excluded]),
            **aggregate_new_task(task, excluded),
        }
    else:
        metrics['lenient_containment_by_variant'] = {
            variant: _mean([float(row['lenient_containment_by_variant'][variant]) for row in rows])
            for variant in SET_VARIANTS}
        if task == 'shortest_path_listing':
            metrics['degenerate_output_fraction'] = _mean([float(row['degenerate_output']) for row in rows])
            metrics['strict_accuracy'] = metrics['raw']['path_exact']
        else:
            metrics['strict'] = (metrics['set_metrics'] if task != 'neighbor_listing'
                                 else {variant: metrics[variant] for variant in SET_VARIANTS})
    return metrics


def levenshtein_distance_at_most(left: str, right: str, limit: int = 2) -> int | None:
    if abs(len(left) - len(right)) > limit:
        return None
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, char_right in enumerate(right, start=1):
            value = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return None
        previous = current
    return previous[-1] if previous[-1] <= limit else None


def near_duplicate_evidence(rows: list[dict]) -> list[dict]:
    evidence = []
    for row in rows:
        gold = {item.strip(): item for item in row["parsed_gold_nodes"]}
        predicted = {item.strip(): item for item in row["parsed_predicted_nodes"]}
        for predicted_exact, predicted_raw in predicted.items():
            if predicted_exact in gold:
                continue
            matches = []
            for gold_exact, gold_raw in gold.items():
                distance = levenshtein_distance_at_most(predicted_exact, gold_exact, 2)
                if distance is not None:
                    matches.append((distance, gold_raw))
            for distance, gold_raw in sorted(matches):
                evidence.append(
                    {
                        "statement_idx": int(row["statement_idx"]),
                        "predicted": predicted_raw,
                        "gold": gold_raw,
                        "distance": distance,
                    }
                )
    return evidence


def _fenced(value: object) -> str:
    return "```text\n" + str(value).replace("```", "` ` `") + "\n```\n"


def write_failure_cases(path: Path, model_name: str, rows_by_task: dict[str, list[dict]],
                        task_types=TASK_TYPES) -> None:
    lines = [
        f"# Stage 1 failure cases: {model_name}\n",
        "Scores below use strict raw correctness to select failures. Token-ceiling hits are shown explicitly.\n",
    ]
    for task in task_types:
        rows = rows_by_task[task]
        candidates = [row for row in rows if not row["is_correct"]]
        wrong = []
        predicates = [
            lambda row: bool(row.get("hit_token_ceiling")),
            lambda row: row.get("parsed_integer", object()) is None,
            lambda row: task == "highest_node_degree"
            and (row.get("parsed_degree") is None or row.get("parsed_name") is None),
            lambda row: task in {"node_description", "triple_listing"}
            and not (
                row.get("parsed_predicted_nodes")
                or row.get("parsed_predicted_triples")
            ),
        ]
        for predicate in predicates:
            match = next((row for row in candidates if row not in wrong and predicate(row)), None)
            if match is not None:
                wrong.append(match)
        wrong.extend(row for row in candidates if row not in wrong)
        wrong = wrong[:5]
        lines.append(f"## {task}\n")
        if not wrong:
            lines.append("No strict-raw failures.\n")
            continue
        for number, row in enumerate(wrong, start=1):
            parsed_fields = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "statement_idx",
                    "task_type",
                    "prompt",
                    "gold",
                    "raw_response",
                    "image",
                    "model_id",
                    "model_revision",
                    "timestamp_utc",
                }
            }
            lines.append(
                f"### {number}. statement_idx={row['statement_idx']} "
                f"(token ceiling: {bool(row.get('hit_token_ceiling'))})\n"
            )
            lines.append("Prompt:\n" + _fenced(row["prompt"]))
            lines.append("Gold:\n" + _fenced(row["gold"]))
            lines.append("Prediction:\n" + _fenced(row["raw_response"]))
            lines.append(
                "Parsed fields and flags:\n```json\n"
                + json.dumps(parsed_fields, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n```\n"
            )

    evidence = near_duplicate_evidence(rows_by_task["node_description"])
    lines.append("## Node-description near-duplicate evidence\n")
    lines.append(
        "This is an evidence dump, not a fuzzy score. A prediction is unmatched "
        "under raw component equality; distance is raw Levenshtein distance at most 2.\n"
    )
    if evidence:
        lines.append("| statement_idx | predicted string | gold string | distance |\n")
        lines.append("| ---: | --- | --- | ---: |\n")
        for item in evidence:
            predicted = str(item["predicted"]).replace("|", "\\|")
            gold = str(item["gold"]).replace("|", "\\|")
            lines.append(
                f"| {item['statement_idx']} | {predicted} | {gold} | {item['distance']} |\n"
            )
    else:
        lines.append("No near-duplicate unmatched predictions were found.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_inputs(records: list[dict], metadata: list[dict], task_types=TASK_TYPES) -> tuple[dict, dict]:
    required = {"statement_idx", "task_type", "prompt", "answer", "image"}
    records_by_key = {}
    for line_number, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Input row {line_number} lacks fields: {sorted(missing)}")
        task = record["task_type"]
        require_task(task, task_types)
        if task in NEW_TASK_TYPES:
            structured_gold(record)
        key = (int(record["statement_idx"]), task)
        if key in records_by_key:
            raise ValueError(f"Duplicate Stage 1 key: {key}")
        records_by_key[key] = record
    metadata_by_idx = {}
    for row in metadata:
        idx = int(row["statement_idx"])
        if idx in metadata_by_idx:
            raise ValueError(f"Duplicate metadata statement_idx={idx}")
        metadata_by_idx[idx] = row
    missing_metadata = sorted({key[0] for key in records_by_key} - metadata_by_idx.keys())
    if missing_metadata:
        raise ValueError(f"Missing graph metadata for statement_idx={missing_metadata[:10]}")
    return records_by_key, metadata_by_idx


def score_files(args: argparse.Namespace) -> tuple[dict, dict[str, list[dict]]]:
    extractor = getattr(args, "extractor", "span")
    records = read_jsonl(args.input_jsonl)
    metadata = read_jsonl(args.graph_metadata)
    task_types = TASK_SETS[getattr(args, "task_set", "paper")]
    records_by_key, metadata_by_idx = validate_inputs(records, metadata, task_types)
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    seen_keys = set()
    for task in task_types:
        path = args.predictions_dir / f"predictions_{args.model_name}_{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        for prediction in read_jsonl(path):
            key = (int(prediction["statement_idx"]), prediction["task_type"])
            if key in seen_keys:
                raise ValueError(f"Duplicate prediction key: {key}")
            seen_keys.add(key)
            if key not in records_by_key:
                raise ValueError(f"Prediction has no matching input record: {key}")
            if key[1] != task:
                raise ValueError(f"Prediction in wrong task file: {key} in {path}")
            source = records_by_key[key]
            scored = {
                **prediction,
                "prompt": source["prompt"],
                "gold": source["answer"],
                **({"structured_gold": source["gold"]} if isinstance(source.get("gold"), dict) else {}),
                **score_record(
                    source,
                    prediction.get("raw_response", ""),
                    metadata_by_idx[key[0]],
                    extractor=extractor,
                ),
            }
            if task in NEW_TASK_TYPES:
                scored.pop("scoring_status", None)  # Earlier Gemma generations were marked pending.
            rows_by_task[task].append(scored)
    missing = sorted(set(records_by_key) - seen_keys)
    if missing:
        raise ValueError(f"Predictions are incomplete; first missing keys: {missing[:10]}")

    for task in task_types:
        rows_by_task[task].sort(key=lambda row: int(row["statement_idx"]))
    metrics = {
        "model_name": args.model_name,
        "input_record_count": len(records),
        "task_record_counts": {
            task: len(rows_by_task[task]) for task in task_types
        },
        "tasks": {
            task: aggregate_task(task, rows_by_task[task], extractor=extractor) for task in task_types
        },
        "scoring_notes": {
            "fuzzy_matching_used": False,
            "set_normalizations": {
                "raw": "outer whitespace only; list/tuple sentence punctuation is parsed as syntax",
                "basic": "raw plus lowercase, collapsed whitespace, and trailing punctuation removal",
                "annotation_stripped": "basic plus removal of trailing [A] or [B,C] labels",
            },
            "numeric_parse": "first signed integer in the response",
            "mean_absolute_error_denominator": "responses containing a parsed integer",
            "highest_node_degree_ties": "all metadata-derived maximum-degree visible nodes are accepted",
            "near_duplicates": "reported only; never counted as correct",
        },
    }
    if task_types == EXTENDED_TASK_TYPES:
        metrics['scoring_notes'].update({
            'relation_identification': 'Extracted relation exact match after lowercase/whitespace removal; per-tier and per-relation accuracy.',
            'neighbor_listing': 'Node-set precision/recall/F1 per tier and gold degree; is_correct is only a full-F1 failure-case diagnostic.',
            'shortest_path_listing': 'Ordered membership in any gold path plus hop-count accuracy and best gold-path node-set overlap F1; truncated gold counted.',
        })
    if extractor == 'span':
        metrics['scoring_notes'].update({
            'extractor': 'span',
            'numeric_parse': 'Last integer in answer_span, else last in response; highest degree uses degree cues.',
            'answer_span': 'Last answer marker first. Without one, numeric/degree/name tasks prefer the last task-specific explicit graph-level answer statement before the last-nonempty-line fallback; sets retain explicit list/tuple regions including groups and trailing commentary. Paths search all traces for the last complete source-to-target chain.',
            'strict': 'Single best extraction; raw is the headline normalization. Highest-degree strict_accuracy is degree-only; strict_joint_accuracy also requires the selected node.',
            'lenient_containment': 'Diagnostic upper bound, NOT accuracy. Numeric: standalone gold integer anywhere. Relation: gold vocabulary label anywhere. Node sets: fraction of gold members mentioned. Triples: fraction with all three components mentioned, without requiring an asserted triple. Paths: any stored gold path mentioned in order, allowing intervening text. Raw headline; set/path tiers reported separately. Not a mathematical upper bound on precision/F1 because it measures gold recall.',
            'highest_node_degree_ties': 'Strict selected name may be any metadata-derived maximum-degree visible node.',
        })
        if task_types == EXTENDED_TASK_TYPES:
            metrics['scoring_notes']['relation_identification'] = 'Alphabetic lowercase vocabulary matching; last-ending match in answer span, longest overlapping label wins. Accuracy includes and excludes relatedto.'
            metrics['scoring_notes']['shortest_path_listing'] = 'Ordered membership in gold.paths; relation labels removed, repeated nodes retained and flagged degenerate. Truncated gold may omit valid paths.'
    return metrics, rows_by_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--task-set", choices=TASK_SETS, default="paper")
    parser.add_argument("--extractor", choices=("legacy", "span"), default="span")
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--failure-cases-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, rows_by_task = score_files(args)
    metrics_out = args.metrics_out or args.predictions_dir / f"metrics_{args.model_name}.json"
    failure_out = (
        args.failure_cases_out
        or args.predictions_dir / f"failure_cases_{args.model_name}.md"
    )
    write_json(metrics_out, metrics)
    write_failure_cases(failure_out, args.model_name, rows_by_task, TASK_SETS[args.task_set])
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Wrote {metrics_out}")
    print(f"Wrote {failure_out}")


if __name__ == "__main__":
    main()
