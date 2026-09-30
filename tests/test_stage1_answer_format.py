"""Answer-format prompt, extraction, and split contracts; CPU only."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(
    0,
    str(
        ROOT
        / "experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts"
    ),
)

import score_stage1 as scorer
import stage1_common as common
from test_gemma3_evaluation import dataset


EXPECTED_SUFFIXES = {
    "node_number": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the integer. Do not write anything after that line."
    ),
    "edge_number": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the integer. Do not write anything after that line."
    ),
    "node_degree": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the integer. Do not write anything after that line."
    ),
    "highest_node_degree": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the node, a comma, and the integer degree. "
        "Do not write anything after that line."
    ),
    "node_description": (
        "You may reason before answering. Finish with exactly one final line that starts "
        "with 'Answer:' followed only by the comma-separated node names, listing every "
        "node exactly once. Do not write anything after that line."
    ),
    "triple_listing": (
        "You may reason before answering. Finish with a final answer block and no text "
        "after it. Put 'Answer:' on its own line, then one triple per line in the form "
        "(head, relation, tail)."
    ),
}


def test_suffixes_are_exact_and_prompt_records_are_not_mutated():
    assert common.CONSTRAINED_ANSWER_SUFFIXES == EXPECTED_SUFFIXES
    for task, suffix in EXPECTED_SUFFIXES.items():
        record = {"task_type": task, "prompt": "Original GraphVis prompt."}
        before = deepcopy(record)
        assert common.raw_image_prompt(record, "none") == "<image>\nOriginal GraphVis prompt."
        assert common.raw_image_prompt(record, "constrained") == (
            "<image>\nOriginal GraphVis prompt.\n\n" + suffix
        )
        assert record == before


def test_deepseek_ablation_prompt_variants_end_exactly_at_completion_prefix():
    record = {
        "task_type": "node_degree",
        "prompt": 'What is the degree of the node labeled "hub"?',
    }
    assert common.deepseek_image_prompt(record, "none", "completion") == (
        '<image>\nThe degree of the node "hub" is'
    )
    assert common.deepseek_image_prompt(record, "none", "describe") == (
        "<image>\nDescribe this image in detail."
    )
    old = common.deepseek_image_prompt(record, "none", "old-placeholder")
    assert old.endswith("Answer: <integer>")
    assert "<integer>" not in common.deepseek_image_prompt(record, "constrained", "standard")


def test_triple_scoring_keeps_valid_lines_and_penalizes_malformed_lines():
    record = {"task_type": "triple_listing", "answer": "(alpha, related to, beta), (beta, is a, gamma)"}
    row = scorer.score_record(
        record,
        "Answer:\n(alpha, related to, beta)\nnot a triple\n(beta, is a, gamma)",
        {},
    )
    assert row["parsed_predicted_triples"] == [
        ["alpha", "related to", "beta"], ["beta", "is a", "gamma"]
    ]
    assert row["malformed_triple_line_count"] == 1
    assert row["set_metrics"]["raw"]["true_positive"] == 2
    assert row["set_metrics"]["raw"]["false_positive"] == 1
    assert not row["is_correct"]


def test_unconstrained_triple_prose_is_not_a_false_positive():
    triples, malformed = scorer.scored_triples(
        "I inspected the graph carefully.\n"
        "(alpha, related to, beta) (This is a duplicate of an earlier edge)"
    )
    assert triples == [("alpha", "related to", "beta")]
    assert malformed == 0


def test_unconstrained_counts_only_comma_groups_with_bad_arity_or_empty_parts():
    triples, malformed = scorer.scored_triples(
        "Comment (without commas). Candidate (alpha, related to). "
        "Candidate (alpha, , beta). Valid (alpha, related to, beta) afterward."
    )
    assert triples == [("alpha", "related to", "beta")]
    assert malformed == 2


def test_constrained_triple_line_with_commentary_is_one_false_positive():
    triples, malformed = scorer.scored_triples(
        "Answer:\n(alpha, related to, beta) commentary (without commas)"
    )
    assert triples == [("alpha", "related to", "beta")]
    assert malformed == 1


def test_constrained_format_rejects_unspecified_extended_task():
    with pytest.raises(ValueError, match="No constrained answer format"):
        common.raw_image_prompt(
            {"task_type": "neighbor_listing", "prompt": "List neighbours."},
            "constrained",
        )


def test_expected_split_is_explicit(dataset):
    records = [dict(row, split="dev") for row in dataset.papers]
    path = dataset.root / "dev.jsonl"
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in records))
    loaded, _, _ = common.validate_stage1(
        path,
        dataset.root / "metadata.jsonl",
        len(records),
        expected_split="dev",
    )
    assert len(loaded) == len(records)
    with pytest.raises(ValueError, match="expected 'test'"):
        common.validate_stage1(
            path,
            dataset.root / "metadata.jsonl",
            len(records),
            expected_split="test",
        )


def highest_fixture():
    record = {
        "statement_idx": 0,
        "task_type": "highest_node_degree",
        "answer": 'One node with the highest degree is "hub" with a degree of 2.',
    }
    metadata = {
        "statement_idx": 0,
        "visible_nodes": [
            {"cid": 1, "name": "hub"},
            {"cid": 2, "name": "leaf"},
            {"cid": 3, "name": "other"},
        ],
        "edges": [
            {"source_cid": 1, "target_cid": 2},
            {"source_cid": 1, "target_cid": 3},
        ],
    }
    return record, metadata


def test_compact_highest_degree_marker_and_annotation_comma():
    record, metadata = highest_fixture()
    row = scorer.score_record(record, "Reasoning mentions leaf 1.\nAnswer: hub, 2", metadata)
    assert row["parsed_name"] == "hub"
    assert row["parsed_degree"] == 2
    assert row["is_correct"]
    assert scorer.compact_highest_answer("node [A,B], 4") == ("node [A,B]", 4)


def test_unconstrained_explicit_statements_precede_last_line_fallback():
    assert scorer.span_integer("There are 7 nodes.\nStep 2 checked labels.", "node_number") == 7
    assert scorer.span_integer(
        "That's 11 nodes in group one.\nThe total number of nodes in the graph is 22.",
        "node_number",
    ) == 22
    assert scorer.span_integer(
        "Group counts: 14, 4, 8, 7, 3, 2, 1.\nTotal = 14 + 4 + 8 + 7 + 3 + 2 + 1 = 39.",
        "node_number",
    ) == 39
    # Scoped subgroup counts are reasoning, not an explicit graph-level answer.
    assert scorer.explicit_numeric_value("That's 7 edges from water.", "edge_number") is None
    assert scorer.span_integer("The node has 4 connections.\nI checked 2 routes.", "node_degree") == 4
    assert scorer.span_degree("The hub has a degree of 5.\nI checked 2 alternatives.") == 5
    record, metadata = highest_fixture()
    row = scorer.score_record(
        record,
        "The node with the highest degree is **hub**.\nIt has a degree of 2.\nChecked 3 candidates.",
        metadata,
    )
    assert row["parsed_name"] == "hub" and row["parsed_degree"] == 2


@pytest.mark.parametrize(
    "task,response",
    [
        ("node_number", "Counted 8 first.\nAnswer: 7"),
        ("edge_number", "Counted 8 first.\nAnswer: 7"),
        ("node_degree", "Counted 8 first.\nAnswer: 7"),
        ("highest_node_degree", "Maybe leaf, 1.\nAnswer: hub, 2"),
        ("node_description", "Candidates were wrong.\nAnswer: hub, leaf"),
        ("triple_listing", "Candidate: (x, wrong, y)\nAnswer:\n(hub, related to, leaf)"),
    ],
)
def test_answer_marker_diagnostics_for_every_paper_task(task, response):
    details = scorer.answer_format_diagnostics(task, response)
    assert details["final_answer_marker_found"]
    assert details["answer_marker_extraction_success"]
    assert details["final_format_compliant"]


def test_empty_prediction_precision_is_zero_and_ocr_is_diagnostic_only():
    metrics = scorer.set_counts(set(), set())
    assert metrics["exact_set_equality"] is True
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.0
    details = scorer.answer_format_diagnostics(
        "node_description", "<table><tr><td>hub</td></tr></table>", "none"
    )
    assert details["ocr_style_dump"]
    assert details["final_format_compliant"] is None
