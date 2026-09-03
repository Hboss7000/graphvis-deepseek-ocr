#!/usr/bin/env python3
"""Model-agnostic prompt construction and answer parsing for zero-shot OBQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator


OPTIONS = ["A", "B", "C", "D", "E"]
CONDITIONS = ("image", "text", "text_noref", "kg_text")
IMAGE_REFERENCE_SENTENCE = (
    "The image represents a knowledge graph relevant to the question, "
    "which may or may not be useful. "
)
KG_TEXT_REFERENCE_SENTENCE = (
    "The following knowledge graph is relevant to the question, "
    "which may or may not be useful. "
)
ANSWER_LETTER_ANNOTATION = re.compile(r"\[[A-D](,[A-D])*\]")

# Copied verbatim from scripts/generate_graphvis_datasets.py. That renderer uses
# RELATION_TEXT.get(relation, relation), which format_kg_block mirrors below.
RELATION_TEXT = {
    "antonym": "antonym",
    "atlocation": "at location",
    "capableof": "capable of",
    "causes": "causes",
    "createdby": "created by",
    "isa": "is a",
    "desires": "desires",
    "hassubevent": "has subevent",
    "partof": "part of",
    "hascontext": "has context",
    "hasproperty": "has property",
    "madeof": "made of",
    "notcapableof": "not capable of",
    "notdesires": "not desires",
    "receivesaction": "receives action",
    "relatedto": "related to",
    "usedfor": "used for",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_graph_metadata(path: Path, records: list[dict]) -> dict[int, dict]:
    """Load metadata and require a unique match for every evaluation record."""
    metadata_by_idx: dict[int, dict] = {}
    for line_number, meta in enumerate(read_jsonl(path), start=1):
        if "statement_idx" not in meta:
            raise ValueError(
                f"Graph metadata row {line_number} lacks field: statement_idx"
            )
        statement_idx = int(meta["statement_idx"])
        if statement_idx in metadata_by_idx:
            raise ValueError(
                f"Graph metadata contains duplicate statement_idx={statement_idx}"
            )
        metadata_by_idx[statement_idx] = meta

    missing = sorted(
        int(record["statement_idx"])
        for record in records
        if int(record["statement_idx"]) not in metadata_by_idx
    )
    if missing:
        raise ValueError(
            "Graph metadata lacks input statement_idx values: "
            + ", ".join(map(str, missing))
        )
    return metadata_by_idx


def format_kg_block(meta: dict) -> str:
    """Linearise exactly the connected edges recorded in graph metadata."""
    names_by_cid: dict[int, str] = {}
    connected_by_cid: dict[int, bool] = {}
    for node in meta["visible_nodes"]:
        cid = int(node["cid"])
        names_by_cid[cid] = str(node["name"]).replace("_", " ")
        connected_by_cid[cid] = bool(node["connected"])

    triples: list[tuple[str, str, str]] = []
    for edge in meta["edges"]:
        source_cid = int(edge["source_cid"])
        target_cid = int(edge["target_cid"])
        try:
            source_name = names_by_cid[source_cid]
            target_name = names_by_cid[target_cid]
            endpoints_connected = (
                connected_by_cid[source_cid] and connected_by_cid[target_cid]
            )
        except KeyError as exc:
            raise ValueError(
                f"Metadata edge references absent visible-node cid={exc.args[0]}"
            ) from exc
        if not endpoints_connected:
            continue
        relation = str(edge["relation"])
        relation_text = RELATION_TEXT.get(relation, relation)
        triples.append((source_name, relation_text, target_name))

    triples.sort()
    if not triples:
        raise ValueError(
            "Cannot construct kg_text prompt from an empty triple block for "
            f"statement_idx={meta.get('statement_idx')}"
        )
    return "\n".join(
        f"({source}, {relation}, {target})"
        for source, relation, target in triples
    )


def assert_no_answer_letter_annotation(prompt: str) -> None:
    match = ANSWER_LETTER_ANNOTATION.search(prompt)
    if match:
        raise RuntimeError(
            "KG-text prompt contains a forbidden answer-letter annotation: "
            f"{match.group(0)}"
        )


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(row["statement_idx"])
        for row in read_jsonl(path)
        if "statement_idx" in row
    }


def parse_answer(pred_text: str, n_choices: int) -> tuple[str, str]:
    """Return the parsed option and the tier that accepted the response."""
    opts = OPTIONS[:n_choices]
    if pred_text in opts:
        return pred_text, "exact"
    if len(pred_text) >= 3 and pred_text[0] in opts and pred_text[1:3] == ". ":
        return pred_text[0], "option_prefix"
    res = re.compile(r"The answer is ([A-Z]).").findall(pred_text)
    if len(res) == 1:
        return res[0], "the_answer_is"
    standalone = re.findall(r"\b([A-D])\b", pred_text)
    if standalone:
        return standalone[-1], "standalone_fallback"
    return "FAILED", "FAILED"


def validate_records(records: list[dict], expected_count: int) -> None:
    required = {"statement_idx", "image", "prompt", "answer"}
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} input rows, found {len(records)}")
    indices = []
    for line_number, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Input row {line_number} lacks fields: {sorted(missing)}")
        indices.append(int(record["statement_idx"]))
    if len(set(indices)) != expected_count:
        raise ValueError("Input does not contain the expected number of unique statement_idx values")


def render_prompt(
    prompt_body: str,
    condition: str,
    kg_block: str | None = None,
) -> str:
    """Build the model-independent prompt body for one condition."""
    if condition == "image":
        return f"<image>\n{prompt_body}"
    if condition == "text":
        return prompt_body
    if condition == "text_noref":
        if not prompt_body.startswith(IMAGE_REFERENCE_SENTENCE):
            raise RuntimeError(
                "Expected prompt body to start with IMAGE_REFERENCE_SENTENCE; "
                "input JSONL does not match the expected Stage 2 template."
            )
        return prompt_body[len(IMAGE_REFERENCE_SENTENCE) :]
    if condition == "kg_text":
        if kg_block is None:
            raise RuntimeError("kg_text requires a formatted knowledge-graph block")
        body = render_prompt(prompt_body, "text_noref")
        prompt = f"{KG_TEXT_REFERENCE_SENTENCE}\n{kg_block}\n\n{body}"
        assert_no_answer_letter_annotation(prompt)
        return prompt
    raise ValueError(f"Unknown condition: {condition}")


def iter_prompt_bodies(
    records: list[dict],
    metadata_by_idx: dict[int, dict],
    conditions: tuple[str, ...] = CONDITIONS,
) -> Iterator[tuple[int, str, str]]:
    """Yield prompt bodies in stable statement-index and condition order."""
    for record in sorted(records, key=lambda row: int(row["statement_idx"])):
        statement_idx = int(record["statement_idx"])
        kg_block = format_kg_block(metadata_by_idx[statement_idx])
        for condition in conditions:
            body = render_prompt(
                record["prompt"],
                condition,
                kg_block=kg_block if condition == "kg_text" else None,
            )
            yield statement_idx, condition, body


def _encoded_prompt_line(prompt_body: str) -> bytes:
    return (json.dumps(prompt_body, ensure_ascii=False) + "\n").encode("utf-8")


def prompt_bodies_sha256(
    records: list[dict],
    metadata_by_idx: dict[int, dict],
    condition: str,
) -> str:
    """Hash one condition's JSON-line-encoded bodies in statement-index order."""
    digest = hashlib.sha256()
    for _, _, body in iter_prompt_bodies(records, metadata_by_idx, (condition,)):
        digest.update(_encoded_prompt_line(body))
    return digest.hexdigest()


def dump_prompt_bodies(
    records: list[dict],
    metadata_by_idx: dict[int, dict],
    output_path: Path,
) -> str:
    """Write all four prompt bodies as one JSON string per physical line."""
    digest = hashlib.sha256()
    with output_path.open("wb") as handle:
        for _, _, body in iter_prompt_bodies(records, metadata_by_idx):
            encoded = _encoded_prompt_line(body)
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--dump-prompt-bodies", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input_jsonl)
    validate_records(records, args.expected_count)
    metadata_by_idx = index_graph_metadata(args.graph_metadata, records)
    digest = dump_prompt_bodies(records, metadata_by_idx, args.dump_prompt_bodies)
    print(f"Wrote {len(records) * len(CONDITIONS)} prompt bodies")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
