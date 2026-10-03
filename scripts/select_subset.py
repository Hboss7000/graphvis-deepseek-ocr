#!/usr/bin/env python3
"""Select the deterministic 50-question OBQA test subset for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


DEFAULT_SOURCE = Path("data_preprocessed_release/obqa/statement/test.statement.jsonl")
DEFAULT_OUTPUT = Path("subsets/obqa_test_subset50_seed13.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_subset(statement_file: Path, seed: int, n: int) -> dict:
    with statement_file.open("r", encoding="utf-8") as handle:
        source_n = sum(1 for line in handle if line.strip())
    if not 0 < n <= source_n:
        raise ValueError(f"Require 0 < n <= {source_n}; found {n}")
    indices = sorted(random.Random(seed).sample(range(source_n), n))
    return {
        "split": "test",
        "seed": seed,
        "n": n,
        "source_n": source_n,
        "indices": indices,
        "sha256_of_statement_file": sha256_file(statement_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement-file", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    subset = build_subset(args.statement_file, args.seed, args.n)
    payload = json.dumps(subset, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite different subset: {args.output}")
        print(f"Subset already exists and is identical: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(f"q00060 in subset: {60 in subset['indices']}")


if __name__ == "__main__":
    main()
