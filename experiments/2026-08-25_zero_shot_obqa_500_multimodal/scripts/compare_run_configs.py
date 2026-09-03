#!/usr/bin/env python3
"""Verify that two runs used byte-identical shared prompt bodies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()

    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    first_condition = first.get("condition")
    second_condition = second.get("condition")
    if first_condition != second_condition:
        raise SystemExit(
            "Cannot compare prompt hashes for different conditions: "
            f"{first_condition!r} != {second_condition!r}"
        )

    key = "prompt_bodies_sha256"
    first_hash = first.get(key)
    second_hash = second.get(key)
    if not first_hash or not second_hash:
        raise SystemExit(f"Both run configs must contain {key!r}")
    if first_hash != second_hash:
        raise SystemExit(
            f"Prompt-body mismatch for condition={first_condition}: "
            f"{first_hash} != {second_hash}"
        )
    print(
        f"Prompt bodies match for condition={first_condition}: {first_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
