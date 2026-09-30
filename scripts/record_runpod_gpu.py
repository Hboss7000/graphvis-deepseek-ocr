#!/usr/bin/env python3
"""Record the visible Runpod GPU in a Stage 1 run_config.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch


def details() -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    inventory = subprocess.run(
        ["nvidia-smi", "-L"], check=True, text=True, capture_output=True
    ).stdout.strip()
    full_pro_6000 = (
        "RTX PRO 6000" in props.name
        and "MIG" not in props.name.upper()
        and props.total_memory >= 90 * 1024**3
        and "MIG" not in inventory.upper()
    )
    return {
        "torch_device_name": props.name,
        "torch_total_memory_bytes": props.total_memory,
        "torch_architecture": f"sm_{major}{minor}",
        "nvidia_smi_inventory": inventory,
        "comparison_eligible": full_pro_6000,
        "comparison_eligibility_rule": (
            "visible CUDA device is a non-MIG RTX PRO 6000 with at least 90 GiB"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path, nargs="+")
    args = parser.parse_args()
    gpu = details()
    print(json.dumps(gpu, sort_keys=True))
    for path in args.run_config:
        config = json.loads(path.read_text(encoding="utf-8"))
        config["runpod_gpu"] = gpu
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
