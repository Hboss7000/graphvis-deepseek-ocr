#!/usr/bin/env python3
"""Print readable task x model tables from the phase-1 dry-run metrics.

Usage (repo root):  python3 phase1_tables.py [outputs_dir]
Reads outputs/phase1_dryrun/runs/*stage1*/metrics_*.json and
outputs/phase1_reading_probe/runs/reading_probe_report.json.
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
MODELS = ["llava", "qwen", "gemma3", "deepseek"]
TASKS = ["node_description", "node_number", "edge_number", "triple_listing",
         "highest_node_degree", "node_degree", "relation_identification",
         "neighbor_listing", "shortest_path_listing"]


def flatten(obj, prefix=""):
    """Yield (dotted_path, value) for every leaf of a nested dict."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            name = value.get("arm") or value.get("name") if isinstance(value, dict) else None
            yield from flatten(value, f"{prefix}.{name or i}" if prefix else str(name or i))
    else:
        yield prefix, obj


def pick(task_metrics, endings, prefer="raw"):
    """First numeric leaf whose name matches `endings` (exact names, or callables); prefer `prefer` paths."""
    leaves = [(p, v) for p, v in flatten(task_metrics)
              if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for ending in endings:
        match = ending if callable(ending) else (lambda name, e=ending: name == e)
        hits = [(p, v) for p, v in leaves if match(p.split(".")[-1])]
        if hits:
            hits.sort(key=lambda pv: (prefer not in pv[0], len(pv[0])))
            return hits[0][1]
    return None


def fmt(value):
    return "  —  " if value is None else f"{100 * value:5.1f}"


metrics = {}
for path in sorted((root / "phase1_dryrun" / "runs").glob("*stage1*/metrics_*.json")):
    data = json.loads(path.read_text())
    metrics[data.get("model_name", path.stem)] = data.get("tasks", {})
models = [m for m in MODELS if m in metrics] + [m for m in metrics if m not in MODELS]

TABLES = [
    ("Strict accuracy (%)  — exact answer, as in paper Table 4", ["strict_accuracy", "exact_accuracy",
      lambda n: "exact" in n and ("accuracy" in n or "equality" in n) and "count" not in n]),
    ("Gold recall (%)  — share of gold items named (set tasks; raw normalization)", ["macro_recall", "recall", "mean_recall"]),
    ("Token-ceiling hits (%)  — output ran to the generation limit", ["hit_token_ceiling_fraction"]),
]
for title, endings in TABLES:
    print(f"\n{title}  [8 graphs per cell: one item = 12.5 points]")
    print(f"{'task':<25}" + "".join(f"{m:>10}" for m in models))
    print("-" * (25 + 10 * len(models)))
    for task in TASKS:
        cells = [fmt(pick(metrics[m].get(task, {}), endings)) for m in models]
        print(f"{task:<25}" + "".join(f"{c:>10}" for c in cells))

print("\nHighest-degree node-only name accuracy (%)  — raw normalization, ties accepted")
print(f"{'task':<25}" + "".join(f"{m:>10}" for m in models))
print("-" * (25 + 10 * len(models)))
cells = [fmt(pick(metrics[m].get("highest_node_degree", {}), ["name_accuracy"], prefer="raw"))
         for m in models]
print(f"{'highest_node_degree':<25}" + "".join(f"{c:>10}" for c in cells))

probe = root / "phase1_reading_probe" / "runs" / "reading_probe_report.json"
if probe.exists():
    print("\nReading probe — node_description gold-node recall (20 graphs)")
    for path, value in flatten(json.loads(probe.read_text())):
        if "recall" in path.split(".")[-1] and isinstance(value, (int, float)):
            print(f"  {path:<60} {100 * value:5.1f} %")
    if not any("recall" in p.split(".")[-1] for p, _ in flatten(json.loads(probe.read_text()))):
        print("  (no recall field found; top-level keys:", list(json.loads(probe.read_text()))[:10], ")")
