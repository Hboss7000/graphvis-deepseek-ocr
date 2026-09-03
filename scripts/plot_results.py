#!/usr/bin/env python3
"""Create thesis-quality figures and summaries for the zero-shot OBQA run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


OPTIONS = ("A", "B", "C", "D")
EXPECTED_COUNT = 500
CONDITIONS = (
    ("image", "Image + text"),
    ("text", "Text only (image sentence kept)"),
    ("text_noref", "Text only (image sentence removed)"),
)
CONDITION_LABELS = dict(CONDITIONS)
CONDITION_COLORS = {
    "image": "#0072B2",
    "text": "#E69F00",
    "text_noref": "#009E73",
}
CONDITION_HATCHES = {
    "image": "//",
    "text": "\\\\",
    "text_noref": "xx",
}


class DataError(ValueError):
    """Raised when an input file cannot support a paired 500-question analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing predictions_image/text/text_noref.jsonl",
    )
    parser.add_argument(
        "--graph-metadata",
        type=Path,
        help="Optional graph_metadata_0_500.jsonl; missing files are silently skipped",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Figure directory (default: <results-dir>/figures)",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise DataError(f"{path}: line {line_number} is not a JSON object")
                rows.append(row)
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc}") from exc
    return rows


def load_predictions(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise DataError(f"Missing predictions file: {path}")

    rows = load_jsonl(path)
    indices = []
    for line_number, row in enumerate(rows, start=1):
        statement_idx = row.get("statement_idx")
        if type(statement_idx) is not int:
            raise DataError(f"{path}: line {line_number} has a non-integer statement_idx")
        indices.append(statement_idx)
    counts = Counter(indices)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    unique_count = len(counts)
    if len(rows) != EXPECTED_COUNT or unique_count != EXPECTED_COUNT or duplicates:
        duplicate_note = f"; duplicate statement_idx values: {duplicates[:10]}" if duplicates else ""
        raise DataError(
            f"{path}: expected {EXPECTED_COUNT} rows with {EXPECTED_COUNT} unique "
            f"statement_idx values, found {len(rows)} rows and {unique_count} unique values"
            f"{duplicate_note}"
        )

    records = {}
    for line_number, row in enumerate(rows, start=1):
        statement_idx = row.get("statement_idx")
        gold = row.get("gold_option")
        predicted = row.get("predicted_option")
        if gold not in OPTIONS:
            raise DataError(f"{path}: statement_idx {statement_idx} has invalid gold_option={gold!r}")
        if predicted is not None and predicted not in OPTIONS:
            raise DataError(
                f"{path}: statement_idx {statement_idx} has invalid predicted_option={predicted!r}"
            )
        records[statement_idx] = row
    return records


def load_all_predictions(results_dir: Path) -> dict[str, dict[int, dict]]:
    predictions = {
        key: load_predictions(results_dir / f"predictions_{key}.jsonl")
        for key, _label in CONDITIONS
    }
    reference_key = CONDITIONS[0][0]
    reference_indices = set(predictions[reference_key])
    reference_gold = {
        index: row["gold_option"] for index, row in predictions[reference_key].items()
    }

    for key, label in CONDITIONS[1:]:
        indices = set(predictions[key])
        if indices != reference_indices:
            missing = sorted(reference_indices - indices)
            extra = sorted(indices - reference_indices)
            raise DataError(
                f"{label}: statement_idx set differs from {CONDITION_LABELS[reference_key]}; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        mismatched_gold = [
            index
            for index in sorted(reference_indices)
            if predictions[key][index]["gold_option"] != reference_gold[index]
        ]
        if mismatched_gold:
            raise DataError(
                f"{label}: gold_option differs from {CONDITION_LABELS[reference_key]} "
                f"for statement_idx values {mismatched_gold[:10]}"
            )
    return predictions


def compute_condition_stats(
    records: dict[int, dict], gold_counts: Counter, total: int
) -> dict[str, float | int | Counter]:
    prediction_counts = Counter(
        row["predicted_option"]
        for row in records.values()
        if row["predicted_option"] in OPTIONS
    )
    unparsed = sum(row["predicted_option"] is None for row in records.values())
    correct = sum(
        row["predicted_option"] == row["gold_option"] for row in records.values()
    )
    expected_correct = sum(
        prediction_counts[letter] * (gold_counts[letter] / total) for letter in OPTIONS
    )
    expected_accuracy = expected_correct / total
    sd = math.sqrt(total * expected_accuracy * (1.0 - expected_accuracy))
    z_score = (correct - expected_correct) / sd if sd else 0.0
    bc_count = prediction_counts["B"] + prediction_counts["C"]
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total,
        "prediction_counts": prediction_counts,
        "unparsed": unparsed,
        "bc_count": bc_count,
        "bc_share": bc_count / total,
        "expected_correct": expected_correct,
        "expected_accuracy": expected_accuracy,
        "z": z_score,
    }


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(first_only, discordant, p=0.5).pvalue)
    except ImportError:
        tail = min(first_only, second_only)
        probability = 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1))
        probability /= 2**discordant
        return min(1.0, probability)


def paired_result(
    first: dict[int, dict], second: dict[int, dict], first_key: str, second_key: str
) -> dict[str, int | float | str]:
    both_correct = 0
    first_only = 0
    second_only = 0
    both_wrong = 0
    for statement_idx in sorted(first):
        first_correct = first[statement_idx]["predicted_option"] == first[statement_idx]["gold_option"]
        second_correct = (
            second[statement_idx]["predicted_option"] == second[statement_idx]["gold_option"]
        )
        if first_correct and second_correct:
            both_correct += 1
        elif first_correct:
            first_only += 1
        elif second_correct:
            second_only += 1
        else:
            both_wrong += 1
    return {
        "first_key": first_key,
        "second_key": second_key,
        "both_correct": both_correct,
        "first_only": first_only,
        "second_only": second_only,
        "both_wrong": both_wrong,
        "p_value": exact_mcnemar_p(first_only, second_only),
    }


def compute_pairwise(predictions: dict[str, dict[int, dict]]) -> list[dict]:
    pairs = (("image", "text"), ("image", "text_noref"), ("text", "text_noref"))
    return [
        paired_result(predictions[first], predictions[second], first, second)
        for first, second in pairs
    ]


def confusion_matrix(records: dict[int, dict]) -> list[list[float]]:
    counts = [[0 for _column in OPTIONS] for _row in OPTIONS]
    for row in records.values():
        predicted = row["predicted_option"]
        if predicted in OPTIONS:
            gold_index = OPTIONS.index(row["gold_option"])
            predicted_index = OPTIONS.index(predicted)
            counts[gold_index][predicted_index] += 1
    normalised = []
    for row in counts:
        valid_total = sum(row)
        normalised.append(
            [value / valid_total if valid_total else 0.0 for value in row]
        )
    return normalised


def load_graph_sizes(path: Path | None, expected_indices: set[int]) -> dict[int, dict] | None:
    if path is None or not path.is_file():
        return None
    rows = load_jsonl(path)
    metadata = {}
    for line_number, row in enumerate(rows, start=1):
        index = row.get("statement_idx")
        if type(index) is not int:
            raise DataError(f"{path}: line {line_number} has a non-integer statement_idx")
        if index in metadata:
            raise DataError(f"{path}: duplicate statement_idx {index}")
        nodes = row.get("visible_nodes")
        edges = row.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise DataError(
                f"{path}: statement_idx {index} must contain visible_nodes and edges lists"
            )
        metadata[index] = {"nodes": len(nodes), "edges": len(edges)}
    if set(metadata) != expected_indices:
        missing = sorted(expected_indices - set(metadata))
        extra = sorted(set(metadata) - expected_indices)
        raise DataError(
            f"{path}: metadata statement_idx set does not match predictions; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return metadata


def make_graph_bins(graph_sizes: dict[int, dict], target_bins: int = 4) -> list[dict]:
    values = sorted(item["nodes"] for item in graph_sizes.values())
    unique_values = sorted(set(values))
    bin_count = min(target_bins, len(unique_values))
    if bin_count == 0:
        return []

    # Start with observation quartiles. If tied quantiles collapse bins, divide
    # the unique node-count values instead so the result stays interpretable.
    upper_bounds = sorted(
        {
            values[min(len(values) - 1, math.ceil(i * len(values) / bin_count) - 1)]
            for i in range(1, bin_count + 1)
        }
    )
    if len(upper_bounds) < bin_count:
        upper_bounds = sorted(
            {
                unique_values[
                    min(
                        len(unique_values) - 1,
                        math.ceil(i * len(unique_values) / bin_count) - 1,
                    )
                ]
                for i in range(1, bin_count + 1)
            }
        )
    if upper_bounds[-1] != unique_values[-1]:
        upper_bounds.append(unique_values[-1])

    bins = []
    lower = unique_values[0]
    for upper in upper_bounds:
        indices = sorted(
            index
            for index, size in graph_sizes.items()
            if lower <= size["nodes"] <= upper
        )
        if indices:
            edge_counts = [graph_sizes[index]["edges"] for index in indices]
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "indices": indices,
                    "edge_min": min(edge_counts),
                    "edge_max": max(edge_counts),
                    "edge_mean": sum(edge_counts) / len(edge_counts),
                }
            )
        lower = upper + 1
    return bins


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    header = " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def format_p_value(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def build_summary(
    stats: dict[str, dict], pairwise: list[dict], graph_bins: list[dict] | None
) -> str:
    condition_headers = [
        "Condition",
        "n",
        "Correct",
        "Accuracy",
        "A",
        "B",
        "C",
        "D",
        "Null",
        "B+C share",
        "Expected correct",
        "Expected acc.",
        "z",
    ]
    condition_rows = []
    for key, label in CONDITIONS:
        item = stats[key]
        counts = item["prediction_counts"]
        condition_rows.append(
            [
                label,
                str(item["n"]),
                str(item["correct"]),
                f"{100 * item['accuracy']:.1f}%",
                str(counts["A"]),
                str(counts["B"]),
                str(counts["C"]),
                str(counts["D"]),
                str(item["unparsed"]),
                f"{100 * item['bc_share']:.1f}%",
                f"{item['expected_correct']:.2f}",
                f"{100 * item['expected_accuracy']:.2f}%",
                f"{item['z']:.2f}",
            ]
        )

    pair_headers = [
        "Pair (first vs second)",
        "Both correct",
        "First only",
        "Second only",
        "Both wrong",
        "McNemar p",
    ]
    pair_rows = []
    for item in pairwise:
        pair_rows.append(
            [
                f"{CONDITION_LABELS[item['first_key']]} vs {CONDITION_LABELS[item['second_key']]}",
                str(item["both_correct"]),
                str(item["first_only"]),
                str(item["second_only"]),
                str(item["both_wrong"]),
                format_p_value(item["p_value"]),
            ]
        )

    sections = [
        "ZERO-SHOT OBQA RESULTS",
        "",
        format_table(condition_headers, condition_rows),
        "",
        "PAIRED EXACT McNEMAR TESTS",
        "",
        format_table(pair_headers, pair_rows),
    ]
    if graph_bins:
        bin_descriptions = []
        for item in graph_bins:
            node_range = (
                str(item["lower"])
                if item["lower"] == item["upper"]
                else f"{item['lower']}–{item['upper']}"
            )
            bin_descriptions.append(
                f"{node_range} nodes: n={len(item['indices'])}, "
                f"edges={item['edge_min']}–{item['edge_max']} "
                f"(mean {item['edge_mean']:.1f})"
            )
        sections.extend(["", "GRAPH-SIZE BINS", "", *bin_descriptions])
    return "\n".join(sections) + "\n"


def write_summary_csv(path: Path, stats: dict[str, dict]) -> None:
    fieldnames = [
        "condition",
        "display_label",
        "n",
        "correct",
        "accuracy_percent",
        "predicted_a",
        "predicted_b",
        "predicted_c",
        "predicted_d",
        "unparsed",
        "bc_count",
        "bc_share_percent",
        "expected_correct",
        "expected_accuracy_percent",
        "z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, label in CONDITIONS:
            item = stats[key]
            counts = item["prediction_counts"]
            writer.writerow(
                {
                    "condition": key,
                    "display_label": label,
                    "n": item["n"],
                    "correct": item["correct"],
                    "accuracy_percent": f"{100 * item['accuracy']:.6f}",
                    "predicted_a": counts["A"],
                    "predicted_b": counts["B"],
                    "predicted_c": counts["C"],
                    "predicted_d": counts["D"],
                    "unparsed": item["unparsed"],
                    "bc_count": item["bc_count"],
                    "bc_share_percent": f"{100 * item['bc_share']:.6f}",
                    "expected_correct": f"{item['expected_correct']:.6f}",
                    "expected_accuracy_percent": f"{100 * item['expected_accuracy']:.6f}",
                    "z": f"{item['z']:.6f}",
                }
            )


def save_figure(fig, out_dir: Path, name: str, plt) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def accuracy_axis_limit(stats: dict[str, dict]) -> float:
    highest = max(
        max(100 * item["accuracy"], 100 * item["expected_accuracy"])
        for item in stats.values()
    )
    return min(100.0, max(50.0, math.ceil((highest + 8.0) / 10.0) * 10.0))


def plot_accuracy(stats: dict[str, dict], out_dir: Path, plt) -> None:
    keys = [key for key, _label in CONDITIONS]
    labels = [label for _key, label in CONDITIONS]
    x_positions = list(range(len(keys)))
    accuracies = [100 * stats[key]["accuracy"] for key in keys]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(
        x_positions,
        accuracies,
        width=0.62,
        color=[CONDITION_COLORS[key] for key in keys],
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.axhline(
        25.0,
        color="#555555",
        linestyle="--",
        linewidth=1.6,
        label="Random guessing (25.0%)",
        zorder=2,
    )
    for index, key in enumerate(keys):
        expected = 100 * stats[key]["expected_accuracy"]
        ax.plot(
            [index - 0.22, index + 0.22],
            [expected, expected],
            color="#CC79A7",
            linewidth=3.2,
            solid_capstyle="butt",
            label="Expected if no signal (per condition)" if index == 0 else None,
            zorder=4,
        )
    for bar, key in zip(bars, keys):
        item = stats[key]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{100 * item['accuracy']:.1f}%\n{item['correct']}/{item['n']}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x_positions, labels, rotation=10, ha="right")
    ax.set_ylim(0, accuracy_axis_limit(stats))
    ax.set_xlabel("Experimental condition")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot DeepSeek-OCR-2 accuracy on OpenBookQA")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, out_dir, "accuracy", plt)


def plot_answer_distribution(
    predictions: dict[str, dict[int, dict]], gold_counts: Counter, out_dir: Path, plt
) -> None:
    x_positions = list(range(len(OPTIONS)))
    width = 0.19
    series = [
        (
            key,
            label,
            Counter(
                row["predicted_option"]
                for row in predictions[key].values()
                if row["predicted_option"] in OPTIONS
            ),
            CONDITION_COLORS[key],
            CONDITION_HATCHES[key],
        )
        for key, label in CONDITIONS
    ]
    series.append(("gold", "Gold answers", gold_counts, "#777777", ".."))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for series_index, (_key, label, counts, color, hatch) in enumerate(series):
        offset = (series_index - (len(series) - 1) / 2) * width
        ax.bar(
            [position + offset for position in x_positions],
            [counts[letter] for letter in OPTIONS],
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            zorder=3,
        )
    ax.set_xticks(x_positions, OPTIONS)
    ax.set_xlabel("Answer option")
    ax.set_ylabel("Number of questions")
    ax.set_title("Predicted and gold answer distributions")
    highest_count = max(max(counts.values(), default=0) for _key, _label, counts, _color, _hatch in series)
    ax.set_ylim(0, max(1, highest_count * 1.28))
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_dir, "answer_distribution", plt)


def plot_confusion(
    predictions: dict[str, dict[int, dict]], stats: dict[str, dict], out_dir: Path, plt
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    images = []
    any_unparsed = any(stats[key]["unparsed"] for key, _label in CONDITIONS)
    for ax, (key, label) in zip(axes, CONDITIONS):
        matrix = confusion_matrix(predictions[key])
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        images.append(image)
        for row_index in range(len(OPTIONS)):
            for column_index in range(len(OPTIONS)):
                value = matrix[row_index][column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value >= 0.5 else "black",
                )
        ax.set_xticks(range(len(OPTIONS)), OPTIONS)
        ax.set_yticks(range(len(OPTIONS)), OPTIONS)
        ax.set_xlabel("Predicted option")
        ax.set_ylabel("Gold option")
        ax.set_title(f"{label}\nAccuracy: {100 * stats[key]['accuracy']:.1f}%")
    fig.suptitle("Row-normalised confusion matrices", fontsize=14)
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.83, pad=0.02)
    colorbar.set_label("Fraction within gold-answer row")
    if any_unparsed:
        fig.text(
            0.5,
            -0.045,
            "Unparsed responses are counted as incorrect and excluded from A–D matrix columns.",
            ha="center",
            fontsize=9,
        )
    save_figure(fig, out_dir, "confusion", plt)


def plot_paired(pairwise: list[dict], out_dir: Path, plt) -> None:
    categories = (
        ("both_correct", "Both correct", "#009E73", ""),
        ("first_only", "Only first condition correct", "#0072B2", "//"),
        ("second_only", "Only second condition correct", "#E69F00", "\\\\"),
        ("both_wrong", "Both wrong", "#999999", "xx"),
    )
    y_positions = list(range(len(pairwise)))
    y_labels = [
        f"{CONDITION_LABELS[item['first_key']]}\nvs {CONDITION_LABELS[item['second_key']]}"
        for item in pairwise
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    left = [0] * len(pairwise)
    for category_key, category_label, color, hatch in categories:
        values = [int(item[category_key]) for item in pairwise]
        bars = ax.barh(
            y_positions,
            values,
            left=left,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            label=category_label,
        )
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
        left = [current + value for current, value in zip(left, values)]

    total = max(left) if left else EXPECTED_COUNT
    for y_position, item in zip(y_positions, pairwise):
        formatted_p = format_p_value(item["p_value"])
        p_expression = f"p{formatted_p}" if formatted_p.startswith("<") else f"p={formatted_p}"
        ax.text(
            total + 5,
            y_position,
            f"McNemar {p_expression}\n"
            f"first-only={item['first_only']}, second-only={item['second_only']}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.set_xlim(0, total + 120)
    ax.set_yticks(y_positions, y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of paired questions")
    ax.set_ylabel("Condition pair (first vs second)")
    ax.set_title("Question-level correctness agreement")
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False)
    save_figure(fig, out_dir, "paired_comparison", plt)


def plot_accuracy_vs_graph_size(
    predictions: dict[str, dict[int, dict]], graph_bins: list[dict], out_dir: Path, plt
) -> None:
    x_positions = list(range(len(graph_bins)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for condition_index, (key, label) in enumerate(CONDITIONS):
        offset = (condition_index - 1) * width
        accuracies = []
        for graph_bin in graph_bins:
            indices = graph_bin["indices"]
            correct = sum(
                predictions[key][index]["predicted_option"]
                == predictions[key][index]["gold_option"]
                for index in indices
            )
            accuracies.append(100 * correct / len(indices))
        ax.bar(
            [position + offset for position in x_positions],
            accuracies,
            width=width,
            label=label,
            color=CONDITION_COLORS[key],
            edgecolor="black",
            linewidth=0.7,
            hatch=CONDITION_HATCHES[key],
            zorder=3,
        )

    tick_labels = []
    for graph_bin in graph_bins:
        node_range = (
            str(graph_bin["lower"])
            if graph_bin["lower"] == graph_bin["upper"]
            else f"{graph_bin['lower']}–{graph_bin['upper']}"
        )
        tick_labels.append(f"{node_range}\n(n={len(graph_bin['indices'])})")
    ax.set_xticks(x_positions, tick_labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Visible nodes per graph (questions per bin)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by rendered graph size")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.015,
        "Graph size can causally affect only the image condition; text conditions "
        "control for overall question difficulty.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, out_dir, "accuracy_vs_graph_size", plt)


def configure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DataError(
            "matplotlib is required to create figures. Install it in the reporting environment."
        ) from exc
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    return plt


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"

    try:
        predictions = load_all_predictions(args.results_dir)
        indices = set(predictions[CONDITIONS[0][0]])
        gold_counts = Counter(
            row["gold_option"] for row in predictions[CONDITIONS[0][0]].values()
        )
        stats = {
            key: compute_condition_stats(records, gold_counts, EXPECTED_COUNT)
            for key, records in predictions.items()
        }
        pairwise = compute_pairwise(predictions)
        graph_sizes = load_graph_sizes(args.graph_metadata, indices)
        graph_bins = make_graph_bins(graph_sizes) if graph_sizes is not None else None
        plt = configure_matplotlib()
    except DataError as exc:
        raise SystemExit(f"error: {exc}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(stats, pairwise, graph_bins)
    print(summary, end="")
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    write_summary_csv(out_dir / "summary.csv", stats)

    plot_accuracy(stats, out_dir, plt)
    plot_answer_distribution(predictions, gold_counts, out_dir, plt)
    plot_confusion(predictions, stats, out_dir, plt)
    plot_paired(pairwise, out_dir, plt)
    if graph_bins:
        plot_accuracy_vs_graph_size(predictions, graph_bins, out_dir, plt)

    print(f"Figures and summaries written to: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
