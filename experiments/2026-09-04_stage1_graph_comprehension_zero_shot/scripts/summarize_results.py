#!/usr/bin/env python3
"""Collate every zero-shot result into one markdown summary, plus an SVG chart.

Reads the prediction and metrics files produced by the Stage 2 (OBQA QA) and
Stage 1 (graph comprehension) experiments. Recomputes accuracies from the
prediction files rather than trusting stored metrics, and runs exact McNemar
tests on the paired conditions.

Standard library only. Runs on the COMA login node with plain python3.

Usage:
    python3 summarize_results.py --out summary.md --chart stage1.svg
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

HOME = Path.home()

# --- result locations -------------------------------------------------------
DS_STAGE2 = HOME / "outputs/2026-08-25_zero_shot_obqa_500"
DS_KGTEXT = HOME / "outputs/experiments/2026-08-25_zero_shot_obqa_500_multimodal/kg_text"
QWEN_STAGE2 = HOME / "outputs/2026-08-25_zero_shot_obqa_500_qwen3vl"
STAGE1 = HOME / "outputs/experiments/2026-09-04_stage1_graph_comprehension_zero_shot"

# Task scores corrupted by response-format parsing bugs, keyed by (model, task).
UNRELIABLE = {
    ("qwen", "edge_number"): (
        "parser takes the first integer; Qwen answers as a numbered list, so "
        "the list index '1' is scored instead of the total"
    ),
    ("qwen", "triple_listing"): (
        "parser expects (a, b, c); Qwen emits backtick-delimited triples, so "
        "only 46 of ~9388 gold triples matched despite 99.1% gold-node recall"
    ),
}

# Metrics computed by a code path the parser bug does not touch, so they stay
# reportable even when the task is otherwise excluded.
RELIABLE_ROWS = {("qwen", "triple_listing", "gold-node recall")}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(path):
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    correct = sum(r.get("is_correct") is True for r in rows)
    return {
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "letters": Counter(r.get("predicted_option") for r in rows),
    }


def mcnemar(path_a, path_b):
    if not (path_a.is_file() and path_b.is_file()):
        return None
    a = {r["statement_idx"]: r.get("is_correct") is True for r in read_jsonl(path_a)}
    b = {r["statement_idx"]: r.get("is_correct") is True for r in read_jsonl(path_b)}
    shared = set(a) & set(b)
    n01 = sum(1 for k in shared if a[k] and not b[k])
    n10 = sum(1 for k in shared if not a[k] and b[k])
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    tail = sum(comb(n, i) for i in range(min(n01, n10) + 1))
    return {"n01": n01, "n10": n10, "p": min(1.0, tail / 2 ** n * 2)}


def fmt_pct(x):
    return "n/a" if x is None else "{:.1f}%".format(100 * x)


def fmt_p(p):
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else "{:.3f}".format(p)


def letter_string(counts):
    return " / ".join(str(counts.get(k, 0)) for k in ("A", "B", "C", "D"))


def load_stage1_metrics():
    metrics = {}
    for model in ("deepseek", "qwen"):
        path = STAGE1 / model / "metrics_{}.json".format(model)
        if path.is_file():
            metrics[model] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


# --- chart ------------------------------------------------------------------

# One headline metric per task, on a common 0-100 scale.
CHART_ROWS = [
    ("node_number", "Node count|(exact acc.)",
     lambda t: 100 * t["exact_accuracy"]),
    ("edge_number", "Edge count|(exact acc.)",
     lambda t: 100 * t["exact_accuracy"]),
    ("node_degree", "Node degree|(exact acc.)",
     lambda t: 100 * t["exact_accuracy"]),
    ("highest_node_degree", "Highest degree|(degree acc.)",
     lambda t: 100 * t["degree_accuracy"]),
    ("node_description", "Node listing|(micro F1)",
     lambda t: 100 * t["set_metrics"]["annotation_stripped"]["micro_f1"]),
    ("triple_listing", "Triple listing|(gold-node recall)",
     lambda t: 100 * t["gold_node_component_recall"]["annotation_stripped"]["micro_recall"]),
]

# Which metric name each chart bar corresponds to, for the RELIABLE_ROWS check.
CHART_METRIC_NAME = {"triple_listing": "gold-node recall"}

COLORS = {"deepseek": "#c1666b", "qwen": "#4a7ba7"}
MODEL_LABEL = {"deepseek": "DeepSeek-OCR-2", "qwen": "Qwen3-VL-8B"}


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_chart(metrics):
    """Hand-rolled SVG grouped bar chart. No plotting dependency."""
    width, height = 920, 520
    left, right, top, bottom = 72, 24, 72, 132
    plot_w = width - left - right
    plot_h = height - top - bottom
    base_y = top + plot_h

    groups = len(CHART_ROWS)
    gw = plot_w / groups
    bar_w = min(46.0, gw / 2 - 12)

    def y_of(value):
        return base_y - (value / 100.0) * plot_h

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}" font-family="Helvetica, Arial, sans-serif">'.format(
            width, height, width, height),
        '<rect width="{}" height="{}" fill="#ffffff"/>'.format(width, height),
        '<text x="{}" y="30" font-size="17" font-weight="bold">Stage 1 graph '
        'comprehension, zero-shot (500 records per task)</text>'.format(left),
        '<text x="{}" y="50" font-size="12" fill="#555">Higher is better. Hatched bars '
        'are excluded as scorer artifacts, not model failures.</text>'.format(left),
        '<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="7" height="7" fill="#d8d8d8"/>'
        '<line x1="0" y1="0" x2="0" y2="7" stroke="#9a9a9a" stroke-width="3"/>'
        '</pattern></defs>',
    ]

    for tick in range(0, 101, 20):
        y = y_of(tick)
        parts.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#e4e4e4" '
                     'stroke-width="1"/>'.format(left, y, left + plot_w, y))
        parts.append('<text x="{}" y="{:.1f}" font-size="11" fill="#555" '
                     'text-anchor="end">{}%</text>'.format(left - 9, y + 4, tick))
    parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" '
                 'stroke-width="1.5"/>'.format(left, base_y, left + plot_w, base_y))

    for i, (task, label, getter) in enumerate(CHART_ROWS):
        gc = left + gw * (i + 0.5)
        for j, model in enumerate(("deepseek", "qwen")):
            cx = gc + (j - 0.5) * (bar_w + 8)
            x = cx - bar_w / 2
            metric_name = CHART_METRIC_NAME.get(task)
            excluded = ((model, task) in UNRELIABLE
                        and (model, task, metric_name) not in RELIABLE_ROWS)
            model_metrics = metrics.get(model)
            value = None
            if model_metrics:
                try:
                    value = getter(model_metrics["tasks"][task])
                except (KeyError, TypeError):
                    value = None

            if excluded or value is None:
                parts.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                             'fill="url(#hatch)" stroke="#9a9a9a"/>'.format(
                                 x, y_of(6), bar_w, base_y - y_of(6)))
                parts.append('<text x="{:.1f}" y="{:.1f}" font-size="10" fill="#777" '
                             'text-anchor="middle">excl.</text>'.format(cx, y_of(6) - 6))
            else:
                parts.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                             'fill="{}"/>'.format(x, y_of(value), bar_w,
                                                  base_y - y_of(value), COLORS[model]))
                parts.append('<text x="{:.1f}" y="{:.1f}" font-size="11" '
                             'text-anchor="middle" fill="#333">{:.1f}</text>'.format(
                                 cx, y_of(value) - 6, value))

        if task == "highest_node_degree":
            baseline = None
            for m in metrics.values():
                baseline = m["tasks"].get(task, {}).get(
                    "constant_degree_5_baseline_accuracy")
                if baseline is not None:
                    break
            if baseline is not None:
                by = y_of(100 * baseline)
                parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                             'stroke="#222" stroke-width="1.6" '
                             'stroke-dasharray="6 4"/>'.format(
                                 gc - gw / 2 + 6, by, gc + gw / 2 - 6, by))
                parts.append('<text x="{:.1f}" y="{:.1f}" font-size="10" '
                             'text-anchor="middle" fill="#222">constant "5" = '
                             '{:.1f}%</text>'.format(gc, by - 5, 100 * baseline))

        for k, line in enumerate(label.split("|")):
            parts.append('<text x="{:.1f}" y="{}" font-size="11.5" text-anchor="middle" '
                         'fill="#333">{}</text>'.format(gc, base_y + 20 + k * 15, esc(line)))

    ly = height - 46
    lx = left
    for model in ("deepseek", "qwen"):
        parts.append('<rect x="{}" y="{}" width="14" height="14" fill="{}"/>'.format(
            lx, ly, COLORS[model]))
        parts.append('<text x="{}" y="{}" font-size="12" fill="#333">{}</text>'.format(
            lx + 20, ly + 12, MODEL_LABEL[model]))
        lx += 175
    parts.append('<rect x="{}" y="{}" width="14" height="14" fill="url(#hatch)" '
                 'stroke="#9a9a9a"/>'.format(lx, ly))
    parts.append('<text x="{}" y="{}" font-size="12" fill="#333">excluded '
                 '(scorer artifact)</text>'.format(lx + 20, ly + 12))
    parts.append('<text x="{}" y="{}" font-size="10.5" fill="#666">Metrics differ by '
                 'task; each bar is on a 0-100 scale. Token-ceiling hits count as wrong, '
                 'so the Qwen edge and triple tasks are floors.</text>'.format(
                     left, height - 14))
    parts.append("</svg>")
    return "\n".join(parts)


# --- report sections --------------------------------------------------------

def section_stage2(out):
    out.append("## Stage 2 - OBQA question answering (500 test questions)\n")
    conditions = [
        ("DeepSeek-OCR-2", "image", DS_STAGE2 / "predictions_image.jsonl"),
        ("DeepSeek-OCR-2", "text", DS_STAGE2 / "predictions_text.jsonl"),
        ("DeepSeek-OCR-2", "text_noref", DS_STAGE2 / "predictions_text_noref.jsonl"),
        ("DeepSeek-OCR-2", "kg_text", DS_KGTEXT / "predictions.jsonl"),
        ("Qwen3-VL-8B", "image", QWEN_STAGE2 / "image/predictions.jsonl"),
        ("Qwen3-VL-8B", "text_noref", QWEN_STAGE2 / "text_noref/predictions.jsonl"),
    ]
    out.append("| Model | Condition | Accuracy | Correct | A / B / C / D |")
    out.append("| --- | --- | --- | --- | --- |")
    loaded = {}
    for model, cond, path in conditions:
        res = load_predictions(path)
        loaded[(model, cond)] = res
        if res is None:
            out.append("| {} | {} | *missing* | | |".format(model, cond))
            continue
        out.append("| {} | {} | {} | {}/{} | {} |".format(
            model, cond, fmt_pct(res["accuracy"]), res["correct"], res["total"],
            letter_string(res["letters"])))
    out.append("")
    out.append("Chance is 25%. A uniform predicted-letter distribution is 125 each.\n")

    out.append("### Paired significance tests (exact McNemar)\n")
    out.append("| Comparison | Better only | Worse only | p |")
    out.append("| --- | --- | --- | --- |")
    for label, pa, pb in [
        ("DeepSeek text_noref vs kg_text",
         DS_STAGE2 / "predictions_text_noref.jsonl", DS_KGTEXT / "predictions.jsonl"),
        ("DeepSeek text_noref vs image",
         DS_STAGE2 / "predictions_text_noref.jsonl", DS_STAGE2 / "predictions_image.jsonl"),
        ("DeepSeek text_noref vs text",
         DS_STAGE2 / "predictions_text_noref.jsonl", DS_STAGE2 / "predictions_text.jsonl"),
        ("Qwen text_noref vs image",
         QWEN_STAGE2 / "text_noref/predictions.jsonl",
         QWEN_STAGE2 / "image/predictions.jsonl"),
    ]:
        m = mcnemar(pa, pb)
        if m is None:
            out.append("| {} | *missing* | | |".format(label))
            continue
        out.append("| {} | {} | {} | {} |".format(label, m["n01"], m["n10"], fmt_p(m["p"])))
    out.append("")
    out.append("\"Better only\" counts questions the first condition got right and the "
               "second got wrong; \"worse only\" is the reverse. Agreements are discarded.\n")

    def delta(model, a, b):
        ra, rb = loaded.get((model, a)), loaded.get((model, b))
        if ra is None or rb is None:
            return None
        return ra["accuracy"] - rb["accuracy"]

    out.append("### Image penalty, both backbones\n")
    for model in ("DeepSeek-OCR-2", "Qwen3-VL-8B"):
        d = delta(model, "text_noref", "image")
        if d is not None:
            out.append("- {}: adding the rendered graph costs {} "
                       "(text_noref -> image).".format(model, fmt_pct(d)))
    d = delta("DeepSeek-OCR-2", "text_noref", "kg_text")
    if d is not None:
        out.append("- DeepSeek-OCR-2: supplying the same subgraph as **text** still costs "
                   "{}, so the failure is not a modality problem.".format(fmt_pct(d)))
    d = delta("DeepSeek-OCR-2", "text_noref", "text")
    if d is not None:
        out.append("- Caveat: merely *referring* to a graph costs {} on its own "
                   "(text_noref -> text), so part of the kg_text gap is framing, not "
                   "triples. A prefix-only control is still needed to separate them.".format(
                       fmt_pct(d)))
    out.append("")


def section_connectivity(out):
    out.append("## Why the subgraph does not help - structural analysis\n")
    path = DS_STAGE2 / "answer_connectivity_pairwise.json"
    if not path.is_file():
        out.append("*Pairwise connectivity metrics not found.*\n")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    out.append("Computed over the pruned subgraphs only; no model involved.\n")
    out.append("- Correct answer's node has **degree zero in {} of {} questions "
               "({})**.".format(m["questions_correct_answer_degree_zero"], m["questions"],
                                fmt_pct(m["fraction_correct_answer_degree_zero"])))
    out.append("- Pairwise degree win rate, correct answer vs each distractor: **{:.3f}** "
               "(chance 0.500, ties counted as half, {} pairs).".format(
                   m["pairwise_win_rate"], m["pairs_compared"]))
    hist = m.get("option_degree_histogram", {})
    if hist:
        out.append("- Degree is compressed by the `max_degree=5` pruning cap: {} "
                   "answer-option slots sit at exactly 5 against {} at 4. The cap, not "
                   "ConceptNet, sets the dynamic range.".format(
                       hist.get("5", 0), hist.get("4", 0)))
    out.append("")
    out.append("Caveat: max-degree is one way to operationalise \"the graph favours this "
               "answer\"; a path-based measure would be equally defensible. The result also "
               "implicates the pruning heuristic (max_nodes=18, max_edges=30, max_degree=5, "
               "which is not from the GraphVis paper) as much as ConceptNet retrieval.\n")


def section_stage1(out, metrics, chart_path):
    out.append("## Stage 1 - graph comprehension, zero-shot (RQ1)\n")
    if not metrics:
        out.append("*Stage 1 metrics not found.*\n")
        return
    out.append("500 records per task, image condition only.\n")
    if chart_path:
        out.append("![Stage 1 comprehension]({})\n".format(chart_path.name))

    out.append("| Task | Metric | DeepSeek-OCR-2 | Qwen3-VL-8B |")
    out.append("| --- | --- | --- | --- |")

    def cell(model, task, metric_name, getter):
        m = metrics.get(model)
        if not m:
            return "n/a"
        if ((model, task) in UNRELIABLE
                and (model, task, metric_name) not in RELIABLE_ROWS):
            return "**unreliable**"
        try:
            return getter(m["tasks"][task])
        except (KeyError, TypeError):
            return "n/a"

    rows = [
        ("node_number", "exact accuracy", lambda t: fmt_pct(t["exact_accuracy"])),
        ("node_number", "mean abs. error",
         lambda t: "{:.1f}".format(t["mean_absolute_error"])),
        ("edge_number", "exact accuracy", lambda t: fmt_pct(t["exact_accuracy"])),
        ("node_degree", "exact accuracy", lambda t: fmt_pct(t["exact_accuracy"])),
        ("node_degree", "mean abs. error",
         lambda t: "{:.1f}".format(t["mean_absolute_error"])),
        ("highest_node_degree", "degree accuracy", lambda t: fmt_pct(t["degree_accuracy"])),
        ("highest_node_degree", "name accuracy (ties ok)",
         lambda t: fmt_pct(t["annotation_stripped"]["name_accuracy"])),
        ("node_description", "micro F1 (annot. stripped)",
         lambda t: "{:.3f}".format(t["set_metrics"]["annotation_stripped"]["micro_f1"])),
        ("triple_listing", "triple micro F1",
         lambda t: "{:.3f}".format(t["set_metrics"]["annotation_stripped"]["micro_f1"])),
        ("triple_listing", "gold-node recall",
         lambda t: fmt_pct(
             t["gold_node_component_recall"]["annotation_stripped"]["micro_recall"])),
    ]
    for task, metric_name, getter in rows:
        out.append("| {} | {} | {} | {} |".format(
            task, metric_name,
            cell("deepseek", task, metric_name, getter),
            cell("qwen", task, metric_name, getter)))
    out.append("")

    out.append("**Gold-node recall is the key row.** It is computed by a code path the "
               "triple-format parser bug does not touch, so it stands even where the "
               "triple F1 does not. Qwen recovers 99.1% of gold node names from the "
               "rendered image while its triple F1 scores 0.010 - that gap is the proof "
               "the F1 is a formatting artifact, and that the model reads the graph.\n")

    any_metrics = next(iter(metrics.values()))
    hnd = any_metrics["tasks"].get("highest_node_degree", {})
    if "constant_degree_5_baseline_accuracy" in hnd:
        out.append("**`highest_node_degree` is near-saturated.** Answering a constant \"5\" "
                   "scores {} because {} of 500 gold degrees sit at the pruning cap. Both "
                   "models score far *below* that trivial strategy, so this task is "
                   "uninformative as currently constructed.\n".format(
                       fmt_pct(hnd["constant_degree_5_baseline_accuracy"]),
                       hnd.get("pruning_cap_5_count", "?")))

    out.append("### Token-ceiling hit rates\n")
    out.append("| Task | DeepSeek-OCR-2 | Qwen3-VL-8B |")
    out.append("| --- | --- | --- |")
    for task in sorted(any_metrics["tasks"]):
        vals = []
        for model in ("deepseek", "qwen"):
            m = metrics.get(model)
            frac = m["tasks"].get(task, {}).get("hit_token_ceiling_fraction") if m else None
            vals.append(fmt_pct(frac))
        out.append("| {} | {} | {} |".format(task, vals[0], vals[1]))
    out.append("")
    out.append("A response that hits the ceiling produces no parseable answer, so a high "
               "rate makes that task's score a floor rather than a measurement. Budgets "
               "differ by necessity: DeepSeek-OCR-2 hardcodes 8192 in its remote code, "
               "Qwen was run at 1024.\n")

    out.append("### Scores excluded as measurement artifacts\n")
    for (model, task), reason in sorted(UNRELIABLE.items()):
        out.append("- **{} / {}**: {}.".format(model, task, reason))
    out.append("")
    out.append("These are parser problems, not model failures, and need a scorer fix "
               "rather than a re-run. The predictions are already on disk.\n")

    out.append("### Qualitative: how the two models answer\n")
    out.append("- **DeepSeek-OCR-2** replies in its pretrained document-layout format, "
               "emitting `<|ref|>Node<|/ref|><|det|>[[20, 20, 60, 100], ...]` bounding "
               "boxes rather than prose, then repeating until the 8192-token ceiling. Its "
               "near-zero scores are as much a format mismatch as a comprehension failure "
               "- which is precisely what Stage 1 fine-tuning would teach.")
    out.append("- **Qwen3-VL** enumerates the graph correctly in prose, and on at least "
               "one record reproduces all 13 triples exactly and then adds a separate "
               "\"No connections found in KG\" section for the disconnected nodes.\n")


def section_caveats(out):
    out.append("## Standing caveats\n")
    out.append("- Visual token budgets are not equal across models and cannot be made "
               "equal: DeepSeek-OCR-2 uses base_size=1024 / image_size=768 / crop_mode "
               "(832 vision tokens); Qwen3-VL uses a pinned dynamic pixel range (1222 "
               "tokens, source 4056x2302 downscaled to 1504x832).")
    out.append("- The rendered graphs embed answer-choice letters in node labels "
               "(`straw [B,C]`), an addition of this implementation rather than the "
               "GraphVis paper. Stage 1 gold answers inherit them, which is why set "
               "metrics are reported with and without annotation stripping.")
    out.append("- The pruning heuristic (max_nodes=18, max_edges=30, max_degree=5, "
               "relation-priority ordering) has no counterpart in GraphVis, which prunes "
               "nothing. It is the prime suspect for the retrieval results above.")
    out.append("- Stage 1 set tasks omit disconnected nodes from the text side while the "
               "image draws them; this asymmetry is recorded in the run notes.")
    out.append("")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the markdown report here")
    parser.add_argument("--chart", type=Path, help="Write the Stage 1 SVG chart here")
    args = parser.parse_args()

    metrics = load_stage1_metrics()

    if args.chart and metrics:
        args.chart.parent.mkdir(parents=True, exist_ok=True)
        args.chart.write_text(build_chart(metrics), encoding="utf-8")

    out = ["# Zero-shot results summary", ""]
    section_stage2(out)
    section_connectivity(out)
    section_stage1(out, metrics, args.chart if metrics else None)
    section_caveats(out)

    report = "\n".join(out)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print("\nWrote {}".format(args.out))
    if args.chart and metrics:
        print("Wrote {}".format(args.chart))


if __name__ == "__main__":
    main()
