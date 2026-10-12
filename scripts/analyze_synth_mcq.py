#!/usr/bin/env python3
"""CPU analysis with graph-level bootstrap intervals and paired comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from synth_mcq_common import LETTERS, CONDITIONS, read_predictions, row_key, write_json, logit_scores, PARSER_PROTOCOL

METRICS = {'parsed_letter': 'is_correct', 'logit_argmax': 'logit_argmax_is_correct'}
PARSE_TIERS = ('exact', 'sole_letter', 'option_prefix', 'explicit_answer', 'bold_letter', 'FAILED')


def csv_output(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, rng, repetitions):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [.025, .975]))


def validate_rows(rows):
    graph_meta = {}
    question_sets = {}
    for row in rows:
        try:
            row.setdefault('prompt_style', 'obqa')
            row.setdefault('render_fontsize', 18)
            row.setdefault('render_style', 'compact')  # Legacy files used the original compact renderer.
            row.setdefault('question_id', row['graph_id'])
            if row['render_style'] not in ('training', 'compact'):
                raise ValueError('Invalid render_style')
            if type(row['render_fontsize']) is not int or row['render_fontsize'] <= 0:
                raise ValueError('Invalid render_fontsize')
            if row['prompt_style'] not in ('obqa', 'needed'):
                raise ValueError('Unknown prompt_style')
            if row['model'] not in ('base', 'tuned') or row['condition'] not in CONDITIONS:
                raise ValueError('Unknown model/condition')
            if type(row['perm']) is not int or row['perm'] not in range(4):
                raise ValueError('Invalid permutation')
            choices = row['choices']
            if set(choices) != set(LETTERS) or len(set(choices.values())) != 4:
                raise ValueError('Invalid choices')
            if choices[row['correct_letter']] != row['correct_node']:
                raise ValueError('Gold mapping mismatch')
            letter = row['predicted_letter']
            if letter is not None and letter not in LETTERS:
                raise ValueError('Invalid predicted letter')
            if type(row['parse_ok']) is not bool or row['parse_ok'] != (letter is not None):
                raise ValueError('Parse status mismatch')
            predicted = choices.get(letter)
            if row['predicted_node'] != predicted or row['is_correct'] != (predicted == row['correct_node']):
                raise ValueError('Node-based score mismatch')
            if row.get('parser_protocol', PARSER_PROTOCOL) != PARSER_PROTOCOL:
                raise ValueError('Expected explicit_v1 parser')
            if row['parse_tier'] not in PARSE_TIERS or (row['parse_tier'] == 'FAILED') != (not row['parse_ok']):
                raise ValueError('Invalid explicit_v1 parse tier/status')
            scores = logit_scores(row['first_letter_logits'], choices, row['correct_node'])
            for key, value in scores.items():
                if key in row and row[key] != value:
                    raise ValueError('Logit argmax mapping/score mismatch')
                row[key] = value
            if row.get('backend') == 'llava':
                from synth_mcq_llava_common import fallback_fields, prefix_scores, prefix_generated_letter, VERIFIED_LETTER_IDS
                extra = fallback_fields(row['response'], choices, row['correct_node'])
                if 'prefix_letter_logits' in row:
                    extra.update(prefix_scores(row['prefix_letter_logits'], choices, row['correct_node']))
                if 'generated_token_ids' in row and 'prefix_token_ids' in row:
                    extra['prefix_generated_letter'] = prefix_generated_letter(
                        row['response'], row['generated_token_ids'], row['prefix_token_ids'], VERIFIED_LETTER_IDS,
                        row.get('generated_prefix_token_count'))
                for key, value in extra.items():
                    if key in row and row[key] != value:
                        raise ValueError('LLaVA prefix/fallback mapping mismatch')
                    row[key] = value
            meta = row['dataset'], row.get('reasoning_hops'), row['correct_node'], row.get('question_type')
            qkey = row['graph_id'], row['question_id']
            if qkey in graph_meta and graph_meta[qkey] != meta:
                raise ValueError('Inconsistent graph metadata across cells')
            graph_meta[qkey] = meta
            if row['dataset'] not in ('original_25', 'expanded_25', 'name_grounded_50'):
                raise ValueError('Invalid dataset')
            if row['dataset'] == 'name_grounded_50':
                ids = row['question_ids']
                if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or row['question_id'] not in ids:
                    raise ValueError('Require two expected question_ids per name-grounded graph')
                if row.get('questions_per_graph') != 2 or not isinstance(row.get('question_type'), str) or row['question_type'].count('+') != 1:
                    raise ValueError('Missing question count or derived relation pair')
            else:
                ids = [row['question_id']]
            if row['graph_id'] in question_sets and question_sets[row['graph_id']] != set(ids):
                raise ValueError('Inconsistent expected question IDs across cells')
            question_sets[row['graph_id']] = set(ids)
            if row['dataset'] == 'expanded_25' and type(row.get('reasoning_hops')) is not int:
                raise ValueError('Missing reasoning_hops')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Prediction {row_key(row)}: {error}') from error
    # The very same permutation mapping must be used in every comparison cell.
    maps = {}
    for row in rows:
        key = row['graph_id'], row['question_id'], row['perm']
        if key in maps and maps[key] != row['choices']:
            raise ValueError(f'Inconsistent permutation mapping: {key}')
        maps[key] = row['choices']
    if len({row['render_fontsize'] for row in rows}) != 1:
        raise ValueError('Analyze different render_fontsize variants in separate files/output directories')
    if len({row['render_style'] for row in rows}) != 1:
        raise ValueError('Analyze different render_style variants in separate files/output directories')


def none_style_agreement(rows):
    """Match model/graph/permutation; compare outputs exactly, excluding runtime and style."""
    by_style = {style: {(r['model'], r['graph_id'], r.get('question_id', r['graph_id']), r['perm']): r for r in rows
                        if r['condition'] == 'none' and r['prompt_style'] == style}
                for style in ('obqa', 'needed')}
    prediction_fields = ('choices', 'correct_node', 'correct_letter', 'raw_response', 'response',
                         'predicted_letter', 'predicted_node', 'parse_ok', 'parse_tier',
                         'parser_protocol', 'is_correct', 'logit_argmax_letter', 'logit_argmax_node',
                         'logit_argmax_is_correct', 'generated_token_count', 'hit_token_ceiling',
                         'input_tokens', 'vision_tokens')
    logit_fields = ('first_letter_token_ids', 'first_letter_logits', 'first_letter_probs')
    common = sorted(set(by_style['obqa']) & set(by_style['needed']))
    mismatches = []
    for key in common:
        a, b = by_style['obqa'][key], by_style['needed'][key]
        changed = [field for field in prediction_fields + logit_fields
                   if (field in a) != (field in b) or a.get(field) != b.get(field)]
        if changed:
            mismatches.append(dict(model=key[0], graph_id=key[1], question_id=key[2], perm=key[3],
                changed_fields=changed, prediction_mismatch=any(f in prediction_fields for f in changed),
                logit_mismatch=any(f in logit_fields for f in changed)))
    return dict(status='compared' if common else 'no_matched_none_rows',
                matched_rows=len(common), mismatch_count=len(mismatches),
                prediction_mismatch_count=sum(r['prediction_mismatch'] for r in mismatches),
                logit_mismatch_count=sum(r['logit_mismatch'] for r in mismatches),
                obqa_only_rows=len(set(by_style['obqa']) - set(by_style['needed'])),
                needed_only_rows=len(set(by_style['needed']) - set(by_style['obqa'])),
                comparison='Exact equality of generated/parsed outputs, scores and A-D logits/probabilities; timing, memory, prompt_style excluded',
                mismatches=mismatches)


def analyze(pred, out, seed=0, repetitions=10000, n_perms=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred, out = Path(pred), Path(out)
    rows = read_predictions(pred)
    if not rows:
        raise ValueError('No predictions to analyze')
    validate_rows(rows)
    metrics = dict(METRICS)
    llava = all(r.get('backend') == 'llava' for r in rows)
    if llava:
        from synth_mcq_llava_common import FALLBACK_PROTOCOL, parser_summary, prefix_agreement, prefix_probe_summary
        metrics[FALLBACK_PROTOCOL] = 'fallback_is_correct'
        if all('prefix_logit_argmax_is_correct' in r for r in rows):
            metrics['prefix_logit_argmax'] = 'prefix_logit_argmax_is_correct'
    grounded = any(r['dataset'] == 'name_grounded_50' for r in rows)
    print('HEADLINE: graph-level bootstrap; two questions share one graph cluster. At 50 graphs, worst-case 95% uncertainty is approximately +/-14 pp.', flush=True)
    config_path = pred.parent / 'config.json'
    if n_perms is None and config_path.exists():
        n_perms = json.loads(config_path.read_text())['identity']['n_perms']
    if n_perms is None:
        n_perms = max(row['perm'] for row in rows) + 1
        print(f'No config.json: inferring n_perms={n_perms}; supply --n-perms if this is a partial run.', flush=True)
    if not 1 <= n_perms <= 4 or any(row['perm'] >= n_perms for row in rows):
        raise ValueError('Prediction permutations disagree with n_perms')
    out.mkdir(parents=True, exist_ok=True)
    if llava:
        parser_rows = parser_summary(rows)
        csv_output(out / 'llava_parser_summary.csv', parser_rows, list(parser_rows[0]))
        probe_rows = prefix_probe_summary(rows)
        csv_output(out / 'llava_prefix_probe_summary.csv', probe_rows, list(probe_rows[0]))
        for probe in probe_rows:
            print('PREFIX PROBE ' + json.dumps(probe), flush=True)
        write_json(out / 'llava_prefix_agreement.json', prefix_agreement(rows))
    agreement = none_style_agreement(rows)
    write_json(out / 'none_style_agreement.json', agreement)
    print(f"NONE STYLE AGREEMENT: matched={agreement['matched_rows']} mismatches={agreement['mismatch_count']} prediction_mismatches={agreement['prediction_mismatch_count']} logit_mismatches={agreement['logit_mismatch_count']} obqa_only={agreement['obqa_only_rows']} needed_only={agreement['needed_only_rows']}")
    rng = np.random.default_rng(seed)
    cells = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        cells[row['prompt_style'], row['model'], row['condition']][row['graph_id']][row['question_id'], row['perm']] = row
    complete, excluded = {}, []
    for cell, graphs in sorted(cells.items()):
        complete[cell] = {}
        for gid, permutations in graphs.items():
            expected_questions = permutations[next(iter(permutations))].get('question_ids', [next(iter(permutations))[0]])
            expected_keys = {(qid, p) for qid in expected_questions for p in range(n_perms)}
            if set(permutations) != expected_keys:
                excluded.append(dict(prompt_style=cell[0], model=cell[1], condition=cell[2], graph_id=gid,
                                     expected_question_ids=expected_questions, observed_question_perms=sorted(permutations)))
                continue
            ordered = [permutations[key] for key in sorted(expected_keys)]
            complete[cell][gid] = dict(rows=ordered, n_questions=len(expected_questions),
                                       accuracy={metric: sum(r[field] for r in ordered) / len(ordered)
                                                              for metric, field in metrics.items()},
                                       dataset=ordered[0]['dataset'], reasoning_hops=ordered[0].get('reasoning_hops'))
    summaries = []
    for (style, model, condition), graphs in sorted(complete.items()):
        splits = [('overall', 'all', '', list(graphs.values()))]
        for dataset in (('name_grounded_50',) if grounded else ('original_25', 'expanded_25')):
            subset = [g for g in graphs.values() if g['dataset'] == dataset]
            splits.append(('dataset', dataset, '', subset))
            if dataset == 'expanded_25':
                for hops in sorted({g['reasoning_hops'] for g in subset}):
                    splits.append(('dataset_reasoning_hops', dataset, hops,
                                   [g for g in subset if g['reasoning_hops'] == hops]))
        for scope, dataset, hops, subset in splits:
            for metric in metrics:
                values = [g['accuracy'][metric] for g in subset]
                lo, hi = bootstrap(values, rng, repetitions)
                summaries.append(dict(prompt_style=style, model=model, condition=condition, metric=metric,
                                      scope=scope, dataset=dataset, reasoning_hops=hops,
                                      n_graphs=len(values), n_rows=sum(len(g['rows']) for g in subset),
                                      accuracy=float(np.mean(values)) if values else None, ci_low=lo, ci_high=hi))
    csv_output(out / 'summary.csv', summaries,
               ['prompt_style', 'model', 'condition', 'metric', 'scope', 'dataset', 'reasoning_hops', 'n_graphs', 'n_rows', 'accuracy', 'ci_low', 'ci_high'])
    per_graph = [dict(prompt_style=style, model=model, condition=condition, graph_id=gid,
                      metric=metric, n_questions=g['n_questions'], n_rows=len(g['rows']), accuracy=g['accuracy'][metric])
                 for (style, model, condition), cell in sorted(complete.items())
                 for gid, g in sorted(cell.items()) for metric in metrics]
    csv_output(out / 'per_graph.csv', per_graph,
               ['prompt_style', 'model', 'condition', 'graph_id', 'metric', 'n_questions', 'n_rows', 'accuracy'])
    type_rows = []
    if grounded:
        for (style, model, condition), cell in sorted(complete.items()):
            typed = defaultdict(list)
            for gid, g in cell.items():
                by_question = defaultdict(list)
                for r in g['rows']: by_question[r['question_id']].append(r)
                for qid, values in by_question.items():
                    typed[values[0]['question_type']].append((gid, values))
            for pair, questions in sorted(typed.items()):
                for metric, field in metrics.items():
                    means = [sum(r[field] for r in values) / len(values) for _, values in questions]
                    type_rows.append(dict(label='descriptive, small n', prompt_style=style, model=model,
                        condition=condition, metric=metric, question_type=pair, n_questions=len(questions),
                        n_graphs=len({gid for gid, _ in questions}), n_rows=sum(len(values) for _, values in questions),
                        accuracy=float(np.mean(means))))
    csv_output(out / 'question_type_breakdown.csv', type_rows,
               ['label', 'prompt_style', 'model', 'condition', 'metric', 'question_type', 'n_questions', 'n_graphs', 'n_rows', 'accuracy'])
    comparisons = []
    pairs = [(("tuned", c), ("base", c)) for c in (CONDITIONS if grounded else ('image', 'text', 'none'))]
    pairs += [((m, 'image'), (m, 'text')) for m in ('base', 'tuned')]
    if grounded:
        pairs += [((m, correct), (m, control)) for m in ('base', 'tuned')
                  for correct in ('image', 'text') for control in ('none', 'wrong_image', 'wrong_text')]
    styles = sorted({r['prompt_style'] for r in rows})
    for style in styles:
        for metric, field in metrics.items():
            for left, right in pairs:
                a, b = complete.get((style, *left), {}), complete.get((style, *right), {})
                gids = sorted(set(a) & set(b))
                differences = [a[g]['accuracy'][metric] - b[g]['accuracy'][metric] for g in gids]
                lo, hi = bootstrap(differences, rng, repetitions)
                paired_rows = [(x[field], y[field]) for g in gids
                               for x, y in zip(a[g]['rows'], b[g]['rows'])]
                comparisons.append(dict(prompt_style=style, metric=metric,
                    left='_'.join(left), right='_'.join(right), n_graphs=len(gids), n_perms=n_perms,
                    mean_difference=float(np.mean(differences)) if gids else None, ci_low=lo, ci_high=hi,
                    graphs_improved=sum(d > 0 for d in differences), graphs_worsened=sum(d < 0 for d in differences),
                    graphs_tied=sum(d == 0 for d in differences),
                    graphs_all_correct_gained=sum(a[g]['accuracy'][metric] == 1 and b[g]['accuracy'][metric] < 1 for g in gids),
                    graphs_all_correct_lost=sum(a[g]['accuracy'][metric] < 1 and b[g]['accuracy'][metric] == 1 for g in gids),
                    paired_perms_wrong_to_correct=sum(x and not y for x, y in paired_rows),
                    paired_perms_correct_to_wrong=sum(y and not x for x, y in paired_rows)))
    csv_output(out / 'paired_comparisons.csv', comparisons, list(comparisons[0]))
    positions, failures, tiers = [], [], []
    for (style, model, condition), graphs in sorted(cells.items()):
        cell_rows = [r for permutations in graphs.values() for r in permutations.values()]
        parsed = sum(r['parse_ok'] for r in cell_rows)
        letter_fields = [('parsed_letter', 'predicted_letter'), ('logit_argmax', 'logit_argmax_letter')]
        if llava:
            letter_fields.append((FALLBACK_PROTOCOL, 'fallback_predicted_letter'))
            if 'prefix_logit_argmax' in metrics:
                letter_fields.append(('prefix_logit_argmax', 'prefix_logit_argmax_letter'))
        for metric, letter_field in letter_fields:
            counts = Counter(r[letter_field] for r in cell_rows if r[letter_field] is not None)
            predictions = sum(counts.values())
            for letter in LETTERS:
                positions.append(dict(prompt_style=style, model=model, condition=condition, metric=metric,
                                      predicted_letter=letter, count=counts[letter], n_rows=len(cell_rows),
                                      n_predictions=predictions, proportion_all_rows=counts[letter] / len(cell_rows),
                                      proportion_predictions=counts[letter] / predictions if predictions else None, chance=.25))
        failures.append(dict(prompt_style=style, model=model, condition=condition, n_rows=len(cell_rows),
                             parse_failures=len(cell_rows) - parsed, parse_failure_rate=1 - parsed / len(cell_rows)))
        counts = Counter(r['parse_tier'] for r in cell_rows)
        for tier in PARSE_TIERS:
            tiers.append(dict(prompt_style=style, model=model, condition=condition, parser_protocol=PARSER_PROTOCOL,
                              parse_tier=tier, count=counts[tier], n_rows=len(cell_rows), proportion=counts[tier] / len(cell_rows)))
        print(f'PARSE TIERS {style}/{model}/{condition} ({PARSER_PROTOCOL}): ' + json.dumps(dict(counts), sort_keys=True))
    csv_output(out / 'position_bias.csv', positions, list(positions[0]))
    csv_output(out / 'parse_failures.csv', failures, list(failures[0]))
    csv_output(out / 'parse_tier_counts.csv', tiers, list(tiers[0]))
    write_json(out / 'analysis_config.json', dict(seed=seed, bootstrap_repetitions=repetitions,
        n_perms=n_perms, prompt_styles=styles, metrics=list(metrics), parser_protocol=PARSER_PROTOCOL,
        render_fontsize=rows[0]['render_fontsize'], render_style=rows[0]['render_style'],
        none_style_agreement={k: v for k, v in agreement.items() if k != 'mismatches'},
        bootstrap_unit='graphs', incomplete_graph_cells_excluded=excluded,
        summary_protocol='mean correctness over all question/permutation rows within each complete graph, then mean over graphs',
        question_type_protocol='sorted pair of non-createdby supporting relations; descriptive, small n; no type-level headline CIs',
        flips_protocol='graph-level sign counts compare fractional accuracies; permutation discordances are descriptive only',
        position_protocol='reported over all available rows, with both valid-prediction and all-row denominators'))
    overall = [r for r in summaries if r['scope'] == 'overall' and r['n_graphs']]
    if not overall:
        raise ValueError('No graph has all expected permutations; incomplete diagnostics have been saved')
    fig, axes = plt.subplots(len(metrics), len(styles), figsize=(10 * len(styles), 4.5 * len(metrics)), squeeze=False)
    for j, style in enumerate(styles):
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            panel = [r for r in overall if r['prompt_style'] == style and r['metric'] == metric]
            x = np.arange(len(panel))
            ax.bar(x, [r['accuracy'] * 100 for r in panel],
                   color=['#527aab' if r['model'] == 'base' else '#df9051' for r in panel])
            ax.vlines(x, [r['ci_low'] * 100 for r in panel], [r['ci_high'] * 100 for r in panel], color='black')
            for k, row in enumerate(panel):
                ax.hlines([row['ci_low'] * 100, row['ci_high'] * 100], k - .07, k + .07, color='black')
                ax.text(k, row['ci_high'] * 100 + 2, f"n={row['n_graphs']}", ha='center', fontsize=9)
            ax.axhline(25, color='gray', linestyle='--', label='25% chance')
            ax.set_xticks(x, [f"{r['model']}\n{r['condition']}" for r in panel])
            ax.set_ylim(0, 110)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{style} / {metric} — 95% graph bootstrap intervals')
            ax.legend(loc='upper left')
    fig.text(.5, .015, 'At n=50 graphs, worst-case 95% uncertainty is roughly ±14 percentage points; actual bootstrap CIs vary.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(out / 'accuracy.png', dpi=180)
    fig.savefig(out / 'accuracy.pdf')
    plt.close(fig)
    for row in overall:
        print(f"{row['prompt_style']}/{row['model']}/{row['condition']}/{row['metric']}: n={row['n_graphs']} graphs, accuracy={row['accuracy']:.1%}, 95% CI [{row['ci_low']:.1%}, {row['ci_high']:.1%}]")
    for row in comparisons:
        if row['n_graphs']:
            print(f"PAIRED {row['prompt_style']}/{row['metric']} {row['left']} - {row['right']}: n={row['n_graphs']}, difference={row['mean_difference']:+.1%}, CI [{row['ci_low']:+.1%}, {row['ci_high']:+.1%}], improved/worsened/tied={row['graphs_improved']}/{row['graphs_worsened']}/{row['graphs_tied']}")
    print(f'Incomplete graph/cell groups excluded from accuracy and pairing: {len(excluded)}')
    print('With n=50 graphs, worst-case 95% uncertainty is roughly ±14 percentage points.')
    return summaries, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--n-perms', type=int, help='Override/inform expected permutations for partial runs')
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100')
    analyze(args.pred, args.out, args.seed, args.bootstrap, args.n_perms)


if __name__ == '__main__':
    main()
