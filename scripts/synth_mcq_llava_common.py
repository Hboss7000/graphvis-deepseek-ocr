"""CPU-only contracts for the accepted LLaVA synthetic variant."""
import json
import re
from collections import Counter
from pathlib import Path

from synth_mcq_common import ROOT, LETTERS, sha256_file, parse_answer, logit_scores
from audit_synth_mcq_llava_images import DATASET_SHA256, graph_checksum
from llava_common import DEFAULT_MODEL_ID, DEFAULT_REVISION

VARIANT = 'dot_lr_tight_natural_n42_e34'
PAIRING_SHA256 = '8260de2296f5f4680e8d439dd6bf9df629d15004ee2dba5f9751f26cb0d8c6ba'
CHECKPOINT_SHA256 = '6864483da020dc07665ca91d611ea26421ce4e416413074000d8d1b556258c83'
STYLE = dict(engine='dot', rankdir='LR', node_fontsize=42, edge_fontsize=34,
             nodesep=.2, ranksep=.15, graph_ratio=None, dpi=200, node_fill='#ADD8E6')
VERIFIED_LETTER_IDS = dict(A=[28741, 330], B=[28760, 365], C=[28743, 334], D=[28757, 384])
ANSWER_PREFIX = 'The correct answer is'
PREFIX_GATE_PROTOCOL = 'llava_prefix_agreement_exact_ties_v2'
PREFIX_ARGMAX_TIE_BREAK = 'first maximum in A, B, C, D order'
# Exact helper used by the first prefix smoke; only its gate/annotation changed.
PRE_TIE_COMMON_SHA256 = '4f6f43b959384b130e826598081633f7c8353ef0292c131aa8906787a8685d66'
MIN_PREFIX_ROWS = 20
FALLBACK_PROTOCOL = 'explicit_v1_llava_fallback'


def token_letter(token_id, variants):
    return next((letter for letter in LETTERS if int(token_id) in variants[letter]), None)


def generated_prefix_length(token_ids, tokenizer):
    """Find the generated token boundary that completes the decoded prefix."""
    for length in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:length], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip()
        if text == ANSWER_PREFIX:
            return length
        if text and not ANSWER_PREFIX.startswith(text):
            return None
    return None


def prefix_generated_letter(response, token_ids, prefix_ids, variants, generated_prefix_token_count=None):
    """Qualify a decoded prefix followed immediately by one letter token."""
    length = generated_prefix_token_count
    if length is None:
        if not prefix_ids or token_ids[:len(prefix_ids)] != prefix_ids:
            return None
        length = len(prefix_ids)
    if (not response.startswith(ANSWER_PREFIX) or type(length) is not int
            or length < 1 or len(token_ids) <= length):
        return None
    return token_letter(token_ids[length], variants)


def prefix_scores(logits, choices, correct_node):
    # Preserve logit_scores' deterministic A/B/C/D tie-break and accuracy.
    scores = {'prefix_' + key: value for key, value in logit_scores(logits, choices, correct_node).items()}
    scores['prefix_logit_tie'] = sum(float(logits[l]) == max(map(float, logits.values())) for l in LETTERS) > 1
    return scores


def compatible_smoke_identity(saved, current):
    """Accept exact identity or the known v1 -> exact-tie annotation-only change."""
    if saved == current:
        return True
    candidate = json.loads(json.dumps(saved))
    key = 'scripts/synth_mcq_llava_common.py'
    if (candidate.get('protocol') != 'synth_mcq_llava_v2'
            or candidate.get('prefix_probe', {}).get('gate_protocol') != 'llava_prefix_agreement_v1'
            or candidate.get('code_sha256', {}).get(key) != PRE_TIE_COMMON_SHA256):
        return False
    candidate['prefix_probe']['gate_protocol'] = PREFIX_GATE_PROTOCOL
    candidate['code_sha256'][key] = current.get('code_sha256', {}).get(key)
    return candidate == current


def prefix_probe_summary(rows):
    """All available prefix probes; argmax accuracy still uses A/B/C/D tie-break."""
    result = []
    for model, condition in sorted({(r['model'], r['condition']) for r in rows}):
        values = [prefix_scores(r['prefix_letter_logits'], r['choices'], r['correct_node'])
                  for r in rows if (r['model'], r['condition']) == (model, condition)
                  and 'prefix_letter_logits' in r]
        result.append(dict(model=model, condition=condition, metric='prefix_logit_argmax', n=len(values),
            accuracy=sum(r['prefix_logit_argmax_is_correct'] for r in values) / len(values) if values else None,
            tied_rows=sum(r['prefix_logit_tie'] for r in values)))
    return result


def parse_llava_fallback(response, choices):
    """Post hoc smoke-derived tiers; explicit_v1 always takes precedence."""
    letter, tier = parse_answer(response, 4)
    if letter != 'FAILED':
        return letter, tier
    letters = set(re.findall(r'\bis\s+([A-D])\.', response))
    if letters:
        return (letters.pop(), 'is_letter') if len(letters) == 1 else ('FAILED', 'FAILED')
    names = [letter for letter, name in choices.items()
             if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', response, re.IGNORECASE)]
    return (names[0], 'choice_name') if len(names) == 1 else ('FAILED', 'FAILED')


def fallback_fields(response, choices, correct_node):
    letter, tier = parse_llava_fallback(response, choices)
    predicted = choices.get(letter)
    return dict(fallback_parser_protocol=FALLBACK_PROTOCOL,
                fallback_predicted_letter=None if letter == 'FAILED' else letter,
                fallback_predicted_node=predicted, fallback_parse_ok=letter != 'FAILED',
                fallback_parse_tier=tier, fallback_is_correct=predicted == correct_node)


def parser_summary(rows):
    """All-row accuracy (failed parses count wrong) and answer rate, by cell."""
    summaries = []
    for model, condition in sorted({(r['model'], r['condition']) for r in rows}):
        group = [r for r in rows if (r['model'], r['condition']) == (model, condition)]
        fallback = [fallback_fields(r['response'], r['choices'], r['correct_node']) for r in group]
        for protocol, values, ok, correct in (
                ('explicit_v1', group, 'parse_ok', 'is_correct'),
                (FALLBACK_PROTOCOL, fallback, 'fallback_parse_ok', 'fallback_is_correct')):
            parsed, matches = sum(r[ok] for r in values), sum(r[correct] for r in values)
            summaries.append(dict(model=model, condition=condition, parser_protocol=protocol,
                n_rows=len(group), parsed_rows=parsed, answer_rate=parsed / len(group),
                correct_rows=matches, accuracy=matches / len(group),
                accuracy_parsed=matches / parsed if parsed else None))
    return summaries


def prefix_agreement(rows):
    def summarize(group):
        qualified = [(r, prefix_generated_letter(r.get('response', ''), r.get('generated_token_ids', []),
                      r.get('prefix_token_ids', []), VERIFIED_LETTER_IDS,
                      r.get('generated_prefix_token_count'))) for r in group]
        qualified = [(r, letter) for r, letter in qualified if letter is not None]
        matches, tied_rows = 0, 0
        for row, letter in qualified:
            logits = row.get('prefix_letter_logits')
            if logits is None:
                continue  # Missing probes are mismatches, never synthetic agreements.
            scores = prefix_scores(logits, row['choices'], row['correct_node'])
            tied_rows += scores['prefix_logit_tie']
            matches += float(logits[letter]) == max(map(float, logits.values()))
        return dict(n=len(qualified), matches=matches, tied_rows=tied_rows,
                    non_tie_mismatches=len(qualified) - matches,
                    fraction=matches / len(qualified) if qualified else None)
    pooled = summarize(rows)
    pooled['passed'] = pooled['n'] >= MIN_PREFIX_ROWS and pooled['non_tie_mismatches'] == 0
    return dict(protocol=PREFIX_GATE_PROTOCOL, prefix=ANSWER_PREFIX, minimum_n=MIN_PREFIX_ROWS,
                required_fraction=1.0, pooled=pooled,
                agreement_rule='generated letter logit equals maximum A-D probe logit exactly; no tolerance; post hoc after first prefix smoke',
                argmax_tie_break=PREFIX_ARGMAX_TIE_BREAK,
                per_condition={c: summarize([r for r in rows if r['condition'] == c])
                               for c in ('image', 'text', 'none', 'wrong_image', 'wrong_text')},
                qualification='Reply starts with decoded prefix and its immediately next generated token is a verified A-D token; pooled gate only')


def check_triple_order(original, candidate, graph_ids):
    for gid in graph_ids:
        if (Path(original) / f'{gid}.txt').read_bytes() != (Path(candidate) / f'{gid}.txt').read_bytes():
            raise ValueError(f'{gid}: text triple order/content differs from Qwen')
        lines = lambda p: [line.strip() for line in p.read_text().splitlines() if ' -> ' in line]
        if lines(Path(original) / f'{gid}.dot') != lines(Path(candidate) / f'{gid}.dot'):
            raise ValueError(f'{gid}: DOT triple order/content differs from Qwen')
    return dict(n_graphs=len(graph_ids), passed=True, dot_order_identical=True, text_byte_identical=True)


def letter_variants(tokenizer, verify_pin=False):
    vocab = tokenizer.get_vocab()
    variants = {}
    for letter in LETTERS:
        ids = [int(vocab[token]) for token in (letter, '▁' + letter) if token in vocab]
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
        variants[letter] = list(dict.fromkeys(ids))
        if len(variants[letter]) < 2:
            raise ValueError(f'Missing bare/SentencePiece letter variants: {letter}')
    if len({token for values in variants.values() for token in values}) != sum(map(len, variants.values())):
        raise ValueError('Letter token variants collide')
    if verify_pin and variants != VERIFIED_LETTER_IDS:
        raise ValueError(f'Pinned LLaVA letter IDs differ: {variants}')
    return variants


def score_letters(logits, variants):
    return {letter: max(float(logits[token]) for token in variants[letter]) for letter in LETTERS}


def range_failures(rows, ranges):
    failures = []
    for row in rows:
        failed = [kind for kind in ('node', 'edge') if not
                  ranges[kind]['min'] <= row[f'{kind}_cap_px_grid'] <= ranges[kind]['max']]
        if failed:
            failures.append(dict(graph_id=row['graph_id'], labels=failed,
                                 node=row['node_cap_px_grid'], edge=row['edge_cap_px_grid']))
    return failures


def verify_assets(dataset, assets, pairing_path, expected_checksum=None):
    assets = Path(assets)
    if sha256_file(dataset) != DATASET_SHA256:
        raise ValueError('Dataset checksum differs from the fixed name-grounded experiment')
    if sha256_file(pairing_path) != PAIRING_SHA256:
        raise ValueError('Wrong-graph pairing hash differs from Qwen')
    manifest = json.loads((assets / 'render_manifest.json').read_text())
    actual = graph_checksum(assets / 'graphs')
    if (manifest.get('dataset_sha256') != DATASET_SHA256 or manifest.get('style') != STYLE
            or manifest.get('variant') != VARIANT or actual != manifest.get('graphs_sha256')
            or (expected_checksum and actual != expected_checksum)):
        raise ValueError('LLaVA render checksum/manifest/settings differ')
    if set(manifest['graphs']) != {f'G{i:02d}' for i in range(1, 51)}:
        raise ValueError('LLaVA render manifest must cover all 50 graphs')
    for gid, entry in manifest['graphs'].items():
        for suffix in ('png', 'dot', 'txt'):
            if sha256_file(assets / 'graphs' / f'{gid}.{suffix}') != entry['files_sha256'][suffix]:
                raise ValueError(f'LLaVA manifest hash mismatch: {gid}.{suffix}')
    return manifest, actual


def validate_gate(path, manifest, assets, allow_failed=False, images_requested=True):
    if path is None:
        if images_requested:
            raise ValueError('LLaVA image conditions require --image-audit with the training-range gate')
        return dict(required=False, override=False, failing_graph_count=None)
    gate = json.loads(Path(path).read_text())
    if (gate.get('backend') != 'llava' or gate.get('criterion') != 'fixed_stage1_reference_min_max'
            or gate.get('dataset_sha256') != DATASET_SHA256
            or gate.get('graphs_sha256') != manifest['graphs_sha256']
            or gate.get('render_manifest_sha256') != sha256_file(Path(assets) / 'render_manifest.json')
            or gate.get('model_revision') != DEFAULT_REVISION or gate.get('n_images') != 50):
        raise ValueError('LLaVA audit provenance differs from the accepted assets')
    reference_path = ROOT / 'outputs/synth_mcq_name_grounded_llava_audit_2026-10-10/audit_config.json'
    reference = json.loads(reference_path.read_text())
    expected_indices = reference['training_sample_indices']
    if gate.get('training_reference_indices') != expected_indices:
        raise ValueError('LLaVA training reference list changed')
    expected_ranges = {kind: {key: reference['summaries']['stage1_training_sample'][kind + '_grid'][key]
                              for key in ('min', 'max')} for kind in ('node', 'edge')}
    if (gate.get('ranges') != expected_ranges
            or gate.get('training_reference_audit_sha256') != sha256_file(reference_path)):
        raise ValueError('LLaVA training reference range/provenance changed')
    # Recompute rather than trusting a passed flag or count in a JSON file.
    measurements = gate['measurements']
    if len(measurements) != 50 or {r['graph_id'] for r in measurements} != set(manifest['graphs']):
        raise ValueError('LLaVA gate measurements must cover all graphs exactly once')
    failures = range_failures(measurements, gate['ranges'])
    if (failures != gate['failures'] or gate['passed'] != (not failures)
            or gate.get('failing_graph_count') != len(failures)):
        raise ValueError('LLaVA gate status/counts inconsistent with measurements')
    if failures and images_requested and not allow_failed:
        raise ValueError('LLaVA training-range gate failed: ' + ', '.join(r['graph_id'] for r in failures)
                         + '; image/wrong_image refused. Text/none remain available; explicit --allow-failed-legibility overrides.')
    return dict(required=images_requested, override=bool(failures and images_requested and allow_failed),
                override_requested=allow_failed, failing_graph_count=len(failures), failures=failures,
                criterion=gate['criterion'], ranges=gate['ranges'], training_reference_indices=expected_indices)


def smoke_report(rows, loading_seconds=0):
    agreement, timings, tiers, probe_accuracy = prefix_agreement(rows), {}, {}, {}
    for condition in ('image', 'text', 'none', 'wrong_image', 'wrong_text'):
        for model in ('base', 'tuned'):
            group = [r for r in rows if r['model'] == model and r['condition'] == condition]
            timings[f'{model}/{condition}'] = dict(n=len(group), mean_seconds=sum(r['item_elapsed_seconds'] for r in group) / len(group) if group else None)
            tiers[f'{model}/{condition}'] = dict(Counter(r['parse_tier'] for r in group))
            probe_accuracy[f'{model}/{condition}'] = {}
            for metric in ('logit_argmax', 'prefix_logit_argmax'):
                values = [r[metric + '_is_correct'] for r in group if metric + '_is_correct' in r]
                probe_accuracy[f'{model}/{condition}'][metric] = dict(
                    n=len(values), accuracy=sum(values) / len(values) if values else None)
    for cell in prefix_probe_summary(rows):
        probe_accuracy[f'{cell["model"]}/{cell["condition"]}']['prefix_logit_argmax'].update(
            n=cell['n'], accuracy=cell['accuracy'], tied_rows=cell['tied_rows'])
    complete = len(rows) == 40 and all(v['n'] == 4 for v in timings.values())
    generation = sum(400 * v['mean_seconds'] for v in timings.values()) if complete else None
    probe_times = [r['prefix_probe_elapsed_seconds'] for r in rows if 'prefix_probe_elapsed_seconds' in r]
    return dict(prefix_agreement=agreement, timings=timings, complete_smoke=complete,
                parse_tier_counts=tiers, parser_summary=parser_summary(rows), probe_accuracy=probe_accuracy,
                passed=complete and agreement['pooled']['passed'],
                prefix_probe_timing=dict(n=len(probe_times), mean_seconds=sum(probe_times) / len(probe_times) if probe_times else None),
                estimated_full_prefix_probe_seconds=4000 * sum(probe_times) / len(probe_times) if len(probe_times) == 40 and complete else None,
                estimated_full_generation_seconds=generation,
                measured_loading_seconds=loading_seconds,
                estimated_full_total_seconds=generation + loading_seconds if generation is not None else None,
                estimate_method='sum of 400 x mean seconds/row over all ten model/condition cells + measured loading; includes wrong-graph cells',
                peak_memory_allocated_bytes=max((r['peak_memory_allocated_bytes'] for r in rows), default=0))
