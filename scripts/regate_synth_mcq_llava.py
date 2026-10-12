#!/usr/bin/env python3
"""CPU-only exact-tie re-gating of saved prefix smoke; preserve original artifacts."""
import argparse
import json
from pathlib import Path

from analyze_synth_mcq import validate_rows
from synth_mcq_common import read_predictions, sha256_file, write_json
from synth_mcq_llava_common import smoke_report, PREFIX_GATE_PROTOCOL


def regate(smoke_out):
    smoke_out = Path(smoke_out)
    pred = smoke_out / 'predictions.jsonl'
    source_report = smoke_out / 'smoke_report.json'
    original = json.loads(source_report.read_text())
    config = json.loads((smoke_out / 'config.json').read_text())
    if original['identity'] != config['identity'] or original['identity'].get('backend') != 'llava':
        raise ValueError('LLaVA smoke/config identity mismatch')
    rows = read_predictions(pred)
    if not rows or any(r.get('backend') != 'llava' or 'prefix_letter_logits' not in r for r in rows):
        raise ValueError('Saved LLaVA prefix logits required for every row')
    validate_rows(rows)  # Derive tie flags, check existing argmax/accuracy, leave originals intact.
    sanity = original.get('first_position_sanity', {})
    if sanity != config.get('first_position_sanity'):
        raise ValueError('Smoke/config logit sanity mismatch')
    sanity_rows = [r for r in rows if all(r.get(key) == sanity.get(key)
                   for key in ('prompt_style', 'condition', 'graph_id', 'question_id', 'perm'))]
    if (len(sanity_rows) != 2 or {r['model'] for r in sanity_rows} != {'base', 'tuned'}
            or any(r['first_letter_logits'] != sanity.get(r['model']) for r in sanity_rows)):
        raise ValueError('Saved sanity does not match the prediction probes')
    differ = max(abs(sanity['base'][l] - sanity['tuned'][l]) for l in 'ABCD') > 1e-6
    if bool(sanity.get('differ')) != differ:
        raise ValueError('Invalid saved base/tuned logit sanity')
    report = smoke_report(rows, original['measured_loading_seconds'])
    report.update(identity=original['identity'], adapter_sha256=original['adapter_sha256'],
                  first_position_sanity=sanity)
    report['passed'] = report['passed'] and differ
    report['offline_regate'] = dict(protocol=PREFIX_GATE_PROTOCOL, post_hoc=True,
        source_predictions_sha256=sha256_file(pred), source_report_sha256=sha256_file(source_report),
        source_gate_protocol=original['prefix_agreement']['protocol'],
        source_pooled=original['prefix_agreement']['pooled'],
        evaluation_code_sha256={p.name: sha256_file(p) for p in
            (Path(__file__), Path(__file__).with_name('synth_mcq_llava_common.py'))})
    enriched = smoke_out / 'predictions_regated.jsonl'
    enriched.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(smoke_out / 'smoke_report_regated.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = regate(args.smoke_out)
    print(json.dumps({**report['prefix_agreement']['pooled'], 'passed': report['passed']}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
