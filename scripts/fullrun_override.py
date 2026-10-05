"""Audited, job-local exception for the interrupted Gemma strict QA parse guard."""
import hashlib
import json
from pathlib import Path
from fullrun_common import frozen_json, sha


def allowed(spec):
    return (spec.get('model'), spec.get('label'), spec.get('stage'), spec.get('condition'),
            spec.get('mode'), spec.get('effective_max_new_tokens')) == (
                'gemma', 'qa_kg_text', 'qa', 'kg_text', 'full', 64)


def audit_override(directory, spec, config):
    path = directory / 'tripwire_override.json'
    enabled = config.get('tripwire_overridden', False)
    if type(enabled) is not bool or enabled != path.exists():
        raise ValueError('Missing/inconsistent tripwire override audit')
    if not enabled:
        return False
    if not allowed(spec):
        raise ValueError('Parse override is restricted to Gemma strict qa_kg_text')
    audit = json.loads(path.read_text())
    if audit['tripwire_overridden'] is not True or audit['disabled_tripwire'] != 'qa_first20_parse_failure':
        raise ValueError('Invalid tripwire override scope')
    original = directory / 'run_config.before_parse_override.json'
    if sha(original) != audit['original_config_sha256'] or sha(directory / 'contract.json') != audit['contract_sha256']:
        raise ValueError('Changed original override evidence')
    if {k:v for k,v in config.items() if k != 'tripwire_overridden'} != json.loads(original.read_text()):
        raise ValueError('Override changed experimental run configuration')
    with (directory / 'predictions.jsonl').open('rb') as handle:
        prefix = handle.read(audit['preserved_prediction_bytes'])
    if hashlib.sha256(prefix).hexdigest() != audit['preserved_predictions_sha256']:
        raise ValueError('Override resume changed original predictions')
    for row in (json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()):
        link = row.get('code_attempt')
        if link:
            path = Path(link['path'])
            if len(path.parts) != 2 or path.parts[0] != 'code_attempts' or path.name in ('.', '..'):
                raise ValueError('Invalid code-attempt path')
            if sha(directory / path) != link['sha256']:
                raise ValueError('Changed code-attempt evidence')
    return True


def prepare_override(root, directory, current):
    from fullrun_common import saved_spec, verify_job, prediction_rows
    if not allowed(current):
        raise ValueError('Parse override is restricted to Gemma strict qa_kg_text')
    recorded = saved_spec(root, directory, current)
    # All integrity/settings/time checks precede any mutation. Only parse is waived.
    verify_job(directory, recorded, complete=False, parse_override=True)
    rows = prediction_rows(directory, recorded)
    if len(rows) < 20:
        raise ValueError('Override requires the existing first 20 predictions')
    config_path = directory / 'run_config.json'
    config = json.loads(config_path.read_text())
    if not config.get('tripwire_overridden', False):
        original = directory / 'run_config.before_parse_override.json'
        if original.exists() and original.read_bytes() != config_path.read_bytes():
            raise ValueError('Original config backup collision')
        if not original.exists():
            original.write_bytes(config_path.read_bytes())
        predictions = directory / 'predictions.jsonl'
        frozen_json(directory / 'tripwire_override.json', {
            'tripwire_overridden': True, 'disabled_tripwire': 'qa_first20_parse_failure',
            'model': 'gemma', 'job': 'qa_kg_text',
            'authorization': '--job qa_kg_text --override-qa-parse-tripwire',
            'reason': 'Known Gemma explanation-before-answer behavior at strict 64 tokens',
            'original_config_sha256': sha(original), 'contract_sha256': sha(directory / 'contract.json'),
            'preserved_rows': len(rows), 'preserved_prediction_bytes': predictions.stat().st_size,
            'preserved_predictions_sha256': sha(predictions),
        })
        config['tripwire_overridden'] = True
        temp = directory / 'run_config.override.tmp'
        temp.write_text(json.dumps(config, indent=2) + '\n')
        temp.replace(config_path)
    audit_override(directory, recorded, config)
    attempts = directory / 'code_attempts'
    attempt = attempts / ('attempt_%04d.json' % (len(list(attempts.glob('attempt_*.json'))) + 1))
    frozen_json(attempt, {'code_files': current['code_files'], 'tripwire_overridden': True,
                          'original_code_files': recorded.get('code_files', {}), 'resume_rows': len(rows)})
    return recorded, attempt
