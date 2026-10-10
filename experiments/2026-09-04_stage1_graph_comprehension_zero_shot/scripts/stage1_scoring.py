"""Dispatch versioned Stage 1 scorers; historical implementations stay frozen."""
from __future__ import annotations

import score_stage1 as historical
import score_stage1_template_v1 as template
import score_stage1_audit_v2 as corrected
import score_stage1_audit_v3 as frozen_v3

EXTRACTORS = ('legacy', 'span', 'span_extended', 'template_v1', 'audit_v2', 'audit_v3')
DEFAULT_EXTRACTOR = 'audit_v3'


def implementation(extractor):
    if extractor not in EXTRACTORS:
        raise ValueError(f'Unknown Stage 1 extractor: {extractor}')
    return {'template_v1': template, 'audit_v2': corrected, 'audit_v3': frozen_v3}.get(extractor, historical)


def score_record(record, response, meta, extractor=DEFAULT_EXTRACTOR):
    return implementation(extractor).score_record(record, response, meta, extractor=extractor)


def aggregate_task(task, rows, extractor=DEFAULT_EXTRACTOR):
    return implementation(extractor).aggregate_task(task, rows, extractor=extractor)


def score_files(args):
    extractor = getattr(args, 'extractor', DEFAULT_EXTRACTOR)
    return implementation(extractor).score_files(args)
