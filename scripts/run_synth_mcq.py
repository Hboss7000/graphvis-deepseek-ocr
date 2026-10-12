#!/usr/bin/env python3
"""Dispatch synthetic MCQ inference; the Qwen v4 implementation is frozen verbatim."""
from pathlib import Path as _DispatchPath
import sys as _dispatch_sys
import argparse as _dispatch_argparse

_QWEN_SOURCE = _DispatchPath(__file__).with_name('_synth_mcq_qwen_v4.py')
_dispatch_module_name = __name__
__name__ = 'run_synth_mcq'
exec(compile(_QWEN_SOURCE.read_text(), str(_QWEN_SOURCE), 'exec'), globals())
__name__ = _dispatch_module_name
_qwen_parse_args, _qwen_main, _qwen_run_identity = parse_args, main, run_identity


def _select_backend(argv):
    parser = _dispatch_argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('qwen', 'llava'), default='qwen')
    options, rest = parser.parse_known_args(list(argv) if argv is not None else _dispatch_sys.argv[1:])
    return options.backend, rest


def parse_args(argv=None):
    backend, rest = _select_backend(argv)
    if backend == 'llava':
        from synth_mcq_llava_backend import parse_args as parse_llava
        return parse_llava(rest)
    args = _qwen_parse_args(rest)
    args.backend = 'qwen'
    return args


def run_identity(args):
    identity = _qwen_run_identity(args)
    # Historical key stays unchanged, but its digest identifies the byte-for-byte
    # frozen Qwen execution source, rather than the new backend-only dispatcher.
    identity['code_sha256']['scripts/run_synth_mcq.py'] = sha256_file(_QWEN_SOURCE)
    return identity


def main(argv=None):
    backend, rest = _select_backend(argv)
    if backend == 'llava':
        from synth_mcq_llava_backend import main as main_llava
        return main_llava(rest)
    return _qwen_main(rest)


if __name__ == '__main__':
    main()
