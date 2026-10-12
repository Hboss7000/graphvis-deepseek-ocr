"""CPU-only audit checks; no pretrained model or GPU inference."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_synth_mcq_llava_images as audit


def test_graph_checksum_matches_shell_listing_and_detects_change(tmp_path):
    listing = ''
    for suffix in ('png', 'dot', 'txt'):
        for index in range(1, 51):
            name = f'G{index:02d}.{suffix}'
            payload = name.encode()
            (tmp_path / name).write_bytes(payload)
            listing += f'{hashlib.sha256(payload).hexdigest()}  {name}\n'
    before = audit.graph_checksum(tmp_path)
    assert before == hashlib.sha256(listing.encode()).hexdigest()
    (tmp_path / 'G01.png').write_bytes(b'corrupt transfer')
    assert audit.graph_checksum(tmp_path) != before


def test_graph_checksum_rejects_incomplete_transfer(tmp_path):
    (tmp_path / 'G01.png').write_bytes(b'image')
    with pytest.raises(ValueError, match='exactly G01-G50'):
        audit.graph_checksum(tmp_path)


def test_summary_uses_strict_ten_pixel_gate():
    assert audit.summary([9.0, 10.0, 11.0]) == dict(min=9.0, median=10.0, max=11.0, below_10px=1, n_images=3)


class FakeImageProcessor:
    size = SimpleNamespace(height=None, width=None, shortest_edge=336)

    def _resize_for_patching(self, image, grid):
        # Padded grid height is 672, but real content height is only 400.
        return SimpleNamespace(shape=(3, 400, 672))


def test_measurement_uses_real_content_height_and_restores_processor(monkeypatch):
    owner = FakeImageProcessor()
    processor = SimpleNamespace(image_processor=owner)
    calls = []

    def prepare(p, prompt, image):
        calls.append((p, prompt, image))
        p.image_processor._resize_for_patching(image, (672, 672))
        return {}

    monkeypatch.setattr(audit, 'prepare_inputs', prepare)
    monkeypatch.setattr(audit, 'format_llava_prompt', lambda p, b, c, t: (b, c, t))
    monkeypatch.setattr(audit, 'image_diagnostics', lambda *a: {'realised_image_tokens': 2500})
    image = SimpleNamespace(height=4000, size=(6720, 4000))
    measured = audit.measure_image(processor, image, '<image>\nQuestion',
                                  {'node': {'reference_cap_height_px': 60}, 'edge': {'reference_cap_height_px': 48}})
    assert measured['node_cap_px_grid'] == 6.0
    assert measured['edge_cap_px_grid'] == 4.8
    assert measured['node_cap_px_base'] == 5.04
    assert measured['realised_image_tokens'] == 2500
    assert calls[0][1] == ('<image>\nQuestion', 'image', 'llava_v1')
    assert '_resize_for_patching' not in vars(owner)


def test_processor_restored_if_preprocessing_fails(monkeypatch):
    processor = SimpleNamespace(image_processor=FakeImageProcessor())

    def fail(*args):
        raise RuntimeError('bad input')

    monkeypatch.setattr(audit, 'format_llava_prompt', lambda *args: 'prompt')
    monkeypatch.setattr(audit, 'prepare_inputs', fail)
    with pytest.raises(RuntimeError, match='bad input'):
        audit.measure_image(processor, None, 'body', {})
    assert '_resize_for_patching' not in vars(processor.image_processor)
