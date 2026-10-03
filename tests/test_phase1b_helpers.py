"""Regression checks for the CPU-side Phase 1b report/launch helpers."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_reading_probe_report_help_does_not_load_graph_generator():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/report_inference50_reading_probe.py'), '--help'],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'LABEL=predictions_node_description.jsonl' in result.stdout
    assert 'generate_graphvis_datasets' not in result.stderr


def test_phase1b_d_runner_path_is_not_duplicated():
    script = (ROOT / 'scripts/phase1b_manual.sh').read_text(encoding='utf-8')
    assert '"${STAGE1_RUNNER}/run_stage1_llava.py"' not in script
    assert 'run_job "llava_probe_D" "${LLAVA_PYTHON}" "${STAGE1_RUNNER}"' in script
