from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_manual_pins_common_stage1_and_qa_limits_and_single_outputs_root():
    text = (ROOT / 'scripts/phase1_manual.sh').read_text()
    assert 'STAGE1_TOKEN_LIMIT=1024' in text
    assert 'QA_TOKEN_LIMIT=64' in text
    assert 'Effective token limits:' in text
    assert '--max-new-tokens 8192' not in text
    assert 'readonly OUTPUTS_ROOT="${REPO_ROOT}/outputs"' in text
    assert '${WORKSPACE}/outputs/' not in text


def test_all_phase1_runners_record_the_effective_token_limit():
    paths = [
        'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_deepseek.py',
        'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_qwen.py',
        'experiments/2026-09-04_stage1_graph_comprehension_zero_shot/scripts/run_stage1_llava.py',
        'scripts/eval_gemma3_stage1.py',
        'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot.py',
        'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot_qwen.py',
        'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts/run_zero_shot_llava.py',
        'scripts/eval_gemma3_stage2.py',
    ]
    for relative in paths:
        assert 'effective_max_new_tokens' in (ROOT / relative).read_text(), relative


def test_phase1b_is_sequential_resumable_offline_and_uses_one_outputs_root():
    text = (ROOT / 'scripts/phase1b_manual.sh').read_text()
    assert 'HF_HUB_OFFLINE=1' in text
    assert 'readonly OUTPUTS_ROOT="${REPO_ROOT}/outputs"' in text
    assert text.count('--resume') >= 3
    assert 'for condition in image text text_noref kg_text' in text
    assert 'for config in A B C' in text
    assert 'rsync -rltvz --no-owner --no-group' in text
