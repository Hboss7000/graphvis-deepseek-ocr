import json, glob, statistics as st

ARMS = {
    'phase1 default render (LLaVA)': 'outputs/phase1_reading_probe/runs/llava_default_render',
    'phase1 TB30 (LLaVA)':           'outputs/phase1_reading_probe/runs/llava_tb30',
    'phase1 TB30 (Qwen)':            'outputs/phase1_reading_probe/runs/qwen_tb30',
    'phase1b A TB30':                'outputs/phase1b/reading_probe/llava_A',
    'phase1b B auto-orient':         'outputs/phase1b/reading_probe/llava_B',
    'phase1b C wrapped':             'outputs/phase1b/reading_probe/llava_C',
    'phase1b D parentheses':         'outputs/phase1b/reading_probe/llava_D',
    'node style legacy (50)':        'outputs/fullrun_s0b_fetch_jcjYPY6T/fullrun_2026-10-04_B_s0b/llava/probe_legacy50',
    'node style uniform (50)':       'outputs/fullrun_s0b_fetch_jcjYPY6T/fullrun_2026-10-04_B_s0b/llava/probe_uniform50',
    'node style white (50)':         'outputs/fullrun_s0b_fetch_jcjYPY6T/fullrun_2026-10-04_B_s0b/llava/probe_white50',
}

def means(path):
    out = {}
    for variant in ('raw', 'basic'):
        vals = []
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                vals.append(float(row['set_metrics'][variant]['recall']))
            except Exception:
                pass
        out[variant] = (len(vals), round(st.mean(vals), 4)) if vals else None
    return out

for name, folder in ARMS.items():
    files = glob.glob(f'{folder}/predictions_*node_description.jsonl')
    print(f'{name:32s}', means(files[0]) if files else 'NO FILE')
