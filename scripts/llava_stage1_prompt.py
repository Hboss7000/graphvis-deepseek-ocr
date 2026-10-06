"""Stage 1 LLaVA diagnostic wrappers; user-turn bodies remain model-independent."""
from llava_common import build_messages, format_prompt

PROMPT_TEMPLATES = ('hf-chat', 'llava_v1')


def format_stage1_prompt(processor, body, prompt_template='hf-chat'):
    return format_llava_prompt(processor, body, 'image', prompt_template)


def format_llava_prompt(processor, body, condition, prompt_template='hf-chat'):
    if prompt_template == 'hf-chat':
        return format_prompt(processor, body, condition)
    if prompt_template != 'llava_v1':
        raise ValueError(f'Unknown prompt template: {prompt_template}')
    build_messages(body, condition)  # Validate the shared image/text body.
    # Source: https://github.com/yihedeng9/GraphVis/blob/main/llava/conversation.py
    # conv_llava_v1 and Conversation.get_prompt(), SeparatorStyle.TWO:
    # system + sep=" " + "USER: " + message + sep + "ASSISTANT:" (empty reply).
    # No sep2/EOS for an empty assistant turn. BOS stays processor-managed, exactly
    # as in the main run: pass this text to the same prepare_inputs() call.
    system = ("A chat between a curious human and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")
    return system + ' USER: ' + body + ' ASSISTANT:'

ASSISTANT_PREFIX_MODES = ('none', 'gold-template')
FIXED_PREFIXES = {
    'node_number': 'There are',
    'edge_number': 'There are',
    'highest_node_degree': 'One node with the highest degree is "',
    'node_description': 'The image depicts the following nodes:',
    'triple_listing': 'The triples in the graph are listed as: (',
}


def gold_template_prefix(task, question):
    """Use task literals and question names only; never accept an answer/metadata."""
    import re
    if task in FIXED_PREFIXES:
        return FIXED_PREFIXES[task]
    names = re.findall(r'"([^"\n]+)"', question)
    expected = {'node_degree': 1, 'relation_identification': 2,
                'neighbor_listing': 1, 'shortest_path_listing': 2}
    if task not in expected or len(names) != expected[task]:
        raise ValueError(f'Cannot derive safe {task} prefix from question')
    # Extra-task wording comes from generate_graphvis_datasets.py,
    # build_stage1_extended_tasks(), stopping before relation/neighbors/path.
    if task == 'node_degree':
        # Explicit diagnostic wording requested by Henrique; the generator's
        # gold uses the shorter 'The degree of the node "X" is'. Both stop
        # before the degree. X is taken only from the question.
        return f'The degree of the node with the name "{names[0]}" is'
    if task == 'relation_identification':
        return f'The relation between "{names[0]}" and "{names[1]}" is "'
    if task == 'neighbor_listing':
        return f'The node "{names[0]}" is directly connected to:'
    return f'The shortest path from "{names[0]}" to "{names[1]}" is:'


def prefix_for_record(record, mode='none'):
    if mode == 'none':
        return ''
    if mode != 'gold-template':
        raise ValueError(f'Unknown assistant prefix mode: {mode}')
    return gold_template_prefix(str(record['task_type']), str(record['prompt']))


def append_assistant_prefix(formatted, prefix, prompt_template):
    if not prefix:
        return formatted
    marker = '[/INST]' if prompt_template == 'hf-chat' else 'ASSISTANT:'
    if not formatted.rstrip().endswith(marker):
        raise ValueError(f'Expected assistant boundary {marker!r}')
    return formatted + ('' if formatted[-1:].isspace() else ' ') + prefix


def diagnostic_arm(prompt_template, assistant_prefix_mode):
    return {('hf-chat','none'): 'main', ('hf-chat','gold-template'): 'P1',
            ('llava_v1','none'): 'P2', ('llava_v1','gold-template'): 'P3'}[
                (prompt_template, assistant_prefix_mode)]


def config_settings(args, user_turn_hash):
    template = getattr(args, 'prompt_template', 'hf-chat')
    mode = getattr(args, 'assistant_prefix_mode', 'none')
    return {'prompt_template': template, 'assistant_prefix_mode': mode,
            'diagnostic_arm': diagnostic_arm(template, mode), 'user_turn_text_sha256': user_turn_hash}


def validate_diagnostic_settings(args):
    import os
    arm = diagnostic_arm(args.prompt_template, args.assistant_prefix_mode)
    if arm == 'main':
        return
    if args.seed != 13 or args.max_new_tokens != 1024 or args.answer_format != 'none':
        raise ValueError('Diagnostic arms require seed=13, max_new_tokens=1024, answer_format=none')
    if os.environ.get('FULLRUN_CONTRACT'):
        raise ValueError('Diagnostic arms require their own outputs, outside the main fullrun contract')
    for name in ('STAGE1_MATRIX_CELL_JSON', 'STAGE1_FROZEN_PROMPT_POLICY_JSON'):
        if os.environ.get(name):
            raise ValueError(f'Diagnostic arm cannot inherit {name}')


def write_compatible_config(directory, config, writer):
    """Keep historical default-run resumes compatible; reject diagnostic drift."""
    import json
    path = directory / 'run_config.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if config['diagnostic_arm'] == 'main':
            # These newly recorded defaults are operational provenance only.
            # Preserve the historical config file; do not relax any old setting.
            defaults = {'prompt_template': 'hf-chat', 'assistant_prefix_mode': 'none',
                        'diagnostic_arm': 'main', 'user_turn_text_sha256': config['prompt_bodies_sha256']}
            config = dict(config)
            for key, value in defaults.items():
                if key not in previous and config.get(key) == value:
                    config.pop(key)
    writer(directory, config)
