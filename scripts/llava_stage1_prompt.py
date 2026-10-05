"""Stage 1 LLaVA diagnostic wrappers; user-turn bodies remain model-independent."""
from llava_common import build_messages, format_prompt

PROMPT_TEMPLATES = ('hf-chat', 'llava_v1')


def format_stage1_prompt(processor, body, prompt_template='hf-chat'):
    if prompt_template == 'hf-chat':
        return format_prompt(processor, body, 'image')
    if prompt_template != 'llava_v1':
        raise ValueError(f'Unknown prompt template: {prompt_template}')
    build_messages(body, 'image')  # Validate exactly one leading image placeholder.
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
