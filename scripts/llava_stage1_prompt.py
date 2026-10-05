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
