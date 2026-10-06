"""Pinned Qwen native Stage 1 collation and approved four-bridge LoRA scope."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/2026-08-25_zero_shot_obqa_500_multimodal/scripts'))
from run_zero_shot_qwen import (format_qwen_prompt, prepare_inputs, load_processor,
                               DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
from llava_stage1_training import projector

DEFAULT_MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
DEFAULT_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
TEMPLATE = 'qwen-native-no-system'


def pinned_processor(source=DEFAULT_MODEL_ID):
    from types import SimpleNamespace
    from transformers import AutoProcessor
    args = SimpleNamespace(model_id=str(source), revision=DEFAULT_REVISION,
        min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS, local_files_only=True)
    return load_processor(AutoProcessor, args)


def training_text(processor, record):
    prompt, answer = str(record['prompt']), str(record['answer'])
    if not answer.strip() or any(x in prompt + answer for x in
        ('<image>', '<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>')):
        raise ValueError('Empty answer or injected chat/image marker')
    prefix = format_qwen_prompt(processor, '<image>\n' + prompt, 'image')
    full = processor.apply_chat_template([
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
        tokenize=False, add_generation_prompt=False)
    # The template's newline AFTER the assistant end token is not supervised.
    eos = processor.tokenizer.eos_token
    if not full.endswith(eos + '\n') or not full.startswith(prefix):
        raise ValueError('Native template answer boundary/end token differs')
    full = full[:-1]
    if '<|im_start|>system' in prefix:
        raise ValueError('A default system prompt must not be injected')
    return prefix, full


class Stage1Collator:
    def __init__(self, processor, image_root, model_max_length=4096,
                 min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
        self.processor, self.image_root = processor, Path(image_root)
        self.model_max_length = model_max_length
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        processor.tokenizer.padding_side = 'right'
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def __call__(self, records):
        import torch
        from PIL import Image
        if not records:
            raise ValueError('Empty batch')
        texts, images = [training_text(self.processor, r) for r in records], []
        for row in records:
            with Image.open(self.image_root / row['image']) as im:
                images.append(im.convert('RGB'))
        batch = self.processor(text=[full for _, full in texts], images=images,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            padding=True, truncation=False, return_tensors='pt')
        batch.pop('token_type_ids', None)
        # mm_token_type_ids is retained: Qwen uses it for multimodal positions.
        labels = torch.full_like(batch['input_ids'], -100)
        marker = self.processor.image_token_id
        merge = self.processor.image_processor.merge_size ** 2
        for i, (prefix, full) in enumerate(texts):
            prefix_ids = self.processor.tokenizer(prefix, truncation=False)['input_ids']
            plain = self.processor.tokenizer(full, truncation=False)['input_ids']
            if plain[:len(prefix_ids)] != prefix_ids:
                raise ValueError('Tokenizer merges across the answer boundary')
            n = int(batch['attention_mask'][i].sum())
            ids = batch['input_ids'][i, :n].tolist()
            if n > self.model_max_length:
                raise ValueError(f'Example exceeds {self.model_max_length}: {n} tokens')
            count = int(batch['image_grid_thw'][i].prod()) // merge
            if ids.count(marker) != count or plain.count(marker) != 1:
                raise ValueError('Image expansion mismatch')
            expanded = [t for x in plain for t in ([x] * count if x == marker else [x])]
            if expanded != ids:
                raise ValueError('Processor differs from exact image expansion')
            boundary = len(prefix_ids) + count - 1
            if boundary >= n - 1 or ids[-1] != self.processor.tokenizer.eos_token_id:
                raise ValueError('Missing answer or final end token')
            labels[i, boundary:n] = batch['input_ids'][i, boundary:n]
        batch['labels'] = labels
        return batch


def attach_lora(model, r=128, alpha=256, dropout=.05):
    import torch
    from peft import LoraConfig, get_peft_model
    names = [n for n, layer in model.named_modules()
             if isinstance(layer, torch.nn.Linear) and '.language_model.' in n]
    if not names or any('.visual.' in n or 'lm_head' in n for n in names):
        raise ValueError('Expected language-only linear layers')
    model.requires_grad_(False)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    adapted = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                              target_modules=names, bias='none'))
    bridges = projector(adapted)
    bridges.requires_grad_(True)
    bridge_ids = {id(p) for p in bridges.parameters()}
    if any(p.requires_grad and id(p) not in bridge_ids and 'lora_' not in n
           for n, p in adapted.named_parameters()):
        raise ValueError('Unexpected trainable vision/language parameter')
    return adapted, names
