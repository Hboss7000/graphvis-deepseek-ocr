"""Shared strict collator, LoRA scope and checkpoint primitives for Stage 1.

These primitives are CPU-tested before the paid trainer is implemented/launched.
No model download, environment install or job launch happens on import.
"""
import hashlib
import json
from pathlib import Path
import random
import shutil

from llava_common import image_token_id
from llava_stage1_prompt import format_stage1_prompt


def training_text(processor, record):
    prompt = str(record['prompt'])
    answer = str(record['answer'])
    if not answer.strip() or any(marker in prompt + answer for marker in
                                 ('<image>', processor.tokenizer.eos_token)):
        raise ValueError('Empty answer or injected image/EOS marker')
    prefix = format_stage1_prompt(processor, '<image>\n' + prompt, 'llava_v1')
    return prefix, prefix + ' ' + answer + processor.tokenizer.eos_token


class Stage1Collator:
    """Same anyres processor as inference; answer+EOS loss; no truncation."""
    def __init__(self, processor, image_root, model_max_length=4096):
        self.processor = processor
        self.image_root = Path(image_root)
        self.model_max_length = model_max_length
        processor.tokenizer.padding_side = 'right'
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def __call__(self, records):
        import torch
        from PIL import Image
        if not records:
            raise ValueError('Empty batch')
        texts = [training_text(self.processor, row) for row in records]
        images = []
        for row in records:
            with Image.open(self.image_root / row['image']) as image:
                images.append(image.convert('RGB'))
        batch = self.processor(text=[full for _, full in texts], images=images,
                               padding=True, truncation=False, return_tensors='pt')
        batch.pop('token_type_ids', None)
        batch.pop('mm_token_type_ids', None)
        labels = torch.full_like(batch['input_ids'], -100)
        marker = image_token_id(self.processor)
        for i, (prefix, full) in enumerate(texts):
            prefix_ids = self.processor.tokenizer(prefix, truncation=False)['input_ids']
            plain = self.processor.tokenizer(full, truncation=False)['input_ids']
            if plain[:len(prefix_ids)] != prefix_ids:
                raise ValueError('Tokenizer merges across the answer boundary; cannot mask safely')
            n = int(batch['attention_mask'][i].sum())
            ids = batch['input_ids'][i, :n].tolist()
            if n > self.model_max_length:
                raise ValueError(f'Example exceeds {self.model_max_length}: {records[i]} ({n} tokens)')
            count = ids.count(marker)
            height, width = batch['image_sizes'][i].tolist()
            tile_height, tile_width = batch['pixel_values'].shape[-2:]
            expected = self.processor._get_number_of_features(height, width, tile_height, tile_width)
            if self.processor.vision_feature_select_strategy == 'default':
                expected -= 1
            if count != expected or plain.count(marker) != 1:
                raise ValueError(f'Image expansion mismatch: {count} != {expected}')
            expanded = [token for x in plain for token in ([x] * count if x == marker else [x])]
            if ids != expanded:
                raise ValueError('Processor tokenization differs from exact image expansion')
            boundary = len(prefix_ids) + count - 1
            if boundary >= n - 1 or ids[-1] != self.processor.tokenizer.eos_token_id:
                raise ValueError('Missing answer or final EOS')
            if ids.count(self.processor.tokenizer.eos_token_id) != 1:
                raise ValueError('Expected exactly one final EOS')
            labels[i, boundary:n] = batch['input_ids'][i, boundary:n]
        batch['labels'] = labels
        return batch


def attach_lora(model, r=128, alpha=256, dropout=.05):
    """GraphVis scope: decoder linear layers; separate full-rank projector.

    The output head is excluded by GraphVis's released find_all_linear_names().
    Record the full resolved names so the scope can be reviewed before a run.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    names = [name for name, layer in model.named_modules()
             if isinstance(layer, torch.nn.Linear) and '.language_model.' in name]
    if not names or any('vision' in name or 'projector' in name for name in names):
        raise ValueError('Failed to resolve language-only linear LoRA scope')
    model.requires_grad_(False)
    model.enable_input_require_grads()
    model.config.use_cache = False
    adapted = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                              target_modules=names, bias='none'))
    projector(adapted).requires_grad_(True)
    trainable = [name for name, value in adapted.named_parameters() if value.requires_grad]
    if any('vision_tower' in name for name in trainable):
        raise ValueError('Vision tower must be frozen')
    if any('lora_' not in name and 'multi_modal_projector' not in name for name in trainable):
        raise ValueError('Unexpected trainable parameter')
    return adapted, names


def projector(model):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return base.model.multi_modal_projector


def tree_sha(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob('*')):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode() + b'\0')
            with file.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(model, optimizer, scheduler, destination, run_config, step, data_cursor):
    """Atomic adapter+projector and optimizer/scheduler/RNG/cursor checkpoint."""
    import numpy as np
    import torch
    destination = Path(destination)
    temporary = destination.with_name(destination.name + '.incomplete')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Checkpoint collision: {destination}')
    temporary.mkdir(parents=True)
    try:
        model.save_pretrained(temporary / 'adapter', safe_serialization=True)
        torch.save({name: value.detach().cpu() for name, value in projector(model).state_dict().items()},
                   temporary / 'projector.pt')
        state = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'step': step, 'data_cursor': data_cursor, 'python_rng': random.getstate(),
                 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state, temporary / 'training_state.pt')
        (temporary / 'run_config.json').write_text(json.dumps(run_config, indent=2, sort_keys=True) + '\n')
        hashes = {str(file.relative_to(temporary)): file_sha(file)
                  for file in sorted(temporary.rglob('*')) if file.is_file()}
        (temporary / 'checkpoint_manifest.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return tree_sha(destination)


def load_checkpoint(model, optimizer, scheduler, source, run_config):
    """Restore a locally created trusted checkpoint; reject drift/corruption."""
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    source = Path(source)
    manifest = json.loads((source / 'checkpoint_manifest.json').read_text())
    actual_files = {str(file.relative_to(source)) for file in source.rglob('*') if file.is_file()}
    if actual_files != set(manifest) | {'checkpoint_manifest.json'}:
        raise ValueError('Checkpoint has missing/unexpected files')
    for name, expected in manifest.items():
        if file_sha(source / name) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    if json.loads((source / 'run_config.json').read_text()) != run_config:
        raise ValueError('Resume configuration differs')
    restored = set_peft_model_state_dict(model, load_peft_weights(str(source / 'adapter'), device='cpu'))
    if restored.unexpected_keys or any('lora_' in name for name in restored.missing_keys):
        raise ValueError(f'Adapter state does not match: {restored}')
    projector(model).load_state_dict(torch.load(source / 'projector.pt', map_location='cpu', weights_only=True),
                                     strict=True)
    # RNG includes numpy/Python state; only load our verified local checkpoint.
    state = torch.load(source / 'training_state.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint cannot resume on CPU')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'], state['data_cursor']
