#!/usr/bin/env python3
"""Run DeepSeek-OCR-2 on paper or extended image-only Stage 1 tasks."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from score_stage1 import (
    TASK_TYPES,
    TASK_SETS,
    answer_format_diagnostics,
    require_task,
    score_record,
)
from stage1_common import (
    ANSWER_FORMATS,
    DEEPSEEK_PROMPT_VARIANTS,
    LEGACY_PLACEHOLDER_ANSWER_SUFFIXES,
    answer_format_provenance,
    completed_keys,
    deepseek_image_prompt,
    output_path,
    print_six_prompt_previews,
    prompts_sha256,
    score_completed_run,
    validate_stage1,
    write_run_config,
)


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
OLD_SCRIPTS_DIR = EXPERIMENTS_DIR / "2026-08-25_zero_shot_obqa_500_multimodal" / "scripts"
sys.path.insert(0, str(OLD_SCRIPTS_DIR))
from prompt_common import sha256_file  # noqa: E402


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
DEFAULT_REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
REQUIRED_TRANSFORMERS_VERSION = "4.46.3"
BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
MAX_NEW_TOKENS = 8192
NO_REPEAT_NGRAM_SIZE = 35
MODEL_NAME = "deepseek"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=3000)
    parser.add_argument("--expected-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--task-set", choices=TASK_SETS, default="paper")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="none")
    parser.add_argument("--prompt-variant", choices=DEEPSEEK_PROMPT_VARIANTS, default="standard")
    parser.add_argument(
        "--behavior-only", action="store_true",
        help="Permit a partial task set and skip aggregate scoring (ablation controls only)",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--approve-prompts",
        action="store_true",
        help="Confirm the printed prompts were reviewed and permit inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.behavior_only and args.prompt_variant != "describe":
        raise SystemExit("--behavior-only is restricted to the describe ablation control")
    task_types = TASK_SETS[args.task_set]
    records, metadata_by_idx, task_counts = validate_stage1(
        args.input_jsonl, args.graph_metadata, args.expected_count, task_types,
        expected_split=args.expected_split,
    )
    prompt_builder = lambda record: deepseek_image_prompt(
        record, args.answer_format, args.prompt_variant
    )
    preview_tasks = tuple(dict.fromkeys(record["task_type"] for record in records))
    print_six_prompt_previews(
        records, task_types=preview_tasks, answer_format=args.answer_format,
        prompt_builder=prompt_builder,
    )
    if args.preview_only:
        print("Preview only: no model was loaded and no predictions were generated.", flush=True)
        return
    if not args.approve_prompts:
        raise SystemExit(
            "Refusing inference until the prompt previews are approved. "
            "Re-run with --approve-prompts."
        )
    if not args.revision:
        raise SystemExit("--revision must pin an immutable model revision")

    # Heavy imports intentionally occur after the preview/approval gate.
    import torch
    import transformers
    transformers.set_seed(args.seed)
    from transformers import AutoModel, AutoTokenizer

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This experiment requires transformers=={REQUIRED_TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU is visible. Run through sbatch.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = {
        task: output_path(args.output_dir, MODEL_NAME, task) for task in task_types
    }
    if not args.resume:
        existing = [path for path in prediction_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Prediction files already exist; use --resume or a new output directory: "
                + ", ".join(map(str, existing))
            )
    done = completed_keys(args.output_dir, MODEL_NAME, task_types) if args.resume else set()
    unknown_done = done - {
        (int(record["statement_idx"]), record["task_type"]) for record in records
    }
    if unknown_done:
        raise ValueError(f"Existing predictions are not in the current input: {sorted(unknown_done)[:10]}")

    load_kwargs = {"trust_remote_code": True, "revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **load_kwargs)
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1

    answer_provenance = answer_format_provenance(args.answer_format, task_types)
    if args.prompt_variant == "old-placeholder":
        answer_provenance = {
            "mode": "legacy-placeholder",
            "application": "Inference-time ablation only; source prompts are not modified.",
            "separator": "\n\n",
            "suffixes": {task: LEGACY_PLACEHOLDER_ANSWER_SUFFIXES[task] for task in task_types},
        }
    run_config = {
        "model_name": MODEL_NAME,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "condition": "image",
        "seed": args.seed,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "input_jsonl": {
            "path": str(args.input_jsonl.resolve()),
            "sha256": sha256_file(args.input_jsonl),
            "record_count": len(records),
            "task_counts": task_counts,
        },
        "graph_metadata": {
            "path": str(args.graph_metadata.resolve()),
            "sha256": sha256_file(args.graph_metadata),
            "record_count": len(metadata_by_idx),
        },
        "prompt": "DeepSeek prompt selected by prompt_variant",
        "prompt_variant": args.prompt_variant,
        "behavior_only": args.behavior_only,
        "answer_format": answer_provenance,
        "prompt_bodies_sha256": prompts_sha256(records, prompt_builder),
        "expected_split": args.expected_split,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "eval_mode": True,
        },
        "effective_max_new_tokens": MAX_NEW_TOKENS,
        "image_processing": {
            "base_size": BASE_SIZE,
            "image_size": IMAGE_SIZE,
            "crop_mode": CROP_MODE,
        },
        "dtype": "torch.bfloat16",
        "attention_implementation": "eager",
        "token_ceiling_detection": (
            "Approximate: response is re-encoded without special tokens; a count "
            ">= 8192 is marked as hitting the remote-code generation ceiling."
        ),
        "source_image_generation": {
            "command_verified_in_shell_history": (
                "python scripts/generate_graphvis_datasets.py --split test --start 0 "
                "--limit 500 --tasks-per-graph 6 --data-root "
                "data_preprocessed_release --out-dir outputs/graphvis_obqa --seed 13"
            ),
            "expected_renderer_and_pruning_settings": {
                "max_nodes": 18,
                "max_edges": 30,
                "max_degree": 5,
                "engine": "dot",
                "dpi": 200,
                "disconnected_rows": 3,
                "hide_relatedto_labels": False,
                "reveal_correct_answer": False,
            },
            "settings_provenance": (
                "Expected generator defaults; these settings are not independently "
                "encoded in graph_metadata_0_500.jsonl."
            ),
        },
    }
    if args.task_set != "paper":
        run_config["task_set"] = args.task_set
    write_run_config(args.output_dir, run_config)

    infer_output_dir = args.output_dir / "infer"
    infer_output_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    with ExitStack() as stack:
        handles = {
            task: stack.enter_context(path.open("a", encoding="utf-8"))
            for task, path in prediction_paths.items()
        }
        for position, record in enumerate(records, start=1):
            idx = int(record["statement_idx"])
            task = record["task_type"]
            require_task(task, task_types)
            key = (idx, task)
            if key in done:
                continue
            image_path = args.image_root / record["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing graph image: {image_path}")
            prompt = prompt_builder(record)
            generation_started = perf_counter()
            infer_response = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(infer_output_dir),
                save_results=False,
                eval_mode=True,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=CROP_MODE,
            )
            generation_elapsed_seconds = perf_counter() - generation_started
            infer_returned_none = infer_response is None
            infer_returned_empty_string = isinstance(infer_response, str) and infer_response == ""
            response = "" if infer_returned_none else str(infer_response).strip()
            generated_tokens = len(tokenizer.encode(response, add_special_tokens=False))
            hit_ceiling = generated_tokens >= MAX_NEW_TOKENS
            format_diagnostics = answer_format_diagnostics(
                task, response, args.answer_format
            )
            format_diagnostics["ceiling_hit_before_complete_answer"] = bool(
                args.answer_format == "constrained"
                and hit_ceiling
                and not format_diagnostics["final_format_compliant"]
            )
            result = {
                "statement_idx": idx,
                "task_type": task,
                "image": record["image"],
                "prompt": record["prompt"],
                "gold": record["answer"],
                **({"structured_gold": record["gold"]} if isinstance(record.get("gold"), dict) else {}),
                "raw_response": response,
                "model_infer_returned_none": infer_returned_none,
                "model_infer_returned_empty_string": infer_returned_empty_string,
                "normalized_response_empty": response == "",
                "generation_elapsed_seconds": generation_elapsed_seconds,
                "generated_token_count_approx": generated_tokens,
                "hit_token_ceiling": hit_ceiling,
                **format_diagnostics,
                **score_record(record, response, metadata_by_idx[idx]),
                "model_id": args.model_id,
                "model_revision": args.revision,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle = handles[task]
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
            generated_count += 1
            print(
                f"[{position}/{len(records)}] task={task} statement_idx={idx} "
                f"correct_raw={result['is_correct']} tokens~={generated_tokens} "
                f"ceiling={hit_ceiling} elapsed_seconds={generation_elapsed_seconds:.3f}",
                flush=True,
            )

    if len(done) != len(records):
        raise RuntimeError(f"Run ended with {len(done)}/{len(records)} completed predictions")
    print(f"Generated {generated_count} new predictions; {len(done)} total complete.", flush=True)
    if args.behavior_only:
        print("Behavior-only ablation complete; aggregate task scoring intentionally skipped.", flush=True)
    else:
        score_completed_run(
            args.input_jsonl, args.graph_metadata, args.output_dir, MODEL_NAME, args.task_set
        )


if __name__ == "__main__":
    main()
