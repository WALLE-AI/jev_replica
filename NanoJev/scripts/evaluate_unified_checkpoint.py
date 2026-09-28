#!/usr/bin/env python3
"""Evaluate a local unified DecisionModel without optimization or calibration.

--stage selects the metric population weights, not the checkpoint's training
history. Use --stage critic to compare an existing SFT checkpoint's Boolean
predictions with critic runs on the same frozen outcome dataset. Every requested
split retains both policy and outcome rows, as in the trainer's final evaluation.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from predict_toy_decisions import local_checkpoint_files
from train_pipeline_decisions import SPLITS, dump, pack_complete_questions
from train_unified_games import (
    evaluate_unified, file_sha256, population_weights, prepare_unified_examples,
    read_unified_dataset,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Frozen five-split dataset directory")
    parser.add_argument("--checkpoint", required=True, help="Complete local checkpoint bundle")
    parser.add_argument("--output-dir", required=True, help="New directory; never overwrite an earlier evaluation")
    parser.add_argument("--stage", choices=("sft", "critic"), required=True,
                        help="Population weighting scheme; does not train or select a checkpoint")
    parser.add_argument("--splits", default="dev,calibration,test,ood", help="Comma-separated split names")
    parser.add_argument("--balance", choices=("task", "task_role"), default="task")
    parser.add_argument("--retention-fraction", type=float, default=.25)
    parser.add_argument("--microbatch-questions", type=int, default=8)
    parser.add_argument("--max-microbatch-tokens", type=int, default=32768)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--disable-native-triton", action="store_true")
    args = parser.parse_args(argv)
    args.splits = [part.strip() for part in args.splits.split(",")]
    if not args.splits or any(split not in SPLITS for split in args.splits):
        parser.error("--splits must contain names from train,dev,calibration,test,ood")
    if len(args.splits) != len(set(args.splits)):
        parser.error("--splits must not contain duplicates")
    if min(args.microbatch_questions, args.max_length) <= 0 or args.max_microbatch_tokens < 0:
        parser.error("Question/max-length limits must be positive; token budget must be nonnegative")
    try:
        population_weights(args.stage, args.balance, args.retention_fraction)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise ValueError("--output-dir must be a new directory, including when an existing directory is empty")
    # These checks are stdlib-only and run before importing Torch or allocating GPU memory.
    root, _ = local_checkpoint_files(args.checkpoint)
    records, manifest, files, schema_audit = read_unified_dataset(args.input)
    data_hashes = {path.name: file_sha256(path) for path in files}
    for split, expected in manifest.get("split_sha256", {}).items():
        if split not in SPLITS or data_hashes.get(f"{split}.jsonl") != expected:
            raise ValueError(f"Dataset file hash disagrees with manifest: {split}")
    selected = [row for row in records if row["split"] in args.splits]
    if any(not any(row["split"] == split for row in selected) for split in args.splits):
        raise ValueError("Every requested split must contain records")
    weights = population_weights(args.stage, args.balance, args.retention_fraction)
    bundle_files = [root / "config.json", root / "best.safetensors"]
    for directory in ("tokenizer", "backbone_config"):
        bundle_files.extend(path for path in sorted((root / directory).rglob("*")) if path.is_file())
    bundle_hashes = {str(path.relative_to(root)): file_sha256(path) for path in bundle_files}

    import torch
    from predict_toy_decisions import DecisionPredictor
    runtime = DecisionPredictor(root, max_length=args.max_length, precision=args.precision,
                                disable_native_triton=args.disable_native_triton)
    runtime.model.requires_grad_(False)
    examples, target_audit = prepare_unified_examples(selected, runtime.tokenizer, args.max_length)
    split_examples = {split: [ex for ex in examples if ex["split"] == split] for split in args.splits}
    # Preflight complete question groups before producing any prediction file.
    for values in split_examples.values():
        pack_complete_questions(values, args.microbatch_questions, args.max_microbatch_tokens)
    out.mkdir(parents=True, exist_ok=False)
    source_files = ("evaluate_unified_checkpoint.py", "train_unified_games.py",
                    "train_pipeline_decisions.py", "predict_toy_decisions.py", "train_toy_decisions.py")
    config = {**vars(args), "schema_version": "nanojev-unified-checkpoint-evaluation-v1",
              "stage": "evaluation", "evaluation_weighting_stage": args.stage,
              "completed_steps": 0, "temperature": 1.0, "temperature_fitted": False,
              "model": runtime.run_config.get("model"), "set_head": runtime.run_config["set_head"],
              "data_sha256": data_hashes, "checkpoint_file_sha256": bundle_hashes,
              "weights_sha256": bundle_hashes["best.safetensors"],
              "implementation_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in source_files},
              "continuation_policy_id": schema_audit["continuation_policy_id"],
              "population_weights": {"/".join(cell): weight for cell, weight in weights.items()},
              "schema_counts": schema_audit, "parameter_storage": "float32",
              "forward_autocast": args.precision,
              "deps": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
              "gpu": torch.cuda.get_device_name(0),
              "selection": "none; evaluate the supplied checkpoint without parameter updates",
              "frozen_continuation_policy_is_updated_by_this_script": False}
    dump(out / "config.json", config)
    dump(out / "target_audit.json", target_audit)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = {}
    for split in args.splits:
        metrics[split] = evaluate_unified(runtime.model, split_examples[split],
                                         runtime.tokenizer.pad_token_id, args, weights,
                                         out / f"predictions_{split}.jsonl", require_all=split == "dev")
        print(json.dumps({"split": split, "metrics": metrics[split]}, allow_nan=False), flush=True)
    torch.cuda.synchronize()
    summary = {"stage": "evaluation", "evaluation_weighting_stage": args.stage,
               "completed_steps": 0, "selected_on": "none; supplied checkpoint",
               "metrics_by_split": metrics, "weights_sha256": bundle_hashes["best.safetensors"],
               "continuation_policy_id": schema_audit["continuation_policy_id"],
               "temperature": 1.0, "temperature_fitted": False,
               "evaluation_seconds": time.perf_counter() - started,
               "max_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
               "config_sha256": file_sha256(out / "config.json"),
               "target_audit_sha256": file_sha256(out / "target_audit.json"),
               "prediction_sha256": {split: file_sha256(out / f"predictions_{split}.jsonl") for split in args.splits},
               "finished": True}
    dump(out / "summary.json", summary)
    print(json.dumps({"done": str(out), **summary}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
