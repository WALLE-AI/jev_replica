#!/usr/bin/env python3
"""Build a fresh DecisionModel init-checkpoint bundle from a raw HF/local backbone.

Produces the exact directory layout train_unified_games.py's --init-checkpoint
expects (config.json, backbone_config/, tokenizer/, best.safetensors), with a
freshly-initialized decision head (LayerNorm+Linear[+attention set head]) on
top of the pretrained backbone weights. No decision-head training happens here.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_SCRIPTS = Path(__file__).resolve().parents[1] / "checkpoints/NanoJev-unified/source/scripts"
sys.path.insert(0, str(REPO_SCRIPTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local path or HF hub id for the base backbone")
    ap.add_argument("--set-head", choices=["none", "attention"], default="attention")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is non-empty")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "openjev_toy_trainer_for_init", REPO_SCRIPTS / "train_toy_decisions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DecisionModel = module.DecisionModel

    print(f"loading backbone from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    body_config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    backbone = AutoModel.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa",
        trust_remote_code=False, local_files_only=True,
    )
    model = DecisionModel(backbone, args.set_head)

    tokenizer.save_pretrained(out / "tokenizer")
    body_config.save_pretrained(out / "backbone_config")
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
        out / "best.safetensors",
    )
    config = {
        "model": args.model,
        "set_head": args.set_head,
        "max_length": 8192,
        "initialization": "fresh backbone + randomly initialized decision head; no training performed",
        "hidden_size": backbone.config.hidden_size,
        "num_hidden_layers": backbone.config.num_hidden_layers,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote fresh init checkpoint to {out}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total DecisionModel parameters: {n_params/1e9:.3f}B")


if __name__ == "__main__":
    main()
