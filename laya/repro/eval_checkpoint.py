"""Argmax accuracy of a trained (or base) checkpoint against the typed-decisions test split."""
import argparse
import json
import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import build_model, collate_items


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="repro/prepared")
    ap.add_argument("--checkpoint-dir", default=None, help="dir with model.safetensors + rl_agent_config.json; omit for zero-shot base encoder")
    ap.add_argument("--encoder-dir", default="models/ModernBERT-large")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    test_items = pickle.load(open(os.path.join(args.data_dir, "test_items.pkl"), "rb"))

    if args.checkpoint_dir:
        cfg = json.load(open(os.path.join(args.checkpoint_dir, "rl_agent_config.json")))
        model = build_model(cfg, pretrained=True)
        sd = load_file(os.path.join(args.checkpoint_dir, "model.safetensors"))
        model.load_state_dict(sd, strict=True)
        tok_dir = cfg["encoder"]
        label = args.checkpoint_dir
    else:
        cfg = {"encoder": args.encoder_dir, "head_layers": 2, "act_costs": {"act": 0.0}}
        model = build_model(cfg, pretrained=True)  # random head, pretrained encoder: zero-shot
        tok_dir = args.encoder_dir
        label = "zero-shot base encoder (random head)"

    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(tok_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    collate_fn = lambda batch: collate_items([[it] for it in batch], pad_id)
    loader = DataLoader(ItemDataset(test_items), batch_size=32, shuffle=False, collate_fn=collate_fn)

    correct, total = 0, 0
    correct_by_type, total_by_type = {}, {}
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(batch_gpu["input_ids"], batch_gpu["attention_mask"],
                                   batch_gpu["marker_pos"], batch_gpu["marker_mask"], batch_gpu["qtype"])
            logits = logits.float().masked_fill(~batch_gpu["marker_mask"], -1e9)
            pred = logits.argmax(-1)
            gold = batch_gpu["target"].argmax(-1)
            ok = (pred == gold)
            correct += ok.sum().item()
            total += ok.numel()
            for qt in batch_gpu["qtype"].unique().tolist():
                sel = batch_gpu["qtype"] == qt
                correct_by_type[qt] = correct_by_type.get(qt, 0) + ok[sel].sum().item()
                total_by_type[qt] = total_by_type.get(qt, 0) + sel.sum().item()

    print(f"=== {label} ===")
    print(f"overall argmax accuracy: {correct/total:.4f} ({correct}/{total})")
    from laya.common import QTYPE_NAMES
    for qt, tot in total_by_type.items():
        print(f"  {QTYPE_NAMES[qt]}: {correct_by_type[qt]/tot:.4f} ({correct_by_type[qt]}/{tot})")


if __name__ == "__main__":
    main()
