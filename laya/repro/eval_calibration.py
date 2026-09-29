"""Compare ECE before vs after temperature calibration on the typed-decisions test split."""
import json
import os
import pickle

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from laya.common import build_model, collate_items, ece_score, temp_bucket


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    device = torch.device("cuda:0")
    ckpt_dir = "repro/checkpoints/english/final"
    cfg = json.load(open(os.path.join(ckpt_dir, "rl_agent_config.json")))
    test_items = pickle.load(open("repro/prepared/test_items.pkl", "rb"))

    model = build_model(cfg, pretrained=True)
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    collate_fn = lambda batch: collate_items([[it] for it in batch], pad_id)
    loader = DataLoader(ItemDataset(test_items), batch_size=32, shuffle=False, collate_fn=collate_fn)

    raw_logits, targets, masks, qtypes = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
            logits = logits.float().cpu()
            for i in range(logits.size(0)):
                k = int(batch["marker_mask"][i].sum().item())
                raw_logits.append(logits[i, :k])
                targets.append(batch["target"][i, :k])
                masks.append(batch["marker_mask"][i, :k])
                qtypes.append(int(batch["qtype"][i].item()))

    temperature_by_options = cfg.get("temperature_by_options", {})
    per_type_temp = cfg.get("temperature", [1.0, 1.0, 1.0])

    def compute(use_temp):
        conf, correct = [], []
        for logit, target, mask, qt in zip(raw_logits, targets, masks, qtypes):
            k = mask.sum().item()
            if use_temp:
                bucket = temp_bucket(qt, k)
                t = temperature_by_options.get(bucket, per_type_temp[qt])
            else:
                t = 1.0
            p = torch.softmax(logit / t, -1).numpy()
            gold = int(target.argmax().item())
            pred = int(p.argmax())
            conf.append(float(p.max()))
            correct.append(1.0 if pred == gold else 0.0)
        conf, correct = np.array(conf), np.array(correct)
        acc = correct.mean()
        ece = ece_score(conf, correct, bins=15)
        return acc, ece, conf.mean()

    acc_raw, ece_raw, mconf_raw = compute(use_temp=False)
    acc_cal, ece_cal, mconf_cal = compute(use_temp=True)

    print(f"Uncalibrated (T=1.0):    accuracy={acc_raw:.4f}  mean_confidence={mconf_raw:.4f}  ECE={ece_raw:.4f}")
    print(f"Calibrated (fitted T):   accuracy={acc_cal:.4f}  mean_confidence={mconf_cal:.4f}  ECE={ece_cal:.4f}")
    print(f"Temperatures used: per_type={per_type_temp}")
    print(f"                   by_options={temperature_by_options}")


if __name__ == "__main__":
    main()
