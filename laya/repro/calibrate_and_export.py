"""Resume from a saved training checkpoint_latest and run only the calibration + export step
(the RLCD training itself already completed -- this just re-runs the part that crashed).
"""
import argparse
import json
import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from laya.common import QTYPES, build_model, collate_items, temp_bucket, clamp_temperature


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def fit_one_temp(logits, target, mask, device):
    if logits.shape[0] < 10:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        z = logits / log_t.exp()
        z = z.masked_fill(~mask, -1e4)
        loss = -(target * torch.log_softmax(z, -1)).sum(-1).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
        return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())
    except Exception as e:
        print("temperature fit failed, falling back to 1.2:", e)
        return 1.2


def pad_stack(rows, pad_value):
    width = max(r.shape[0] for r in rows)
    padded = [F.pad(r, (0, width - r.shape[0]), value=pad_value) for r in rows]
    return torch.stack(padded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="repro/prepared")
    ap.add_argument("--checkpoint-dir", default="repro/checkpoints/english/checkpoint_latest")
    ap.add_argument("--out-dir", default="repro/checkpoints/english")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cfg = json.load(open(os.path.join(args.checkpoint_dir, "rl_agent_config.json")))
    cal_items = pickle.load(open(os.path.join(args.data_dir, "cal_items.pkl"), "rb"))

    model = build_model(cfg, pretrained=True)
    sd = torch.load(os.path.join(args.checkpoint_dir, "model_state.pt"), map_location="cpu")
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"loaded checkpoint from {args.checkpoint_dir}, epoch={cfg.get('epoch')}")

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    cal_collate = lambda batch: collate_items([[it] for it in batch], pad_id)
    cal_loader = DataLoader(ItemDataset(cal_items), batch_size=16, shuffle=False, collate_fn=cal_collate)

    buckets = {}
    with torch.no_grad():
        for batch in cal_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(batch["input_ids"], batch["attention_mask"],
                                   batch["marker_pos"], batch["marker_mask"], batch["qtype"])
            logits = logits.float()
            for i in range(logits.size(0)):
                kq = int(batch["marker_mask"][i].sum().item())
                qt = int(batch["qtype"][i].item())
                bucket = temp_bucket(qt, kq)
                buckets.setdefault(bucket, {"logits": [], "target": [], "mask": []})
                buckets[bucket]["logits"].append(logits[i, :kq].cpu())
                buckets[bucket]["target"].append(batch["target"][i, :kq].cpu())
                buckets[bucket]["mask"].append(batch["marker_mask"][i, :kq].cpu())

    temperature_by_options = {}
    for bucket, d in buckets.items():
        L = pad_stack(d["logits"], -1e4).to(device)
        T = pad_stack(d["target"], 0.0).to(device)
        M = pad_stack(d["mask"], False).to(device)
        t = fit_one_temp(L, T, M, device)
        temperature_by_options[bucket] = clamp_temperature(t)
        print(f"bucket {bucket}: n={L.shape[0]} fitted_temp={t:.3f} clamped={temperature_by_options[bucket]:.3f}")

    fitted_temps = [1.2, 1.2, 1.2]
    for qt_name, qt in QTYPES.items():
        sel_logits, sel_target, sel_mask = [], [], []
        for bucket, d in buckets.items():
            if bucket.startswith(qt_name + ":"):
                sel_logits.extend(d["logits"])
                sel_target.extend(d["target"])
                sel_mask.extend(d["mask"])
        if sel_logits:
            L = pad_stack(sel_logits, -1e4).to(device)
            T = pad_stack(sel_target, 0.0).to(device)
            M = pad_stack(sel_mask, False).to(device)
            fitted_temps[qt] = fit_one_temp(L, T, M, device)

    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    from safetensors.torch import save_file
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(final_dir, "model.safetensors"))

    final_cfg = {
        **cfg,
        "fine_tuned": True,
        "temperature": [clamp_temperature(t) for t in fitted_temps],
        "temperature_by_options": temperature_by_options,
    }
    final_cfg.pop("epoch", None)
    json.dump(final_cfg, open(os.path.join(final_dir, "rl_agent_config.json"), "w"), indent=2)
    print("wrote final checkpoint to", final_dir)
    print("per-type temperatures:", final_cfg["temperature"])
    print("temperature_by_options:", final_cfg["temperature_by_options"])


if __name__ == "__main__":
    main()
