"""RLCD fine-tuning loop: GRPO-style policy gradient over a strictly-proper-scoring-rule
reward, plus a full-weight soft cross-entropy anchor -- mirrors
notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb's train_ddp.py cell.

Launch with:
  CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 repro/train_ddp.py \
      --data-dir repro/prepared --encoder-dir models/ModernBERT-large --out-dir repro/checkpoints/english
"""
import argparse
import json
import os
import pickle
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from laya.common import QTYPES, build_model, collate_items, proper_reward, temp_bucket, clamp_temperature

GROUP_SIZE = 4
SIGMA_START, SIGMA_END = 0.4, 0.1


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world_size, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank):
    return rank == 0


def fit_one_temp(logits, target, mask, device):
    """LBFGS fit of a single temperature scalar in log-space against held-out soft targets."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="repro/prepared")
    ap.add_argument("--encoder-dir", default="models/ModernBERT-large")
    ap.add_argument("--out-dir", default="repro/checkpoints/english")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--head-layers", type=int, default=2)
    ap.add_argument("--head-checkpointing", action="store_true", default=True)
    args = ap.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    max_len, head_max_len = meta["max_len"], meta["head_max_len"]

    tok = AutoTokenizer.from_pretrained(args.encoder_dir)
    train_items = pickle.load(open(os.path.join(args.data_dir, "train_items.pkl"), "rb"))
    cal_items = pickle.load(open(os.path.join(args.data_dir, "cal_items.pkl"), "rb"))

    if is_main(rank):
        print(f"rank0: {len(train_items)} fit items, {len(cal_items)} calibration items, "
              f"world_size={world_size}")

    ds = ItemDataset(train_items)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=13)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    collate_fn = lambda batch: collate_items([[it] for it in batch], pad_id)
    loader = DataLoader(ds, batch_size=args.micro_batch, sampler=sampler, collate_fn=collate_fn,
                         num_workers=2, drop_last=True)

    cfg = {
        "encoder": os.path.abspath(args.encoder_dir),
        "head_layers": args.head_layers,
        "act_costs": {"act": 0.0},
        "max_len": max_len,
        "head_max_len": head_max_len,
        "amp_dtype": "fp16",
    }
    # No `encoder_dir=` here: passing it makes build_model() take the "load architecture only,
    # caller loads weights separately" branch (AutoModel.from_config + skipped init), which is for
    # loading an already-fine-tuned Laya checkpoint's *own* encoder subfolder, not for training
    # from a base encoder. cfg["encoder"] is a local path, so the pretrained branch's
    # AutoModel.from_pretrained(cfg["encoder"]) loads the real weights correctly.
    model = build_model(cfg, pretrained=True)
    model.head_checkpointing = args.head_checkpointing
    model.encoder.gradient_checkpointing_enable()
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    encoder_params = list(model.module.encoder.parameters())
    head_params = [p for n, p in model.module.named_parameters() if not n.startswith("encoder.")]
    optimizer = AdamW([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    global_step = 0
    micro_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss, running_n = 0.0, 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            mask = batch["marker_mask"]
            target = batch["target"]
            progress = global_step / total_steps
            sigma = SIGMA_START - (SIGMA_START - SIGMA_END) * progress

            with torch.autocast("cuda", dtype=torch.float16):
                logits, act_logits = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["marker_pos"], batch["marker_mask"], batch["qtype"],
                )
            logits_f = logits.float()

            k = mask.sum(-1, keepdim=True).clamp(min=1).float()
            eps = torch.randn((GROUP_SIZE,) + logits_f.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits_f.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"], mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits_f.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * F.log_softmax(logits_f.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / args.grad_accum + 0.0 * act_logits.sum()

            scaler.scale(loss).backward()
            running_loss += float(loss_ce.detach()) * target.size(0)
            running_n += target.size(0)
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if is_main(rank) and global_step % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"ce_loss={running_loss/max(1,running_n):.4f} sigma={sigma:.3f} "
                          f"lr={scheduler.get_last_lr()} elapsed={elapsed:.0f}s", flush=True)
                    running_loss, running_n = 0.0, 0

        if is_main(rank):
            ckpt_dir = os.path.join(args.out_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_dir, "model_state.pt"))
            json.dump({**cfg, "epoch": epoch}, open(os.path.join(ckpt_dir, "rl_agent_config.json"), "w"), indent=2)
            print(f"epoch {epoch} done, checkpoint saved to {ckpt_dir}", flush=True)
        if world_size > 1:
            dist.barrier()

    # --- Calibration: fit temperature per (question type, option count) bucket on the
    # never-trained-on calibration slice, rank0 only. ---
    if is_main(rank):
        model.eval()
        cal_collate = lambda batch: collate_items([[it] for it in batch], pad_id)
        cal_loader = DataLoader(ItemDataset(cal_items), batch_size=16, shuffle=False, collate_fn=cal_collate)
        buckets = {}  # bucket -> (logits list, target list, mask list)
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
                    # Slice to this item's real option count: different mini-batches pad to
                    # different widths (kmax varies per DataLoader batch), so storing the
                    # already-batch-padded row would make later torch.stack() fail across
                    # batches with different padding widths, even within the same bucket.
                    buckets[bucket]["logits"].append(logits[i, :kq].cpu())
                    buckets[bucket]["target"].append(batch["target"][i, :kq].cpu())
                    buckets[bucket]["mask"].append(batch["marker_mask"][i, :kq].cpu())

        def pad_stack(rows, pad_value):
            width = max(r.shape[0] for r in rows)
            padded = [F.pad(r, (0, width - r.shape[0]), value=pad_value) for r in rows]
            return torch.stack(padded)

        temperature_by_options = {}
        for bucket, d in buckets.items():
            # A size bucket (e.g. "3-5") pools items with different real option counts, so pad
            # every row to the bucket's own max width before stacking.
            L = pad_stack(d["logits"], -1e4).to(device)
            T = pad_stack(d["target"], 0.0).to(device)
            M = pad_stack(d["mask"], False).to(device)
            t = fit_one_temp(L, T, M, device)
            temperature_by_options[bucket] = clamp_temperature(t)
            print(f"bucket {bucket}: n={L.shape[0]} fitted_temp={t:.3f} clamped={temperature_by_options[bucket]:.3f}")

        # Per-type fallback temperature (three QTYPES), also fit on the same slice.
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
        state_dict = {k: v.contiguous() for k, v in model.module.state_dict().items()}
        save_file(state_dict, os.path.join(final_dir, "model.safetensors"))

        final_cfg = {
            **cfg,
            "fine_tuned": True,
            "temperature": [clamp_temperature(t) for t in fitted_temps],
            "temperature_by_options": temperature_by_options,
        }
        json.dump(final_cfg, open(os.path.join(final_dir, "rl_agent_config.json"), "w"), indent=2)
        print("wrote final checkpoint to", final_dir)
        print("per-type temperatures:", final_cfg["temperature"])

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
