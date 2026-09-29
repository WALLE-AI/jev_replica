"""Build RLCD training items from the downloaded typed-decisions dataset.

Reads data/typed-decisions/all/{train,test}-*.parquet (state/questions/gold as JSON strings,
schema documented in Laya项目深度技术解读报告.md), tokenizes each (state, question) pair with
`laya.common.build_sequence`, and writes a pickled list of training items plus a held-out
calibration slice -- mirroring notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb.
"""
import argparse
import json
import os
import pickle
import random

import pandas as pd
from transformers import AutoTokenizer

from laya.common import QTYPES, build_sequence

CAL_SEED = 20260922


def build_training_item(tok, state, q, gold_q, max_len, head_max_len):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        raise ValueError("unknown question type: %r" % t)

    qint = {"t": t, "ins": q["instructions"], "crit": crit}
    ids, markers = build_sequence(tok, state, qint, max_len=max_len, head_max_len=head_max_len)
    if len(markers) != len(target):
        return None  # option budget truncation collapsed some option spans; drop the item
    return {"ids": ids, "markers": markers, "qtype": QTYPES[t], "target": target}


def build_items_from_df(tok, df, max_len, head_max_len):
    items = []
    dropped = 0
    for _, row in df.iterrows():
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            gold_q = gold.get(qid)
            if gold_q is None:
                continue
            item = build_training_item(tok, state, q, gold_q, max_len, head_max_len)
            if item is None:
                dropped += 1
                continue
            item["workflow"] = row["workflow"]
            item["qid"] = qid
            item["case_id"] = row["id"]
            items.append(item)
    return items, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/typed-decisions/all")
    ap.add_argument("--encoder-dir", default="models/ModernBERT-large")
    ap.add_argument("--out-dir", default="repro/prepared")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--cal-fraction", type=float, default=0.10)
    ap.add_argument("--cal-max", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.encoder_dir)

    train_df = pd.read_parquet(os.path.join(args.data_dir, "train-00000-of-00001.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, "test-00000-of-00001.parquet"))
    print(f"raw cases: train={len(train_df)} test={len(test_df)}")

    train_items, dropped_train = build_items_from_df(tok, train_df, args.max_len, args.head_max_len)
    test_items, dropped_test = build_items_from_df(tok, test_df, args.max_len, args.head_max_len)
    print(f"train items: {len(train_items)} (dropped {dropped_train})")
    print(f"test items:  {len(test_items)} (dropped {dropped_test})")

    # Hold out a calibration slice BEFORE any weight update ever sees it -- temperature fitting
    # on trained-on items measures the fit, not the calibration (docs/finetune.md).
    rng = random.Random(CAL_SEED)
    idx = list(range(len(train_items)))
    rng.shuffle(idx)
    n_cal = min(args.cal_max, int(len(train_items) * args.cal_fraction))
    cal_idx = set(idx[:n_cal])
    cal_items = [train_items[i] for i in sorted(cal_idx)]
    fit_items = [train_items[i] for i in range(len(train_items)) if i not in cal_idx]
    print(f"train split: fit={len(fit_items)} calibration(held-out)={len(cal_items)}")

    with open(os.path.join(args.out_dir, "train_items.pkl"), "wb") as f:
        pickle.dump(fit_items, f)
    with open(os.path.join(args.out_dir, "cal_items.pkl"), "wb") as f:
        pickle.dump(cal_items, f)
    with open(os.path.join(args.out_dir, "test_items.pkl"), "wb") as f:
        pickle.dump(test_items, f)

    meta = {
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "encoder_dir": os.path.abspath(args.encoder_dir),
        "n_fit": len(fit_items),
        "n_cal": len(cal_items),
        "n_test": len(test_items),
        "cal_seed": CAL_SEED,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", args.out_dir)
    print(meta)


if __name__ == "__main__":
    main()
