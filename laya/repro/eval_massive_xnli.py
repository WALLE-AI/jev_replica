"""Evaluate a trained repro checkpoint on the downloaded MASSIVE-intent / XNLI data,
offline, using the same prompt format / sampling as research/eval/laya_eval.py
(seed 13, n_opts, first-K rows) but reading the locally downloaded files directly
instead of going through datasets.load_dataset (avoids network dependency).

NOTE: our checkpoints are fine-tuned on typed-decisions (four agent/business
workflows), not on MASSIVE/XNLI -- this is a zero-shot cross-task check of the
eval harness plumbing + the base encoder's own capability, not a like-for-like
comparison with the shipped laya / laya-multilingual base checkpoints.
"""
import argparse
import gzip
import json
import os
import random

import numpy as np
import pandas as pd

import laya
from laya.common import ece_score

SEED = 13
NLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
NLI_CRIT = {
    "entailment": "the premise implies the hypothesis is true",
    "neutral": "the premise neither implies nor contradicts the hypothesis",
    "contradiction": "the premise implies the hypothesis is false",
}


def render_label(label):
    return label.replace("_", " ").replace(".", ": ")


def eval_massive(agent, lang, per_lang, n_opts, data_dir):
    path = os.path.join(data_dir, "test", f"{lang}.json.gz")
    with gzip.open(path) as f:
        rows = [json.loads(l) for l in f][:per_lang]
    all_labels = sorted({r["label_text"] for r in rows})
    rng = random.Random(SEED)

    conf, correct = [], []
    for r in rows:
        gold = r["label_text"]
        pool = [l for l in all_labels if l != gold]
        distractors = rng.sample(pool, min(n_opts - 1, len(pool)))
        options = distractors + [gold]
        rng.shuffle(options)
        criteria = {o: render_label(o) for o in options}
        questions = {"intent": {"type": "choice",
                                 "instructions": "What is the user asking for in `utterance`?",
                                 "criteria": criteria}}
        result = agent.predict({"utterance": r["text"]}, questions)
        ans = result["answers"]["intent"]
        conf.append(ans["answer_confidence"])
        correct.append(1.0 if ans["choice"] == gold else 0.0)
    conf, correct = np.array(conf), np.array(correct)
    acc = correct.mean()
    ece = ece_score(conf, correct, bins=15)
    return {"n": len(rows), "accuracy": float(acc), "ece": float(ece),
            "mean_confidence": float(conf.mean())}


def eval_xnli(agent, lang, per_lang, data_dir):
    path = os.path.join(data_dir, lang, "test-00000-of-00001.parquet")
    df = pd.read_parquet(path).iloc[:per_lang]

    conf, correct = [], []
    for _, row in df.iterrows():
        gold = NLI_LABELS[int(row["label"])]
        questions = {"relation": {
            "type": "choice",
            "instructions": "What is the relationship between `premise` and `hypothesis`?",
            "criteria": NLI_CRIT,
        }}
        result = agent.predict({"premise": row["premise"], "hypothesis": row["hypothesis"]}, questions)
        ans = result["answers"]["relation"]
        conf.append(ans["answer_confidence"])
        correct.append(1.0 if ans["choice"] == gold else 0.0)
    conf, correct = np.array(conf), np.array(correct)
    acc = correct.mean()
    ece = ece_score(conf, correct, bins=15)
    return {"n": len(df), "accuracy": float(acc), "ece": float(ece),
            "mean_confidence": float(conf.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--massive-langs", default="en,de,fr,es,zh-CN,hi,ar,sw")
    ap.add_argument("--xnli-langs", default="en,de,fr,es,zh,hi,ar,sw")
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--n-opts", type=int, default=20)
    ap.add_argument("--massive-dir", default="data/massive_intent")
    ap.add_argument("--xnli-dir", default="data/xnli")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agent = laya.load(args.checkpoint_dir, device="cuda")
    report = {"config": vars(args), "massive": {}, "xnli": {}}

    print(f"=== MASSIVE intent ({args.checkpoint_dir}) ===")
    for lang in args.massive_langs.split(","):
        r = eval_massive(agent, lang, args.per_lang, args.n_opts, args.massive_dir)
        report["massive"][lang] = r
        print(f"  {lang:6s} n={r['n']:4d} acc={r['accuracy']:.4f} ece={r['ece']:.4f} "
              f"mean_conf={r['mean_confidence']:.4f}")

    print(f"=== XNLI ({args.checkpoint_dir}) ===")
    for lang in args.xnli_langs.split(","):
        r = eval_xnli(agent, lang, args.per_lang, args.xnli_dir)
        report["xnli"][lang] = r
        print(f"  {lang:6s} n={r['n']:4d} acc={r['accuracy']:.4f} ece={r['ece']:.4f} "
              f"mean_conf={r['mean_confidence']:.4f}")

    massive_accs = [v["accuracy"] for v in report["massive"].values()]
    xnli_accs = [v["accuracy"] for v in report["xnli"].values()]
    print(f"MASSIVE macro-accuracy: {sum(massive_accs)/len(massive_accs):.4f}")
    print(f"XNLI macro-accuracy:    {sum(xnli_accs)/len(xnli_accs):.4f}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
