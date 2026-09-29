"""Latency/throughput benchmark for the repro checkpoints, mirroring the methodology
in BENCHMARKS.md (single-question latency, batched per-question latency, predict_batch
throughput) -- the repo's own benchmarks/research/scripts/bench_*.py are hardwired to
the official `convaiinnovations/laya` Hub bundle (subfolder english/multilingual/
typed-decisions), so this is a small standalone script pointed at our local checkpoints
instead of patching those.
"""
import argparse
import statistics
import time

import laya

STATE = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}
ONE_QUESTION = {"department": QUESTIONS["department"]}


def median_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--batch-sizes", default="1,10,32,64")
    args = ap.parse_args()

    agent = laya.load(args.checkpoint_dir, device=args.device)
    print(f"=== {args.checkpoint_dir} (device={args.device}, amp_dtype={agent.cfg.get('amp_dtype')}) ===")

    # Warm up: first call pays kernel/cuDNN autotune + tokenizer JIT.
    agent.predict(STATE, ONE_QUESTION)
    agent.predict(STATE, QUESTIONS)

    t_one_q = median_ms(lambda: agent.predict(STATE, ONE_QUESTION), args.repeats)
    print(f"1 question, single request:       {t_one_q:.2f} ms (median of {args.repeats})")

    t_four_q = median_ms(lambda: agent.predict(STATE, QUESTIONS), args.repeats)
    print(f"4 questions, single request:       {t_four_q:.2f} ms  ({t_four_q/4:.2f} ms/question)")

    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        states = [STATE] * bs

        def _loop():
            for s in states:
                agent.predict(s, ONE_QUESTION)

        t_loop = median_ms(_loop, max(3, args.repeats // 5))

        def _batch():
            agent.predict_batch(states, ONE_QUESTION)

        t_batch = median_ms(_batch, max(3, args.repeats // 5))
        speedup = t_loop / t_batch if t_batch > 0 else float("nan")
        print(f"batch={bs:3d}: loop={t_loop:8.2f} ms ({t_loop/bs:.2f} ms/req)  "
              f"predict_batch={t_batch:8.2f} ms ({t_batch/bs:.2f} ms/req)  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
