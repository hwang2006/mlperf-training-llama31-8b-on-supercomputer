#!/usr/bin/env python3
"""Progress / throughput / eval curve / time-to-train of one or more runs.

  python3 tools/ttq_status.py results/runs/kv6-l8b-*.out
  python3 tools/ttq_status.py --summary results/runs/kv6-l8b-*.out   # one line per run
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllog import parse, ttt_min  # noqa: E402

args = sys.argv[1:]
summary = "--summary" in args
paths = [a for a in args if not a.startswith("--")]
if summary:
    print(f"{'file':45s} {'GBS':>4s} {'status':>8s} {'TTT[min]':>9s} {'samples':>8s} {'last_eval':>9s} {'samp/s':>8s} {'train/s':>8s} {'init[min]':>9s}")

for p in paths:
    r = parse(p)
    ttt = ttt_min(r)
    last = r["evals"][-1] if r["evals"] else (None, None, None)
    rate = None          # average incl. eval, from cumulative samples since run_start
    pts = [(t, s) for t, s, *_ in r["train"] if s] + [(t, s) for s, _, t in r["evals"] if s]
    if r["run_start"] and pts:
        t, s = max(pts, key=lambda p: (p[1], p[0]))   # furthest point in samples, not the latest record
        if t > r["run_start"]:
            rate = s / ((t - r["run_start"]) / 1000.0)
    steps = sorted(st for _, s, st, _ in r["train"] if st and s)   # skip step 0 (graph capture)
    step_rate = r["gbs"] / steps[len(steps) // 2] if steps and r["gbs"] else None
    init = (r["run_start"] - r["init_start"]) / 60000.0 if r["run_start"] and r["init_start"] else None
    if summary:
        f = lambda v, fmt: (fmt % v) if v is not None else "-"
        print(f"{os.path.basename(p)[:45]:45s} {f(r['gbs'],'%4d'):>4s} {str(r['status'] or 'running'):>8s} "
              f"{f(ttt,'%9.2f'):>9s} {f(last[0],'%8d'):>8s} {f(last[1],'%9.4f'):>9s} {f(rate,'%8.2f'):>8s} {f(step_rate,'%8.2f'):>8s} {f(init,'%9.1f'):>9s}")
        continue
    print(f"## {p}  (GBS={r['gbs']})  hp={r['hp']}")
    if init is not None:
        print(f"   init (init_start -> run_start): {init:.1f} min  (not part of TTT)")
    if rate:
        print(f"   avg throughput since run_start (incl. eval): {rate:.2f} samples/s")
    if step_rate:
        print(f"   train throughput (median step, excl. eval):  {step_rate:.2f} samples/s")
    tr = [x for x in r["train"] if x[1] and x[2] and x[3] is not None]
    if tr:
        t, s, st, loss = tr[-1]
        print(f"   last train: samples={s} step_time={st:.3f}s loss={loss:.4f}")
    if r["evals"]:
        print("   EVAL", [(s, round(v, 4)) for s, v, _ in r["evals"]])
        if last[1] and last[1] > 3.3 and rate:
            print("   target 3.3; Cisco H200 (GBS 32) converged at 172k-205k samples "
                  f"-> ~{max(0, 190874 - last[0]) / rate / 60:.0f} min left at current rate")
    if ttt is not None:
        print(f"   RUN_STOP status={r['status']}  TTT={ttt:.3f} min")
