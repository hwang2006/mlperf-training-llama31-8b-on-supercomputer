#!/usr/bin/env python3
"""Recompute the official Cisco MLPerf Training v6.0 Llama 3.1 8B score on 2x8xH200
(16 Hopper GPUs, GBS 32) from the result logs fetched by scripts/00_fetch_sources.sh.

  source env.sh && python3 tools/baseline_cisco.py "$CISCO_RESULTS"
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllog import olympic_mean, parse, ttt_min  # noqa: E402

d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CISCO_RESULTS", ".")
runs = [parse(p) for p in sorted(glob.glob(os.path.join(d, "result_*.txt")))]
ttts, samples = [], []
for r in runs:
    t = ttt_min(r)
    s = r["evals"][-1][0] if r["evals"] else None
    rate = s / (t * 60) if t and s else None
    print(f"{os.path.basename(r['path']):14s} GBS={r['gbs']} status={r['status']} "
          f"TTT={t:7.2f} min  samples={s}  {rate:.2f} samples/s (incl. eval)")
    if r["status"] == "success":
        ttts.append(t)
        samples.append(s)
if ttts:
    hp = runs[0]["hp"]
    print(f"\nScore (mean w/o fastest+slowest of {len(ttts)}): {olympic_mean(ttts):.2f} min")
    print(f"Samples to target: min {min(samples)}, max {max(samples)}, mean {sum(samples)/len(samples):.0f}")
    print(f"LR={hp.get('opt_base_learning_rate')} warmup={hp.get('opt_learning_rate_warmup_steps')} "
          f"eval every {runs[0]['evals'][1][0]-runs[0]['evals'][0][0]} samples")
    per_gpu = sum(samples) / sum(t * 60 for t in ttts) / 16
    print(f"~{per_gpu:.2f} samples/s per H200 (incl. eval) -> GH200 TTT estimate @GBS32, ~{sum(samples)/len(samples):.0f} samples:")
    for n in (1, 2, 4, 8, 16, 32):
        print(f"   {n:3d} nodes ({4*n:4d} GPUs): {sum(samples)/len(samples)/(per_gpu*4*n)/60:7.1f} min  (linear, upper bound on speed)")
