#!/usr/bin/env python3
"""Regenerate docs/img/*.png from results/summary.csv:  python3 tools/plot_results.py"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(root, "results/summary.csv"))))
out = os.path.join(root, "docs/img"); os.makedirs(out, exist_ok=True)
ok = [r for r in rows if r["status"] == "success"]
f = lambda r, k: float(r[k])

def best(sel):
    b = {}
    for r in sel:
        n = int(r["nodes"])
        if n not in b or f(r, "ttt_min") < f(b[n], "ttt_min"): b[n] = r
    return dict(sorted(b.items()))

def finish(ax, path, title):
    ax.set_title(title); ax.grid(alpha=.3, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(out, path), dpi=150); plt.close()

# 1. best TTT per node count
b = best(ok)
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = [str(n) for n in b]; ys = [f(r, "ttt_min") for r in b.values()]
bars = ax.bar(xs, ys, color=["#c0392b" if r["job_id"] == "1198603" else "#2e86c1" for r in b.values()])
for bar, r in zip(bars, b.values()):
    ax.annotate(f'{f(r,"ttt_min"):.1f}\nGBS{r["gbs"]}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(10, 1500)
ax.set_xlabel("nodes (4x GH200 each)"); ax.set_ylabel("time to train [min], log scale")
finish(ax, "ttt_best_per_scale.png", "Best time-to-train per node count (target log ppl 3.3)")

# 2. GBS 32 training throughput vs linear scaling from 4 nodes
g32 = best([r for r in ok if r["gbs"] == "32"])
ref = f(g32[4], "train_samples_per_s") / 4
fig, ax = plt.subplots(figsize=(8, 4.2))
n = list(g32); ax.plot(n, [f(r, "train_samples_per_s") for r in g32.values()], "o-", label="measured")
ax.plot(n, [ref * k for k in n], "--", color="gray", label="linear (from 4 nodes)")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(n); ax.set_xticklabels(n)
ax.set_xlabel("nodes"); ax.set_ylabel("training samples/s"); ax.legend()
finish(ax, "throughput_gbs32.png", "GBS 32: training throughput")

# 3. GBS 32 vs 64 at 32-128 nodes
fig, ax = plt.subplots(figsize=(7, 4))
nodes = [32, 64, 128]; w = .38
g64 = best([r for r in ok if r["gbs"] == "64"])
for i, (lab, g) in enumerate([("GBS 32", g32), ("GBS 64", g64)]):
    vals = [f(g[k], "ttt_min") for k in nodes]
    xs = [j + (i - .5) * w for j in range(len(nodes))]
    ax.bar(xs, vals, w, label=lab)
    for x, v in zip(xs, vals): ax.annotate(f"{v:.1f}", (x, v), ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(nodes))); ax.set_xticklabels([f"{k} nodes" for k in nodes])
ax.set_ylabel("time to train [min]"); ax.legend()
finish(ax, "ttt_gbs32_vs_gbs64.png", "Larger global batch at 32-128 nodes")

# 4. 256-node throughput vs per-GPU micro-batches
t = {r["job_id"]: r for r in rows}
ids = ["1199551", "1199782", "1199866"]
fig, ax = plt.subplots(figsize=(7, 4))
labels = [f'GBS {t[i]["gbs"]}\n(MINIBS {t[i]["minibs"]})' for i in ids]
vals = [f(t[i], "train_samples_per_s") for i in ids]
ax.bar(labels, vals, color="#27ae60")
for j, v in enumerate(vals): ax.annotate(f"{v:.0f}", (j, v), ha="center", va="bottom")
ax.set_ylabel("training samples/s")
finish(ax, "throughput_256nodes.png", "256 nodes (1,024 GH200), TP1 CP4: throughput only")
print("wrote", sorted(os.listdir(out)))
