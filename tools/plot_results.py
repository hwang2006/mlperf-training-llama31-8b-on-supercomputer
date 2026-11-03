#!/usr/bin/env python3
"""Regenerate docs/img/*.png from results/summary.csv and results/eval_curves.csv.

  python3 tools/plot_results.py
TTT plots use only runs with the MLPerf evaluation schedule (eval_schedule == mlperf).
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(root, "results/summary.csv"))))
curves = list(csv.DictReader(open(os.path.join(root, "results/eval_curves.csv"))))
out = os.path.join(root, "docs/img"); os.makedirs(out, exist_ok=True)
f = lambda r, k: float(r[k])
ok = [r for r in rows if r["status"] == "success" and r["eval_schedule"] == "mlperf"]
thr = [r for r in rows if r["train_samples_per_s"]]
BLUE, ORANGE = "#2a78d6", "#eb6834"
GBS_COLORS = {"32": "#2a78d6", "64": "#eb6834", "128": "#1baf7a", "256": "#6250d6",
              "512": "#e87ba4", "1024": "#e34948"}

def best(sel, key="ttt_min", low=True):
    b = {}
    for r in sel:
        n = int(r["nodes"])
        if n not in b or (f(r, key) < f(b[n], key) if low else f(r, key) > f(b[n], key)):
            b[n] = r
    return dict(sorted(b.items()))

def save(fig, ax, name, title):
    ax.set_title(title); plt.tight_layout(); fig.savefig(os.path.join(out, name), dpi=150); plt.close(fig)

# 1. best TTT per node count
b = best(ok)
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar([str(n) for n in b], [f(r, "ttt_min") for r in b.values()],
              color=[ORANGE if r is min(b.values(), key=lambda r: f(r, "ttt_min")) else BLUE for r in b.values()])
for bar, r in zip(bars, b.values()):
    ax.annotate(f'{f(r,"ttt_min"):.1f}\nGBS{r["gbs"]}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(10, 1500); ax.grid(alpha=.3, which="both", axis="y")
ax.set_xlabel("nodes (4x GH200 each)"); ax.set_ylabel("time to train [min], log scale")
save(fig, ax, "ttt_best_per_scale.png", "Best time-to-train per node count (target log ppl 3.3)")

# 2. GBS 32 training throughput vs linear scaling from 4 nodes
g32 = best([r for r in ok if r["gbs"] == "32"])
ref = f(g32[4], "train_samples_per_s") / 4
fig, ax = plt.subplots(figsize=(8, 4.2))
n = list(g32); ax.plot(n, [f(r, "train_samples_per_s") for r in g32.values()], "o-", color=BLUE, label="measured")
ax.plot(n, [ref * k for k in n], "--", color="gray", label="linear (from 4 nodes)")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(n); ax.set_xticklabels(n)
ax.grid(alpha=.3, which="both"); ax.set_xlabel("nodes"); ax.set_ylabel("training samples/s"); ax.legend()
save(fig, ax, "throughput_gbs32.png", "GBS 32: training throughput")

# 3. GBS 32 vs 64 at 32-128 nodes
g64 = best([r for r in ok if r["gbs"] == "64"])
fig, ax = plt.subplots(figsize=(7, 4)); nodes = [32, 64, 128]; w = .38
for i, (lab, g, c) in enumerate([("GBS 32", g32, BLUE), ("GBS 64", g64, ORANGE)]):
    vals = [f(g[k], "ttt_min") for k in nodes]; xs = [j + (i - .5) * w for j in range(len(nodes))]
    ax.bar(xs, vals, w, label=lab, color=c)
    for x, v in zip(xs, vals): ax.annotate(f"{v:.1f}", (x, v), ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(nodes))); ax.set_xticklabels([f"{k} nodes" for k in nodes])
ax.set_ylabel("time to train [min]"); ax.legend(); ax.grid(alpha=.3, axis="y")
save(fig, ax, "ttt_gbs32_vs_gbs64.png", "Larger global batch at 32-128 nodes")

# 4. 256-node throughput vs per-GPU micro-batches
t = {r["job_id"]: r for r in rows}; ids = ["1199551", "1199782", "1199866"]
fig, ax = plt.subplots(figsize=(7, 4))
vals = [f(t[i], "train_samples_per_s") for i in ids]
ax.bar([f'GBS {t[i]["gbs"]}\n(MINIBS {t[i]["minibs"]})' for i in ids], vals, color="#1baf7a")
for j, v in enumerate(vals): ax.annotate(f"{v:.0f}", (j, v), ha="center", va="bottom")
ax.set_ylabel("training samples/s"); ax.grid(alpha=.3, axis="y")
save(fig, ax, "throughput_256nodes.png", "256 nodes (1,024 GH200), TP1 CP4: throughput only")

# 5. TTT (left) and throughput of fastest-to-train vs highest-throughput config (right)
bt = best(ok); bmax = best(thr, "train_samples_per_s", low=False)
ns = [n for n in bt]; labels = [f"{n} node" + ("s" if n > 1 else "") for n in ns]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
y = range(len(ns))
v1 = [f(bt[n], "ttt_min") for n in ns]
a1.barh(y, v1, color=BLUE, height=.6)
for yi, v in zip(y, v1): a1.text(v * 1.05, yi, f"{v:.1f}", va="center", fontsize=8)
a1.set_xscale("log"); a1.set_xlim(10, 2000); a1.set_yticks(list(y)); a1.set_yticklabels(labels); a1.invert_yaxis()
a1.set_xlabel("time to train [min], log scale"); a1.set_title("Best time-to-train"); a1.grid(alpha=.3, axis="x", which="both")
h = .38
vt = [f(bt[n], "train_samples_per_s") for n in ns]; vm = [f(bmax[n], "train_samples_per_s") for n in ns]
a2.barh([yi - h / 2 for yi in y], vt, height=h, color=BLUE, label="fastest-to-train config")
a2.barh([yi + h / 2 for yi in y], vm, height=h, color=ORANGE, alpha=.35, edgecolor=ORANGE, label="highest-throughput config")
for yi, v in zip(y, vt): a2.text(v + 8, yi - h / 2, f"{v:.0f}" if v >= 100 else f"{v:.1f}", va="center", fontsize=7)
for yi, v in zip(y, vm): a2.text(v + 8, yi + h / 2, f"{v:.0f}" if v >= 100 else f"{v:.1f}", va="center", fontsize=7)
a2.set_xlim(0, 820); a2.set_xlabel("training samples/s"); a2.set_title("Training throughput"); a2.legend(loc="upper right", fontsize=8)
a2.grid(alpha=.3, axis="x")
plt.tight_layout(); fig.savefig(os.path.join(out, "ttt_and_throughput.png"), dpi=150); plt.close(fig)

# 6. throughput vs nodes per GBS (best layout per nodes x GBS)
fig, ax = plt.subplots(figsize=(8, 4.6))
for g in sorted({r["gbs"] for r in thr}, key=int):
    pts = best([r for r in thr if r["gbs"] == g], "train_samples_per_s", low=False)
    xs = list(pts); ys = [f(r, "train_samples_per_s") for r in pts.values()]
    ax.plot(xs, ys, "o-", color=GBS_COLORS.get(g, "gray"), label=f"GBS {g}")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = sorted({int(r["nodes"]) for r in thr}); ax.set_xticks(ticks); ax.set_xticklabels(ticks)
ax.grid(alpha=.3, which="both"); ax.set_xlabel("nodes"); ax.set_ylabel("training samples/s"); ax.legend(fontsize=8)
save(fig, ax, "throughput_by_gbs.png", "Training throughput by global batch size (best layout per point)")

# 7. convergence vs samples, and vs time at 64 nodes
runs = {}
for c in curves: runs.setdefault(c["run"], []).append(c)
label = {"cisco-h200-r0": "GBS 32 (Cisco H200, same hyperparameters)", "1198603": "GBS 64 (64 nodes)",
         "1199958": "GBS 128 (64 nodes)", "1203405": "GBS 256 (64 nodes, tail only)"}
for name, xkey, xlabel, title, only64 in [
        ("convergence_samples.png", "samples", "training samples", "Validation log perplexity vs samples", False),
        ("convergence_time_64nodes.png", "minutes_since_run_start", "minutes since run_start (approx.)",
         "Same 64 nodes, different GBS: log perplexity vs time", True)]:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for run, pts in runs.items():
        if only64 and pts[0]["nodes"] != "64": continue
        g = pts[0]["gbs"]
        ax.plot([float(p[xkey]) for p in pts], [float(p["log_ppl"]) for p in pts], "o-", ms=3,
                color=GBS_COLORS.get(g, "gray"), label=label.get(run, run))
    ax.axhline(3.3, ls="--", color="gray", lw=1); ax.text(ax.get_xlim()[1], 3.3, " target 3.3", va="center", fontsize=8)
    ax.set_ylim(3.2, 7.0); ax.set_xlabel(xlabel); ax.set_ylabel("validation log perplexity"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    save(fig, ax, name, title)
print("wrote", sorted(os.listdir(out)))
