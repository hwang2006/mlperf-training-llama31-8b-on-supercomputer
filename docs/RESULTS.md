# Results on Hangang (KISTI 6th supercomputer)

All runs reached the MLPerf Training Llama 3.1 8B target (validation log perplexity < 3.3)
with the NVIDIA MLPerf Training v6.0 implementation, FP8 (hybrid, tensorwise) on GH200 96 GB.
Measured 2026-11-01/02. Raw numbers: [`results/summary.csv`](../results/summary.csv);
plots regenerate with `python3 tools/plot_results.py`.

**Best: 19.14 min on 64 nodes (256 GH200), GBS 64, CP4.** Single runs, not MLPerf-reviewed
results; an official score averages 10 runs.

![best TTT per node count](img/ttt_best_per_scale.png)

![TTT and throughput per node count](img/ttt_and_throughput.png)

Up to 32 nodes the highest-throughput configuration is also the fastest to train. From 64
nodes on they split: larger global batches process more samples per second but need more
samples to reach the target (see *Throughput vs convergence* below).

| Nodes | GPUs | GBS | Parallelism | TTT [min] | Samples to target | Train samples/s |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 4 | 32 | TP2 | 683.7 | 208,896 | 5.2 |
| 2 | 8 | 32 | DP8 | 269.9 | 196,608 | 12.4 |
| 4 | 16 | 32 | DP16 | 140.6 | 196,608 | 23.9 |
| 8 | 32 | 32 | DP32 | 74.7 | 196,608 | 45.0 |
| 16 | 64 | 32 | CP2 | 43.4 | 196,608 | 77.2 |
| 32 | 128 | 64 | CP2 | 24.3 | 221,184 | 155.9 |
| 64 | 256 | 64 | CP4 | **19.1** | 221,184 | 206.4 |
| 128 | 512 | 64 | TP2·CP4 | 20.5 | 221,184 | 188.9 |
| 256 | 1,024 | 128 | TP4·CP2 | 29.6 | 344,064 | 199.0 |

## Strong scaling at GBS 32

Throughput is near-linear to 8 nodes (94% of linear), 81% at 16 and 60% at 32 nodes, then
flat. With GBS fixed at 32 the per-step gradient sync does not shrink, and DP is capped at 32,
so extra GPUs only add CP/TP communication.

![GBS 32 throughput](img/throughput_gbs32.png)

## A larger global batch above 32 nodes

GBS 64 needs 12-27% more samples to converge but raises throughput by 33-66%, so it wins at
every scale from 32 nodes. 64 nodes is the sweet spot: 128 nodes needs TP2·CP4 and is slower.

![GBS 32 vs 64](img/ttt_gbs32_vs_gbs64.png)

GBS 128 converges only with the MLPerf reference hyperparameters (LR 2e-3, 256 warmup steps;
see [`recipes/README.md`](../recipes/README.md)): 64 nodes 22.4 min, 256 nodes 29.6 min.

## 256 nodes: throughput only

300-step runs with all TP/CP traffic inside a node (TP1·CP4). Per-step gradient sync across
256 nodes costs about 0.7 s (about 0.1 s on 64 nodes); more micro-batches per GPU hide it.
GBS 512 and 1,024 have no MLPerf reference hyperparameters, and both runs hung after 64 and
32 steps, so these are not time-to-train results.

![256-node throughput](img/throughput_256nodes.png)

## Throughput vs convergence: why more nodes stop helping

![throughput by GBS](img/throughput_by_gbs.png)

At a fixed GBS, throughput peaks and then falls as nodes are added: DP cannot exceed GBS, so
extra GPUs go to TP/CP and their communication outweighs the extra compute. At a fixed node
count, a larger GBS raises throughput (64 nodes: GBS 32/64/128/256 → 123/206/283/339 samples/s).
But the samples needed to reach the target grow faster than throughput above GBS 64:

| GBS | Samples to target (Hangang) | vs GBS 64 | MLPerf reference mean |
|---:|---|---:|---:|
| 32 | 184k–209k | 0.9× | 179k |
| 64 | 221k–233k | 1.0× | 227k |
| 128 | 344k–402k | ~1.7× | 346k |
| 256 | 578k (LR 2e-3), 762k (LR 3e-3) | ~2.6× | none |

![convergence vs samples](img/convergence_samples.png)

On the same 64 nodes, GBS 128 and 256 process 37% and 64% more samples per second than GBS 64,
yet GBS 64 reaches 3.3 first (19.1 vs 22.4 vs 29.3 min). Time axis below is approximate
(evaluations placed at uniform throughput, last point = measured TTT); regenerate exact curves
from your logs with `tools/export_eval_curves.py`.

![convergence vs time at 64 nodes](img/convergence_time_64nodes.png)

## Beyond the MLPerf rules: 128 to 512 nodes

With any parallelism, batch size, hyperparameters and evaluation schedule allowed, and the
same stop condition (log perplexity < 3.3 on the same 1,024 validation samples):

| Nodes | Best configuration | TTT | Note |
|---:|---|---:|---|
| 128 | GBS 64, TP2·CP4, first eval at 196,608 then every 3,072 samples | 20.3 min | MLPerf schedule: 20.5 min |
| 256 | GBS 128, TP4·CP2 | 29.6 min | A TP2·CP4 run with a dense late schedule took 32.6 min: 402k samples vs 344k |
| 512 | (not run) GBS 256, TP2·CP4 at 223 samples/s × ~578k samples | ~43 min projected | GBS 128: 111 samples/s → ~52 min |

100-step throughput probes (training samples/s):

| Nodes | GBS 64 | GBS 128 | GBS 256 |
|---:|---|---|---|
| 128 | TP2·CP4 189, TP4·CP2 139, TP1·CP8 114 | TP1·CP4 235 | — |
| 256 | TP2·CP8 103, TP4·CP4 95 | TP2·CP4 202, TP4·CP2 199 | TP1·CP4 271 |
| 512 | TP4·CP8 54 | TP4·CP4 111, TP2·CP8 109 | TP2·CP4 223 |

Above 64 nodes the run-to-run spread in samples-to-target (±10–15%) matters more than layout
tuning. Moving the evaluation schedule saved 0.25 min at 128 nodes. The optional schedule is
`EVAL_START=<samples> EVAL_EVERY=<samples>` in `submit.sh`; it is not MLPerf-compliant.

## Comparison with MLPerf v6.0 Hopper submissions

| GPUs | Hangang GH200 | Published Hopper result |
|---:|---|---|
| 16 | 140.6 min (4 nodes) | 122.2 min, Cisco 2x8 H200 SXM |
| 64 | 43.4 min (16 nodes, GBS 32) | 33.1 min, Cisco 8x8 H100 (GBS 32) |
| 128 | 24.3 min (32 nodes, GBS 64) | 22.0 min, Cisco 16x8 H100 (GBS 64) |

Cisco evaluated every 8,192 samples; the v6.0 and v6.1 rules (and our runs) use 12,288, which
can add up to one evaluation interval of extra training. Cisco also packs 8 GPUs per node
behind NVSwitch, so 64 GPUs span 8 nodes instead of 16.

## Other measurements

| Measurement | Result |
|---|---|
| TP vs CP inside a node (64 nodes, GBS 64) | CP4 19.14 min, TP4 19.64 min |
| GBS 64 hyperparameters (32 nodes) | LR 8e-4 + 96 warmup steps: 221k samples; LR 1.2e-3: 233k |
| CXI profile `base` vs `cscs` (4 nodes) | 140.6 vs 140.8 min, no throughput cost |
| `base` vs `cscs` (2 nodes) | 269.9 vs 275.3 min (node placement noise) |
| 2-node NCCL all-reduce, 2 GB | 298 GB/s bus bandwidth with CXI, 11.4 GB/s over TCP |
