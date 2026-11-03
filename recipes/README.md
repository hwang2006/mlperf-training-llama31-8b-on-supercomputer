# Validated recipes

Every line below reached the MLPerf target (validation log perplexity < 3.3) on Hangang
on 2026-11-01/02. Pass a line to `./submit.sh`, or a whole file to `scripts/submit_queue.sh`.

| Nodes | GPUs | `./submit.sh` arguments | GBS | Parallelism | TTT [min] | Samples to target | Network profile used |
|---:|---:|---|---:|---|---:|---:|---|
| 1 | 4 | `1 TP=2 TIME=12:00:00` | 32 | TP2 | 683.7 | 208,896 | (single node) |
| 2 | 8 | `2` | 32 | DP8 | 269.9 | 196,608 | base |
| 4 | 16 | `4` | 32 | DP16 | 140.6 / 140.8 | 196,608 | base / cscs |
| 8 | 32 | `8` | 32 | DP32 | 74.7 | 196,608 | cscs |
| 16 | 64 | `16 CP=2 TIME=01:20:00` | 32 | CP2 | 43.4 | 196,608 | cscs |
| 32 | 128 | `32 GBS=64 CP=2 LR=0.0008 WARMUP_STEPS=96 TIME=01:00:00` | 64 | CP2 | 24.3 | 221,184 | cscs |
| **64** | **256** | `64 GBS=64 CP=4 LR=0.0008 WARMUP_STEPS=64 TIME=00:50:00` | 64 | CP4 | **19.1** | 221,184 | cscs |
| 128 | 512 | `128 GBS=64 TP=2 CP=4 LR=0.0008 WARMUP_STEPS=64 TIME=00:45:00` | 64 | TP2·CP4 | 20.5 | 221,184 | cscs-nordzv |
| 256 | 1,024 | `256 GBS=128 TP=4 CP=2 LR=0.002 WARMUP_STEPS=256 TIME=00:50:00` | 128 | TP4·CP2 | 29.6 | 344,064 | cscs-nordzv |

`./submit.sh` now defaults to `NETPROFILE=cscs-nordzv`, the setting that removed the
collective hangs up to 64 nodes with no throughput cost. Each TTT is a single run; MLPerf
scores average 10 runs, and samples-to-target alone varies by about ±10% between seeds.

## Beyond MLPerf rules

`recipes/relaxed_eval.txt` holds the 128-node run with a late, dense evaluation schedule
(`EVAL_START`, `EVAL_EVERY`; 20.3 min). GBS 256 converged in 578k samples with LR 2e-3 and 128
warmup steps (LR 3e-3: 762k), so GBS above 128 does not pay off on Hangang; see docs/RESULTS.md.

## Choosing hyperparameters for other batch sizes

Use the MLCommons reference convergence points (`mlcommons/logging`,
`mlperf_logging/rcp_checker/training_6.1.0/rcps_llama31_8b.json`). Our GBS 32 and GBS 64 runs
used the Cisco and NVIDIA v6.0 submission values instead; the RCP values for GBS 32 converge
in fewer samples on average and are worth trying.

| GBS | LR | Warmup (samples) | `WARMUP_STEPS` | Reference samples to target (mean, range) |
|---:|---:|---:|---:|---|
| 16 | 4e-4 | 256 | 16 | 161,587 (159,744–172,032) |
| 32 | 8e-4 | 4,096 | 128 | 179,405 (172,032–196,608) |
| 64 | 8e-4 | 6,144 | 96 | 226,714 (208,896–245,760) |
| 96 | 1e-3 | 16,348 | 170 | 278,640 (260,064–297,216) |
| 128 | 2e-3 | 32,768 | 256 | 345,907 (307,200–405,504) |
| 256 | 2e-3 (Hangang, not a reference) | 32,768 | 128 | 577,536 (one run) |

There is no reference above GBS 128. With LR 8e-4 and 64 warmup steps, GBS 128 did not
converge in time (3.47 after 245,760 samples); the reference values above fixed that.

## Parallelism rules of thumb on Hangang (4 GPUs per node)

- `GBS = MINIBS × DP` and `DP = 4 × nodes / (TP × CP)`; `submit.sh` stops a job whose GBS does not divide.
- Keep `TP × CP ≤ 4` so TP/CP traffic stays inside a node. Within a node CP4 was 2.6% faster than TP4 (64 nodes, GBS 64).
- Every NaN run we saw had CP spanning nodes (`TP × CP > 4`); keep the node exclude list in `env.sh` when you need that.
- One node needs TP=2: with TP1 the 96 GB GPUs run out of memory (optimizer state over only 4 data-parallel ranks).
