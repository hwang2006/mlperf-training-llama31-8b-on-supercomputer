# Validated recipes

Every line reached the stop condition (validation log perplexity < 3.3 on the 1,024 MLPerf
validation sequences) on Hangang. Pass a line to `./submit.sh`, or a file to
`scripts/submit_queue.sh`. Track definitions: [docs/RESULTS.md](../docs/RESULTS.md).

## Track 1: MLPerf evaluation schedule, GBS ≤ 128 (`best_per_scale.txt`)

| Nodes | GPUs | `./submit.sh` arguments | TTT [min] | Samples |
|---:|---:|---|---:|---:|
| 1 | 4 | `1 TP=2 TIME=12:00:00` | 683.7 | 208,896 |
| 2 | 8 | `2` | 269.9 | 196,608 |
| 4 | 16 | `4` | 140.6 | 196,608 |
| 8 | 32 | `8` | 74.7 | 196,608 |
| 16 | 64 | `16 CP=2 TIME=01:20:00` | 43.4 | 196,608 |
| 32 | 128 | `32 GBS=64 CP=2 LR=0.0008 WARMUP_STEPS=96 TIME=01:00:00 NETPROFILE=cscs` | 24.3 | 221,184 |
| 64 | 256 | `64 GBS=64 TP=2 CP=2 LR=0.001 WARMUP_STEPS=64 TIME=00:40:00 NETPROFILE=cscs` | 17.1 / 19.6 | 221,184 / 258,048 |
| **128** | **512** | `128 GBS=128 TP=1 CP=4 LR=0.001 WARMUP_STEPS=256 BUCKET_SIZE=2000000000 MAX_STEPS=1200000 TIME=00:40:00 NETPROFILE=cscs` | **13.8 / 14.3** | 319,488 / 331,776 |
| 256 | 1,024 | `256 GBS=128 TP=4 CP=2 LR=0.001 WARMUP_STEPS=256 BUCKET_SIZE=2000000000 MAX_STEPS=1200000 TIME=00:50:00 NETPROFILE=cscs` | 15.3 | 319,488 |
| 512 | 2,048 | `512 GBS=128 TP=4 CP=4 LR=0.001 WARMUP_STEPS=256 BUCKET_SIZE=2000000000 MAX_STEPS=1200000 TIME=00:50:00 NETPROFILE=cscs` | 21.1 | 331,776 |

The 1–32 node rows were measured before the LR 1e-3 and bucket findings; LR 1e-3 with the
reference warmup is worth trying there too. `MAX_STEPS=1200000` keeps the MLPerf value; without
it `MAX_STEPS` is capped for GBS ≥ 128 (shorter init, same training).

## Track 2: relaxed (`track2_best.txt`)

| Nodes | Configuration | TTT [min] | Samples |
|---:|---|---:|---:|
| 64 | GBS 64, TP1·CP4, LR 1e-3, eval from 184,320 every 3,072 | 17.2 | 208,896 |
| 128 | GBS 128, TP1·CP4, bucket 2e9, LR 1e-3, eval from 258,048 every 3,072 | 14.1 | 328,704 |
| 256 | GBS 256, TP4·CP1, bucket 2e9, LR 1e-3, eval from 307,200 every 6,144 | 17.2 | 577,536 |
| 512 | GBS 512, TP4·CP1, bucket 2e9, LR 1e-3, eval from 491,520 every 6,144 | 19.1 | 952,320 |

## Hyperparameters

| GBS | LR / warmup steps used on Hangang | Samples to target on Hangang | MLPerf reference (LR / warmup samples / mean samples) |
|---:|---|---|---|
| 16 | — | — | 4e-4 / 256 / 161,587 |
| 32 | 4e-4 / 20 | 184k–209k | 8e-4 / 4,096 / 179,405 |
| 64 | 1e-3 / 64 (8e-4: similar, 1.2e-3: worse) | 209k–258k | 8e-4 / 6,144 / 226,714 |
| 96 | — | — | 1e-3 / 16,348 / 278,640 |
| 128 | 1e-3 / 256 (1.5e-3: 326k, 2e-3: 344k–402k) | 307k–393k, median ~330k | 2e-3 / 32,768 / 345,907 |
| 256 | 1e-3 / 128 (8e-4: 522k, 2e-3: 578k, 3e-3: 762k) | 516k–578k | none |
| 512 | 1e-3 / 128 | 0.93M–1.15M | none |

## Parallelism and buckets on Hangang (4 GPUs per node)

- `GBS = MINIBS × DP`, `DP = 4 × nodes / (TP × CP)`; `submit.sh` stops a job whose GBS does not divide.
- Up to 128 nodes, keep TP/CP inside a node and favour CP: TP2·CP2 (GBS 64, 64 nodes) and TP1·CP4 (GBS 128, 128 nodes) were fastest.
- From 256 nodes, gradient synchronisation across nodes dominates: use TP4 (a quarter of the parameters per GPU), even if CP then spans nodes.
- `BUCKET_SIZE` counts parameters (default 768M). Use 2e9 with TP1, and with any TP once DP ≥ 128; keep the default for TP ≥ 2 at small DP (64 nodes).
- One node needs TP=2 (96 GB GPUs).
