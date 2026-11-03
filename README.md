# MLPerf Training Llama 3.1 8B on KISTI Hangang (GH200)

Reproducible recipe for the **MLPerf Training Llama 3.1 8B pretraining** benchmark on KISTI's
6th supercomputer **Hangang** (NVIDIA GH200, HPE Slingshot), using the NVIDIA MLPerf Training
v6.0 implementation, from 1 node (4 GPUs) to 256 nodes (1,024 GPUs).

**Best time-to-train: 19.1 min on 64 nodes (256 GH200).** Every scale from 1 to 256 nodes
reached the MLPerf quality target (validation log perplexity < 3.3).

![TTT and throughput per node count](docs/img/ttt_and_throughput.png)

> Unofficial results: single runs, not reviewed by MLCommons. See [docs/RESULTS.md](docs/RESULTS.md).

Companion repositories:
[mlperf-training-llama2-lora-on-supercomputer](https://github.com/hwang2006/mlperf-training-llama2-lora-on-supercomputer)
(Llama 2 70B LoRA on Hangang) and
[kisti-neuron-mlperf-training](https://github.com/hwang2006/kisti-neuron-mlperf-training)
(reference implementation on NEURON).

More nodes need a larger global batch to raise throughput, and above GBS 64 the extra
samples needed to converge outgrow the speedup, so 64 nodes is the sweet spot
([docs/RESULTS.md](docs/RESULTS.md)).

## The benchmark in one paragraph

Pretrain Llama 3.1 8B (32 layers, GQA, 128K vocab, sequence length 8,192) **from random
weights** on a preprocessed C4 subset (`c4-train.en_6`, ~21B tokens, 84 GB) until the
validation log perplexity on 1,024 C4 validation sequences drops below 3.3. Evaluation runs
every 12,288 sequences and its time counts. Fixed: AdamW (β 0.9/0.95, wd 0.1), clipping 1.0,
cosine decay. Free: global batch (GBS), base LR, warmup, parallelism. We train in FP8 (hybrid
E4M3/E5M2, tensorwise scaling) on a BF16 base, as NVIDIA's Hopper submissions do.

## What you need

| Item | Value |
|---|---|
| Account | Hangang account with access to the `normal` (GH200) partition and a valid `--comment` (`showappl`) |
| Scratch space | ~100 GB data (84 GB C4 + validation + tokenizer), ~25 GB containers |
| Time | About 2-3 h of setup (downloads and builds), mostly unattended |

## Quick start

Run everything from a directory under `/scratch` (Hangang only accepts jobs submitted there).

### 0. Clone and configure

```bash
cd /scratch/$USER
git clone https://github.com/hwang2006/mlperf-training-llama31-8b-on-supercomputer.git
cd mlperf-training-llama31-8b-on-supercomputer
vi env.sh        # SLURM_COMMENT (required), DATA_ROOT, CONT_DIR; review EXCLUDE_NODES
source env.sh
```

### 1. Upstream code and data (login node)

```bash
./scripts/00_fetch_sources.sh            # NVIDIA v6.0 implementation + Cisco H200 reference logs
setsid nohup ./scripts/04_prepare_data.sh > results/data-download.log 2>&1 < /dev/null &
```

Data download takes an hour or more; it ends with `DATA_READY` after checking file sizes.
`setsid nohup … &` detaches the job from your terminal so it keeps running after logout.

### 2. Containers

```bash
sbatch --comment="$SLURM_COMMENT" scripts/01_import_base_image.sbatch     # NGC 26.04 + driver check (~15 min)
./scripts/03_build_ofi_plugin.sh 2>&1 | tee results/ofi-build.log        # aws-ofi-nccl for Slingshot (~5 min)
setsid nohup ./scripts/02_build_container.sh > results/build.log 2>&1 < /dev/null &   # ~40-70 min
```

Check: the import log shows `available True` and `cc (9, 0)`; the plugin log ends with
`OFI_BUILD_OK`; the build log contains `FINAL torch …` and `BUILD_EXIT=0`.

### 3. Network check (2 nodes, ~5 min)

```bash
sbatch -N 2 -p normal --comment="$SLURM_COMMENT" slurm/nccl_test.sbatch
cat $(ls -t results/kv6-nccl-*.log | head -1)
```

Expect about 280-300 GB/s bus bandwidth at 512-2,048 MB with `Using network Libfabric`.

### 4. Smoke test (4 nodes, ~6 min)

```bash
./submit.sh 4 MAX_STEPS=200 TIME=00:40:00
python3 tools/ttq_status.py results/runs/kv6-l8b-<job>.out
```

Expect a first loss of ~12.58 and about 24 training samples/s. `"status": "aborted"` is normal
for a `MAX_STEPS` run.

### 5. Time-to-train runs

```bash
./submit.sh 64 GBS=64 CP=4 LR=0.0008 WARMUP_STEPS=64 TIME=00:50:00     # best: ~19 min
setsid nohup ./scripts/submit_queue.sh recipes/best_per_scale.txt > results/queue.log 2>&1 < /dev/null &
```

[`recipes/README.md`](recipes/README.md) lists the command for every scale, the MLPerf
reference hyperparameters per batch size, and parallelism rules. Hangang allows 5 submitted
jobs per user; `submit_queue.sh` submits a recipe file as slots free up.

## Monitoring

`submit.sh` starts a watchdog for every job. It cancels the job on a NaN loss or if the log
stops for 5 minutes, which is how collective hangs show up on this system.

```bash
squeue -u $USER
python3 tools/ttq_status.py --summary results/runs/kv6-l8b-*.out     # TTT, samples, last eval, throughput
python3 tools/ttq_status.py results/runs/kv6-l8b-<job>.out           # eval curve of one run
tail -n2 results/watch_<job>.log                                     # watchdog verdict
```

## How a run is launched

```
submit.sh NODES KEY=VALUE…  →  sbatch slurm/run_ttq.sbatch  (+ tools/watch_job.sh)
   ├─ configs/config_GH200_8b.sh    NVIDIA 8B configs minus FP4, GBS/TP/CP/LR from KEY=VALUE
   ├─ srun 1 task/node              unpack the container on every node (abort if any node fails)
   └─ srun 4 tasks/node, hostfile   slurm2pytorch → run_and_time.sh → pretrain.py
          rank 0 pinned to MASTER_ADDR, 16 CPUs per rank, Slingshot via self-built aws-ofi-nccl
```

| `submit.sh` key | Meaning |
|---|---|
| `GBS`, `TP`, `CP`, `MBS` | Global batch and parallelism; `GBS = MINIBS × 4·nodes / (TP·CP)` |
| `LR`, `WARMUP_STEPS` | Base LR and warmup steps (defaults 4e-4 and 20, the Cisco GBS 32 values) |
| `MAX_STEPS`, `VAL_CHECK_INTERVAL` | Short tests; evaluation defaults to every 12,288 samples. `MAX_STEPS` is capped automatically for GBS ≥ 128 |
| `EVAL_START`, `EVAL_EVERY` | Non-MLPerf evaluation schedule in samples (first evaluation, then interval) |
| `NETPROFILE` | CXI environment: `cscs-nordzv` (default), `cscs`, `cscs-rdzv`, `base` |
| `TIME`, `SEED`, `NCCL_DEBUG`, `WATCH=0`, `STALL=<min>` | Slurm time limit, seed, NCCL logging, watchdog control |

## Things that are different on Hangang

Summary of [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md); the scripts already handle each:

1. The aws-ofi-nccl plugin shipped in NGC 26.04 aborts on Slingshot CXI and NCCL silently uses TCP.
2. Without `--cpus-per-task`, NCCL reaches 40 GB/s instead of ~300 GB/s.
3. Task 0 is not always on the first node of the sorted node list.
4. Collectives can hang silently; the CSCS Slingshot settings fix this up to 64 nodes.
5. Some nodes produced NaN loss in runs with context parallelism across nodes; they are excluded in `env.sh`.
6. One node needs TP=2 (96 GB GPUs); builds on login nodes need `OPENBLAS_NUM_THREADS=1`.
7. GBS ≥ 256 timed out during initialisation (dataset index for `MAX_STEPS × GBS` samples); `MAX_STEPS` is now capped.

## Repository layout

```
env.sh                         paths, Slurm settings, node exclude list
submit.sh                      submit one run (+ watchdog)
configs/config_GH200_8b.sh     GH200 config on top of NVIDIA's 8B configs
slurm/run_ttq.sbatch           time-to-train job
slurm/nccl_test.sbatch         2-node NCCL bandwidth and CXI profile check
scripts/00-04_*                fetch code, import/build containers, build OFI plugin, get data
scripts/submit_queue.sh        submit a recipe file within the 5-job limit
container/                     build steps run inside enroot
recipes/                       validated commands and hyperparameters
tools/                         log parser, status, watchdog, plots, Cisco baseline
results/summary.csv            all measured runs
results/eval_curves.csv        evaluation curves used in docs/RESULTS.md
docs/                          results, troubleshooting, Korean quick guide
```

## Acknowledgements

NVIDIA's MLPerf Training v6.0 implementation and Cisco's v6.0 results are fetched from
[mlcommons/training_results_v6.0](https://github.com/mlcommons/training_results_v6.0) under
their own licenses. Data and tokenizer come from MLCommons via
[r2-downloader](https://github.com/mlcommons/r2-downloader). Slingshot NCCL settings follow the
[CSCS Alps documentation](https://docs.cscs.ch/software/communication/nccl).
Measurements used KISTI Hangang resources.
