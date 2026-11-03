# Hangang pitfalls and how this repository handles them

Each item below cost us at least one failed multi-node run. All fixes are already in the
scripts; this page explains what to look for if you adapt them.

| # | Symptom | Cause | Fix in this repo |
|---|---|---|---|
| 1 | Inter-node NCCL at ~5 GB/s, `NET/Socket` in logs | NGC 26.04's aws-ofi-nccl 1.17.3 aborts on CXI | Self-built plugin (`scripts/03_build_ofi_plugin.sh`) |
| 2 | NCCL at 40 GB/s instead of ~300 GB/s | Each rank pinned to 1 CPU | `--cpus-per-task=16` on sbatch and srun |
| 3 | `client socket has timed out ... (nodeXXXX, 29500)` at 16+ nodes | Rank 0 not on the first sorted node | Hostfile + `--distribution=arbitrary` |
| 4 | Log silently stops mid-training, job runs to TIME | Collective hang on Slingshot | `NETPROFILE=cscs-nordzv` + `tools/watch_job.sh` |
| 5 | `reduced_train_loss: NaN` from the first step | Specific nodes, only with CP across nodes | `EXCLUDE_NODES` in `env.sh`, watchdog cancels on NaN |
| 6 | `task N launch failed: Error configuring interconnect` | Node-side Slingshot setup failure | Exclude the node; job fails fast |
| 7 | CUDA OOM on 1 node | 96 GB GPU; optimizer state over 4 DP ranks only | `TP=2` on 1 node |
| 8 | Container build dies with `KeyboardInterrupt` | Login-node thread limit hit by OpenBLAS | `OPENBLAS_NUM_THREADS=1` in the build |
| 9 | GBS ≥ 256 dies before `run_start` with NCCL `checkTimeout` | Rank 0 builds a dataset index for `MAX_STEPS × GBS` samples | `MAX_STEPS` capped automatically for GBS ≥ 128 |
| 10 | Hang preceded by `NET/OFI Request ... completed with error ... (UNDELIVERABLE)` | CXI could not deliver messages between specific nodes | Exclude the reporting nodes; report time and nodes to the system team |

## 1. aws-ofi-nccl in NGC 26.04 does not work on Slingshot

```
NET/OFI Selected provider is cxi, fabric is cxi (found 4 nics)
NET/OFI Caught exception in plugin init: Failed to open /sys/class/infiniband/cxi0/node_guid
```

NGC builds the plugin with the AWS platform code, which reads an EFA sysfs file that CXI NICs
lack and then throws. NCCL silently falls back to TCP. 1.18.0 and master behave the same;
1.14.0 (NGC 25.04) only warned. `scripts/03_build_ofi_plugin.sh` builds 1.17.3 with
`--disable-platform-aws --with-libfabric=/opt/cray/libfabric/2.3.1` inside the base image and
mounts it at `/ofi`; `NCCL_NET_PLUGIN=ofi` selects it. Check a run with:

```bash
grep -m4 -E "Using network|Selected provider|transport protocol" results/runs/kv6-l8b-<job>.out
```

Expect `Using network Libfabric`, `provider is cxi`, `SENDRECV`. `OFI_NCCL_PROTOCOL=RDMA` fails
to initialise on this system. GPUDirect RDMA is enabled by default; no `NCCL_NET_GDR_LEVEL` needed.

## 2. CPU binding

With 4 tasks per node and no `--cpus-per-task`, each rank appears to get one core and NCCL's
proxy thread competes with the main thread: 40.7 GB/s vs 298 GB/s (2 nodes, 2 GB all-reduce).
`slurm/run_ttq.sbatch` requests 16 CPUs per task and passes `--cpus-per-task=$SLURM_CPUS_PER_TASK`
to srun explicitly. GPU *i* then lands on the Grace cores of NUMA node *i*.
`sbatch -N 2 slurm/nccl_test.sbatch` prints each rank's CPU affinity and the bus bandwidth.

## 3. Rank 0 placement

Slurm 26.05 on Hangang may order tasks by network topology, so task 0 is not always on
`$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)`. Every rank then waits 10 min for
a TCPStore that nobody opened. The sbatch writes a hostfile (each node repeated 4 times),
sets `MASTER_ADDR` to its first entry and launches with
`SLURM_HOSTFILE=... srun --distribution=arbitrary`, as NVIDIA's `run.sub` does.

## 4. Collective hangs

With the default CXI settings, 5 of 6 runs at 8-64 nodes stopped right after an evaluation:
no error, the log simply stops. Steps run as CUDA graphs, which the PyTorch NCCL watchdog does
not monitor, so most hangs held their nodes until the time limit.

`NETPROFILE` selects the CXI environment (`slurm/run_ttq.sbatch`):

| Profile | Adds to the base CXI settings | 8-node stress test (eval every 8 steps) |
|---|---|---|
| `base` | nothing | hung after 31 evaluations |
| `cscs-nordzv` (default) | `FI_CXI_RX_MATCH_MODE=software`, `FI_CXI_DEFAULT_TX_SIZE=16384`, `NCCL_PROTO=^LL128` | 2 runs, ~158 evaluations, no hang |
| `cscs` | the above + `FI_CXI_RDZV_GET_MIN=0`, `FI_CXI_RDZV_THRESHOLD=0`, `FI_CXI_RDZV_EAGER_SIZE=0` | 80/80, no hang |

These follow the [CSCS Alps NCCL guide](https://docs.cscs.ch/software/communication/nccl).
Neither profile cost throughput (4 nodes: 140.6 vs 140.8 min). At 128 nodes and above, hangs
still occurred with both profiles (one 128-node and two 256-node runs), while other 256-node
runs completed, including a 36-minute time-to-train run.

`submit.sh` starts `tools/watch_job.sh` for every job: it cancels the job if the loss is NaN or
if the log has not changed for 5 minutes after `run_start` (`STALL=<min>` to change). Reproduce:

```bash
./submit.sh 8 VAL_CHECK_INTERVAL=8 MAX_STEPS=640 TIME=00:40:00 NETPROFILE=base
```

## 5. NaN from the first step

Six 64- and 256-node runs had `NaN` loss from step 0, with no error. All of them used CP
groups that span nodes (`TP × CP > 4`). Whether they failed matched whether certain nodes were
in the allocation, not the CXI profile:

- Set A: node0320-0324, node0541, node0739 were in all four 64-node NaN runs and in no passing
  64-node run; excluding them made the same configuration pass.
- Set B: 14 nodes were in both 256-node NaN runs and in none of ~25 passing jobs (560 nodes).
- With both sets excluded, 34 jobs on 2026-11-03 (64–512 nodes, CP up to 8 across nodes) had no NaN.

Both sets plus node0943 and three precautionary nodes are in `EXCLUDE_NODES` in `env.sh`. The
system team has been asked to check them; Slurm still shows them as healthy. If you must run CP
across nodes, check the first loss (~12.58) within minutes; the watchdog cancels NaN runs.

```bash
grep -m2 -oE '"reduced_train_loss": [^,]+' results/runs/kv6-l8b-<job>.out
```

## 6. Node fails to start

`srun: error: task N launch failed: Error configuring interconnect` (node0943). The other
ranks used to wait 10 minutes; the sbatch now aborts if the container step fails on any node
and runs the training step with `--kill-on-bad-exit=1`. Add the node to `EXCLUDE_NODES`.

## 7. One node runs out of memory

`torch.OutOfMemoryError` at 86.7 GiB allocated, before `run_start`. With TP1 the FP32 master
weights and Adam state are split over only 4 DP ranks (~24 GB per GPU). `TP=2` halves
parameters, gradients and logits per GPU. Two nodes (DP 8) fit with TP1.

## 8. Building on a login node

Hangang login nodes limit processes per user (RLIMIT_NPROC 16384). The build verification
imports SciPy; OpenBLAS then tries to start one thread per core and the step dies with
`KeyboardInterrupt` after `pthread_create failed`. `container/build_inside.sh` sets
`OPENBLAS_NUM_THREADS=1`. Run long builds with `setsid nohup ... &` so they keep running after logout.

## 9. Large GBS times out during initialisation

`init_start` but no `run_start`, then 255 of 256 ranks print `terminate called after throwing
an instance of 'c10::DistBackendError'` from `ProcessGroupNCCL.cpp ... checkTimeout`. Before
training, rank 0 builds the shuffled sample index for `MAX_STEPS × GBS` samples while the other
ranks wait in a collective. With MLPerf's 1.2M steps that is 38M samples at GBS 32 (init ~1.5
min), 154M at GBS 128 (~5 min) and 307M at GBS 256 (>10 min, the NCCL timeout).
`configs/config_GH200_8b.sh` now sets `MAX_STEPS = 4,000,000 / GBS` for GBS ≥ 128; the LR decay
length stays at 1.2M steps, so training is unchanged. Override with `MAX_STEPS=<steps>`.
The cap was first 2M samples, which would have stopped GBS 1,024 (~2M samples to target) short
of the target; keep the cap well above the samples a run needs.

## 10. CXI delivery errors before a hang

Job 1205461 (64 nodes, 2026-11-03 14:47:42) stopped after these warnings, then the watchdog
cancelled it five minutes later:

```
NET/OFI Request ... completed with error. RC: 113. Error: 2 (UNDELIVERABLE) ... direction: SEND
NET/OFI Request ... completed with error. RC: 5. Error: 53 (CONN_CLOSED) ... direction: SEND
```

They came from node0113 (GPU 0), node0114 (GPU 2) and node0115 (GPU 1), three consecutive
nodes on different NICs, which points at a shared switch or link. The three nodes are now in
`EXCLUDE_NODES`. Count these warnings when a run stalls:

```bash
grep -c 'NCCL WARN NET/OFI Request' results/runs/kv6-l8b-<job>.out
grep 'NCCL WARN NET/OFI Request' results/runs/kv6-l8b-<job>.out | grep -oE 'node[0-9]+' | sort | uniq -c
```

## Options that fail with this software stack

| Option | Error |
|---|---|
| `FP8_ATTN=1` with CP > 1 | `ValueError: No dot product attention backend is available for the provided inputs` |
| `OVERLAP_PG_OPTIM=True` | `IndexError: list index out of range` at init |

## Other Hangang notes

- Five submitted jobs per user (`QOSMaxSubmitJobPerUserLimit`): use `scripts/submit_queue.sh`.
- Heavy use lowers your fair-share and job priority (`sshare -U -u $USER`); large jobs then wait longer.
- `squeue` shows only your own jobs. Free nodes: `sinfo -p normal -h -o "%t %D" | awk '$1=="idle"'`
  (`sinfo -t idle` also counts drained and planned nodes).
- The container needs CUDA forward compatibility (driver 570.124.06 vs CUDA 13.2). It works on
  compute nodes; the login node prints `compatibility mode is UNAVAILABLE`, which is harmless.
- MLPerf code does not stop on NaN and logs `"status": "aborted"` for `MAX_STEPS` runs; both are expected.
