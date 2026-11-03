# Data

| File | What it is | How it is made |
|---|---|---|
| `summary.csv` | Curated table of the runs discussed in `docs/RESULTS.md`, with notes | By hand |
| `eval_curves.csv` | Evaluation curves used for the convergence plots (Cisco GBS 32 reference + Hangang runs) | By hand / `tools/export_eval_curves.py` |
| `raw/runs.csv` | **Every job**: configuration as actually run, status, TTT, throughput, evaluations, network errors, nodes | `tools/collect_runs.py` |
| `raw/eval_curves.csv` | Every evaluation of every job (samples, log perplexity, minutes since `run_start`) | `tools/collect_runs.py` |
| `raw/train_curves.csv` | Every logged training block (samples, step time, training loss) | `tools/collect_runs.py` |
| `raw/mllog/<job>.txt.gz` | The job's `:::MLLOG` lines plus its configuration, NCCL network warnings and exit lines | `tools/collect_runs.py` |

Regenerate the raw data on the Hangang login node after new runs (full job logs stay in
`results/runs/`, which is not committed):

```bash
python3 tools/collect_runs.py results/runs /scratch/$USER/<older run directory>/results/runs --sacct
git add results/raw && git commit -m "Update raw run data"
```

## `raw/runs.csv` columns

| Column | Meaning |
|---|---|
| `job_id`, `tag` | Slurm job and the run tag (`n<nodes>-gbs<GBS>-tp<TP>cp<CP>-mb<MINIBS>-lr<LR>-wu<warmup>-net<mode>-<profile>`) |
| `nodes`, `gbs`, `tp`, `cp`, `minibs`, `mbs` | Scale and parallelism (`DP = 4 × nodes / (TP × CP)`, `GBS = MINIBS × DP`) |
| `lr`, `warmup_steps`, `seed` | From the MLPerf log (`opt_base_learning_rate`, ...) |
| `eval_schedule` | `mlperf` (every 12,288 samples), `relaxed` (`EVAL_START`/`EVAL_EVERY`) or `none` (short throughput probe) |
| `val_check_interval`, `eval_interval_after`, `max_steps` | Evaluation and step settings as run |
| `bucket_size`, `num_distopt_instances`, `tp_comm_overlap`, `fp8_dpa`, `overlap_pg_optim`, `netprofile` | Communication options as run (empty: NVIDIA default) |
| `status` | `success`, `aborted` (`MAX_STEPS` reached), `cancelled-nan`, `cancelled-hang`, `nan`, `init-error`, `no-run-start`, `incomplete` |
| `ttt_min` | `run_start` to `run_stop` in minutes (initialisation excluded, as in MLPerf) |
| `samples_last_eval`, `log_ppl_last_eval`, `n_evals` | Last evaluation; for `success` the samples to target |
| `train_samples_per_s` | GBS ÷ median logged step time (excludes evaluation) |
| `avg_samples_per_s` | Samples ÷ wall time since `run_start` (includes evaluation) |
| `init_min`, `hello_ranks`, `net_errors`, `run_exit` | Initialisation time, ranks that started, `NCCL WARN NET/OFI Request` count, exit code |
| `slurm_state`, `start`, `end`, `elapsed`, `nodelist` | From `sacct` (with `--sacct`) |
