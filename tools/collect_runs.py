#!/usr/bin/env python3
"""Collect every run's configuration, outcome and curves into results/raw/ (commit these).

  python3 tools/collect_runs.py results/runs [OTHER_RUN_DIR ...] [--sacct]

Writes
  results/raw/runs.csv          one row per job: configuration, status, TTT, throughput, nodes
  results/raw/eval_curves.csv   every evaluation (samples, log perplexity, minutes since run_start)
  results/raw/train_curves.csv  every logged training step block (samples, step time, loss)
  results/raw/mllog/<job>.txt.gz  the job's MLPerf log lines plus its configuration lines
--sacct adds node list, start/end and Slurm state (run it on the Hangang login node).
Configuration is read from the job log itself (sbatch trace and environment dump), so it is
what the job actually ran with; the last assignment of each variable wins.
"""
import csv
import glob
import gzip
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllog import parse, ttt_min  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "raw")
CFG_KEYS = ["DGXNNODES", "TENSOR_MODEL_PARALLEL", "CONTEXT_PARALLEL", "MINIBS", "MICRO_BATCH_SIZE",
            "LR", "WARMUP_STEPS", "VAL_CHECK_INTERVAL", "EVAL_INTERVAL_AFTER", "MAX_STEPS",
            "BUCKET_SIZE", "NUM_DISTOPT_INSTANCES", "TP_COMM_OVERLAP",
            "OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP", "FP8_DPA", "NETPROFILE", "NETMODE"]
CFG_RE = {k: re.compile(r"(?<![A-Za-z0-9_])" + k + r"=([^\s,'\"|)]+)") for k in CFG_KEYS}
KEEP_RE = re.compile(r":::MLLOG|EXPTAG=|RUN_EXIT=|NCCL WARN NET/OFI|CONTAINER_SETUP_FAILED|"
                     r"Error configuring interconnect|CANCELLED|Watchdog caught|"
                     + "|".join(r"(?<![A-Za-z0-9_])" + k + "=" for k in CFG_KEYS))


def scan(path):
    cfg, tag, run_exit, hello, neterr, nan, tb = {}, "", "", 0, 0, False, 0
    keep = []
    with open(path, errors="ignore") as f:
        for line in f:
            if "EXPTAG=" in line and not tag:
                tag = line.strip().split("EXPTAG=", 1)[1]
            if "Hello from" in line:
                hello += 1
            if "NCCL WARN NET/OFI Request" in line:
                neterr += 1
            if '"reduced_train_loss": NaN' in line:
                nan = True
            if line.startswith("Traceback"):
                tb += 1
            m = re.search(r"RUN_EXIT=(\d+)", line)
            if m:
                run_exit = m.group(1)
            if "=" in line and len(line) < 2000:      # config lines are short and contain "="
                for k, rx in CFG_RE.items():
                    if k in line:
                        for v in rx.findall(line):
                            cfg[k] = v
            if KEEP_RE.search(line):
                keep.append(line)
    return cfg, tag, run_exit, hello, neterr, nan, tb, keep


def watch_verdict(job, run_dir):
    for d in (os.path.dirname(run_dir.rstrip("/")), run_dir):
        p = os.path.join(d, f"watch_{job}.log")
        if os.path.exists(p):
            txt = open(p, errors="ignore").read()
            if "NaN loss detected" in txt:
                return "cancelled-nan"
            if "no log update" in txt:
                return "cancelled-hang"
            return ""
    return ""


def sacct(job):
    try:
        out = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o", "State,Start,End,Elapsed,NodeList%20000"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return dict(zip(["slurm_state", "start", "end", "elapsed", "nodelist"], out[0].split("|"))) if out else {}
    except Exception:
        return {}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")] or [os.path.join(ROOT, "results", "runs")]
    use_sacct = "--sacct" in sys.argv
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "kv6-l8b-*.out"))) if os.path.isdir(a) else [a]
    os.makedirs(os.path.join(OUT, "mllog"), exist_ok=True)
    runs, evals, trains = [], [], []
    for p in files:
        job = os.path.basename(p).rsplit("-", 1)[-1].split(".")[0]
        cfg, tag, run_exit, hello, neterr, nan, tb, keep = scan(p)
        r = parse(p)
        rs = r["run_start"]
        steps = sorted(st for _, s, st, _ in r["train"] if st and s)
        train_rate = r["gbs"] / steps[len(steps) // 2] if steps and r["gbs"] else None
        pts = [(t, s) for t, s, *_ in r["train"] if s] + [(t, s) for s, _, t in r["evals"] if s]
        avg_rate = None
        if rs and pts:
            t, s = max(pts)
            avg_rate = s / ((t - rs) / 1000.0) if t > rs else None
        maxsteps = int(cfg.get("MAX_STEPS", "0") or 0)
        vci = int(cfg.get("VAL_CHECK_INTERVAL", "0") or 0)
        if "EVAL_INTERVAL_AFTER" in cfg:
            sched = "relaxed"
        elif 0 < maxsteps <= 1000 or vci >= 1000:
            sched = "none"
        else:
            sched = "mlperf"
        # NaN never recovers, so it outranks the MLPerf stop status (short runs log "aborted")
        status = "nan" if nan else (r["status"] or watch_verdict(job, os.path.dirname(p)) or
                                    ("no-run-start" if not rs else "incomplete"))
        if tb and not rs and status == "no-run-start":
            status = "init-error"
        last = r["evals"][-1] if r["evals"] else (None, None, None)
        row = {"job_id": job, "tag": tag, "nodes": cfg.get("DGXNNODES", ""), "gbs": r["gbs"] or "",
               "tp": cfg.get("TENSOR_MODEL_PARALLEL", ""), "cp": cfg.get("CONTEXT_PARALLEL", ""),
               "minibs": cfg.get("MINIBS", ""), "mbs": cfg.get("MICRO_BATCH_SIZE", ""),
               "lr": r["hp"].get("opt_base_learning_rate", cfg.get("LR", "")),
               "warmup_steps": r["hp"].get("opt_learning_rate_warmup_steps", cfg.get("WARMUP_STEPS", "")),
               "seed": r["hp"].get("seed", ""), "eval_schedule": sched,
               "val_check_interval": vci or "", "eval_interval_after": cfg.get("EVAL_INTERVAL_AFTER", ""),
               "max_steps": maxsteps or "", "bucket_size": cfg.get("BUCKET_SIZE", ""),
               "num_distopt_instances": cfg.get("NUM_DISTOPT_INSTANCES", ""),
               "tp_comm_overlap": cfg.get("TP_COMM_OVERLAP", ""), "fp8_dpa": cfg.get("FP8_DPA", ""),
               "overlap_pg_optim": cfg.get("OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP", ""),
               "netprofile": cfg.get("NETPROFILE", ""), "status": status, "run_exit": run_exit,
               "ttt_min": round(ttt_min(r), 3) if ttt_min(r) else "",
               "samples_last_eval": last[0] or "", "log_ppl_last_eval": round(last[1], 4) if last[1] else "",
               "n_evals": len(r["evals"]),
               "train_samples_per_s": round(train_rate, 2) if train_rate else "",
               "avg_samples_per_s": round(avg_rate, 2) if avg_rate else "",
               "init_min": round((rs - r["init_start"]) / 60000, 2) if rs and r["init_start"] else "",
               "hello_ranks": hello, "net_errors": neterr, "log": os.path.relpath(p, ROOT)}
        if use_sacct:
            row.update(sacct(job))
        runs.append(row)
        for s, v, t in r["evals"]:
            evals.append({"job_id": job, "gbs": r["gbs"], "samples": s, "log_ppl": round(v, 4),
                          "minutes_since_run_start": round((t - rs) / 60000, 3) if rs else ""})
        for t, s, st, loss in r["train"]:
            trains.append({"job_id": job, "samples": s, "minutes_since_run_start": round((t - rs) / 60000, 3) if rs else "",
                           "step_time_s": st, "train_loss": loss})
        with gzip.open(os.path.join(OUT, "mllog", f"{job}.txt.gz"), "wt") as g:
            g.writelines(keep)
    for name, rows in (("runs.csv", runs), ("eval_curves.csv", evals), ("train_curves.csv", trains)):
        if not rows:
            continue
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(runs)} runs, {len(evals)} evaluations, {len(trains)} training records -> {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
