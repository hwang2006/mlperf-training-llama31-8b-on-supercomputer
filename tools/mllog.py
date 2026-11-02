"""Minimal :::MLLOG parser shared by the tools in this directory."""
import json
import re

PAT = re.compile(r":::MLLOG (\{.*\})")


def parse(path):
    r = {"path": path, "gbs": None, "run_start": None, "run_stop": None, "status": None,
         "evals": [], "train": [], "hp": {}, "init_start": None}
    hp_keys = {"opt_base_learning_rate", "opt_learning_rate_warmup_steps", "seed",
               "gradient_accumulation_steps", "eval_samples", "number_of_nodes",
               "submission_platform"}
    with open(path, errors="ignore") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            try:
                j = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            k, md = j["key"], j.get("metadata", {}) or {}
            if k == "global_batch_size":
                r["gbs"] = j["value"]
            elif k == "init_start" and r["init_start"] is None:
                r["init_start"] = j["time_ms"]
            elif k == "run_start":
                r["run_start"] = j["time_ms"]
            elif k == "run_stop":
                r["run_stop"], r["status"] = j["time_ms"], md.get("status")
            elif k == "eval_accuracy":
                r["evals"].append((md.get("samples_count"), j["value"], j["time_ms"]))
            elif k == "tracked_stats" and isinstance(j["value"], dict):
                # cumulative samples_count + per-step time (NVIDIA callback_logging.py)
                v = j["value"]
                r["train"].append((j["time_ms"], md.get("samples_count"),
                                   v.get("train_step_time"), v.get("reduced_train_loss")))
            elif k in hp_keys:
                r["hp"][k] = j["value"]
    return r


def ttt_min(r):
    if r["run_start"] and r["run_stop"]:
        return (r["run_stop"] - r["run_start"]) / 60000.0
    return None


def olympic_mean(xs):
    """MLPerf Training score for LLM benchmarks: 10 runs, drop fastest and slowest."""
    xs = sorted(xs)
    if len(xs) >= 3:
        xs = xs[1:-1]
    return sum(xs) / len(xs) if xs else None
