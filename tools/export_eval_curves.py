#!/usr/bin/env python3
"""Export every evaluation of your runs to CSV (real timestamps from the MLPerf log).

  python3 tools/export_eval_curves.py results/runs/kv6-l8b-*.out > results/eval_curves_mine.csv
"""
import csv, json, os, re, sys

PAT = re.compile(r":::MLLOG (\{.*\})")
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["run", "tag", "gbs", "samples", "log_ppl", "minutes_since_run_start", "minutes_source"])
for path in sys.argv[1:]:
    tag, gbs, start = "", "", None
    for line in open(path, errors="ignore"):
        if line.startswith("EXPTAG=") and not tag:
            tag = line.strip().split("=", 1)[1]
        m = PAT.search(line)
        if not m:
            continue
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if d["key"] == "global_batch_size":
            gbs = d["value"]
        elif d["key"] == "run_start":
            start = d["time_ms"]
        elif d["key"] == "eval_accuracy" and start:
            w.writerow([os.path.basename(path).split("-")[-1].split(".")[0], tag, gbs,
                        d["metadata"].get("samples_count"), round(d["value"], 4),
                        round((d["time_ms"] - start) / 60000, 2), "measured"])
