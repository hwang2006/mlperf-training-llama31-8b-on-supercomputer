#!/bin/bash
# Auto-cancel a run on NaN loss or a stalled log. Stalls inside CUDA graphs are not seen
# by the PyTorch NCCL watchdog, so without this a hung job holds its nodes until TIME.
#   tools/watch_job.sh JOBID [stall_minutes]      (submit.sh starts it automatically)
cd "$(dirname "$0")/.."
J=${1:?usage: watch_job.sh JOBID [stall_minutes]}; STALL=${2:-5}
f=results/runs/kv6-l8b-$J.out
echo "$(date '+%F %T') watching job $J (stall limit ${STALL} min)"
while [ -n "$(squeue -h -j "$J" 2>/dev/null)" ]; do
  if [ -f "$f" ]; then
    if grep -q '"reduced_train_loss": NaN' "$f"; then
      echo "$(date '+%F %T') NaN loss detected -> scancel $J"; scancel "$J"; break
    fi
    if grep -q '"key": "run_start"' "$f"; then
      age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      if [ "$age" -gt $((STALL * 60)) ]; then
        echo "$(date '+%F %T') no log update for ${age}s -> scancel $J"; scancel "$J"; break
      fi
    fi
  fi
  sleep 30
done
echo "$(date '+%F %T') watch end. $(grep RUN_EXIT "$f" 2>/dev/null | tail -1)"
