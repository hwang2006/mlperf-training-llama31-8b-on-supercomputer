#!/bin/bash
# Submit one MLPerf Training Llama 3.1 8B run on Hangang and (by default) start the
# auto-watchdog that cancels it on NaN loss or a silent hang.
#
#   ./submit.sh NODES [KEY=VALUE ...]
#
# Keys that override configs/config_GH200_8b.sh:
#   GBS TP CP MBS LR WARMUP_STEPS VAL_CHECK_INTERVAL MAX_STEPS FP8_ATTN=1 BINDCMD=...
# Run-time keys:
#   NETPROFILE=cscs-nordzv|cscs|cscs-rdzv|base   (default cscs-nordzv, see docs/TROUBLESHOOTING.md)
#   NETMODE=C|A        C = Slingshot CXI (default), A = TCP sockets
#   SEED=<int>  NCCL_DEBUG=INFO|WARN  TIME=HH:MM:SS (default depends on NODES)
#   WATCH=0            do not start tools/watch_job.sh
#   STALL=<minutes>    watchdog stall limit (default 5)
#
# Validated recipes: recipes/README.md.  Examples:
#   ./submit.sh 4 MAX_STEPS=200 TIME=00:40:00          # smoke test
#   ./submit.sh 64 GBS=64 CP=4 LR=0.0008 WARMUP_STEPS=64 TIME=00:50:00
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
NODES=${1:?usage: ./submit.sh NODES [KEY=VALUE ...]}; shift
case "$PWD" in /scratch/*|/appsdata/*) ;; *) echo "Submit from /scratch or /appsdata (Hangang policy)"; exit 1;; esac

case $NODES in 1) TIME=12:00:00;; 2) TIME=06:30:00;; 4) TIME=03:30:00;; 8) TIME=02:00:00;; *) TIME=01:30:00;; esac
EXPORTS="ALL"; NETPROFILE=cscs-nordzv; WATCH=1; STALL=5
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  case "$k" in
    GBS|TP|CP|MBS|LR|WARMUP_STEPS|VAL_CHECK_INTERVAL|MAX_STEPS|FP8_ATTN|BINDCMD) EXPORTS="$EXPORTS,OV_$k=$v" ;;
    NETMODE|SEED|NCCL_DEBUG) EXPORTS="$EXPORTS,$k=$v" ;;
    NETPROFILE) NETPROFILE=$v ;;
    TIME) TIME=$v ;;
    WATCH) WATCH=$v ;;
    STALL) STALL=$v ;;
    *) echo "Unknown key: $k"; exit 1 ;;
  esac
done
EXPORTS="$EXPORTS,NETPROFILE=$NETPROFILE"
mkdir -p "$RESULTS"
EXCL=(); [ -n "$EXCLUDE_NODES" ] && EXCL=(--exclude="$EXCLUDE_NODES")
JOB=$(sbatch --parsable -N "$NODES" -p "$SLURM_PARTITION" --comment="$SLURM_COMMENT" -t "$TIME" \
  "${EXCL[@]}" --export="$EXPORTS" slurm/run_ttq.sbatch)
JOB=${JOB%%;*}
echo "Submitted job $JOB: $NODES nodes, TIME=$TIME, NETPROFILE=$NETPROFILE, overrides: ${*:-none}"
echo "$(date '+%F %T') $JOB $NODES $* NETPROFILE=$NETPROFILE TIME=$TIME" >> results/submitted.log
if [ "$WATCH" != 0 ]; then
  setsid nohup tools/watch_job.sh "$JOB" "$STALL" > "results/watch_$JOB.log" 2>&1 < /dev/null &
  echo "Watchdog: results/watch_$JOB.log   Log: results/runs/kv6-l8b-$JOB.out"
fi
