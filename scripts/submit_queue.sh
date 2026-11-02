#!/bin/bash
# Hangang allows 5 submitted jobs per user (QOSMaxSubmitJobPerUserLimit). This submits the
# lines of a queue file one by one whenever a slot is free.
#   setsid nohup ./scripts/submit_queue.sh recipes/gbs32_scaling.txt > results/queue.log 2>&1 < /dev/null &
# Queue file: one "NODES [KEY=VALUE ...]" per line, '#' comments allowed.
cd "$(dirname "$0")/.."
Q=${1:?usage: submit_queue.sh QUEUE_FILE}; MAXJOBS=${MAXJOBS:-5}
grep -vE '^\s*(#|$)' "$Q" | while read -r spec; do
  tries=0
  while :; do
    n=$(squeue -u "$USER" -h | wc -l)
    if [ "$n" -lt "$MAXJOBS" ]; then
      echo "$(date '+%F %T') slot free ($n jobs) -> ./submit.sh $spec"
      if ./submit.sh $spec < /dev/null; then break; fi
      echo "$(date '+%F %T') submit failed, retry later"
    fi
    tries=$((tries + 1))
    [ "$tries" -gt 300 ] && { echo "$(date '+%F %T') GIVE UP: $spec"; break; }
    sleep 300
  done
  sleep 30
done
echo "$(date '+%F %T') QUEUE_DONE"
