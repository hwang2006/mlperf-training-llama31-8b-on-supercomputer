#!/bin/bash
# Sparse checkout of MLCommons Training v6.0 results (a few MB):
#   * NVIDIA llama31_8b NeMo implementation (Dockerfile recipe, pretrain.py, configs)
#   * Cisco 2x8xH200 implementation + result logs (the only Hopper 8B submission in v6.0)
# Run on a login node from the repository root:  ./scripts/00_fetch_sources.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p "$THIRD_PARTY" && cd "$THIRD_PARTY"
if [ ! -d v60 ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/mlcommons/training_results_v6.0 v60
fi
( cd v60 && git sparse-checkout set --no-cone \
    "/NVIDIA/benchmarks/llama31_8b/implementations/nemo/" \
    "/Cisco/benchmarks/llama31_8b/implementations/nemo/" \
    "/Cisco/results/2xC885A_M8-8xH200-SXM-141GB/" \
    "/Cisco/systems/2xC885A_M8-8xH200-SXM-141GB.json" \
  && git rev-parse HEAD > ../v60.commit )
echo "== NVIDIA v6.0 llama31_8b implementation"; ls "$NV_IMPL"
echo "== Cisco H200 results";                    ls "$CISCO_RESULTS"
echo "training_results_v6.0 commit: $(cat "$THIRD_PARTY/v60.commit")"

# Patched callback for the optional non-MLPerf evaluation schedule (EVAL_START/EVAL_EVERY).
# Without EVAL_INTERVAL_AFTER in the environment it behaves exactly like the original.
mkdir -p "$REPO_DIR/patches"
python3 - "$NV_IMPL/callback_logging.py" "$REPO_DIR/patches/callback_logging.py" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = '        if not os.environ.get("VAL_CHECK_INTERVAL"):\n'
new = ('        if os.environ.get("EVAL_INTERVAL_AFTER"):\n'
       '            # first eval at VAL_CHECK_INTERVAL, then every EVAL_INTERVAL_AFTER steps\n'
       '            context.state.cfg.validation.eval_interval = int(os.environ["EVAL_INTERVAL_AFTER"])\n'
       '        elif not os.environ.get("VAL_CHECK_INTERVAL"):\n')
assert s.count(old) == 1, "callback_logging.py changed upstream; patch not applied"
open(dst, "w").write(s.replace(old, new)); print("wrote", dst)
PY
