#!/bin/bash
# Where am I in the setup? Prints one line per step and the next command to run.
#   ./scripts/check_setup.sh
cd "$(dirname "$0")/.."
source ./env.sh
PENDING=0
ok()   { printf "  [ OK ] %s\n" "$*"; }
todo() { printf "  [ .. ] %s\n" "$*"; PENDING=1; }
bad()  { printf "  [FAIL] %s\n" "$*"; PENDING=1; }
NEXT=""
next() { [ -z "$NEXT" ] && NEXT="$*"; }
echo "user=$USER partition=$SLURM_PARTITION comment=$SLURM_COMMENT"

# 0. sources and patch
if [ -f "$NV_IMPL/pretrain.py" ] && [ -f patches/callback_logging.py ]; then ok "0 sources: $NV_IMPL"
else todo "0 sources missing"; next "./scripts/00_fetch_sources.sh"; fi

# 1. data
missing=0
for p in "$PREPROC_DATA/c4-train.en_6_text_document.bin" "$PREPROC_DATA/c4-validation-91205-samples.en_text_document.bin" "$TOKENIZER/tokenizer.json"; do
  [ -f "$p" ] || missing=1; done
if [ $missing = 0 ] && [ "$(stat -c %s "$PREPROC_DATA/c4-train.en_6_text_document.bin")" = 83976774280 ]; then ok "1 data: $PREPROC_DATA"
elif pgrep -u "$(id -u)" -f "04_prepare_data.sh|mlc-r2-downloader" >/dev/null; then todo "1 data: download running ($(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1) so far)"; next "wait for the data download (tail -n2 results/data-download.log shows DATA_READY)"
else todo "1 data missing or incomplete"; next "setsid nohup ./scripts/04_prepare_data.sh > results/data-download.log 2>&1 < /dev/null &"; fi

# 2. base image
if squeue -u "$USER" -h -n kv6-import | grep -q .; then todo "2 base image: import job running ($(du -sh "$ENROOT_CACHE_PATH" 2>/dev/null | cut -f1) downloaded)"; next "wait for the kv6-import job (squeue -u \$USER)"
elif [ -f "$BASE_IMG" ]; then
  IL=$(ls -t results/kv6-import-*.log 2>/dev/null | head -1)
  if [ -n "$IL" ] && ! grep -q "CHECK_EXIT=0" "$IL"; then bad "2 base image exists but $IL has no CHECK_EXIT=0"; next "tail -n20 $IL"
  else ok "2 base image: $BASE_IMG ($(du -h "$BASE_IMG" | cut -f1))"; fi
else todo "2 base image missing"; next "sbatch -p \"\$SLURM_PARTITION\" --comment=\"\$SLURM_COMMENT\" scripts/01_import_base_image.sbatch"; fi

# 3. OFI plugin
if [ -f "$OFI_DIR/lib/libnccl-net-ofi.so" ]; then ok "3 aws-ofi-nccl: $OFI_DIR/lib/libnccl-net-ofi.so"
elif [ -f "$BASE_IMG" ]; then todo "3 aws-ofi-nccl not built"; next "./scripts/03_build_ofi_plugin.sh 2>&1 | tee results/ofi-build.log | tail -3"
else todo "3 aws-ofi-nccl: needs the base image first"; fi

# 4. benchmark container
if pgrep -u "$(id -u)" -f 02_build_container.sh >/dev/null; then todo "4 container: build running ($(tail -n1 results/build.log 2>/dev/null | cut -c1-60))"; next "wait for the container build (tail results/build.log)"
elif [ -f "$IMG" ] && grep -q "BUILD_EXIT=0" results/build.log 2>/dev/null; then ok "4 container: $IMG ($(du -h "$IMG" | cut -f1))"
elif [ -f "$IMG" ]; then ok "4 container: $IMG (no build log here)"
elif grep -q "BUILD_EXIT=" results/build.log 2>/dev/null; then bad "4 container build failed: $(grep -m1 -E 'Error|error' results/build.log | cut -c1-80)"; next "tail -n30 results/build.log"
elif [ -f "$BASE_IMG" ]; then todo "4 container not built"; next "setsid nohup ./scripts/02_build_container.sh > results/build.log 2>&1 < /dev/null &"
else todo "4 container: needs the base image first"; fi

# 5. network check
L=$(ls -t results/kv6-nccl-*.log 2>/dev/null | head -1)
if squeue -u "$USER" -h -n kv6-nccl | grep -q .; then todo "5 NCCL check: job running or queued"; next "wait for the kv6-nccl job (squeue -u \$USER)"
elif [ -n "$L" ] && grep -q "NCCL_TEST_DONE" "$L"; then
  bw=$(awk '/^########## C2/{f=1;next} /^##########/{f=0} f' "$L" | grep -oE "allreduce +2048 MB.*busbw +[0-9.]+" | awk '{print $NF}' | head -1)
  if [ -n "$bw" ] && [ "${bw%.*}" -ge 200 ]; then ok "5 NCCL check: $bw GB/s over CXI ($L)"; else bad "5 NCCL check: ${bw:-no} GB/s over CXI ($L)"; next "grep -E '^##########|allreduce' $L"; fi
elif [ -f "$IMG" ] || [ -f "$BASE_IMG" ]; then todo "5 NCCL check not run"; next "sbatch -N 2 -p \"\$SLURM_PARTITION\" --comment=\"\$SLURM_COMMENT\" --exclude=\"\$EXCLUDE_NODES\" slurm/nccl_test.sbatch"; fi

echo
if [ -n "$NEXT" ]; then echo "next: $NEXT"
elif [ $PENDING = 0 ]; then echo "setup complete: run the smoke test (README step 4)"
else echo "next: see the [ .. ] / [FAIL] lines above"; fi
