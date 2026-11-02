#!/bin/bash
# Build the benchmark image ($IMG) from $BASE_IMG with enroot on a Hangang LOGIN node
# (aarch64, Internet access):
#   nohup ./scripts/02_build_container.sh > results/kv6-build.log 2>&1 &
# NCCL is compiled from source (sm_90 only), so expect ~40-70 min in total.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

[ "$(uname -m)" = "aarch64" ] || { echo "Must run on an aarch64 node (got $(uname -m))"; exit 1; }
[ -f "$BASE_IMG" ] || { echo "Base image not found: $BASE_IMG (run 01_import_base_image.sbatch)"; exit 1; }
[ -d "$NV_IMPL" ]  || { echo "NVIDIA sources not found (run 00_fetch_sources.sh)"; exit 1; }

export ENROOT_TEMP_PATH=$ENROOT_LOCAL/enroot-tmp
export ENROOT_DATA_PATH=$ENROOT_LOCAL/enroot-data
export ENROOT_RUNTIME_PATH=$ENROOT_LOCAL/enroot-run
export ENROOT_MAX_PROCESSORS=8          # login-node per-user thread limit
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH"
df -h /tmp

NAME=kv6-l8b-build
enroot remove -f "$NAME" 2>/dev/null || true
enroot create -n "$NAME" "$BASE_IMG"

enroot start --rw --root \
  -e UV_EXCLUDE_NEWER="$UV_EXCLUDE_NEWER" \
  --mount "$NV_IMPL:/src" \
  --mount "$REPO_DIR/container:/build" \
  "$NAME" bash /build/build_inside.sh

rm -f "$IMG"
enroot export -o "$IMG" "$NAME"
enroot remove -f "$NAME"
ls -lh "$IMG"
echo "BUILD_EXIT=0"
