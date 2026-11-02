#!/bin/bash
# Build aws-ofi-nccl for Slingshot CXI into $OFI_DIR (login node, ~5-10 min):
#   ./scripts/03_build_ofi_plugin.sh 2>&1 | tee results/kv6-ofi-build.log
set -euo pipefail
source "$(dirname "$0")/../env.sh"
[ -f "$BASE_IMG" ] || { echo "Base image not found: $BASE_IMG"; exit 1; }
[ -f "$CRAY_LIBFABRIC/include/rdma/fabric.h" ] || { echo "No libfabric headers in $CRAY_LIBFABRIC/include"; exit 1; }
# Host dependencies of Cray libfabric (same collection as slurm/run_ttq.sbatch)
if [ ! -f "$HOSTLIBS/.done" ]; then
  mkdir -p "$HOSTLIBS"
  ldd "$CRAY_LIBFABRIC/lib64/libfabric.so.1" | awk '/=> \//{print $3}' \
    | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux|libgcc_s|libstdc\+\+)[.-]' \
    | xargs -r -I{} cp -L {} "$HOSTLIBS/" && touch "$HOSTLIBS/.done"
fi
export ENROOT_TEMP_PATH=$ENROOT_LOCAL/enroot-tmp ENROOT_DATA_PATH=$ENROOT_LOCAL/enroot-data
export ENROOT_RUNTIME_PATH=$ENROOT_LOCAL/enroot-run ENROOT_MAX_PROCESSORS=8
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$OFI_DIR"
NAME=kv6-ofi-build
enroot remove -f "$NAME" 2>/dev/null || true
enroot create -n "$NAME" "$BASE_IMG"
enroot start --rw --root -e CRAY_LIBFABRIC="$CRAY_LIBFABRIC" \
  --mount /opt/cray:/opt/cray --mount "$HOSTLIBS:/hostlibs" --mount "$OFI_DIR:/out" \
  --mount "$REPO_DIR/container:/build" "$NAME" bash /build/build_ofi_inside.sh
enroot remove -f "$NAME"
ls -la "$OFI_DIR/lib"
