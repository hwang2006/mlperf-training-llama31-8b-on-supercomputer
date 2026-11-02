#!/bin/bash
# Runs INSIDE an enroot container of the NGC base image (fake root), with the host's
# /opt/cray and $HOSTLIBS (host deps of Cray libfabric) mounted, and /out = $OFI_DIR.
# Builds aws-ofi-nccl WITHOUT the AWS platform code: NGC's build (1.17.3, platform-aws)
# throws in plugin init on Slingshot because it reads /sys/class/infiniband/<nic>/node_guid,
# which CXI NICs do not have.
set -euxo pipefail
: "${OFI_VERSION:=v1.17.3}" "${CRAY_LIBFABRIC:?}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends libhwloc-dev autoconf automake libtool pkg-config

# Link-time view of Cray libfabric: libfabric.so (dev symlink may be missing on the host)
# + its host-only dependencies (libcxi, libnl, json-c, ...) collected in /hostlibs.
FABLINK=/tmp/fablink && rm -rf $FABLINK && mkdir -p $FABLINK
ln -s "$CRAY_LIBFABRIC/lib64/libfabric.so.1" $FABLINK/libfabric.so
ls -la "$CRAY_LIBFABRIC/lib64/" /hostlibs/
export LD_LIBRARY_PATH=$CRAY_LIBFABRIC/lib64:/hostlibs:${LD_LIBRARY_PATH:-}
export LDFLAGS="-L$FABLINK -Wl,-rpath-link,/hostlibs -Wl,-rpath-link,$CRAY_LIBFABRIC/lib64"

cd /tmp && rm -rf aws-ofi-nccl
git clone --depth 1 --branch "$OFI_VERSION" https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
./autogen.sh
if ! ./configure --prefix=/out \
      --with-libfabric="$CRAY_LIBFABRIC" --with-cuda=/usr/local/cuda \
      --disable-platform-aws --disable-tests; then
  echo "===== config.log: libfabric link test ====="
  grep -n -A25 "checking for library containing fi_getinfo" config.log | head -60
  exit 1
fi
grep -E "Platform-specific optimizations" config.log /dev/null || true
make -j8
make install
ls -la /out/lib
ldd /out/lib/libnccl-net-ofi.so | grep -E "fabric|hwloc|cxi|not found" || true
echo "OFI_BUILD_OK $OFI_VERSION platform=none libfabric=$CRAY_LIBFABRIC"
