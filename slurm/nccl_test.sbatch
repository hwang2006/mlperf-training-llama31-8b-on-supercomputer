#!/bin/bash
# 2-node NCCL all-reduce over Slingshot CXI (self-built aws-ofi-nccl in $OFI_DIR):
# CPU binding / GPUDirect RDMA / plugin protocol variants, plus TCP as reference.
#   usage: sbatch -N 2 -p normal --comment="field=efficientai;appl=megatron-lm" slurm/nccl_test.sbatch
#   TEST_IMG defaults to the NGC base image (works before 02_build finishes).
#SBATCH -J kv6-nccl
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -o results/kv6-nccl-%j.log

set -x
source "$SLURM_SUBMIT_DIR/env.sh"
TEST_IMG=${TEST_IMG:-$BASE_IMG}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29511
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
srun --version; lscpu | grep -E "^CPU\(s\)|NUMA node[0-9]"

if [ ! -f "$HOSTLIBS/.done" ]; then
  mkdir -p "$HOSTLIBS"
  ldd "$CRAY_LIBFABRIC/lib64/libfabric.so.1" | awk '/=> \//{print $3}' \
    | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux|libgcc_s|libstdc\+\+)[.-]' \
    | xargs -r -I{} cp -L {} "$HOSTLIBS/" && touch "$HOSTLIBS/.done"
fi
MNT=/opt/cray:/opt/cray:ro,$HOSTLIBS:/hostlibs:ro,$OFI_DIR:/ofi:ro,$REPO_DIR/tools:/tools:ro
CNAME=kv6nccl_$SLURM_JOB_ID
srun --ntasks-per-node=1 --container-image="$TEST_IMG" --container-name="$CNAME" --no-container-mount-home true

CXI="export LD_LIBRARY_PATH=/ofi/lib:$CRAY_LIBFABRIC/lib64:/hostlibs:\$LD_LIBRARY_PATH NCCL_NET_PLUGIN=ofi FI_PROVIDER=cxi FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072 NCCL_CROSS_NIC=1"

run() {  # $1 = label, $2 = extra srun options, $3 = env exports evaluated inside the container
  echo "########## $1"
  NET="$3" srun --ntasks-per-node=4 $2 --mpi=none --container-name="$CNAME" --no-container-mount-home \
    --container-mounts="$MNT" bash -c 'eval "$NET"; python /tools/nccl_bench.py' 2>&1 \
    | grep -E "AFFINITY|allreduce|world=|Using network|Using transport protocol|GPU Direct RDMA|GDR|NCCL WARN|Error" \
    | sed 's/^\[[^]]*\] //; s/^node[0-9]*:[0-9]*:[0-9]* //' | sort | uniq -c | sort -rn | head -30
}
run "C1: CXI, srun default CPU binding (as previous test)" "" "$CXI"
run "C2: CXI, --cpus-per-task=16"                        "--cpus-per-task=16" "$CXI"
run "C3: CXI, --cpus-per-task=16, NCCL_NET_GDR_LEVEL=PHB" "--cpus-per-task=16" "$CXI NCCL_NET_GDR_LEVEL=PHB"
run "C4: C3 + OFI_NCCL_PROTOCOL=RDMA"                     "--cpus-per-task=16" "$CXI NCCL_NET_GDR_LEVEL=PHB OFI_NCCL_PROTOCOL=RDMA"
run "A: TCP sockets, --cpus-per-task=16 (reference)"      "--cpus-per-task=16" "export NCCL_NET_PLUGIN=none NCCL_NET=Socket"
echo "NCCL_TEST_DONE"
