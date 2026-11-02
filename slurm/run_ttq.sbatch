#!/bin/bash
# MLPerf Training Llama 3.1 8B time-to-train on Hangang GH200 nodes.
# Use ./submit.sh (sets -N, partition, comment, time and OV_* overrides).
# Launch model = NVIDIA run.sub: one Slurm task per GPU, slurm2pytorch -> run_and_time.sh
#SBATCH -J kv6-l8b
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH -o results/runs/%x-%j.out

set -x
source "$SLURM_SUBMIT_DIR/env.sh"
export NNODES=$SLURM_JOB_NUM_NODES
NETMODE=${NETMODE:-C}
source "$REPO_DIR/configs/config_GH200_8b.sh"

TAG=n${NNODES}-gbs${EXP_GBS}-tp${TENSOR_MODEL_PARALLEL}cp${CONTEXT_PARALLEL}-mb${MINIBS}-lr${LR}-wu${WARMUP_STEPS}${OV_FP8_ATTN:+-fp8attn}-net${NETMODE}${NETPROFILE:+-$NETPROFILE}
LOGDIR=$RESULTS/$TAG-$SLURM_JOB_ID
NPYDIR=$NPY_INDEX_ROOT/$SLURM_JOB_ID
mkdir -p "$LOGDIR/.cache" "$NPYDIR"
echo "EXPTAG=$TAG"

# ---- Runtime environment --------------------------------------------------------
export SEED=${SEED:-$RANDOM}
# Rank 0 must run on MASTER_ADDR. Slurm may order the allocation by network topology,
# so pin task placement with an explicit hostfile (as NVIDIA run.sub does).
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${NODES[0]}
HOSTFILE=$LOGDIR/hostfile
for n in "${NODES[@]}"; do for i in $(seq 1 $DGXNGPU); do echo "$n"; done; done > "$HOSTFILE"
export MASTER_PORT=29500
export PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM
export TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
export HF_HOME=/results/.cache/huggingface XDG_CACHE_HOME=/results/.cache HOME=/results
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
env | grep -E '^(DGX|MINIBS|TENSOR_MODEL|CONTEXT_PARALLEL|MICRO_BATCH|LR|WARMUP_STEPS|VAL_CHECK|MAX_STEPS|SEED|FP8|FP8_DPA|MCORE_CUDA_GRAPH|FULL_CUDA_GRAPH|BINDCMD)=' | sort

MNT=$LOGDIR:/results,$NPYDIR:/npy_index,$PREPROC_DATA:/preproc_data:ro,$TOKENIZER:/workspace/llm/nemo_tokenizer:ro

if [ "$NETMODE" = "C" ]; then
  if [ ! -f "$HOSTLIBS/.done" ]; then
    mkdir -p "$HOSTLIBS"
    ldd "$CRAY_LIBFABRIC/lib64/libfabric.so.1" | awk '/=> \//{print $3}' \
      | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux|libgcc_s|libstdc\+\+)[.-]' \
      | xargs -r -I{} cp -L {} "$HOSTLIBS/" && touch "$HOSTLIBS/.done"
  fi
  MNT=$MNT,/opt/cray:/opt/cray:ro,$HOSTLIBS:/hostlibs:ro,$OFI_DIR:/ofi:ro
  # Self-built aws-ofi-nccl (no AWS platform code) from $OFI_DIR, see 03_build_ofi_plugin.sh
  # (NGC 26.04 plugin 1.17.3 aborts on CXI: reads /sys/class/infiniband/cxi0/node_guid)
  export NETENV="export LD_LIBRARY_PATH=/ofi/lib:$CRAY_LIBFABRIC/lib64:/hostlibs:\$LD_LIBRARY_PATH \
    NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN_NAME:-ofi} FI_PROVIDER=cxi \
    FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd \
    FI_CXI_DEFAULT_CQ_SIZE=131072 NCCL_CROSS_NIC=1"
  # NETPROFILE=cscs: CSCS Alps (GH200 + Slingshot) NCCL recommendations, incl. the
  # documented workaround for NCCL hangs / watchdog timeouts (disable CXI eager messages)
  case "${NETPROFILE:-base}" in
    cscs) NETENV="$NETENV FI_CXI_DEFAULT_TX_SIZE=16384 FI_CXI_RX_MATCH_MODE=software NCCL_PROTO=^LL128 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0" ;;
    cscs-nordzv) NETENV="$NETENV FI_CXI_DEFAULT_TX_SIZE=16384 FI_CXI_RX_MATCH_MODE=software NCCL_PROTO=^LL128" ;;
    cscs-rdzv)   NETENV="$NETENV FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0" ;;
    base) ;;
    *) echo "unknown NETPROFILE=$NETPROFILE"; exit 1 ;;
  esac
  echo "NETPROFILE=${NETPROFILE:-base}"
else
  export NETENV=':'
fi

CNAME=kv6l8b_$SLURM_JOB_ID
# 1) Create the container once per node (removes start-time skew from the timed part)
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 --container-image="$IMG" --container-name="$CNAME" \
     --container-writable --no-container-mount-home true || { echo "CONTAINER_SETUP_FAILED"; exit 1; }
srun -N1 -n1 --container-name="$CNAME" --container-writable --no-container-mount-home \
     bash -c 'nvidia-smi -L; cat /workspace/kv6-build-info.txt'

# 2) Benchmark: 4 tasks/node, slurm2pytorch sets RANK/LOCAL_RANK/WORLD_SIZE from Slurm
SLURM_HOSTFILE=$HOSTFILE srun -n $((NNODES * DGXNGPU)) --distribution=arbitrary --kill-on-bad-exit=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --mpi=none \
  --container-name="$CNAME" --container-writable --no-container-mount-home \
  --container-mounts="$MNT" --container-workdir=/workspace/llm \
  bash -c 'export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:$PATH
           eval "$NETENV"
           exec slurm2pytorch ./run_and_time.sh' \
  2>&1 | tee "$LOGDIR/run.log"
echo "RUN_EXIT=${PIPESTATUS[0]}"
