#!/bin/bash
# Import nvcr.io/nvidia/pytorch:26.04-py3 (arm64) as squashfs, then check on a GH200
# compute node that the host driver can run this CUDA 13.x container.
# Usage: source env.sh; then submit with -p "$SLURM_PARTITION" --comment="$SLURM_COMMENT" (see README)
# The NGC download took 15-60 min on Hangang. Layers are cached in $ENROOT_CACHE_PATH, so after a
# TIMEOUT simply resubmit: finished layers are not downloaded again.
#SBATCH -J kv6-import
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --comment="field=efficientai;appl=megatron-lm"
#SBATCH -o results/kv6-import-%j.log

set -x
source "$SLURM_SUBMIT_DIR/env.sh"
mkdir -p "$CONT_DIR" "$ENROOT_CACHE_PATH"
AVAIL_GB=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
if [ "$AVAIL_GB" -ge 60 ]; then LOCAL=/tmp/$USER; else LOCAL=/dev/shm/$USER; fi
export ENROOT_TEMP_PATH=$LOCAL/enroot-tmp
mkdir -p "$ENROOT_TEMP_PATH"

if [ ! -f "$BASE_IMG" ]; then
  enroot import -o "$BASE_IMG" "docker://nvcr.io#nvidia/pytorch:${NGC_TAG}"
  echo "IMPORT_EXIT=$?"
fi
rm -rf "$ENROOT_TEMP_PATH"
ls -lh "$BASE_IMG"

# ---- Host driver / container compatibility check -----------------------------
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun -N1 -n1 --container-image="$BASE_IMG" --no-container-mount-home bash -c '
  nvidia-smi -L
  python - <<PY
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok:", bool((x @ x).float().norm().item() > 0))
import transformer_engine as te
print("TE", te.__version__)
try:
    from transformer_engine.pytorch.fp8 import check_fp8_support
    print("TE fp8 support:", check_fp8_support())
except Exception as e:
    print("TE fp8 check skipped:", e)
print("NCCL (torch)", torch.cuda.nccl.version())
PY
  echo "== aws-ofi-nccl plugin candidates"
  ls -la /opt/amazon/aws-ofi-nccl/lib* 2>/dev/null
  find / -xdev \( -name "libnccl-net*.so*" -o -name "libnccl-ofi*.so*" \) 2>/dev/null | head
  echo "container LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
'
echo "CHECK_EXIT=$?"
