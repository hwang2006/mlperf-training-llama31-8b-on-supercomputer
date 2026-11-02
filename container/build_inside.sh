#!/bin/bash
# Runs INSIDE the writable enroot container (fake root). Reproduces the RUN steps of
#   training_results_v6.0/NVIDIA/benchmarks/llama31_8b/implementations/nemo/Dockerfile
# on top of nvcr.io/nvidia/pytorch:26.04-py3 (arm64 / sbsa).
#
# Deviations from a plain `docker build`, all to get a reproducible GH200 image:
#   1. PIP_CONSTRAINT="" in the Dockerfile disables NGC's pins. We pin the NGC GPU stack
#      (torch, TE, triton, nvidia-*) to the versions in the base image and resolve all
#      other packages as of $UV_EXCLUDE_NEWER (uv --exclude-newer). Abort if torch changes.
#   2. NCCL v2.30.4-1 is built for sm_90 only (GH200) to keep the build short.
#   3. TE_REVISION=SKIP as upstream -> NGC's TE (has sm_90). Never rebuild TE with the
#      upstream NVTE_CUDA_ARCHS="100a;103a" (Blackwell only).
# Mounts:  /src = NVIDIA implementation dir (COPY . /workspace/llm),  /build = this dir
set -euxo pipefail

export MAX_JOBS=8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # login-node per-user thread limit
export PIP_CONSTRAINT=""
export DEBIAN_FRONTEND=noninteractive
: "${UV_EXCLUDE_NEWER:=2026-05-15}"
git config --global user.name a && git config --global user.email a

# --- 0. Pin NGC GPU stack ------------------------------------------------------
pip list --format=freeze 2>/dev/null \
  | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton|transformer[-_]engine|numpy|apex|flash[-_]attn|nvidia-|cuda-|cupy|tensorrt|onnx)' \
  | grep -viE '^(torchao|nvidia-mathdx|nvidia-resiliency-ext)' \
  > /workspace/ngc-pins.txt
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
pip install uv
UV="uv pip install --system --break-system-packages -c /workspace/ngc-pins.txt --exclude-newer ${UV_EXCLUDE_NEWER}"

# --- 1. cuDNN 9.21.1.3 (Dockerfile: libcudnn9-*-cuda-13=9.21.1.3-1) -------------
NVARCH=$(uname -m | grep x86_64 2>/dev/null || echo sbsa)
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list
apt-get update
apt-get install -y --allow-downgrades libcudnn9-cuda-13=9.21.1.3-1 libcudnn9-dev-cuda-13=9.21.1.3-1 libcudnn9-headers-cuda-13=9.21.1.3-1

# --- 2. NCCL v2.30.4-1 from source (sm_90) -------------------------------------
cd /workspace
apt-get install -y --no-install-recommends fakeroot devscripts debhelper
apt-get purge -y libnccl-dev libnccl2 || true
rm -rf nccl && git clone https://github.com/NVIDIA/nccl.git && cd nccl
git checkout v2.30.4-1
make -j${MAX_JOBS} pkg.debian.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
dpkg -i build/pkg/deb/*.deb
cd /workspace && rm -rf nccl
apt-get purge -y fakeroot devscripts debhelper || true

# --- 3. PyTorch checkpoint size patch -------------------------------------------
SITE=$(python -c "import torch,os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
patch --forward --directory="$SITE" -p1 < /src/pytorch_ckpt.patch || echo "WARN: patch already applied?"

$UV numcodecs==0.16.3 nvidia-mathdx==25.1.1

# --- 4. Megatron-Bridge + Megatron-LM 26.04-alpha.rc1 ---------------------------
uv pip uninstall --system --break-system-packages megatron-core || true
rm -rf /workspace/Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge
cd /workspace/Megatron-Bridge
git checkout 26.04-alpha.rc1
git submodule update --init --recursive
git rev-parse HEAD > /MBRIDGE_COMMIT_HASH.env
cd 3rdparty/Megatron-LM
git checkout 26.04-alpha.rc1
git rev-parse HEAD > /MCORE_COMMIT_HASH.env
$UV --no-build-isolation -e .
( cd megatron/core/datasets && [ -f Makefile ] && make ) || echo "WARN: dataset helpers will be JIT-compiled"

# --- 5. TE: TE_REVISION=SKIP -> keep NGC's Transformer Engine -------------------

# --- 6. requirements.txt ---------------------------------------------------------
pip uninstall -y transformers torchao || true
$UV -r /src/requirements.txt

# --- 7. Benchmark code (COPY . /workspace/llm) + custom embedding (sm_90 included)
mkdir -p /workspace/llm && cp -a /src/. /workspace/llm/
cd /workspace/llm/embedding_lib
TORCH_CUDA_ARCH_LIST="9.0" pip install --force-reinstall --no-deps --no-build-isolation .

# --- 8. ENV (enroot reads /etc/environment) --------------------------------------
PP=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM
grep -q 'Megatron-Bridge/src' /etc/environment || echo "PYTHONPATH=$PP" >> /etc/environment

# --- 9. Verify ---------------------------------------------------------------------
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || { echo "ERROR: torch changed $TORCH_BEFORE -> $TORCH_AFTER"; exit 1; }
cd /workspace/llm
PYTHONPATH=$PP python - <<'PY' 2>&1 | grep -vE 'UserWarning|getDeviceCount'
import torch, transformer_engine as te, megatron.core as mc, megatron.bridge as mb
import mlperf_logging, mlperf_common, transformers, custom_embedding
print("FINAL torch", torch.__version__, "cuda", torch.version.cuda, "| TE", te.__version__,
      "| MCore", mc.__version__, "| MBridge", getattr(mb, "__version__", "?"),
      "| transformers", transformers.__version__, "| NCCL", torch.cuda.nccl.version())
PY
which slurm2pytorch bindpcie
echo "== aws-ofi-nccl after NCCL rebuild"
ls -la /opt/amazon/aws-ofi-nccl/lib* 2>/dev/null || true
find / -xdev \( -name "libnccl-net*.so*" -o -name "libnccl-ofi*.so*" \) 2>/dev/null | head

# --- 10. Record and clean -----------------------------------------------------------
pip list --format=freeze 2>/dev/null > /workspace/kv6-pip-freeze.txt
echo "MBRIDGE=$(cat /MBRIDGE_COMMIT_HASH.env) MCORE=$(cat /MCORE_COMMIT_HASH.env) NCCL=v2.30.4-1 EXCLUDE_NEWER=$UV_EXCLUDE_NEWER" > /workspace/kv6-build-info.txt
rm -rf /root/.cache/pip /root/.cache/uv /var/lib/apt/lists/*
apt-get clean || true
echo "BUILD_INSIDE_OK"
