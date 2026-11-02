# =============================================================================
# env.sh -- central configuration (edit paths, then: source env.sh)
# MLPerf Training Llama 3.1 8B (NVIDIA v6.0 implementation) on KISTI Hangang (GH200)
#   * Jobs must be submitted from under /scratch or /appsdata (cli_filter).
#   * Every job needs --comment="field=<field>;appl=<program>" (see `showappl`).
# =============================================================================
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Containers ---------------------------------------------------------------
export CONT_DIR=/scratch/$USER/containers
export NGC_TAG=26.04-py3                                         # NVIDIA v6.0 8B base
export BASE_IMG=$CONT_DIR/kv6-pytorch-${NGC_TAG}-arm64.sqsh
export IMG=$CONT_DIR/kv6-nv60-llama31-8b-arm64.sqsh
# Resolve every non-NGC Python dependency as of this date (~ v6.0 submission time)
export UV_EXCLUDE_NEWER=2026-05-15

# --- Data (already downloaded + checksum-verified with mlc-r2-downloader) ------
export DATA_ROOT=/scratch/$USER/mlperf-llama31
export PREPROC_DATA=$DATA_ROOT/data/llama3_1_8b_preprocessed_c4_dataset   # -> /preproc_data
export TOKENIZER=$DATA_ROOT/tokenizer/llama3_1_8b_tokenizer_only          # -> /workspace/llm/nemo_tokenizer (6 tokenizer files only)
export NPY_INDEX_ROOT=$DATA_ROOT/npy_indices/nv60_gh200                   # new dir, do NOT reuse NEURON caches

# --- Upstream sources (scripts/00_fetch_sources.sh) ---------------------------
export THIRD_PARTY=$REPO_DIR/third_party
export NV_IMPL=$THIRD_PARTY/v60/NVIDIA/benchmarks/llama31_8b/implementations/nemo
export CISCO_IMPL=$THIRD_PARTY/v60/Cisco/benchmarks/llama31_8b/implementations/nemo
export CISCO_RESULTS=$THIRD_PARTY/v60/Cisco/results/2xC885A_M8-8xH200-SXM-141GB/llama31_8b

# --- Slingshot (CXI) host libraries (same as the LoRA repo) -------------------
export CRAY_LIBFABRIC=/opt/cray/libfabric/2.3.1
export HOSTLIBS=$REPO_DIR/hostlibs

# --- Outputs ------------------------------------------------------------------
export RESULTS=$REPO_DIR/results/runs

# --- Slurm --------------------------------------------------------------------
export SLURM_PARTITION=normal
export SLURM_COMMENT="field=efficientai;appl=megatron-lm"   # REQUIRED on Hangang: use your own values (see `showappl`)
# Nodes that broke multi-node runs on Hangang (2026-11-02; see docs/TROUBLESHOOTING.md):
#   NaN from the first step (cross-node CP runs): node0320-0324, node0541, node0739 (set A)
#   and node0104,0130,0334,0543,0752,0753,0846,0847,0849,1004,1048,1049,1083,1151 (set B);
#   "Error configuring interconnect": node0943; precautionary: node0316, node1119, node1125.
# Re-check with the system team before relying on this list; empty it to use all nodes.
export EXCLUDE_NODES="node0316,node1119,node1125,node0943,node0320,node0321,node0322,node0323,node0324,node0541,node0739,node0104,node0130,node0334,node0543,node0752,node0753,node0846,node0847,node0849,node1004,node1048,node1049,node1083,node1151"

# --- Enroot (build on a login node; temp/data must be node-local + exec-able) --
export ENROOT_CACHE_PATH=/scratch/$USER/enroot-tmp/cache
export ENROOT_LOCAL=/tmp/$USER

# --- aws-ofi-nccl built for Slingshot CXI (scripts/03_build_ofi_plugin.sh) ----
export OFI_DIR=$REPO_DIR/ofi-plugin
