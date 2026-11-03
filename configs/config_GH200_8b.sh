# GH200 (Hopper, 4 GPUs/node) config for the NVIDIA v6.0 llama31_8b implementation.
# Sourced on the HOST by slurm/run_ttq.sbatch; all exported vars reach the container.
# Inputs: NNODES, NV_IMPL (env.sh), optional OV_* overrides from submit.sh.
: "${NNODES:?}" "${NV_IMPL:?}"

# Same order as NVIDIA's 8B configs, WITHOUT config_common_fp4.sh (Hopper has no FP4)
source "$NV_IMPL/config_common.sh"
source "$NV_IMPL/config_common_cg.sh"
source "$NV_IMPL/config_common_8b.sh"
[ "${OV_FP8_ATTN:-0}" = 1 ] && source "$NV_IMPL/config_common_fp8attn.sh"

export DGXNNODES=$NNODES
export DGXNGPU=4
export GPU_ARCH=h100                          # tp-overlap yaml lookup (optional)

export TENSOR_MODEL_PARALLEL=${OV_TP:-1}
export CONTEXT_PARALLEL=${OV_CP:-1}
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
if [ "$TENSOR_MODEL_PARALLEL" -gt 1 ]; then export SEQ_PARALLEL=True; else export SEQ_PARALLEL=False; fi
export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=${OV_MBS:-1}

# GBS = MINIBS * DP,  DP = 4*N / (TP*CP)
GBS=${OV_GBS:-32}
DP=$(( DGXNGPU * DGXNNODES / (TENSOR_MODEL_PARALLEL * CONTEXT_PARALLEL) ))
if [ "$DP" -lt 1 ] || [ $(( GBS % (DP * MICRO_BATCH_SIZE) )) -ne 0 ]; then
  echo "ERROR: GBS=$GBS not divisible by DP*MBS=$DP*$MICRO_BATCH_SIZE (N=$NNODES TP=$TENSOR_MODEL_PARALLEL CP=$CONTEXT_PARALLEL)"; exit 1
fi
export MINIBS=$(( GBS / DP ))

# Hyperparameters: Cisco v6.0 2x8xH200 (GBS 32) = LR 4e-4, warmup 20
export LR=${OV_LR:-0.0004}
export WARMUP_STEPS=${OV_WARMUP_STEPS:-20}
# Eval every 12288 sequences (training_rules v6.1). Cisco v6.0 used 8192 (VAL_CHECK_INTERVAL=8192/GBS).
export VAL_CHECK_INTERVAL=${OV_VAL_CHECK_INTERVAL:-$(( (12288 + GBS - 1) / GBS ))}
# Smoke tests only (changes nothing else; LR decay stays at 1.2M steps)
# MAX_STEPS sizes the dataset index rank 0 builds before training. With the MLPerf value
# (1.2M steps) a GBS of 256 needs >10 min and the other ranks hit the NCCL timeout, so for
# GBS >= 128 cap it at 2M samples (5x what GBS 128 needs). LR decay still uses 1.2M steps.
if [ -n "${OV_MAX_STEPS:-}" ]; then export MAX_STEPS=$OV_MAX_STEPS
elif [ "$GBS" -ge 128 ]; then export MAX_STEPS=$(( 2000000 / GBS )); fi
# Non-MLPerf evaluation schedule: first eval near EVAL_START samples, then every EVAL_EVERY
# samples (needs patches/callback_logging.py, created by scripts/00_fetch_sources.sh)
if [ -n "${OV_EVAL_EVERY:-}" ]; then
  K=$(( (OV_EVAL_EVERY + GBS - 1) / GBS )); S0=$(( (${OV_EVAL_START:-12288} + GBS - 1) / GBS )); S0=$(( (S0 + K - 1) / K * K ))
  export VAL_CHECK_INTERVAL=$S0 EVAL_INTERVAL_AFTER=$K
fi

# NVIDIA default "bindpcie --cpu=node" is untested on Grace; off unless requested
export BINDCMD=${OV_BINDCMD:-}

export DGXSYSTEM=GH200_${DGXNNODES}x4x${MINIBS}xtp${TENSOR_MODEL_PARALLEL}pp1cp${CONTEXT_PARALLEL}_8b
export EXP_GBS=$GBS EXP_DP=$DP
