#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${TS_REPO_ROOT:-/home/u2601112087/tokenspeed}"
VENV_ROOT="${TS_VENV_ROOT:-/home/u2601112087/venvs/tokenspeed}"
CANN_ROOT="${TS_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
MODEL_DIR="${TS_MODEL_DIR:-/home/u2601112087/models/Qwen3.5-0.8B}"

source "${VENV_ROOT}/bin/activate"
source "${CANN_ROOT}/set_env.sh"
set -u

cd "${REPO_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export CPATH="/opt/miniconda3/envs/aics/include/python3.11:/opt/miniconda3/envs/aics/include:${CPATH:-}"
export LD_LIBRARY_PATH="/opt/miniconda3/envs/aics/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tokenspeed-kernel/python:${REPO_ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec python -m tokenspeed.cli serve "${MODEL_DIR}" \
  --served-model-name "${TS_SERVED_MODEL_NAME:-qwen3.5-0.8b}" \
  --host "${TS_HOST:-127.0.0.1}" \
  --port "${TS_PORT:-8000}" \
  --device npu \
  --world-size 1 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend mha \
  --sampling-backend greedy \
  --disable-prefill-graph \
  --disable-pdl \
  --disable-kvstore \
  --disable-prefix-caching \
  --enforce-eager \
  --language-model-only \
  --max-model-len "${TS_MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${TS_MAX_NUM_SEQS:-1}" \
  --gpu-memory-utilization "${TS_GPU_MEMORY_UTILIZATION:-0.30}" \
  "$@"
