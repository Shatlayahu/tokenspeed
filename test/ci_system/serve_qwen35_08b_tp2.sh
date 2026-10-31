#!/usr/bin/env bash
set -eo pipefail

# Repository and environment configuration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${TS_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_ROOT="${TS_VENV_ROOT:-$HOME/ts-env}"
CANN_ROOT="${TS_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
MODEL_DIR="${TS_MODEL_DIR:-}"

# Activate the existing Python and Ascend environments.
source "${VENV_ROOT}/bin/activate"
source "${CANN_ROOT}/set_env.sh"
set -u

# Fail early instead of starting with an invalid model path.
if [[ -z "${MODEL_DIR}" ]]; then
    echo "ERROR: Set TS_MODEL_DIR to the local Qwen3.5-0.8B model directory." >&2
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory does not exist: ${MODEL_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

# Use the two physical NPUs previously verified by the HCCL TP=2 test.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6,7}"

# Use the current repository's Python packages, including Ascend kernels.
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tokenspeed-kernel/python:${REPO_ROOT}/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"

# Keep model loading offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The full serving entry point requires the SMG gateway packages.
python - <<'PY'
import importlib.util
import sys

missing = []
for module in ("smg", "smg_grpc_servicer.tokenspeed.server"):
    try:
        available = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        missing.append(module)

if missing:
    print(
        "ERROR: Missing modules required by the full serving entry point:",
        *missing,
        sep="\n  ",
        file=sys.stderr,
    )
    print(
        "Install or configure the matching TokenSpeed serving dependencies "
        "before launching the model.",
        file=sys.stderr,
    )
    sys.exit(2)
PY

# TokenSpeed creates its own Scheduler processes; do not wrap this in torchrun.
exec python -m tokenspeed.cli serve "${MODEL_DIR}" \
    --served-model-name "${TS_SERVED_MODEL_NAME:-qwen3.5-0.8b}" \
    --host "${TS_HOST:-127.0.0.1}" \
    --port "${TS_PORT:-8000}" \
    --device npu \
    --world-size 2 \
    --tensor-parallel-size 2 \
    --nprocs-per-node 2 \
    --base-gpu-id 0 \
    --gpu-id-step 1 \
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
