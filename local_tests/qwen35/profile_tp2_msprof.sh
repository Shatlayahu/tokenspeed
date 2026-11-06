#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${TS_REPO_ROOT:?Set TS_REPO_ROOT}"
VENV_ROOT="${TS_VENV_ROOT:?Set TS_VENV_ROOT}"
CANN_ROOT="${TS_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
PROFILE_OUTPUT="${TS_PROFILE_OUTPUT:?Set TS_PROFILE_OUTPUT}"
PROFILE_DELAY="${TS_PROFILE_DELAY:?Set TS_PROFILE_DELAY}"
PROFILE_DURATION="${TS_PROFILE_DURATION:?Set TS_PROFILE_DURATION}"

source "${CANN_ROOT}/set_env.sh"
source "${VENV_ROOT}/bin/activate"
set -u
cd "${REPO_ROOT}"
mkdir -p "${PROFILE_OUTPUT}"

# Delay avoids most startup tuning. The profile is diagnostic, not a latency baseline.
exec msprof \
  --output="${PROFILE_OUTPUT}" \
  --hccl=on --task-time=l1 --runtime-api=on \
  --delay="${PROFILE_DELAY}" \
  --duration="${PROFILE_DURATION}" \
  bash test/ci_system/serve_qwen35_08b_tp2.sh
