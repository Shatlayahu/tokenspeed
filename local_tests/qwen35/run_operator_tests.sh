#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${TS_REPO_ROOT:-/home/u2601112087/tokenspeed}"
VENV_ROOT="${TS_VENV_ROOT:-/home/u2601112087/venvs/tokenspeed}"
CANN_ROOT="${TS_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
LOG_DIR="${OPERATOR_LOG_DIR:-/home/u2601112087/tokenspeed-logs/hw1/operators}"

source "${VENV_ROOT}/bin/activate"
source "${CANN_ROOT}/set_env.sh"
set -u
cd "${REPO_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export CPATH="/opt/miniconda3/envs/aics/include/python3.11:/opt/miniconda3/envs/aics/include:${CPATH:-}"
export LD_LIBRARY_PATH="/opt/miniconda3/envs/aics/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tokenspeed-kernel/python:${REPO_ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
mkdir -p "${LOG_DIR}"

run_pytest() {
  local test_path="$1"
  local log_name
  log_name="$(basename "${test_path}" .py).log"
  echo "[RUN] ${test_path}"
  pytest -q "${test_path}" 2>&1 | tee "${LOG_DIR}/${log_name}"
}

run_python() {
  local test_path="$1"
  local log_name
  log_name="$(basename "${test_path}" .py).log"
  echo "[RUN] ${test_path}"
  python "${test_path}" 2>&1 | tee "${LOG_DIR}/${log_name}"
}

# Run legacy local tests in separate processes: several execute assertions at
# import time, and Triton-Ascend autotuners cannot be registered repeatedly in
# one pytest collection process.
for test_path in local_tests/conv1d/test_*.py; do
  run_python "${test_path}"
done
for test_path in local_tests/attention/test_*.py; do
  case "${test_path}" in
    *test_gemma_add_norm_npu.py|*test_layernorm_gated_npu.py)
      run_pytest "${test_path}"
      ;;
    *)
      run_python "${test_path}"
      ;;
  esac
done

run_pytest tokenspeed-kernel-npu/test/test_layernorm.py
run_pytest tokenspeed-kernel-npu/test/test_mha.py

echo "[RUN] tokenspeed-kernel/test/standalone_gdn_ascend.py"
python tokenspeed-kernel/test/standalone_gdn_ascend.py \
  2>&1 | tee "${LOG_DIR}/standalone_gdn_ascend.log"

echo "All Ascend operator tests passed."
