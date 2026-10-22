#!/usr/bin/env bash
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${HERE}/results}"
REFERENCE_JSON="${REFERENCE_JSON:-${HERE}/results/hf_reference.json}"
SERVER_LOG="${RESULTS_DIR}/server.log"
mkdir -p "${RESULTS_DIR}"

source "${TS_VENV_ROOT:-/home/u2601112087/venvs/tokenspeed}/bin/activate"
source "${TS_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}/set_env.sh"
set -u
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "${RUN_OPERATOR_TESTS:-1}" == "1" ]]; then
  "${HERE}/run_operator_tests.sh"
fi

if [[ "${GENERATE_HF_REFERENCE:-1}" == "1" ]]; then
  python "${HERE}/hf_reference.py" \
    --output "${REFERENCE_JSON}" \
    --max-new-tokens "${E2E_MAX_TOKENS:-32}"
fi

"${HERE}/serve_qwen35_08b_npu.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -INT "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 120 "${SERVER_LOG}"
    exit 1
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null

python "${HERE}/e2e_correctness.py" \
  --reference-json "${REFERENCE_JSON}" \
  --output "${RESULTS_DIR}/e2e_correctness.json" \
  --max-tokens "${E2E_MAX_TOKENS:-32}"

python "${HERE}/benchmark.py" \
  --input-lengths "${BENCH_INPUT_LENGTHS:-32,1024}" \
  --output-lengths "${BENCH_OUTPUT_LENGTHS:-32,128}" \
  --warmups "${BENCH_WARMUPS:-3}" \
  --repeats "${BENCH_REPEATS:-10}" \
  --output-dir "${RESULTS_DIR}/benchmark"
