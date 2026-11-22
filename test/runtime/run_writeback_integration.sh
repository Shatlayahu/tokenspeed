#!/usr/bin/env bash
set -e
cd "${1:-$(dirname -- "${BASH_SOURCE[0]}")/../..}"
export PYTHONPATH="$PWD/tokenspeed-kernel/python:$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -v -s \
  'test/runtime/test_host_executor_npu.py::NPUCompactLayoutRoundTripTest::test_npu_writeback_ack_releases_real_scheduler_blocks'
