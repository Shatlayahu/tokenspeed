#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/tokenspeed"
source /usr/local/Ascend/cann-9.0.0/set_env.sh

export TS_REPO_ROOT="$HOME/tokenspeed"
export TS_VENV_ROOT="$HOME/envs/tokenspeed"
export ASCEND_RT_VISIBLE_DEVICES=6,7
export TOKENSPEED_SKIP_GRPC_WARMUP=1
export TS_MAX_MODEL_LEN=4096
export TS_MAX_NUM_SEQS=1
export TS_PORT=8002
export TS_LOG_REPLAY_TIMING=0
export TS_RETRACTION_TRACE=1

OUT="$HOME/retraction_e2e"
mkdir -p "$OUT"

PID=""

stop_service() {
    if [[ -n "$PID" ]]; then
        kill -TERM -- "-$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=""
    fi
}

trap stop_service EXIT

if curl -fsS --max-time 2 \
    http://127.0.0.1:8002/v1/models \
    >/dev/null 2>&1; then
    echo "ERROR: Stop existing service on port 8002."
    exit 1
fi

for SIZE in 0.8B 4B; do
    if [[ "$SIZE" == "0.8B" ]]; then
        export TS_MODEL_DIR="$HOME/models/Qwen3.5-0.8B"
        export TS_SERVED_MODEL_NAME="qwen3.5-0.8b"
    else
        export TS_MODEL_DIR="$HOME/models/Qwen3.5-4B"
        export TS_SERVED_MODEL_NAME="qwen3.5-4b"
    fi

    for PHASE in baseline forced; do
        echo "========== $SIZE / $PHASE =========="

        EXTRA=()
        if [[ "$PHASE" == "forced" ]]; then
            EXTRA=(--debug-force-retraction-interval 25)
        fi

        LOG="$OUT/${SIZE}_${PHASE}.log"

        setsid bash test/ci_system/serve_qwen35_hosthit_b.sh \
            --kvstore-ratio 4 \
            --max-total-tokens 16384 \
            "${EXTRA[@]}" \
            >"$LOG" 2>&1 &

        PID=$!
        READY=0

        for i in $(seq 1 180); do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "Service exited. Last log lines:"
                tail -80 "$LOG"
                exit 1
            fi

            if curl -fsS --max-time 2 \
                http://127.0.0.1:8002/v1/models \
                >/dev/null 2>&1; then
                READY=1
                break
            fi

            sleep 2
        done

        if [[ "$READY" != 1 ]]; then
            echo "Service startup timeout"
            tail -80 "$LOG"
            exit 1
        fi

        PYTHONDONTWRITEBYTECODE=1 \
        python -B test/runtime/test_real_retraction_e2e.py \
            --model "$TS_SERVED_MODEL_NAME" \
            --phase "$PHASE" \
            --out "$OUT"

        stop_service

        for i in $(seq 1 30); do
            if ! curl -fsS --max-time 2 \
                http://127.0.0.1:8002/v1/models \
                >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done

        if curl -fsS --max-time 2 \
            http://127.0.0.1:8002/v1/models \
            >/dev/null 2>&1; then
            echo "ERROR: Old service still running"
            exit 1
        fi
    done
done

python -B - <<'PY'
import json
import re
from pathlib import Path

root = Path.home() / "retraction_e2e"

for size, model in [
    ("0.8B", "qwen3.5-0.8b"),
    ("4B", "qwen3.5-4b"),
]:
    cold = json.loads(
        (root / f"{model}_baseline.json").read_text()
    )
    restored = json.loads(
        (root / f"{model}_forced.json").read_text()
    )

    log = (root / f"{size}_forced.log").read_text(
        errors="replace"
    )

    events = re.findall(
        r"\[RetractionE2E\] phase=(submit|ack) "
        r"rank=(\d+) kind=(\w+)"
        r"(?: request=\S+)? op_id=(\d+)",
        log,
    )

    expected = {
        ("submit", "SnapshotOp"),
        ("ack", "SnapshotDoneEvent"),
        ("submit", "RestoreOp"),
        ("ack", "RestoreDoneEvent"),
    }

    print(f"\n=== {size} TP=2 ===")

    for rank in (0, 1):
        actual = {
            (phase, kind)
            for phase, r, kind, op_id in events
            if int(r) == rank
        }
        print(f"Rank {rank}: {sorted(actual)}")
        assert expected <= actual, (
            f"Missing events on rank {rank}: "
            f"{expected - actual}"
        )

    identical = cold["output"] == restored["output"]
    print("Output identical:", identical)
    print("Cold tokens:", cold["completion_tokens"])
    print("Restored tokens:", restored["completion_tokens"])

    assert identical, f"{size}: output mismatch"
    print(f"PASS: {size} Retraction E2E")

print("\nALL RETRACTION E2E TESTS PASSED")
PY
