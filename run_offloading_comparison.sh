#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/tokenspeed"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
# Use the already activated tokenspeed environment.
if [[ "${CONDA_DEFAULT_ENV:-}" != "tokenspeed" ]]; then
    echo "ERROR: Activate tokenspeed before running this script."
    exit 1
fi

export TS_LOG_REPLAY_TIMING=0
export TS_REPO_ROOT="$HOME/tokenspeed"
export TS_VENV_ROOT="$HOME/envs/tokenspeed"
export ASCEND_RT_VISIBLE_DEVICES=6,7
export TOKENSPEED_SKIP_GRPC_WARMUP=1
export TS_MODEL_DIR="$HOME/models/Qwen3.5-0.8B"
export TS_SERVED_MODEL_NAME="qwen3.5-0.8b"
export TS_MAX_MODEL_LEN=4096
export TS_MAX_NUM_SEQS=2
export TS_PORT=8002

OUT="$HOME/offloading_bench"
mkdir -p "$OUT"

PID=""

cleanup() {
    if [[ -n "$PID" ]]; then
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=""
    fi
}
trap cleanup EXIT INT TERM

# Do not silently benchmark an existing, unrelated server.
if curl -fsS --max-time 2 \
    "http://127.0.0.1:$TS_PORT/v1/models" \
    >/dev/null 2>&1; then
    echo "ERROR: Stop the existing service on port $TS_PORT first."
    exit 1
fi

for MODE in ON4 OFF ON1; do
    echo "================ $MODE ================"

    case "$MODE" in
        ON4) EXTRA=(--kvstore-ratio 4) ;;
        OFF) EXTRA=(--disable-kvstore) ;;
        ON1) EXTRA=(--kvstore-ratio 1) ;;
    esac

    LOG="$OUT/$MODE.log"

    bash /tmp/serve_qwen35_hosthit_b.sh \
        --max-total-tokens 16384 \
        "${EXTRA[@]}" >"$LOG" 2>&1 &
    PID=$!

    READY=0
    for i in $(seq 1 180); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "ERROR: Service exited during startup."
            tail -60 "$LOG"
            exit 1
        fi

        if curl -fsS --max-time 2 \
            "http://127.0.0.1:$TS_PORT/v1/models" \
            >/dev/null 2>&1; then
            READY=1
            break
        fi
        sleep 2
    done

    if [[ "$READY" != 1 ]]; then
        echo "ERROR: Startup timeout for $MODE"
        tail -60 "$LOG"
        exit 1
    fi

    # Confirm that the loaded model is actually 0.8B.
    python - <<'PY'
import requests
d = requests.get("http://127.0.0.1:8002/v1/models",
                 timeout=20).json()
ids = [x["id"] for x in d["data"]]
assert "qwen3.5-0.8b" in ids, ids
print("Model verified:", ids)
PY

    PYTHONDONTWRITEBYTECODE=1 \
    python bench_offloading.py \
        --mode "$MODE" \
        --model "$TS_SERVED_MODEL_NAME" \
        --port "$TS_PORT" \
        --output "$OUT/$MODE.csv"

    cleanup

    # Ensure the old service released the port before next mode.
    for i in $(seq 1 30); do
        if ! curl -fsS --max-time 2 \
            "http://127.0.0.1:$TS_PORT/v1/models" \
            >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    if curl -fsS --max-time 2 \
        "http://127.0.0.1:$TS_PORT/v1/models" \
        >/dev/null 2>&1; then
        echo "ERROR: Service still owns port $TS_PORT."
        echo "Stop it before continuing."
        exit 1
    fi
done

python - <<'PY'
import csv
import re
from pathlib import Path
from statistics import mean

base = Path.home() / "offloading_bench"
result = []

for mode in ("ON4", "OFF", "ON1"):
    with (base / f"{mode}.csv").open() as f:
        requests = list(csv.DictReader(f))

    log = (base / f"{mode}.log").read_text(errors="replace")

    transfers = {}
    for direction in ("D2H", "H2D"):
        found = re.findall(
            rf"direction={direction}\s+attn_tp_rank=(\d+)"
            rf".*?payload_bytes=(\d+)"
            rf"\s+submit_to_ready_ms=([\d.]+)",
            log,
        )
        # Keep rank-specific numbers: TP transfers occur in parallel.
        transfers[direction] = {
            "bytes": sum(int(x[1]) for x in found),
            "count": len(found),
            "mean_ms": mean(float(x[2]) for x in found)
                       if found else 0,
        }

    host = re.findall(
        r"Host Cache Stats: total_lcm=(\d+) used_lcm=(\d+) "
        r"free_lcm=(\d+) cached_entries=(\d+) "
        r"pinned_entries=(\d+) evictions=(\d+)",
        log,
    )
    max_used = max((int(x[1]) for x in host), default=0)
    max_evictions = max((int(x[5]) for x in host), default=0)

    for workload in ("high", "low"):
        group = [r for r in requests if r["workload"] == workload]
        latency = [float(r["latency_s"]) for r in group]
        prompt = sum(int(r["prompt_tokens"]) for r in group)
        cached = sum(int(r["cached_tokens"]) for r in group)
        output = sum(int(r["output_tokens"]) for r in group)

        result.append({
            "config": mode,
            "workload": workload,
            "requests": len(group),
            "mean_latency_s": round(mean(latency), 4),
            "max_latency_s": round(max(latency), 4),
            "cached_tokens": cached,
            "prompt_tokens": prompt,
            "cache_hit_ratio": round(cached / prompt, 4)
                               if prompt else 0,
            "output_tokens_per_s": round(
                output / sum(latency), 4
            ),
            "host_max_used_lcm": max_used,
            "host_evictions_log_max": max_evictions,
            "D2H_bytes_full_run": transfers["D2H"]["bytes"],
            "H2D_bytes_full_run": transfers["H2D"]["bytes"],
            "D2H_mean_submit_ready_ms":
                round(transfers["D2H"]["mean_ms"], 3),
            "H2D_mean_submit_ready_ms":
                round(transfers["H2D"]["mean_ms"], 3),
        })

dest = base / "comparison.csv"
with dest.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)

print("\n========== COMPARISON ==========")
for r in result:
    print(
        r["config"], r["workload"],
        "latency=", r["mean_latency_s"],
        "cached_ratio=", r["cache_hit_ratio"],
        "tok/s=", r["output_tokens_per_s"],
    )
print("\nSaved:", dest)
PY

echo "ALL EXPERIMENTS COMPLETED"
