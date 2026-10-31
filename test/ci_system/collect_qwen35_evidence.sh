#!/usr/bin/env bash
set -Eeuo pipefail

# ==========================================
# Qwen3.5 Unified Evidence Collector
# Supports: 0.8B / 4B
# TP=2
# ==========================================

# 1. Model selection

MODEL_KEY=""

if [[ "${1:-}" == "--model" && -n "${2:-}" ]]; then
    MODEL_KEY="$2"
else
    echo "Usage: bash $0 --model {0.8b|4b}"
    exit 1
fi

case "$MODEL_KEY" in
    0.8b)
        MODEL_LABEL="Qwen3.5-0.8B"
        MODEL="$HOME/models/Qwen3.5-0.8B"
        SERVED_MODEL="qwen3.5-0.8b"
        ;;
    4b)
        MODEL_LABEL="Qwen3.5-4B"
        MODEL="$HOME/models/Qwen3.5-4B"
        SERVED_MODEL="qwen3.5-4b"
        ;;
    *)
        echo "ERROR: Unsupported model: $MODEL_KEY"
        echo "Supported models: 0.8b, 4b"
        exit 1
        ;;
esac

# 2. Common configuration

REPO="$HOME/tokenspeed-github-work-20261020"
VENV="$HOME/ts-env"
CANN="/usr/local/Ascend/cann-9.0.0"

LAUNCH_SCRIPT="$REPO/test/ci_system/serve_qwen35_08b_tp2.sh"

TS_PORT="${TS_PORT:-8002}"
HTTP_PORT="${HTTP_PORT:-8003}"

TS_GPU_MEMORY_UTILIZATION="${TS_GPU_MEMORY_UTILIZATION:-0.30}"
TS_MAX_MODEL_LEN="${TS_MAX_MODEL_LEN:-4096}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6,7}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUT="$HOME/qwen35_evidence/$MODEL_KEY/$TIMESTAMP"

mkdir -p "$OUT"

SERVER_LOG="$OUT/server_rank_logs.log"

echo "=========================================="
echo "Unified Qwen3.5 Evidence Collection"
echo "Model: $MODEL_LABEL"
echo "=========================================="
echo "Output: $OUT"

# 3. Check environment

if [[ ! -d "$REPO" || ! -d "$MODEL" ]]; then
    echo "ERROR: Repository or model directory missing."
    exit 1
fi

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "ERROR: Launch script missing: $LAUNCH_SCRIPT"
    exit 1
fi

source "$VENV/bin/activate"
source "$CANN/set_env.sh"

cd "$REPO"

{
    echo "Model: $MODEL_LABEL"
    echo "Model path: $MODEL"
    echo "Served model: $SERVED_MODEL"
    echo "Timestamp: $(date)"
    echo "Hostname: $(cat /proc/sys/kernel/hostname)"
    echo "Python: $(python3 --version)"
    echo "Repository: $REPO"
    echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    echo "Tensor parallel size: 2"
    echo "TS port: $TS_PORT"
    echo "HTTP port: $HTTP_PORT"
    echo "GPU memory utilization: $TS_GPU_MEMORY_UTILIZATION"
    echo "Max model length: $TS_MAX_MODEL_LEN"
} > "$OUT/environment.txt"

# 4. Check port conflicts. Never kill an existing server.

port_open() {
    python3 - "$1" <<'PY'
import socket
import sys

s = socket.socket()
s.settimeout(1)

try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

if port_open "$TS_PORT" || port_open "$HTTP_PORT"; then
    echo "ERROR: Port $TS_PORT or $HTTP_PORT is already occupied."
    echo "Stop the existing server before running this collector."
    exit 2
fi

# 5. Collect pre-start device snapshot

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info > "$OUT/npu_smi_before.txt" 2>&1 || true
else
    echo "npu-smi not found" > "$OUT/npu_smi_before.txt"
fi

{
    echo "Timestamp: $(date)"
    echo "Hostname: $(cat /proc/sys/kernel/hostname)"
    echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    echo
    echo "===== NPU-SMI device information ====="
    cat "$OUT/npu_smi_before.txt"
} > "$OUT/device_mapping.txt"

# 6. Launch model and capture BOTH rank logs

echo "Starting $MODEL_LABEL TP=2..."

TS_MODEL_DIR="$MODEL" \
TS_PORT="$TS_PORT" \
TS_SERVED_MODEL_NAME="$SERVED_MODEL" \
TS_GPU_MEMORY_UTILIZATION="$TS_GPU_MEMORY_UTILIZATION" \
TS_MAX_MODEL_LEN="$TS_MAX_MODEL_LEN" \
bash "$LAUNCH_SCRIPT" \
    > "$SERVER_LOG" 2>&1 < /dev/null &

SERVER_PID=$!

echo "$SERVER_PID" > "$OUT/server.pid"

echo "Server PID: $SERVER_PID"
echo "Waiting for server readiness..."

# 7. Wait for HTTP server

READY=0

for ((i=1; i<=900; i++)); do

    if grep -q "control server ready on http://127.0.0.1:$HTTP_PORT" \
        "$SERVER_LOG"; then
        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Model server exited unexpectedly."
        tail -100 "$SERVER_LOG"
        exit 3
    fi

    sleep 1
done

if [[ "$READY" != "1" ]]; then
    echo "ERROR: Server readiness timeout."
    tail -100 "$SERVER_LOG"
    exit 4
fi

echo "Model server ready."

# 8. Capture post-start device snapshot

npu-smi info > "$OUT/npu_smi_after_startup.txt" 2>&1 || true

# 9. Run reproducible continuous multi-turn inference

echo "Running multi-turn inference..."

python3 - "$HTTP_PORT" "$SERVED_MODEL" "$MODEL_LABEL" \
    <<'PY' > "$OUT/multiturn_output.txt" 2>&1

import json
import sys
import urllib.request

port = int(sys.argv[1])
served_model = sys.argv[2]
model_label = sys.argv[3]

url = f"http://127.0.0.1:{port}/v1/chat/completions"

messages = []

prompts = [
    "What is a GPU? Answer in two sentences.",
    "My name is Alex, and I am studying computer systems.",
    "What is my name?",
    "I am currently learning distributed inference with two Ascend NPUs.",
    "What hardware am I using? Only use information I provided.",
    "Summarize what I have told you about my studies in two sentences."
]

print("=" * 60)
print(f"{model_label} Continuous Multi-turn Inference")
print(f"Endpoint: {url}")
print("=" * 60)

for turn, prompt in enumerate(prompts, 1):

    messages.append({
        "role": "user",
        "content": prompt
    })

    payload = {
        "model": served_model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        reply = result["choices"][0]["message"]["content"]

        messages.append({
            "role": "assistant",
            "content": reply
        })

        print(f"\n===== TURN {turn} =====")
        print(f"User: {prompt}")
        print(f"Assistant: {reply}")
        print(f"Usage: {result.get('usage', {})}")

    except Exception as exc:
        messages.pop()
        print(f"ERROR at turn {turn}: {exc}")
        raise

print("\nMulti-turn inference completed successfully.")

PY

echo "Multi-turn inference finished."

# 10. Capture post-inference device snapshot

npu-smi info > "$OUT/npu_smi_after_inference.txt" 2>&1 || true

# 11. Extract Rank 0 and Rank 1 logs

grep -E \
'ATTN TP RANK 0|rank=0/2|global rank: 0|rank 0' \
"$SERVER_LOG" > "$OUT/rank0.log" || true

grep -E \
'ATTN TP RANK 1|rank=1/2|global rank: 1|rank 1' \
"$SERVER_LOG" > "$OUT/rank1.log" || true

# 12. Extract TP-sharded tensor shape evidence

grep 'TP_SHARD_CHECK' "$SERVER_LOG" \
    > "$OUT/tp_shard_shapes.log" || true

grep 'TP_SHARD_CHECK.*rank=0/2' \
    "$OUT/tp_shard_shapes.log" \
    > "$OUT/rank0_shapes.txt" || true

grep 'TP_SHARD_CHECK.*rank=1/2' \
    "$OUT/tp_shard_shapes.log" \
    > "$OUT/rank1_shapes.txt" || true

R0=$(grep -c 'TP_SHARD_CHECK' "$OUT/rank0_shapes.txt" || true)
R1=$(grep -c 'TP_SHARD_CHECK' "$OUT/rank1_shapes.txt" || true)

# 13. Extract actual HBM information

python3 - "$OUT" <<'PYTHON' > "$OUT/hbm_snapshot.txt"

import os
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])

files = [
    "npu_smi_before.txt",
    "npu_smi_after_startup.txt",
    "npu_smi_after_inference.txt",
]

visible = set(
    x.strip()
    for x in os.environ.get(
        "ASCEND_RT_VISIBLE_DEVICES", ""
    ).split(",")
    if x.strip()
)

for filename in files:

    print(f"\n===== {filename} =====")

    path = out / filename

    if not path.exists():
        print("SNAPSHOT MISSING")
        continue

    current_npu = None

    for line in path.read_text(errors="replace").splitlines():

        parts = line.split("|")

        if len(parts) < 4:
            continue

        fields = parts[1].split()

        if len(fields) >= 2 and fields[1] == "Ascend910":
            current_npu = int(fields[0])
            continue

        if len(fields) != 2 or not all(x.isdigit() for x in fields):
            continue

        if current_npu is None:
            continue

        chip_id, phy_id = map(int, fields)

        matches = re.findall(r"(\d+)\s*/\s*(\d+)", line)

        if not matches:
            continue

        used, total = map(int, matches[-1])

        percentage = used / total * 100 if total else 0

        tag = " [VISIBLE]" if str(phy_id) in visible else ""

        print(
            f"NPU={current_npu} "
            f"Chip={chip_id} "
            f"Phy-ID={phy_id} "
            f"HBM={used}/{total} MB "
            f"Usage={percentage:.2f}%{tag}"
        )

PYTHON

# 14. Generate summary

{
    echo "$MODEL_LABEL TP=2 Evidence Summary"
    echo "Timestamp: $(date)"
    echo "Model: $MODEL"
    echo "Served model: $SERVED_MODEL"
    echo "Visible devices: $ASCEND_RT_VISIBLE_DEVICES"
    echo "Tensor parallel size: 2"
    echo "TS port: $TS_PORT"
    echo "HTTP port: $HTTP_PORT"
    echo "Server PID: $SERVER_PID"
    echo
    echo "Rank 0 TP_SHARD_CHECK lines: $R0"
    echo "Rank 1 TP_SHARD_CHECK lines: $R1"
    echo
    echo "Multi-turn inference: completed"
    echo "Server log: $SERVER_LOG"
    echo "Server remains running."
    echo
    echo "Evidence files:"
    find "$OUT" -maxdepth 1 -type f -printf '%f\n' | sort
} > "$OUT/summary.txt"

echo
echo "=========================================="
echo "Evidence collection completed."
echo "=========================================="
cat "$OUT/summary.txt"

echo
echo "Evidence directory: $OUT"
