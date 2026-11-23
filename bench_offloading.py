import argparse
import csv
import time
import requests

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True)
p.add_argument("--model", default="qwen3.5-0.8b")
p.add_argument("--port", type=int, default=8002)
p.add_argument("--output", required=True)
args = p.parse_args()

url = f"http://127.0.0.1:{args.port}"
s = requests.Session()
rows = []

def flush():
    r = s.post(url + "/flush_cache", timeout=120)
    r.raise_for_status()
    assert r.json().get("status") == "success"

def request(workload, trial, label, prompt, record=True):
    t0 = time.perf_counter()
    r = s.post(
        url + "/v1/completions",
        json={
            "model": args.model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=300,
    )
    if not r.ok:
        print(
            f"FAILED: workload={workload}, trial={trial}, "
            f"label={label}, status={r.status_code}, "
            f"response={r.text[:500]!r}",
            flush=True,
        )
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    data = r.json()
    usage = data["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get(
        "cached_tokens", 0
    )
    prompt_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    print(
        f"{args.mode} {workload} {trial} {label}: "
        f"{elapsed:.3f}s cached={cached}/{prompt_tokens}",
        flush=True,
    )

    if record:
        rows.append({
            "config": args.mode,
            "workload": workload,
            "trial": trial,
            "request": label,
            "latency_s": round(elapsed, 6),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached,
            "output_tokens": output_tokens,
        })

def prefix(tag, n):
    return (
        f"/no_think\nExperiment {tag}.\n"
        + f" context_{tag} alpha beta gamma delta" * n
        + "\nStored answer:"
    )

for trial in range(1, 4):
    # High reuse: prime A, displace Device cache, then revisit A.
    flush()
    a = prefix("A", 110)
    request("high", trial, "prime", a, record=False)
    request("high", trial, "pressure_B",
            prefix(f"B{trial}", 100), record=False)
    request("high", trial, "pressure_C",
            prefix(f"C{trial}", 100), record=False)
    request("high", trial, "return_A",
            a + " ORANGE.\nAnswer in one word:")

    # Low reuse: every request has a different long prefix.
    flush()
    for j in range(3):
        request(
            "low", trial, f"unique_{j}",
            prefix(f"UNIQUE_{args.mode}_{trial}_{j}", 110)
            + " BLUE.\nAnswer in one word:"
        )

with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "config", "workload", "trial", "request",
        "latency_s", "prompt_tokens",
        "cached_tokens", "output_tokens",
    ])
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", args.output, flush=True)
