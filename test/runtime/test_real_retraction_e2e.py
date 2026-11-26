import argparse
import json
import sys
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--phase", choices=["baseline", "forced"], required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

URL = "http://127.0.0.1:8002"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

r = requests.get(URL + "/v1/models", timeout=20)
r.raise_for_status()
models = [item["id"] for item in r.json()["data"]]
assert args.model in models, f"Model mismatch: {models}"

prompt = (
    "/no_think\n"
    "Continue the following numerical sequence using the same format. "
    "Keep writing numbers without explaining anything.\n"
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "
    + "11, 12, 13, 14, 15, 16, 17, 18, 19, 20, " * 12
    + "\nContinue: "
)

flush = requests.post(URL + "/flush_cache", timeout=120)
flush.raise_for_status()
assert flush.json().get("status") == "success"

payload = {
    "model": args.model,
    "prompt": prompt,
    "temperature": 0,
    "max_tokens": 64,
    "ignore_eos": True,
}

print(f"Running {args.model} {args.phase}", flush=True)

response = requests.post(
    URL + "/v1/completions",
    json=payload,
    timeout=600,
)

if not response.ok:
    print(
        f"HTTP {response.status_code}: "
        f"{response.text[:2000]}",
        file=sys.stderr,
    )
response.raise_for_status()

data = response.json()
choice = data["choices"][0]
usage = data.get("usage", {})

result = {
    "model": args.model,
    "phase": args.phase,
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "finish_reason": choice.get("finish_reason"),
    "output": choice["text"],
}

path = out / f"{args.model}_{args.phase}.json"
path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)

print(json.dumps(
    {
        k: v for k, v in result.items() if k != "output"
    },
    ensure_ascii=False,
    indent=2,
), flush=True)

print("Output:", repr(result["output"]), flush=True)
print("Saved:", path, flush=True)

assert result["completion_tokens"] >= 32, (
    "Generation ended too early to exercise decode retraction"
)
