"""End-to-end correctness and request-isolation checks for TokenSpeed."""

from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from common import correctness_workloads, workload_dict, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--reference-json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def generate(args: argparse.Namespace, prompt: str) -> dict:
    started = time.perf_counter()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "output_text": payload["choices"][0]["message"]["content"].strip(),
        "finish_reason": payload["choices"][0]["finish_reason"],
        "usage": payload.get("usage", {}),
        "elapsed_seconds": time.perf_counter() - started,
    }


def load_references(path: str | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in payload["results"]}


def normalized(text: str) -> str:
    return "".join(text.split()).casefold()


def main() -> None:
    args = parse_args()
    references = load_references(args.reference_json)
    failures = []
    results = []

    for workload in correctness_workloads():
        actual = generate(args, workload.prompt)
        expected_present = normalized(workload.expected_substring) in normalized(
            actual["output_text"]
        )
        if not expected_present:
            failures.append(f"{workload.name}: expected answer missing")
        reference = references.get(workload.name)
        comparison = None
        if reference is not None:
            actual_norm = normalized(actual["output_text"])
            reference_norm = normalized(reference["output_text"])
            comparison = {
                "exact_match": actual_norm == reference_norm,
                "similarity": SequenceMatcher(
                    None, actual_norm, reference_norm
                ).ratio(),
                "reference_output": reference["output_text"],
            }
        result = {
            **workload_dict(workload),
            **actual,
            "expected_substring_present": expected_present,
            "hf_comparison": comparison,
        }
        results.append(result)
        print(
            f"{workload.name}: pass={expected_present} "
            f"output={actual['output_text']!r}",
            flush=True,
        )

    # Greedy generation must not leak recurrent state between requests.
    probe = correctness_workloads()[0]
    first = generate(args, probe.prompt)["output_text"]
    generate(args, correctness_workloads()[1].prompt)
    second = generate(args, probe.prompt)["output_text"]
    isolation_passed = first == second
    if not isolation_passed:
        failures.append("request isolation: repeated greedy output changed")

    report = {
        "backend": "tokenspeed",
        "base_url": args.base_url,
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "results": results,
        "request_isolation": {
            "passed": isolation_passed,
            "first_output": first,
            "output_after_interfering_request": second,
        },
        "failures": failures,
    }
    write_json(args.output, report)
    if failures:
        raise SystemExit("; ".join(failures))
    print("All end-to-end correctness checks passed.", flush=True)


if __name__ == "__main__":
    main()
