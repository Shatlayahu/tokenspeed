"""Measure TTFT, inter-token latency, total latency, and HBM usage."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from common import write_json


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument(
        "--model-dir",
        default="/home/u2601112087/models/Qwen3.5-0.8B",
    )
    parser.add_argument("--input-lengths", type=comma_ints, default=[32, 1024])
    parser.add_argument("--output-lengths", type=comma_ints, default=[32, 128])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--npu-card", type=int, default=0)
    parser.add_argument("--npu-chip", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def chat_token_count(tokenizer, content: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def make_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    overhead = chat_token_count(tokenizer, "")
    wanted_content_tokens = max(target_tokens - overhead, 1)
    filler_ids = tokenizer.encode(
        "性能基线测试上下文。" * (wanted_content_tokens + 16),
        add_special_tokens=False,
    )[:wanted_content_tokens]
    content = tokenizer.decode(filler_ids, skip_special_tokens=True)
    content += "\n请简要概括上述内容。"
    actual = chat_token_count(tokenizer, content)
    return content, actual


def read_hbm_mb(card: int, chip: int) -> int | None:
    try:
        output = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    current_card = None
    for line in output.splitlines():
        card_match = re.match(r"^\|\s*(\d+)\s+Ascend", line)
        if card_match:
            current_card = int(card_match.group(1))
            continue
        chip_match = re.match(r"^\|\s*(\d+)\s+\d+\s+\|", line)
        if current_card != card or not chip_match:
            continue
        if int(chip_match.group(1)) != chip:
            continue
        usages = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if usages:
            return int(usages[-1][0])
    return None


class HbmSampler:
    def __init__(self, card: int, chip: int) -> None:
        self.card = card
        self.chip = chip
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            value = read_hbm_mb(self.card, self.chip)
            if value is not None:
                self.samples.append(value)
            self.stop_event.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_event.set()
        self.thread.join(timeout=6)


def stream_request(args, tokenizer, prompt: str, output_tokens: int) -> dict:
    started = time.perf_counter()
    event_times = []
    pieces = []
    usage = None
    with HbmSampler(args.npu_card, args.npu_chip) as hbm:
        response = requests.post(
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": output_tokens,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            stream=True,
            timeout=args.timeout,
        )
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if payload.get("usage"):
                usage = payload["usage"]
            if not payload.get("choices"):
                continue
            delta = payload["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                pieces.append(content)
                event_times.append(time.perf_counter())
    finished = time.perf_counter()
    text = "".join(pieces)
    completion_tokens = (
        usage["completion_tokens"]
        if usage and usage.get("completion_tokens") is not None
        else len(tokenizer.encode(text, add_special_tokens=False))
    )
    ttft = None if not event_times else event_times[0] - started
    decode_latency = None
    if len(event_times) >= 2 and completion_tokens >= 2:
        decode_latency = (event_times[-1] - event_times[0]) / (completion_tokens - 1)
    return {
        "ttft_seconds": ttft,
        "decode_seconds_per_token": decode_latency,
        "e2e_seconds": finished - started,
        "completion_tokens": completion_tokens,
        "output_text": text,
        "hbm_peak_mb": max(hbm.samples) if hbm.samples else None,
        "hbm_samples": len(hbm.samples),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    baseline_hbm = read_hbm_mb(args.npu_card, args.npu_chip)
    rows = []

    for target_input in args.input_lengths:
        prompt, actual_input = make_prompt(tokenizer, target_input)
        for target_output in args.output_lengths:
            for _ in range(args.warmups):
                stream_request(args, tokenizer, prompt, target_output)
            for repeat in range(args.repeats):
                result = stream_request(args, tokenizer, prompt, target_output)
                row = {
                    "target_input_tokens": target_input,
                    "actual_input_tokens": actual_input,
                    "target_output_tokens": target_output,
                    "repeat": repeat,
                    **result,
                }
                rows.append(row)
                print(
                    f"input={actual_input} output={result['completion_tokens']} "
                    f"ttft={result['ttft_seconds']:.4f}s "
                    f"decode={result['decode_seconds_per_token']} "
                    f"hbm={result['hbm_peak_mb']}MB",
                    flush=True,
                )

    summary = []
    for target_input in args.input_lengths:
        for target_output in args.output_lengths:
            group = [
                row
                for row in rows
                if row["target_input_tokens"] == target_input
                and row["target_output_tokens"] == target_output
            ]
            ttft = [row["ttft_seconds"] for row in group if row["ttft_seconds"]]
            decode = [
                row["decode_seconds_per_token"]
                for row in group
                if row["decode_seconds_per_token"]
            ]
            e2e = [row["e2e_seconds"] for row in group]
            hbm = [row["hbm_peak_mb"] for row in group if row["hbm_peak_mb"]]
            summary.append(
                {
                    "target_input_tokens": target_input,
                    "actual_input_tokens": group[0]["actual_input_tokens"],
                    "target_output_tokens": target_output,
                    "runs": len(group),
                    "ttft_p50_seconds": statistics.median(ttft),
                    "ttft_p95_seconds": percentile(ttft, 0.95),
                    "decode_p50_seconds_per_token": statistics.median(decode),
                    "decode_p95_seconds_per_token": percentile(decode, 0.95),
                    "e2e_p50_seconds": statistics.median(e2e),
                    "e2e_p95_seconds": percentile(e2e, 0.95),
                    "hbm_peak_mb": max(hbm) if hbm else None,
                }
            )

    write_json(
        output_dir / "benchmark.json",
        {
            "configuration": vars(args),
            "hbm_baseline_mb": baseline_hbm,
            "summary": summary,
            "runs": rows,
        },
    )
    with (output_dir / "benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Input tokens | Output tokens | TTFT P50 (s) | "
        "Decode P50 (s/token) | E2E P50 (s) | Peak HBM (MB) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['actual_input_tokens']} | {item['target_output_tokens']} "
            f"| {item['ttft_p50_seconds']:.4f} "
            f"| {item['decode_p50_seconds_per_token']:.4f} "
            f"| {item['e2e_p50_seconds']:.4f} "
            f"| {item['hbm_peak_mb']} |"
        )
    (output_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
