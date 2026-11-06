"""HW2 TP benchmark: concurrent streaming requests and two-chip HBM sampling."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def comma_ints(value: str) -> list[int]:
    result = [int(part) for part in value.split(",")]
    if not result or any(number <= 0 for number in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--physical-devices", type=comma_ints, required=True)
    parser.add_argument("--input-lengths", type=comma_ints, default=[32, 1024])
    parser.add_argument("--concurrencies", type=comma_ints, default=[1, 8])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--hbm-interval", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.physical_devices) != args.tp:
        parser.error("--physical-devices must list exactly one device per TP rank")
    if args.output_tokens < 2 or args.warmups < 0 or args.repeats < 1:
        parser.error("output-tokens >= 2, warmups >= 0, repeats >= 1 required")
    return args


def parse_npu_smi_hbm(output: str) -> dict[int, int]:
    """Parse physical Phy-ID and HBM (not Memory-Usage) from npu-smi."""
    result = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) < 5:
            continue
        ids = fields[1].split()
        if len(ids) != 2 or not all(part.isdigit() for part in ids):
            continue
        usages = re.findall(r"(\d+)\s*/\s*\d+", line)
        if usages:
            result[int(ids[1])] = int(usages[-1])
    return result


def device_hbm_mb() -> dict[int, int]:
    output = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=15
    ).stdout
    return parse_npu_smi_hbm(output)


class HbmSampler:
    def __init__(self, devices: list[int], interval: float):
        self.devices = devices
        self.interval = interval
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                usage = device_hbm_mb()
                self.samples.append(
                    {
                        "time": time.time(),
                        "hbm_mb": {str(d): usage.get(d) for d in self.devices},
                    }
                )
            except (OSError, subprocess.SubprocessError):
                pass
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join(timeout=20)


def prompt_for_length(tokenizer, target: int, variant: int) -> tuple[str, int]:
    prefix = f"实验请求 {variant}。"
    filler = "分布式推理需要比较计算、通信与等待。" * (target + 16)
    empty_template = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    overhead = len(tokenizer.encode(empty_template, add_special_tokens=False))
    tokens = tokenizer.encode(prefix + filler, add_special_tokens=False)[
        : max(1, target - overhead)
    ]
    content = tokenizer.decode(tokens, skip_special_tokens=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    actual = len(tokenizer.encode(rendered, add_special_tokens=False))
    return content, actual


def request_one(args, content: str, input_tokens: int, request_index: int) -> dict:
    start = time.perf_counter()
    first = None
    last = None
    usage = None
    chunks = []
    error = None
    try:
        with requests.post(
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": args.output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
            timeout=args.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    part = delta.get("content", "")
                    if part:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        chunks.append(part)
    except (requests.RequestException, ValueError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    completion = usage.get("completion_tokens") if usage else None
    tpot = (
        (last - first) / (completion - 1)
        if first is not None and last is not None and completion and completion > 1
        else None
    )
    return {
        "request_index": request_index,
        "input_tokens": input_tokens,
        "completion_tokens": completion,
        "ttft_s": first - start if first is not None else None,
        "tpot_s": tpot,
        "e2e_s": end - start,
        "stream_chunks": len(chunks),
        "success": error is None and completion is not None and first is not None,
        "error": error,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def summarize(
    rows: list[dict], elapsed: float, samples: list[dict], devices: list[int]
) -> dict:
    good = [row for row in rows if row["success"]]
    return {
        "requests": len(rows),
        "successes": len(good),
        "wall_s": elapsed,
        "output_tokens_per_s": sum(row["completion_tokens"] for row in good) / elapsed,
        "ttft_p50_s": percentile([row["ttft_s"] for row in good], 0.5),
        "ttft_p95_s": percentile([row["ttft_s"] for row in good], 0.95),
        "tpot_p50_s": percentile(
            [row["tpot_s"] for row in good if row["tpot_s"] is not None], 0.5
        ),
        "tpot_p95_s": percentile(
            [row["tpot_s"] for row in good if row["tpot_s"] is not None], 0.95
        ),
        "e2e_p50_s": percentile([row["e2e_s"] for row in good], 0.5),
        "hbm_peak_total_mb": {
            str(d): max(
                (
                    s["hbm_mb"][str(d)]
                    for s in samples
                    if s["hbm_mb"][str(d)] is not None
                ),
                default=None,
            )
            for d in devices
        },
        "hbm_samples": len(samples),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    baseline = device_hbm_mb()
    missing = [device for device in args.physical_devices if device not in baseline]
    if missing:
        raise RuntimeError(f"npu-smi did not report physical devices: {missing}")
    groups = []
    for target in args.input_lengths:
        for concurrency in args.concurrencies:
            prompts = [
                prompt_for_length(tokenizer, target, i) for i in range(concurrency)
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for _ in range(args.warmups):
                    list(pool.map(lambda item: request_one(args, *item, -1), prompts))
                with HbmSampler(args.physical_devices, args.hbm_interval) as sampler:
                    started = time.perf_counter()
                    rows = []
                    for repeat in range(args.repeats):
                        futures = [
                            pool.submit(
                                request_one,
                                args,
                                content,
                                tokens,
                                repeat * concurrency + i,
                            )
                            for i, (content, tokens) in enumerate(prompts)
                        ]
                        rows.extend(future.result() for future in futures)
                    elapsed = time.perf_counter() - started
            if not sampler.samples:
                raise RuntimeError("no HBM samples collected; check npu-smi access")
            group = {
                "target_input_tokens": target,
                "actual_input_tokens": [tokens for _, tokens in prompts],
                "concurrency": concurrency,
                "summary": summarize(
                    rows, elapsed, sampler.samples, args.physical_devices
                ),
                "requests": rows,
                "hbm_samples": sampler.samples,
            }
            groups.append(group)
            print(
                json.dumps(
                    {"input": target, "concurrency": concurrency, **group["summary"]}
                ),
                flush=True,
            )
    payload = {
        "configuration": {**vars(args), "output_dir": str(args.output_dir)},
        "hbm_pre_run_total_mb": {
            str(d): baseline.get(d) for d in args.physical_devices
        },
        "hbm_note": "npu-smi device totals include other processes; peak is not model-exclusive",
        "timing_note": "TTFT uses first nonempty content chunk; TPOT uses usage token count and first/last content chunk timestamps",
        "groups": groups,
    }
    (args.output_dir / "benchmark_tp.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
