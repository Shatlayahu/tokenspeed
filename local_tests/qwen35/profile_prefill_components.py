# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Measure representative Qwen3.5-0.8B Prefill components on Ascend."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill
from tokenspeed_kernel_npu.ops.mha import mha_prefill


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _measure(
    operation: Callable[[], torch.Tensor | object], warmups: int, repeats: int
) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _linear_case(
    tokens: int,
    input_features: int,
    output_features: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    inputs = torch.randn((tokens, input_features), dtype=torch.bfloat16, device=device)
    weight = torch.randn(
        (output_features, input_features), dtype=torch.bfloat16, device=device
    )
    return _measure(lambda: F.linear(inputs, weight), warmups, repeats)


def _activation_case(
    tokens: int,
    features: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    gate = torch.randn((tokens, features), dtype=torch.bfloat16, device=device)
    up = torch.randn_like(gate)
    return _measure(lambda: F.silu(gate) * up, warmups, repeats)


def _gdn_core_case(
    tokens: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    heads = 16
    head_dim = 128
    q = torch.randn((1, tokens, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn((1, tokens, heads), device=device))
    beta = torch.rand((1, tokens, heads), dtype=torch.bfloat16, device=device).sigmoid()
    initial_state = torch.zeros(
        (1, heads, head_dim, head_dim), dtype=torch.bfloat16, device=device
    )
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    def run() -> object:
        return gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=head_dim**-0.5,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=True,
            output_final_state=True,
            solution="triton",
        )

    return _measure(run, warmups, repeats)


def _mha_core_case(
    tokens: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    head_dim = 256
    q = torch.randn((tokens, 8, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((tokens, 2, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return mha_prefill(
            q,
            k,
            v,
            cu_seqlens,
            [0, tokens],
            tokens,
            softmax_scale=head_dim**-0.5,
        )

    return _measure(run, warmups, repeats)


def _profile_length(
    tokens: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    cases = [
        (
            "gdn_input_projection",
            18,
            lambda: _linear_case(tokens, 1024, 8224, warmups, repeats, device),
        ),
        (
            "gdn_core",
            18,
            lambda: _gdn_core_case(tokens, warmups, repeats, device),
        ),
        (
            "gdn_output_projection",
            18,
            lambda: _linear_case(tokens, 2048, 1024, warmups, repeats, device),
        ),
        (
            "ffn_gate_up_projection",
            24,
            lambda: _linear_case(tokens, 1024, 7168, warmups, repeats, device),
        ),
        (
            "ffn_silu_multiply",
            24,
            lambda: _activation_case(tokens, 3584, warmups, repeats, device),
        ),
        (
            "ffn_down_projection",
            24,
            lambda: _linear_case(tokens, 3584, 1024, warmups, repeats, device),
        ),
        (
            "full_attention_core",
            6,
            lambda: _mha_core_case(tokens, warmups, repeats, device),
        ),
    ]
    rows = []
    for name, count, benchmark in cases:
        measurement = benchmark()
        rows.append(
            {
                "tokens": tokens,
                "component": name,
                "layer_count": count,
                **measurement,
                "estimated_model_ms": count * measurement["p50_ms"],
            }
        )
    return rows


def _write_markdown(path: Path, results: dict) -> None:
    lines = [
        "# Qwen3.5-0.8B representative Prefill component profile",
        "",
        (
            f"Configuration: BF16, warmups={results['configuration']['warmups']}, "
            f"repeats={results['configuration']['repeats']}."
        ),
        "",
        (
            "The layer-count estimate is a diagnostic microbenchmark, not a sum of "
            "a whole-model trace. It excludes norms, cache writes, framework overhead, "
            "and kernel overlap."
        ),
        "",
        (
            "| Tokens | Component | Layers | P50/layer (ms) | P95/layer (ms) "
            "| Count × P50 (ms) |"
        ),
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in results["measurements"]:
        lines.append(
            f"| {row['tokens']} | {row['component']} | {row['layer_count']} "
            f"| {row['p50_ms']:.4f} | {row['p95_ms']:.4f} "
            f"| {row['estimated_model_ms']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="39,1031")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_tests/qwen35/results/prefill_components"),
    )
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("An available Ascend NPU is required")
    device = torch.device("npu:0")
    lengths = [int(value) for value in args.lengths.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measurements = []
    for tokens in lengths:
        measurements.extend(_profile_length(tokens, args.warmups, args.repeats, device))
    results = {
        "configuration": {
            "device": str(device),
            "dtype": "bfloat16",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "sequence_lengths": lengths,
        },
        "measurements": measurements,
    }
    json_path = args.output_dir / "prefill_components.json"
    markdown_path = args.output_dir / "prefill_components.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(markdown_path, results)
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
