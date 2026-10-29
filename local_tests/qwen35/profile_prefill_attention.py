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

"""Profile the Qwen3.5 Full Attention Prefill path on Ascend."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.mha import mha_prefill


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _time_call(callable_obj) -> float:
    torch.npu.synchronize()
    start = time.perf_counter()
    callable_obj()
    torch.npu.synchronize()
    return time.perf_counter() - start


def _make_inputs(
    seq_len: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seq_len + head_dim)
    q = torch.randn(
        (seq_len, 8, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        (seq_len, 2, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (seq_len, 2, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens


def _profile_whole_path(
    seq_len: int,
    head_dim: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    q, k, v, cu_seqlens = _make_inputs(seq_len, head_dim, device)

    def run() -> torch.Tensor:
        return mha_prefill(
            q,
            k,
            v,
            cu_seqlens,
            [0, seq_len],
            seq_len,
            softmax_scale=head_dim**-0.5,
        )

    for _ in range(warmups):
        run()
    samples = [_time_call(run) for _ in range(repeats)]
    return {
        "sequence_length": seq_len,
        "head_dim": head_dim,
        "path": "fallback" if head_dim == 256 else "fused",
        "p50_ms": statistics.median(samples) * 1000,
        "p95_ms": _percentile(samples, 0.95) * 1000,
        "min_ms": min(samples) * 1000,
        "max_ms": max(samples) * 1000,
    }


def _profile_fallback_stages(
    seq_len: int,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int]:
    head_dim = 256
    q, k, v, _ = _make_inputs(seq_len, head_dim, device)

    def run_once() -> tuple[float, float, float]:
        stage_start = time.perf_counter()
        q_seq = q.transpose(0, 1)
        k_seq = k.transpose(0, 1).repeat_interleave(4, dim=0)
        v_seq = v.transpose(0, 1).repeat_interleave(4, dim=0)
        torch.npu.synchronize()
        layout_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        scores = torch.matmul(q_seq.float(), k_seq.float().transpose(-2, -1))
        scores.mul_(head_dim**-0.5)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        scores.masked_fill_(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(v_seq.dtype)
        torch.npu.synchronize()
        scores_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        output = torch.matmul(probabilities, v_seq).transpose(0, 1)
        torch.cat([output], dim=0)
        torch.npu.synchronize()
        values_seconds = time.perf_counter() - stage_start
        return layout_seconds, scores_seconds, values_seconds

    for _ in range(warmups):
        run_once()
    samples = [run_once() for _ in range(repeats)]
    layout_p50 = statistics.median(sample[0] for sample in samples) * 1000
    scores_p50 = statistics.median(sample[1] for sample in samples) * 1000
    values_p50 = statistics.median(sample[2] for sample in samples) * 1000
    total_p50 = layout_p50 + scores_p50 + values_p50
    score_bytes = 8 * seq_len * seq_len * torch.float32.itemsize
    return {
        "sequence_length": seq_len,
        "layout_p50_ms": layout_p50,
        "qk_mask_softmax_p50_ms": scores_p50,
        "pv_p50_ms": values_p50,
        "staged_total_p50_ms": total_p50,
        "qk_mask_softmax_share": scores_p50 / total_p50,
        "score_matrix_mib": score_bytes / (1024**2),
    }


def _write_markdown(path: Path, results: dict) -> None:
    lines = [
        "# Qwen3.5 Full Attention Prefill profile",
        "",
        (
            f"Configuration: BF16, Q heads=8, KV heads=2, warmups="
            f"{results['configuration']['warmups']}, repeats="
            f"{results['configuration']['repeats']}."
        ),
        "",
        "## Whole operator",
        "",
        "| Tokens | Head dim | Path | P50 (ms) | P95 (ms) |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in results["whole_operator"]:
        lines.append(
            f"| {row['sequence_length']} | {row['head_dim']} | {row['path']} "
            f"| {row['p50_ms']:.4f} | {row['p95_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## D=256 fallback stage breakdown",
            "",
            (
                "Stage measurements intentionally synchronize between stages; "
                "use them for attribution rather than end-to-end latency."
            ),
            "",
            (
                "| Tokens | Layout/repeat P50 (ms) | QK+mask+softmax P50 (ms) "
                "| PV P50 (ms) | QK stage share | Score matrix (MiB) |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["fallback_stages"]:
        lines.append(
            f"| {row['sequence_length']} | {row['layout_p50_ms']:.4f} "
            f"| {row['qk_mask_softmax_p50_ms']:.4f} | {row['pv_p50_ms']:.4f} "
            f"| {row['qk_mask_softmax_share']:.1%} "
            f"| {row['score_matrix_mib']:.2f} |"
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
        default=Path("local_tests/qwen35/results/prefill_profile"),
    )
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("An available Ascend NPU is required")
    device = torch.device("npu:0")
    lengths = [int(value) for value in args.lengths.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "configuration": {
            "device": str(device),
            "dtype": "bfloat16",
            "q_heads": 8,
            "kv_heads": 2,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "sequence_lengths": lengths,
        },
        "whole_operator": [
            _profile_whole_path(
                seq_len,
                head_dim,
                args.warmups,
                args.repeats,
                device,
            )
            for seq_len in lengths
            for head_dim in (128, 256)
        ],
        "fallback_stages": [
            _profile_fallback_stages(seq_len, args.warmups, args.repeats, device)
            for seq_len in lengths
        ],
    }
    json_path = args.output_dir / "prefill_attention_profile.json"
    markdown_path = args.output_dir / "prefill_attention_profile.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(markdown_path, results)
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
