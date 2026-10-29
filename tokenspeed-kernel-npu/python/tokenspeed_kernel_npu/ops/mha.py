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

"""Ascend multi-head attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu

_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    if window_left >= 0:
        raise NotImplementedError("Ascend MHA does not support sliding windows")
    if logit_cap:
        raise NotImplementedError("Ascend MHA does not support logit caps")
    if sinks is not None:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if return_lse:
        raise NotImplementedError("Ascend MHA does not return LSE")


def _mha_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_cpu: list[int],
    softmax_scale: float | None,
) -> torch.Tensor:
    """Torch fallback for head sizes unsupported by the fused Ascend operator."""
    outputs = []
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError(
            f"query heads ({num_heads}) must be divisible by KV heads ({num_kv_heads})"
        )
    kv_repeats = num_heads // num_kv_heads
    scale = _scale(q, softmax_scale)
    for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        q_seq = q[start:end].transpose(0, 1)
        k_seq = k[start:end].transpose(0, 1)
        v_seq = v[start:end].transpose(0, 1)
        if kv_repeats != 1:
            k_seq = k_seq.repeat_interleave(kv_repeats, dim=0)
            v_seq = v_seq.repeat_interleave(kv_repeats, dim=0)
        seq_len = end - start
        scores = torch.matmul(q_seq.float(), k_seq.float().transpose(-2, -1))
        scores.mul_(scale)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores.masked_fill_(causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v_seq.dtype)
        outputs.append(torch.matmul(probs, v_seq).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run causal variable-length MHA over uncached K/V."""
    del cu_seqlens, max_seqlen
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q.shape[-1] not in (64, 128, 192):
        return _mha_prefill_reference(q, k, v, cu_seqlens_cpu, softmax_scale)
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=_causal_mask(q.device),
        actual_seq_lengths=cu_seqlens_cpu[1:],
        actual_seq_lengths_kv=cu_seqlens_cpu[1:],
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=2,
    )
    return output


def mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run variable-length MHA over a paged K/V cache."""
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=3 if is_causal else 0,
        block_size=k_cache.shape[1],
    )
    return output


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run fixed-shape, graph-capturable decode over a paged K/V cache."""
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if max_seqlen_q != 1:
        raise NotImplementedError("Ascend MHA decode supports one query per request")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    del max_seqlen_k, enable_pdl
    batch_size = cache_seqlens.shape[0]
    actual_seq_lengths_kv = (
        [1] * batch_size if torch.npu.is_current_stream_capturing() else cache_seqlens
    )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.reshape(batch_size, 1, -1),
        k_cache.flatten(2),
        v_cache.flatten(2),
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        block_size=k_cache.shape[1],
    )
    return output.reshape_as(q)


__all__ = [
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]
