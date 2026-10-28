# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
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

# -*- coding: utf-8 -*-


import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.gdn._triton.index import prepare_chunk_indices
from tokenspeed_kernel.ops.attention.gdn._triton.op import safe_exp
from tokenspeed_kernel.platform import current_platform, pdl_enabled


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g_cumsum,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NUM_COL_TILES: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    tile_id = tl.program_id(2)
    tile_m = tile_id // NUM_COL_TILES
    tile_n = tile_id % NUM_COL_TILES

    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    row_offsets = tile_m * BM + tl.arange(0, BM)
    col_offsets = tile_n * BN + tl.arange(0, BN)

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT + tile_m * BM,),
        (BM,),
        (0,),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BM, BN], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        k_base = k + (bos * Hg + i_h // (H // Hg)) * K

        p_k_m = tl.make_block_ptr(
            k_base,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + tile_m * BM, i_k * BK),
            (BM, BK),
            (1, 0),
        )
        p_k_n = tl.make_block_ptr(
            k_base,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + tile_n * BN, i_k * BK),
            (BN, BK),
            (1, 0),
        )

        b_k_m = tl.load(p_k_m, boundary_check=(0, 1))
        b_k_n = tl.load(p_k_n, boundary_check=(0, 1))

        b_km_beta = b_k_m * b_beta[:, None]
        b_A += tl.dot(b_km_beta.to(b_k_m.dtype), tl.trans(b_k_n))

    if USE_G:
        p_g_m = tl.make_block_ptr(
            g_cumsum + bos * H + i_h,
            (T,),
            (H,),
            (i_t * BT + tile_m * BM,),
            (BM,),
            (0,),
        )
        p_g_n = tl.make_block_ptr(
            g_cumsum + bos * H + i_h,
            (T,),
            (H,),
            (i_t * BT + tile_n * BN,),
            (BN,),
            (0,),
        )

        b_g_m = tl.load(p_g_m, boundary_check=(0,))
        b_g_n = tl.load(p_g_n, boundary_check=(0,))
        b_g_diff = b_g_m[:, None] - b_g_n[None, :]
        b_A = b_A * safe_exp(b_g_diff)

    b_A = tl.where(
        row_offsets[:, None] > col_offsets[None, :],
        b_A,
        0,
    )

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (BT * H, 1),
        (i_t * BT + tile_m * BM, tile_n * BN),
        (BM, BN),
        (1, 0),
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g_cumsum (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`.
            Default: None
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """

    enable_pdl = pdl_enabled()
    B, T, Hg, K = k.shape

    H = beta.shape[-1]
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[
        (NT, B * H, triton.cdiv(BT, 32) * triton.cdiv(BT, 32))
    ](
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=64,
        BM=32,
        BN=32,
        NUM_COL_TILES=triton.cdiv(BT, 32),
        # gfx950: four waves are faster than eight.
        num_warps=4 if current_platform().is_cdna4 else 8,
        num_stages=1,
        ENABLE_PDL=enable_pdl,
        enable_ubuf_saving=True,
        multibuffer=False,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return A
