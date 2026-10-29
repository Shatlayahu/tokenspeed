import math

import torch
import torch_npu
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
    gqa_prologue,
)

DEVICE = "npu:0"
DTYPE = torch.bfloat16

Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 128
PREFILL_LEN = 5
DECODE_STEPS = 3
TOTAL = PREFILL_LEN + DECODE_STEPS

torch.manual_seed(42)


def make_rope_table(max_positions=128, rotary_dim=64):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(
        torch.arange(max_positions, dtype=torch.float32),
        inv_freq,
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DEVICE)


def reference_attention(q, k, v):
    """
    PyTorch reference:
    q: [query_tokens, q_heads, head_dim]
    k/v: [all_visible_tokens, kv_heads, head_dim]
    """
    q = q.float()
    k = k.float().repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
    v = v.float().repeat_interleave(Q_HEADS // KV_HEADS, dim=1)

    scores = torch.einsum("thd,shd->hts", q, k)
    scores = scores / math.sqrt(HEAD_DIM)

    # Query tokens are the last query_len tokens in visible KV.
    nq, nk = q.shape[0], k.shape[0]
    q_positions = nk - nq + torch.arange(nq)
    k_positions = torch.arange(nk)
    mask = k_positions[None, :] > q_positions[:, None]
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    result = torch.einsum("hts,shd->thd", probs, v)

    return result


# Qwen3.5-style per-head RMSNorm weights.
norm = HeadNorm(
    q_weight=torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE) * 0.1,
    k_weight=torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE) * 0.1,
    weight_offset=1.0,
    eps=1e-6,
)

rope_table = make_rope_table(rotary_dim=64)

# Flat view for GQA Prologue writes.
# Paged view for MHA Decode reads.
k_pages = torch.zeros(
    1,
    PAGE_SIZE,
    KV_HEADS,
    HEAD_DIM,
    device=DEVICE,
    dtype=DTYPE,
)
v_pages = torch.zeros_like(k_pages)

k_flat = k_pages.view(PAGE_SIZE, KV_HEADS, HEAD_DIM)
v_flat = v_pages.view(PAGE_SIZE, KV_HEADS, HEAD_DIM)

page_table = torch.tensor([[0]], dtype=torch.int32, device=DEVICE)

# Keep the independent reference history on CPU.
history_q = []
history_k = []
history_v = []

# Create Q/K/V and a 3D strided gate.
packed = torch.randn(
    TOTAL,
    Q_HEADS,
    2,
    HEAD_DIM,
    device=DEVICE,
    dtype=DTYPE,
)
q_all, gate_all = packed.unbind(dim=2)

k_all = torch.randn(
    TOTAL,
    KV_HEADS * HEAD_DIM,
    device=DEVICE,
    dtype=DTYPE,
)
v_all = torch.randn_like(k_all)


def prepare(start, length, return_kv):
    positions = torch.arange(start, start + length, dtype=torch.int64, device=DEVICE)

    rotary = Rotary(
        cos_sin_cache=rope_table,
        positions=positions,
        style=RopeStyle.NEOX,
        mrope=None,
    )

    slots = torch.arange(start, start + length, dtype=torch.int32, device=DEVICE)

    cache = HeadKVCache(
        k_cache=k_flat,
        v_cache=v_flat,
        scales=None,
        slots=slots,
    )

    output = gqa_prologue(
        q=q_all[start : start + length].reshape(length, -1),
        k=k_all[start : start + length],
        v=v_all[start : start + length],
        norm=norm,
        rotary=rotary,
        cache=cache,
        return_kv=return_kv,
        solution=None,
        override=None,
    )

    torch.npu.synchronize()

    q = output.q.reshape(length, Q_HEADS, HEAD_DIM)

    # Use the actual cache contents for K/V, including Decode.
    # Prior GQA tests already compared these against independent
    # RMSNorm + RoPE reference calculations.
    new_k = k_flat[start : start + length].float().cpu()
    new_v = v_flat[start : start + length].float().cpu()

    history_k.extend([row.clone() for row in new_k])
    history_v.extend([row.clone() for row in new_v])

    return q, output


def check_attention(actual, q, gate, label):
    actual_flat = actual.reshape(q.shape[0], -1).contiguous()

    # Reference MHA from processed Q and stored K/V.
    k_cpu = torch.stack(history_k)
    v_cpu = torch.stack(history_v)

    expected = reference_attention(q.float().cpu(), k_cpu, v_cpu).reshape(
        q.shape[0], -1
    )

    # Apply the same BF16 output rounding as an attention kernel.
    expected = expected.to(DTYPE).float()

    # Reference gate multiplication in FP32.
    expected_gated = (
        expected * torch.sigmoid(gate.reshape(q.shape[0], -1).float().cpu())
    ).to(DTYPE)

    # Actual in-place Ascend output gate.
    original_ptr = actual_flat.data_ptr()
    sigmoid_mul(actual_flat, gate)
    torch.npu.synchronize()

    assert actual_flat.data_ptr() == original_ptr

    actual_cpu = actual_flat.float().cpu()
    expected_cpu = expected_gated.float()

    error = (actual_cpu - expected_cpu).abs().max().item()

    torch.testing.assert_close(
        actual_cpu,
        expected_cpu,
        atol=0.06,
        rtol=0.06,
    )

    print(
        f"PASSED | {label} | max_error={error:.6f}",
        flush=True,
    )


# --------------------------------------------------
# 1. Prefill
# --------------------------------------------------

q, prepared = prepare(0, PREFILL_LEN, return_kv=True)

assert prepared.k is not None
assert prepared.v is not None

cu = torch.tensor([0, PREFILL_LEN], dtype=torch.int32, device=DEVICE)

prefill_result = mha_prefill(
    q=q,
    k=prepared.k.reshape(PREFILL_LEN, KV_HEADS, HEAD_DIM),
    v=prepared.v.reshape(PREFILL_LEN, KV_HEADS, HEAD_DIM),
    cu_seqlens=cu,
    cu_seqlens_cpu=[0, PREFILL_LEN],
    max_seqlen=PREFILL_LEN,
    solution=None,
)

check_attention(
    prefill_result,
    q,
    gate_all[:PREFILL_LEN],
    "Prefill",
)


# --------------------------------------------------
# 2. Sequential Decode
# --------------------------------------------------

for step in range(DECODE_STEPS):
    token_index = PREFILL_LEN + step

    q, prepared = prepare(token_index, 1, return_kv=False)

    assert prepared.k is None
    assert prepared.v is None

    visible_length = token_index + 1

    decode_result = mha_decode_with_kvcache(
        q=q,
        k_cache=k_pages,
        v_cache=v_pages,
        page_table=page_table,
        cache_seqlens=torch.tensor(
            [visible_length],
            dtype=torch.int32,
            device=DEVICE,
        ),
        max_seqlen_k=PAGE_SIZE,
        max_seqlen_q=1,
        solution=None,
    )

    check_attention(
        decode_result,
        q,
        gate_all[token_index : token_index + 1],
        f"Decode step {step + 1}",
    )


print("\nALL GATED ATTENTION INTEGRATION CASES PASSED")
