import torch
import torch_npu
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
    gqa_prologue,
)

DEVICE = "npu:0"
HEAD_DIM = 128
Q_HEADS = 8
KV_HEADS = 2
EPS = 1e-6


def reference_norm(x, weight):
    """CPU FP32 reference with Ascend's BF16 weight rounding."""
    x = x.float()
    w = (weight.float() + 1.0).to(weight.dtype).float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w).to(
        torch.bfloat16
    )


def reference_rope(x, cache, positions, rotary_dim):
    """Independent CPU NeoX partial RoPE reference."""
    x = x.float()
    rot = x[..., :rotary_dim]
    remaining = x[..., rotary_dim:]

    half = rotary_dim // 2
    left, right = rot[..., :half], rot[..., half:]
    rotated = torch.cat([-right, left], dim=-1)

    table = cache[positions]
    cos, sin = table.chunk(2, dim=-1)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)

    out = rot * cos + rotated * sin
    return torch.cat([out, remaining], dim=-1).to(torch.bfloat16)


def run_case(tokens, factor, return_kv):
    torch.manual_seed(42 + tokens)

    rotary_dim = int(HEAD_DIM * factor)
    slots_count = 32

    # Use BF16: the registered composite Ascend path supports BF16/FP16.
    q = torch.randn(tokens, Q_HEADS * HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(tokens, KV_HEADS * HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(tokens, KV_HEADS * HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    q_weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k_weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16) * 0.1

    norm = HeadNorm(
        q_weight=q_weight,
        k_weight=k_weight,
        weight_offset=1.0,
        eps=EPS,
    )

    positions = torch.arange(3, 3 + tokens, device=DEVICE, dtype=torch.int64)

    # Build an ordinary NeoX cos/sin table.
    base = 10000.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(torch.arange(64, dtype=torch.float32), inv_freq)
    cos_sin_cpu = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    rotary = Rotary(
        cos_sin_cache=cos_sin_cpu.to(DEVICE),
        positions=positions,
        style=RopeStyle.NEOX,
        mrope=None,
    )

    slots_cpu = torch.arange(1, 1 + tokens, dtype=torch.int32) * 2
    slots = slots_cpu.to(DEVICE)

    k_cache = torch.zeros(
        slots_count, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)

    cache = HeadKVCache(
        k_cache=k_cache,
        v_cache=v_cache,
        scales=None,
        slots=slots,
    )

    # Independent CPU reference
    q_ref = reference_norm(q.cpu().view(tokens, Q_HEADS, HEAD_DIM), q_weight.cpu())
    k_ref = reference_norm(k.cpu().view(tokens, KV_HEADS, HEAD_DIM), k_weight.cpu())

    pos_cpu = positions.cpu()
    q_ref = reference_rope(q_ref, cos_sin_cpu, pos_cpu, rotary_dim)
    k_ref = reference_rope(k_ref, cos_sin_cpu, pos_cpu, rotary_dim)
    v_ref = v.cpu().view(tokens, KV_HEADS, HEAD_DIM)

    # The real TokenSpeed public API.
    result = gqa_prologue(
        q=q,
        k=k,
        v=v,
        norm=norm,
        rotary=rotary,
        cache=cache,
        return_kv=return_kv,
        solution=None,
        override=None,
    )

    torch.npu.synchronize()

    atol = 0.04
    rtol = 0.04

    torch.testing.assert_close(
        result.q.float().cpu().reshape(tokens, Q_HEADS, HEAD_DIM),
        q_ref.float(),
        atol=atol,
        rtol=rtol,
    )

    # Check written cache regardless of return_kv.
    torch.testing.assert_close(
        k_cache[slots.long()].float().cpu(),
        k_ref.float(),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        v_cache[slots.long()].cpu(),
        v_ref,
        atol=0,
        rtol=0,
    )

    if return_kv:
        assert result.k is not None
        assert result.v is not None

        torch.testing.assert_close(
            result.k.float().cpu().reshape(tokens, KV_HEADS, HEAD_DIM),
            k_ref.float(),
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            result.v.cpu().reshape(tokens, KV_HEADS, HEAD_DIM),
            v_ref,
            atol=0,
            rtol=0,
        )
    else:
        assert result.k is None
        assert result.v is None

    # Ensure unused cache slots were not modified.
    used = set(slots_cpu.tolist())
    unused = [i for i in range(slots_count) if i not in used]
    assert torch.count_nonzero(k_cache[unused]).item() == 0
    assert torch.count_nonzero(v_cache[unused]).item() == 0

    q_error = (
        (result.q.float().cpu().reshape(tokens, Q_HEADS, HEAD_DIM) - q_ref.float())
        .abs()
        .max()
        .item()
    )

    print(
        f"PASSED | tokens={tokens}, factor={factor}, "
        f"return_kv={return_kv}, q_max_error={q_error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    count = 0

    for tokens in [1, 7]:
        for factor in [0.25, 1.0]:
            for return_kv in [True, False]:
                run_case(tokens, factor, return_kv)
                count += 1

    print(f"\nALL {count} GQA PROLOGUE CASES PASSED")
