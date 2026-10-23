import torch
import torch_npu

from tokenspeed.runtime.layers.rotary_embedding import get_rope

DEVICE = "npu:0"
torch.manual_seed(42)


def reference_rope(x, positions, cos_sin_cache, rotary_dim, head_dim):
    """Independent PyTorch reference for NeoX-style partial RoPE."""
    tokens = x.shape[0]
    heads = x.shape[1] // head_dim

    x3 = x.float().reshape(tokens, heads, head_dim)

    # RoPE rotates only the first rotary_dim features.
    rotated_part = x3[..., :rotary_dim]
    unchanged_part = x3[..., rotary_dim:]

    cache = cos_sin_cache.float()[positions.long()]
    cos, sin = cache.chunk(2, dim=-1)

    # NeoX rotation pairs the two halves.
    half = rotary_dim // 2
    left = rotated_part[..., :half]
    right = rotated_part[..., half:]

    rotated_half = torch.cat((-right, left), dim=-1)

    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)

    result = rotated_part * cos + rotated_half * sin

    return (
        torch.cat((result, unchanged_part), dim=-1)
        .reshape(tokens, heads * head_dim)
        .to(x.dtype)
    )


def run_case(dtype, factor, tokens):
    head_dim = 128
    q_heads = 8
    kv_heads = 2

    rotary_dim = int(head_dim * factor)
    assert rotary_dim % 2 == 0

    # Tests ordinary NeoX RoPE; model-specific scaling
    # still needs to be checked using its real config.
    rope = get_rope(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=128,
        base=10000,
        is_neox_style=True,
        rope_scaling=None,
        dtype=torch.float32,
        partial_rotary_factor=factor,
    ).to(DEVICE)

    positions = torch.arange(5, 5 + tokens, dtype=torch.int64, device=DEVICE)

    q = torch.randn(tokens, q_heads * head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(tokens, kv_heads * head_dim, dtype=dtype, device=DEVICE)

    expected_q = reference_rope(q, positions, rope.cos_sin_cache, rotary_dim, head_dim)
    expected_k = reference_rope(k, positions, rope.cos_sin_cache, rotary_dim, head_dim)

    actual_q, actual_k = rope(positions, q.clone(), k.clone())
    torch.npu.synchronize()

    atol = rtol = 0.03 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(
        actual_q.float().cpu(),
        expected_q.float().cpu(),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        actual_k.float().cpu(),
        expected_k.float().cpu(),
        atol=atol,
        rtol=rtol,
    )

    # The non-rotary dimensions must remain unchanged.
    if factor < 1:
        for actual, original, heads in [
            (actual_q, q, q_heads),
            (actual_k, k, kv_heads),
        ]:
            a = actual.reshape(tokens, heads, head_dim)
            b = original.reshape(tokens, heads, head_dim)
            torch.testing.assert_close(
                a[..., rotary_dim:].cpu(),
                b[..., rotary_dim:].cpu(),
                atol=0,
                rtol=0,
            )

    err_q = (actual_q.float().cpu() - expected_q.float().cpu()).abs().max().item()

    print(
        f"PASSED | dtype={dtype}, factor={factor}, "
        f"tokens={tokens}, rotary_dim={rotary_dim}, "
        f"q_max_error={err_q:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    count = 0

    for dtype in [torch.float32, torch.bfloat16]:
        for factor in [0.25, 0.5, 1.0]:
            for tokens in [1, 7]:
                run_case(dtype, factor, tokens)
                count += 1

    print(f"\nALL {count} ROPE CASES PASSED")
