"""Standalone numerical tests for GDN Triton kernels on Ascend NPU."""

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from tokenspeed_kernel.ops.attention.gdn import (
    gdn_chunk_prefill,
    gdn_decode_step,
)


DEVICE = torch.device("npu:0")
DTYPE = torch.bfloat16
ATOL = 2e-2
RTOL = 2e-2


def check_close(name, actual, expected, *, atol=ATOL, rtol=RTOL):
    actual = actual.float().cpu()
    expected = expected.float().cpu()

    max_err = (actual - expected).abs().max().item()
    print(f"{name}: max_abs_error={max_err:.8g}", flush=True)

    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=name
    )


def l2norm_reference(x):
    """Match the existing GDN test helper's Q/K normalization."""
    x_float = x.float()
    return (
        x_float
        * torch.rsqrt(
            x_float.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        )
    ).to(x.dtype)


def torch_prefill_reference(q, k, v, g, beta, initial_state, cu_seqlens, scale):
    """Independent token-by-token GDN recurrence; all inputs are on CPU."""
    qf, kf, vf = q.float(), k.float(), v.float()
    gf, betaf = g.float(), beta.float()

    # Public state layout: [num_sequences, HV, V, K].
    # Reference math uses [num_sequences, HV, K, V].
    state = initial_state.float().transpose(-2, -1).clone()

    _, total_tokens, num_q_heads, _ = q.shape
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[-1]
    group_size = num_v_heads // num_q_heads

    out = torch.zeros(
        (1, total_tokens, num_v_heads, head_v_dim),
        dtype=torch.float32,
        device="cpu",
    )

    starts = cu_seqlens[:-1].tolist()
    ends = cu_seqlens[1:].tolist()
    final_states = []

    for seq_idx, (start, end) in enumerate(zip(starts, ends)):
        state_seq = state[seq_idx]

        for t in range(start, end):
            for hv in range(num_v_heads):
                qh = hv // group_size
                q_t = qf[0, t, qh]
                k_t = kf[0, t, qh]
                v_t = vf[0, t, hv]

                state_h = torch.exp(gf[0, t, hv]) * state_seq[hv]
                delta = betaf[0, t, hv] * (
                    v_t - k_t @ state_h
                )
                state_h = state_h + k_t[:, None] * delta[None, :]

                out[0, t, hv] = scale * (q_t @ state_h)
                state_seq[hv] = state_h

        final_states.append(state_seq.clone())

    final_state = (
        torch.stack(final_states)
        .transpose(-2, -1)
        .to(initial_state.dtype)
    )
    return out.to(q.dtype), final_state


def test_gdn_prefill():
    print("\n===== GDN Prefill: Ascend Triton =====", flush=True)
    torch.manual_seed(1234)

    # Match the geometry of the existing prefill correctness test.
    seq_len = 130
    num_q_heads = 16
    num_v_heads = 32
    head_dim = 128
    scale = head_dim ** -0.5

    q = torch.randn(
        1, seq_len, num_q_heads, head_dim,
        device=DEVICE, dtype=DTYPE,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, seq_len, num_v_heads, head_dim,
        device=DEVICE, dtype=DTYPE,
    ) * 0.5
    g = F.logsigmoid(
        torch.randn(1, seq_len, num_v_heads, device=DEVICE)
    )
    beta = torch.rand(
        1, seq_len, num_v_heads, device=DEVICE, dtype=DTYPE
    ).sigmoid()
    initial_state = torch.randn(
        1, num_v_heads, head_dim, head_dim,
        device=DEVICE, dtype=DTYPE,
    ) * 0.01
    cu_seqlens = torch.tensor(
        [0, seq_len], device=DEVICE, dtype=torch.int32
    )

    result = gdn_chunk_prefill(
        q, k, v, g, beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    torch.npu.synchronize()

    assert result.final_state is not None
    assert result.out.shape == v.shape
    assert result.final_state.shape == initial_state.shape

    # Independent CPU oracle. Normalize Q/K as the existing reference does.
    q_ref = l2norm_reference(q.cpu())
    k_ref = l2norm_reference(k.cpu())

    ref_out, ref_state = torch_prefill_reference(
        q_ref,
        k_ref,
        v.cpu(),
        g.cpu(),
        beta.cpu(),
        initial_state.cpu(),
        cu_seqlens.cpu(),
        scale,
    )

    check_close("Prefill output", result.out, ref_out)
    check_close(
        "Prefill final state",
        result.final_state,
        ref_state,
        atol=3e-2,
    )

    print("PASS: GDN Prefill output and final state.", flush=True)


def torch_decode_reference(
    q, k, v, a, b, A_log, dt_bias, initial_state, scale
):
    """Independent sequential recurrent-decode reference on CPU."""
    qf, kf, vf = q.float(), k.float(), v.float()
    af, bf = a.float(), b.float()
    A_log = A_log.float()
    dt_bias = dt_bias.float()

    batch, steps, num_q_heads, _ = q.shape
    num_v_heads = v.shape[2]
    group_size = num_v_heads // num_q_heads

    # K-last pool rows: [B, HV, V, K].
    state = initial_state.float().transpose(-2, -1).clone()
    out = torch.zeros(
        batch, steps, num_v_heads, v.shape[-1],
        dtype=torch.float32,
        device="cpu",
    )
    states = []

    for t in range(steps):
        x = af[:, t] + dt_bias
        softplus_x = torch.where(
            x <= 20.0, torch.log1p(torch.exp(x)), x
        )
        g = -torch.exp(A_log) * softplus_x
        beta = torch.sigmoid(bf[:, t])

        for bi in range(batch):
            for hv in range(num_v_heads):
                qh = hv // group_size
                q_t = qf[bi, t, qh]
                k_t = kf[bi, t, qh]

                qn = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                kn = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)

                state_h = state[bi, hv] * torch.exp(g[bi, hv])
                delta = beta[bi, hv] * (
                    vf[bi, t, hv] - kn @ state_h
                )
                state_h = state_h + kn[:, None] * delta[None, :]

                out[bi, t, hv] = scale * (qn @ state_h)
                state[bi, hv] = state_h

        states.append(state.transpose(-2, -1).clone())

    return out, states


def test_gdn_recurrent_decode(state_dtype):
    print(
        f"\n===== GDN Recurrent Decode: state_dtype={state_dtype} =====",
        flush=True,
    )
    torch.manual_seed(7)

    batch = 4
    steps = 16
    num_q_heads = 4
    num_v_heads = 8
    head_dim = 128
    pool_size = 16
    scale = head_dim ** -0.5

    q = torch.randn(
        batch, steps, num_q_heads, head_dim,
        device=DEVICE, dtype=DTYPE,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch, steps, num_v_heads, head_dim,
        device=DEVICE, dtype=DTYPE,
    ) * 0.5
    a = torch.randn(
        batch, steps, num_v_heads, device=DEVICE, dtype=torch.float32
    )
    b = torch.randn_like(a)
    A_log = torch.randn(num_v_heads, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(
        num_v_heads, device=DEVICE, dtype=torch.float32
    )

    pool = torch.randn(
        pool_size, num_v_heads, head_dim, head_dim,
        device=DEVICE, dtype=state_dtype,
    ) * 0.02
    original_pool = pool.clone()
    read_idx = torch.tensor(
        [1, 3, 5, 7], device=DEVICE, dtype=torch.int32
    )
    scale = float(scale)

    ref_out, ref_states = torch_decode_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        a.cpu(),
        b.cpu(),
        A_log.cpu(),
        dt_bias.cpu(),
        original_pool[read_idx].cpu(),
        scale,
    )

    # Run the actual T=1 recurrent kernel repeatedly, updating the same
    # state-pool rows after every token.
    max_output_error = 0.0
    max_state_error = 0.0

    for t in range(steps):
        out_t = gdn_decode_step(
            q[:, t:t + 1].contiguous(),
            k[:, t:t + 1].contiguous(),
            v[:, t:t + 1].contiguous(),
            A_log=A_log,
            a=a[:, t:t + 1].contiguous(),
            dt_bias=dt_bias,
            b=b[:, t:t + 1].contiguous(),
            initial_state=pool,
            initial_state_indices=read_idx,
            scale=scale,
            use_qk_l2norm=True,
            solution="triton",
        )
        torch.npu.synchronize()

        actual_out = out_t[:, 0].float().cpu()
        expected_out = ref_out[:, t]
        torch.testing.assert_close(
            actual_out, expected_out, rtol=RTOL, atol=ATOL
        )

        output_error = (actual_out - expected_out).abs().max().item()
        max_output_error = max(max_output_error, output_error)

        actual_state = pool[read_idx].float().cpu()
        expected_state = ref_states[t].to(state_dtype).float()
        torch.testing.assert_close(
            actual_state, expected_state, rtol=RTOL, atol=3e-2
        )

        state_error = (
            actual_state - expected_state
        ).abs().max().item()
        max_state_error = max(max_state_error, state_error)

    # The decode kernel must not change pool rows it never reads or writes.
    untouched_rows = [i for i in range(pool_size) if i not in [1, 3, 5, 7]]
    assert torch.equal(
        pool[untouched_rows].cpu(),
        original_pool[untouched_rows].cpu(),
    ), "Untouched state-pool rows were modified."

    print(f"Steps verified: {steps}", flush=True)
    print(f"Maximum output error: {max_output_error:.8g}", flush=True)
    print(f"Maximum state error: {max_state_error:.8g}", flush=True)
    print("PASS: GDN recurrent decode.", flush=True)



def main():
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available.")

    print(f"Device: {DEVICE}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)

    test_gdn_prefill()
    test_gdn_recurrent_decode(torch.float32)
    test_gdn_recurrent_decode(torch.bfloat16)

    print("\nALL STANDALONE GDN TESTS PASSED.", flush=True)


if __name__ == "__main__":
    main()
