#!/usr/bin/env python3
"""GDN recurrent-state isolation test for Ascend NPU."""

import torch
import torch_npu  # noqa: F401

import standalone_gdn_ascend as base


DEVICE = base.DEVICE
DTYPE = base.DTYPE

STEPS = 6
NUM_Q_HEADS = 4
NUM_V_HEADS = 8
HEAD_DIM = 128
POOL_SIZE = 4

ROW_A = 1
ROW_B = 3
UNUSED_ROWS = [0, 2]

RTOL = 1e-5
ATOL = 1e-6


def make_sequence():
    """Create one independent request sequence."""
    return {
        "q": torch.randn(
            1, STEPS, NUM_Q_HEADS, HEAD_DIM,
            device=DEVICE, dtype=DTYPE,
        ),
        "k": torch.randn(
            1, STEPS, NUM_Q_HEADS, HEAD_DIM,
            device=DEVICE, dtype=DTYPE,
        ),
        "v": torch.randn(
            1, STEPS, NUM_V_HEADS, HEAD_DIM,
            device=DEVICE, dtype=DTYPE,
        ) * 0.5,
        "a": torch.randn(
            1, STEPS, NUM_V_HEADS,
            device=DEVICE, dtype=torch.float32,
        ),
        "b": torch.randn(
            1, STEPS, NUM_V_HEADS,
            device=DEVICE, dtype=torch.float32,
        ),
    }


def run_token(pool, seq, t, row, A_log, dt_bias, scale):
    """Execute one decode step and update the selected state row."""
    indices = torch.tensor(
        [row], device=DEVICE, dtype=torch.int32
    )

    out = base.gdn_decode_step(
        seq["q"][:, t:t + 1].contiguous(),
        seq["k"][:, t:t + 1].contiguous(),
        seq["v"][:, t:t + 1].contiguous(),
        A_log=A_log,
        a=seq["a"][:, t:t + 1].contiguous(),
        dt_bias=dt_bias,
        b=seq["b"][:, t:t + 1].contiguous(),
        initial_state=pool,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm=True,
        solution="triton",
    )

    torch.npu.synchronize()
    return out[:, 0].float().cpu()


def run_sequence(pool, seq, row, A_log, dt_bias, scale):
    """Run a complete request on its assigned state row."""
    outputs = []

    for t in range(STEPS):
        outputs.append(
            run_token(pool, seq, t, row, A_log, dt_bias, scale)
        )

    return (
        torch.stack(outputs, dim=1),
        pool[row].float().cpu().clone(),
    )


def assert_same(name, actual, expected):
    """Require exact equality for identical inputs and state."""
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()

    if not torch.equal(actual_cpu, expected_cpu):
        diff = (
            actual_cpu.float() - expected_cpu.float()
        ).abs().max().item()

        raise AssertionError(
            f"{name} changed unexpectedly; max_abs_diff={diff:.8g}"
        )

    print(f"PASS: {name} unchanged", flush=True)


def compare_results(name, actual, expected):
    """Compare results from independent and interleaved execution."""
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()

    max_error = (actual - expected).abs().max().item()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
        msg=f"{name} differs between isolated and interleaved runs",
    )

    print(f"{name} max_abs_diff: {max_error:.8g}", flush=True)


def test_gdn_state_isolation(state_dtype):
    print(
        f"\n===== GDN State Isolation: {state_dtype} =====",
        flush=True,
    )

    torch.manual_seed(20261026)

    scale = float(HEAD_DIM ** -0.5)

    # A and B have different token sequences.
    seq_a = make_sequence()
    seq_b = make_sequence()

    A_log = torch.randn(
        NUM_V_HEADS, device=DEVICE, dtype=torch.float32
    )
    dt_bias = torch.randn(
        NUM_V_HEADS, device=DEVICE, dtype=torch.float32
    )

    # All scenarios start with exactly the same state pool.
    initial_pool = torch.randn(
        POOL_SIZE,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=DEVICE,
        dtype=state_dtype,
    ) * 0.02

    # ---------------------------------------------------------
    # 1. Independent baselines
    # ---------------------------------------------------------
    pool_a = initial_pool.clone()
    ref_a_out, ref_a_state = run_sequence(
        pool_a, seq_a, ROW_A, A_log, dt_bias, scale
    )

    pool_b = initial_pool.clone()
    ref_b_out, ref_b_state = run_sequence(
        pool_b, seq_b, ROW_B, A_log, dt_bias, scale
    )

    # ---------------------------------------------------------
    # 2. Interleaved execution: A -> B -> A
    # ---------------------------------------------------------
    interleaved_pool = initial_pool.clone()

    # A executes its first token.
    a_outputs = [
        run_token(
            interleaved_pool, seq_a, 0,
            ROW_A, A_log, dt_bias, scale,
        )
    ]

    # A's first step must not modify B's state.
    assert_same(
        "B state after A's first token",
        interleaved_pool[ROW_B],
        initial_pool[ROW_B],
    )

    # Save A's state before B runs.
    a_state_before_b = interleaved_pool[ROW_A].clone()

    # B executes its complete sequence.
    b_outputs = []
    for t in range(STEPS):
        b_outputs.append(
            run_token(
                interleaved_pool, seq_b, t,
                ROW_B, A_log, dt_bias, scale,
            )
        )

    # B must not modify A's state.
    assert_same(
        "A state while B executes",
        interleaved_pool[ROW_A],
        a_state_before_b,
    )

    # A resumes from its saved state.
    for t in range(1, STEPS):
        a_outputs.append(
            run_token(
                interleaved_pool, seq_a, t,
                ROW_A, A_log, dt_bias, scale,
            )
        )

    interleaved_a_out = torch.stack(a_outputs, dim=1)
    interleaved_b_out = torch.stack(b_outputs, dim=1)

    # ---------------------------------------------------------
    # 3. Compare outputs and final states
    # ---------------------------------------------------------
    compare_results(
        "A output",
        interleaved_a_out,
        ref_a_out,
    )
    compare_results(
        "A final state",
        interleaved_pool[ROW_A],
        ref_a_state,
    )

    compare_results(
        "B output",
        interleaved_b_out,
        ref_b_out,
    )
    compare_results(
        "B final state",
        interleaved_pool[ROW_B],
        ref_b_state,
    )

    # Neither request should modify unrelated state rows.
    for row in UNUSED_ROWS:
        assert_same(
            f"Unused state row {row}",
            interleaved_pool[row],
            initial_pool[row],
        )

    print(
        f"PASS: GDN state isolation ({state_dtype})",
        flush=True,
    )


def main():
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available.")

    for state_dtype in (torch.float32, torch.bfloat16):
        test_gdn_state_isolation(state_dtype)

    print(
        "\nALL GDN STATE ISOLATION TESTS PASSED",
        flush=True,
    )


if __name__ == "__main__":
    main()
