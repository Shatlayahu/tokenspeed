#!/usr/bin/env python3
"""Extended GDN recurrent-state isolation tests for Ascend NPU."""

import torch
import torch_npu  # noqa: F401

import standalone_gdn_ascend as base


DEVICE = base.DEVICE
DTYPE = base.DTYPE

STEPS = 3
NUM_Q_HEADS = 4
NUM_V_HEADS = 8
HEAD_DIM = 128
POOL_SIZE = 4

ROW_A = 1
ROW_B = 3
UNUSED_ROWS = [0, 2]

RTOL = 1e-5
ATOL = 1e-5

SEQ_KEYS = ("q", "k", "v", "a", "b")


def make_sequence(seed):
    """Create a deterministic three-token request."""
    torch.manual_seed(seed)

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


def make_pool(state_dtype, seed=100):
    """Create a reproducible nonzero state pool."""
    torch.manual_seed(seed)
    return torch.randn(
        POOL_SIZE,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=DEVICE,
        dtype=state_dtype,
    ) * 0.02


def make_parameters(seed=200):
    torch.manual_seed(seed)

    A_log = torch.randn(
        NUM_V_HEADS, device=DEVICE, dtype=torch.float32
    )
    dt_bias = torch.randn(
        NUM_V_HEADS, device=DEVICE, dtype=torch.float32
    )

    return A_log, dt_bias


def run_step(pool, seq, t, rows, A_log, dt_bias, scale):
    """Run one decode step for the supplied request rows."""
    indices = torch.tensor(
        rows, device=DEVICE, dtype=torch.int32
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
    """Run one request independently for three steps."""
    outputs = []
    states = []

    for t in range(STEPS):
        out = run_step(
            pool, seq, t, [row], A_log, dt_bias, scale
        )
        outputs.append(out[0])
        states.append(pool[row].detach().cpu().clone())

    return torch.stack(outputs, dim=0), states


def assert_close(name, actual, expected):
    """Compare numerical outputs or states."""
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()

    max_error = (actual - expected).abs().max().item()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
        msg=f"{name} does not match its reference",
    )

    print(f"PASS: {name}; max_abs_diff={max_error:.8g}", flush=True)


def assert_unchanged(name, actual, expected):
    """Require an untouched state row to remain bitwise identical."""
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()

    if not torch.equal(actual, expected):
        max_error = (
            actual.float() - expected.float()
        ).abs().max().item()

        raise AssertionError(
            f"{name} was modified; max_abs_diff={max_error:.8g}"
        )

    print(f"PASS: {name}", flush=True)


def test_two_requests_same_batch(state_dtype):
    """Two requests share a batch but use distinct state-pool rows."""
    print(
        f"\n===== Same-batch Decode: {state_dtype} =====",
        flush=True,
    )

    seq_a = make_sequence(301)
    seq_b = make_sequence(302)
    initial_pool = make_pool(state_dtype, seed=303)
    A_log, dt_bias = make_parameters(seed=304)
    scale = float(HEAD_DIM ** -0.5)

    # Baseline A: run A independently from its initial state.
    pool_a = initial_pool.clone()
    ref_a_out, ref_a_states = run_sequence(
        pool_a, seq_a, ROW_A, A_log, dt_bias, scale
    )

    # Baseline B: run B independently from its initial state.
    pool_b = initial_pool.clone()
    ref_b_out, ref_b_states = run_sequence(
        pool_b, seq_b, ROW_B, A_log, dt_bias, scale
    )

    # Combine A and B into one batch.
    seq_ab = {
        key: torch.cat([seq_a[key], seq_b[key]], dim=0)
        for key in SEQ_KEYS
    }

    batch_pool = initial_pool.clone()
    batch_outputs = []

    for t in range(STEPS):
        # Both requests execute in the same kernel call.
        out = run_step(
            batch_pool,
            seq_ab,
            t,
            [ROW_A, ROW_B],
            A_log,
            dt_bias,
            scale,
        )
        batch_outputs.append(out)

        # Each request's state must match its independent baseline
        # after every token, not merely at the end.
        assert_close(
            f"A state after batch step {t + 1}",
            batch_pool[ROW_A],
            ref_a_states[t],
        )
        assert_close(
            f"B state after batch step {t + 1}",
            batch_pool[ROW_B],
            ref_b_states[t],
        )

    batch_outputs = torch.stack(batch_outputs, dim=1)

    assert_close("A same-batch output", batch_outputs[0], ref_a_out)
    assert_close("B same-batch output", batch_outputs[1], ref_b_out)

    for row in UNUSED_ROWS:
        assert_unchanged(
            f"Unused row {row} after same-batch Decode",
            batch_pool[row],
            initial_pool[row],
        )

    print(
        "PASS: two-request same-batch Decode, three steps",
        flush=True,
    )


def test_single_request_update(state_dtype):
    """Update A alone and ensure B and unused rows remain unchanged."""
    print(
        f"\n===== Single-request Update: {state_dtype} =====",
        flush=True,
    )

    seq_a = make_sequence(401)
    initial_pool = make_pool(state_dtype, seed=402)
    A_log, dt_bias = make_parameters(seed=403)
    scale = float(HEAD_DIM ** -0.5)

    # Baseline: execute A's first token independently.
    baseline_pool = initial_pool.clone()
    ref_out = run_step(
        baseline_pool, seq_a, 0, [ROW_A],
        A_log, dt_bias, scale,
    )
    ref_state = baseline_pool[ROW_A].detach().cpu().clone()

    # The real test pool also contains B's live state.
    test_pool = initial_pool.clone()
    b_before = test_pool[ROW_B].detach().clone()
    unused_before = {
        row: test_pool[row].detach().clone()
        for row in UNUSED_ROWS
    }

    actual_out = run_step(
        test_pool, seq_a, 0, [ROW_A],
        A_log, dt_bias, scale,
    )
    actual_state = test_pool[ROW_A].detach().cpu().clone()

    assert_close("A single-request output", actual_out, ref_out)
    assert_close("A single-request updated state", actual_state, ref_state)

    assert_unchanged(
        "B state after A-only update",
        test_pool[ROW_B],
        b_before,
    )

    for row in UNUSED_ROWS:
        assert_unchanged(
            f"Unused row {row} after A-only update",
            test_pool[row],
            unused_before[row],
        )

    print("PASS: single-request update isolation", flush=True)


def test_cleared_slot_reuse(state_dtype):
    """Clear a released slot, reuse it for C, and compare with zero-state C."""
    print(
        f"\n===== Cleared-slot Reuse: {state_dtype} =====",
        flush=True,
    )

    initial_pool = make_pool(state_dtype, seed=501)
    A_log, dt_bias = make_parameters(seed=502)
    seq_c = make_sequence(503)
    scale = float(HEAD_DIM ** -0.5)

    # Simulate the pool containing an old request's state.
    actual_pool = initial_pool.clone()

    if torch.count_nonzero(actual_pool[ROW_A]).item() == 0:
        raise AssertionError("Expected the old slot to contain nonzero state.")

    b_before = actual_pool[ROW_B].detach().clone()
    unused_before = {
        row: actual_pool[row].detach().clone()
        for row in UNUSED_ROWS
    }

    # Simulate releasing A and explicitly clearing its state slot.
    actual_pool[ROW_A].zero_()
    torch.npu.synchronize()

    assert_unchanged(
        "Released row is explicitly zeroed",
        actual_pool[ROW_A],
        torch.zeros_like(actual_pool[ROW_A]),
    )

    # Independent baseline: C starts from a zero state.
    baseline_pool = torch.zeros_like(initial_pool)
    ref_c_out, ref_c_states = run_sequence(
        baseline_pool, seq_c, ROW_A, A_log, dt_bias, scale
    )

    # Reuse A's cleared slot for new request C, while other pool
    # rows still contain their original states.
    actual_c_out, actual_c_states = run_sequence(
        actual_pool, seq_c, ROW_A, A_log, dt_bias, scale
    )

    assert_close("New request C output after slot reuse", actual_c_out, ref_c_out)

    for t in range(STEPS):
        assert_close(
            f"New request C state after slot-reuse step {t + 1}",
            actual_c_states[t],
            ref_c_states[t],
        )

    assert_close(
        "New request C final state after slot reuse",
        actual_pool[ROW_A],
        baseline_pool[ROW_A],
    )

    # Reusing A's slot must not modify B or unused pool rows.
    assert_unchanged(
        "B state after C reuses A's slot",
        actual_pool[ROW_B],
        b_before,
    )

    for row in UNUSED_ROWS:
        assert_unchanged(
            f"Unused row {row} after slot reuse",
            actual_pool[row],
            unused_before[row],
        )

    print(
        "PASS: cleared-slot reuse for a new request",
        flush=True,
    )


def run_all(state_dtype):
    """Run all three isolation scenarios for one state dtype."""
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available.")

    test_two_requests_same_batch(state_dtype)
    test_single_request_update(state_dtype)
    test_cleared_slot_reuse(state_dtype)

    print(
        f"\nALL EXTENDED ISOLATION TESTS PASSED: {state_dtype}",
        flush=True,
    )


if __name__ == "__main__":
    run_all(torch.float32)
    run_all(torch.bfloat16)
    print("\nALL FP32/BF16 SCENARIOS PASSED", flush=True)
