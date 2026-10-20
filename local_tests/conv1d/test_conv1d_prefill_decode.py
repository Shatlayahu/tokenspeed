import torch
import torch_npu
import torch.nn.functional as F

from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

DEVICE = "npu:0"
WIDTH = 4
DIM = 128
DECODE_STEPS = 10


def run_case(prefill_len, dtype):
    torch.manual_seed(42 + prefill_len)

    total_len = prefill_len + DECODE_STEPS

    # Generate all inputs upfront, ensuring both
    # the NPU and CPU reference see identical values.
    tokens = torch.randn(
        total_len, DIM, dtype=dtype, device="cpu"
    )
    weight_cpu = (
        torch.randn(DIM, WIDTH, dtype=torch.float32) * 0.1
    ).to(dtype)

    weight = weight_cpu.to(DEVICE)

    # Two cache slots: slot 0 unused, slot 1 active.
    state = torch.zeros(
        2, DIM, WIDTH - 1, dtype=dtype, device=DEVICE
    )

    # CPU reference: full continuous sequence.
    history = torch.zeros(DIM, WIDTH - 1, dtype=torch.float32)
    reference_outputs = []
    reference_states = []

    for t in range(total_len):
        current = tokens[t].float()
        window = torch.cat(
            [history, current[:, None]], dim=-1
        )

        output = (window * weight_cpu.float()).sum(-1)
        output = F.silu(output).to(dtype)

        reference_outputs.append(output)
        history = window[:, 1:].to(dtype).float().clone()
        reference_states.append(history.to(dtype).clone())

    reference_outputs = torch.stack(reference_outputs)

    # 1. Prefill
    prefill_x = tokens[:prefill_len].to(DEVICE).T

    lengths = torch.tensor([prefill_len], dtype=torch.int32)
    boundaries = torch.tensor(
        [0, prefill_len], dtype=torch.int32, device=DEVICE
    )

    metadata = build_causal_conv1d_prefill_metadata(
        boundaries, lengths, CAUSAL_CONV1D_BLOCK_M
    )

    prefill_output = causal_conv1d_fn(
        x=prefill_x,
        weight=weight,
        bias=None,
        conv_states=state,
        query_start_loc=boundaries,
        cache_indices=torch.tensor(
            [1], dtype=torch.int32, device=DEVICE
        ),
        has_initial_state=torch.tensor(
            [False], dtype=torch.bool, device=DEVICE
        ),
        activation="silu",
        pad_slot_id=-1,
        validate_data=False,
        prefill_metadata=metadata,
    )

    torch.npu.synchronize()

    atol = 0.05 if dtype == torch.bfloat16 else 1e-4
    rtol = 0.05 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(
        prefill_output.T.float().cpu(),
        reference_outputs[:prefill_len].float(),
        atol=atol, rtol=rtol,
    )

    torch.testing.assert_close(
        state[1].cpu(),
        reference_states[prefill_len - 1],
        atol=0, rtol=0,
    )

    # 2. Decode, reusing the state written by Prefill.
    max_error = 0.0

    for t in range(prefill_len, total_len):
        x = tokens[t:t+1].to(DEVICE).clone()

        output = causal_conv1d_update(
            x=x,
            conv_state=state,
            weight=weight,
            bias=None,
            activation="silu",
            conv_state_indices=torch.tensor(
                [1], dtype=torch.int32, device=DEVICE
            ),
            parent_indices=None,
        )

        torch.npu.synchronize()

        actual = output[0].float().cpu()
        expected = reference_outputs[t].float()

        max_error = max(
            max_error,
            (actual - expected).abs().max().item()
        )

        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            state[1].cpu(),
            reference_states[t],
            atol=0, rtol=0,
        )

    print(
        f"PASSED | prefill={prefill_len}, "
        f"decode={DECODE_STEPS}, dtype={dtype}, "
        f"decode_max_error={max_error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    for dtype in [torch.float32, torch.bfloat16]:
        for prefill_len in [3, 20, 33]:
            run_case(prefill_len, dtype)

    print("\nALL 6 PREFILL -> DECODE CASES PASSED")
