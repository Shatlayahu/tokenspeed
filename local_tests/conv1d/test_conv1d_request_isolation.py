"""Conv1D isolation: interleaved requests vs CPU and standalone execution."""
import torch
import torch_npu

from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

DEVICE = "npu:0"
DTYPE = torch.bfloat16
C, W = 128, 4
ATOL = RTOL = 0.035


def check_close(actual, expected, label):
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=ATOL,
        rtol=RTOL,
        msg=lambda m: f"{label}: {m}",
    )


def cpu_reference(seq, weight, bias):
    x = seq.float().cpu()
    w = weight.float().cpu()
    b = bias.float().cpu()
    outputs = []

    for t in range(len(x)):
        window = torch.zeros(C, W)
        for j in range(W):
            pos = t - W + 1 + j
            if pos >= 0:
                window[:, j] = x[pos]
        y = (window * w).sum(-1) + b
        outputs.append(torch.nn.functional.silu(y))

    return torch.stack(outputs)


def make_inputs():
    torch.manual_seed(101)
    weight = (torch.randn(C, W) * 0.08).to(DEVICE, dtype=DTYPE)
    bias = (torch.randn(C) * 0.03).to(DEVICE, dtype=DTYPE)
    data = {
        "A": (torch.randn(9, C) * 0.5).to(DTYPE),
        "B": (torch.randn(7, C) * 0.5 + 0.3).to(DTYPE),
    }
    return weight, bias, data


def prefill(seq, count, state, slot, weight, bias):
    # Keep the transposed stride; do NOT append .contiguous().
    x = seq[:count].to(DEVICE).T
    boundaries = torch.tensor(
        [0, count], dtype=torch.int32, device=DEVICE
    )
    lengths = torch.tensor(
        [count], dtype=torch.int32, device="cpu"
    )
    metadata = build_causal_conv1d_prefill_metadata(
        boundaries, lengths, CAUSAL_CONV1D_BLOCK_M
    )

    y = causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=state,
        query_start_loc=boundaries,
        cache_indices=torch.tensor(
            [slot], dtype=torch.int32, device=DEVICE
        ),
        has_initial_state=torch.tensor(
            [False], dtype=torch.bool, device=DEVICE
        ),
        activation="silu",
        prefill_metadata=metadata,
    )
    return y.T


def decode(token, state, slot, weight, bias):
    out = causal_conv1d_update(
        x=token.unsqueeze(0).to(DEVICE),
        conv_state=state,
        weight=weight,
        bias=bias,
        activation="silu",
        conv_state_indices=torch.tensor(
            [slot], dtype=torch.int32, device=DEVICE
        ),
        parent_indices=None,
    )
    return out[0]


def execute(schedule, data, prefill_sizes, slots, weight, bias):
    states = torch.zeros(
        5, C, W - 1, dtype=DTYPE, device=DEVICE
    )
    counts = {name: 0 for name in data}
    results = {name: [] for name in data}
    initialized = set()

    for name in schedule:
        slot = slots[name]

        # Snapshot other request's state before this operation.
        other_slots = [
            v for k, v in slots.items() if k != name
        ]
        before = {
            s: states[s].clone() for s in other_slots
        }

        if name not in initialized:
            count = prefill_sizes[name]
            out = prefill(
                data[name], count, states, slot, weight, bias
            )
            results[name].extend(list(out.unbind(0)))
            counts[name] = count
            initialized.add(name)
        else:
            index = counts[name]
            out = decode(
                data[name][index], states, slot, weight, bias
            )
            results[name].append(out)
            counts[name] += 1

        for s in other_slots:
            check_close(
                states[s], before[s],
                f"{name} changed another request's state"
            )

    torch.npu.synchronize()
    return states, results, counts


def main():
    weight, bias, data = make_inputs()
    prefill_sizes = {"A": 5, "B": 3}
    slots = {"A": 1, "B": 3}

    schedule = ["A", "B"]
    remaining = {
        name: len(seq) - prefill_sizes[name]
        for name, seq in data.items()
    }

    while any(remaining.values()):
        for name in ("B", "A"):
            if remaining[name]:
                schedule.append(name)
                remaining[name] -= 1

    states, results, counts = execute(
        schedule, data, prefill_sizes, slots, weight, bias
    )

    for name in ("A", "B"):
        assert counts[name] == len(data[name])

        standalone_schedule = [name] * (
            1 + len(data[name]) - prefill_sizes[name]
        )
        _, standalone, _ = execute(
            standalone_schedule,
            {name: data[name]},
            {name: prefill_sizes[name]},
            {name: 1},
            weight,
            bias,
        )

        check_close(
            torch.stack(results[name]),
            torch.stack(standalone[name]),
            f"{name} interleaved vs standalone",
        )

        check_close(
            torch.stack(results[name]),
            cpu_reference(data[name], weight, bias),
            f"{name} CPU reference",
        )

        expected_state = data[name][-W + 1:].T

        print(f"{name} actual state [0, :]:",
              states[slots[name]][0].float().cpu())
        print(f"{name} expected state [0, :]:",
              expected_state[0].float().cpu())

        print(
            f"{name} state max error:",
            (
                states[slots[name]].float().cpu()
                - expected_state.float().cpu()
            ).abs().max().item(),
        )

        check_close(
            states[slots[name]],
            expected_state,
            f"{name} final state",
        )

        print(
            f"PASS Conv1D request {name}: "
            f"prefill={prefill_sizes[name]}, "
            f"decode={len(data[name]) - prefill_sizes[name]}, "
            f"slot={slots[name]}",
            flush=True,
        )

    for slot in (0, 2, 4):
        check_close(
            states[slot],
            torch.zeros_like(states[slot]),
            f"unused state slot {slot}",
        )

    print(
        "PASS: Conv1D interleaved requests, "
        "CPU reference, state isolation"
    )


if __name__ == "__main__":
    main()
