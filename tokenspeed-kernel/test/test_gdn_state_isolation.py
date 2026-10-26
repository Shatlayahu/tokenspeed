"""GDN recurrent state isolation: batched requests, slot independence, reuse."""

import torch
import torch_npu  # noqa: F401

from tokenspeed_kernel.ops.attention.gdn import gdn_decode_step


DEVICE = "npu:0"


def inputs(seed, batch=2):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*size, dtype=torch.bfloat16):
        return torch.randn(*size, generator=gen).to(DEVICE, dtype=dtype)

    return dict(
        q=rand(batch, 1, 2, 128),
        k=rand(batch, 1, 2, 128),
        v=rand(batch, 1, 4, 128) * 0.5,
        a=rand(batch, 1, 4, dtype=torch.float32),
        b=rand(batch, 1, 4, dtype=torch.float32),
    )


def call(data, pool, slots, alog, bias):
    out = gdn_decode_step(
        data["q"].contiguous(),
        data["k"].contiguous(),
        data["v"].contiguous(),
        A_log=alog,
        a=data["a"].contiguous(),
        dt_bias=bias,
        b=data["b"].contiguous(),
        initial_state=pool,
        initial_state_indices=torch.tensor(
            slots, device=DEVICE, dtype=torch.int32
        ),
        scale=128 ** -0.5,
        use_qk_l2norm=True,
        solution="triton",
    )
    torch.npu.synchronize()
    return out


def close(name, x, y):
    torch.testing.assert_close(
        x.float().cpu(),
        y.float().cpu(),
        atol=3e-2,
        rtol=2e-2,
        msg=name,
    )


def single(data, row):
    return {
        k: v[row:row + 1].contiguous()
        for k, v in data.items()
    }


def test_isolation(dtype):
    torch.manual_seed(2026)

    alog = torch.randn(4, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(4, device=DEVICE, dtype=torch.float32)

    slots = [1, 3]
    pool = (
        torch.randn(6, 4, 128, 128, device=DEVICE, dtype=dtype)
        * 0.02
    )
    initial = pool.clone()
    untouched = [0, 2, 4, 5]

    isolated_pools = [initial.clone(), initial.clone()]

    # Test 1: A/B in the same Decode batch
    for step in range(3):
        data = inputs(100 + step)

        standalone = [
            call(
                single(data, i),
                isolated_pools[i],
                [slots[i]],
                alog,
                bias,
            )
            for i in range(2)
        ]

        together = call(data, pool, slots, alog, bias)

        for i in range(2):
            close(
                f"request {i} output step {step}",
                together[i:i + 1],
                standalone[i],
            )
            close(
                f"request {i} state step {step}",
                pool[slots[i]],
                isolated_pools[i][slots[i]],
            )

        assert torch.equal(
            pool[untouched].cpu(),
            initial[untouched].cpu(),
        ), "Unused slots changed"

    # Test 2: Updating A must not affect B
    other_before = pool[slots[1]].clone()

    call(inputs(345, batch=1), pool, [slots[0]], alog, bias)

    assert torch.equal(
        pool[slots[1]].cpu(),
        other_before.cpu(),
    ), "Other request slot changed"

    assert torch.equal(
        pool[untouched].cpu(),
        initial[untouched].cpu(),
    ), "Unused slots changed"

    # Test 3: Reuse a slot after explicitly resetting its state
    pool[slots[0]].zero_()
    fresh_pool = pool.clone()

    assert torch.count_nonzero(pool[slots[0]]).item() == 0

    new_req = inputs(999, batch=1)
    other_before = pool[slots[1]].clone()

    reused = call(new_req, pool, [slots[0]], alog, bias)
    fresh = call(new_req, fresh_pool, [slots[0]], alog, bias)

    close("Reused slot output", reused, fresh)
    close(
        "Reused slot final state",
        pool[slots[0]],
        fresh_pool[slots[0]],
    )

    assert torch.equal(
        pool[slots[1]].cpu(),
        other_before.cpu(),
    ), "Reuse changed other request"

    assert torch.equal(
        pool[untouched].cpu(),
        initial[untouched].cpu(),
    ), "Unused slots changed"

    print(f"PASS: recurrent state isolation {dtype}", flush=True)


if __name__ == "__main__":
    assert torch.npu.is_available(), "Requires Ascend NPU"

    for dtype in [torch.float32, torch.bfloat16]:
        test_isolation(dtype)

    print("ALL GDN STATE ISOLATION TESTS PASSED.", flush=True)
