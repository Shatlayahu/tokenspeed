"""Paged MHA KV-cache isolation across two interleaved requests."""
import math
import torch
import torch_npu

from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
)
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

DEVICE = "npu:0"
DTYPE = torch.bfloat16
NQ, NKV, D = 8, 2, 128
PAGE_SIZE = 128
ATOL = RTOL = 0.045


def check_close(actual, expected, label):
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=ATOL,
        rtol=RTOL,
        msg=lambda m: f"{label}: {m}",
    )


def new_cache():
    shape = (5, PAGE_SIZE, NKV, D)
    return (
        torch.zeros(shape, dtype=DTYPE, device=DEVICE),
        torch.zeros(shape, dtype=DTYPE, device=DEVICE),
    )


def cpu_attention(q, k, v):
    q = q.float().cpu().reshape(NKV, NQ // NKV, D)
    k = k.float().cpu()
    v = v.float().cpu()

    scores = (
        torch.einsum("hgd,thd->hgt", q, k)
        / math.sqrt(D)
    )
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum(
        "hgt,thd->hgd", probs, v
    )
    return output.reshape(NQ, D)


def make_data():
    torch.manual_seed(2026)
    data = {}

    for name, length, prefill in (
        ("A", 8, 4),
        ("B", 6, 2),
    ):
        data[name] = {
            "q": (
                torch.randn(length, NQ, D) * 0.3
            ).to(DEVICE, dtype=DTYPE),
            "k": (
                torch.randn(length, NKV, D) * 0.3
            ).to(DEVICE, dtype=DTYPE),
            "v": (
                torch.randn(length, NKV, D) * 0.3
            ).to(DEVICE, dtype=DTYPE),
            "gate": (
                torch.randn(length, NQ, D) * 0.3
            ).to(DEVICE, dtype=DTYPE),
            "prefill": prefill,
        }

    return data


def execute(schedule, data, pages):
    k_cache, v_cache = new_cache()
    counts = {name: 0 for name in pages}
    outputs = {name: [] for name in pages}

    for name in schedule:
        item = data[name]
        page = pages[name]
        start = counts[name]

        # First operation inserts the prompt; later operations
        # append one decode token.
        count = item["prefill"] if start == 0 else 1
        stop = start + count

        assert stop <= len(item["q"])

        # Write this request's K/V only to its own page.
        k_cache[page, start:stop].copy_(
            item["k"][start:stop]
        )
        v_cache[page, start:stop].copy_(
            item["v"][start:stop]
        )

        # Query each newly visible prefix through real NPU
        # paged-decode operator. This is a KV isolation test,
        # not an actual MHA prefill-path test.
        for pos in range(start, stop):
            visible = pos + 1

            result = mha_decode_with_kvcache(
                q=item["q"][pos:pos + 1],
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=torch.tensor(
                    [[page]], dtype=torch.int32,
                    device=DEVICE,
                ),
                cache_seqlens=torch.tensor(
                    [visible], dtype=torch.int32,
                    device=DEVICE,
                ),
                max_seqlen_k=visible,
                max_seqlen_q=1,
            )

            if isinstance(result, tuple):
                result = result[0]

            gated = result[0].clone()
            sigmoid_mul(gated, item["gate"][pos])

            reference = cpu_attention(
                item["q"][pos],
                item["k"][:visible],
                item["v"][:visible],
            ).to(DEVICE, dtype=DTYPE)

            reference = (
                reference
                * torch.sigmoid(item["gate"][pos])
            )

            check_close(
                gated,
                reference,
                f"{name} token {pos}: CPU reference",
            )

            outputs[name].append(
                gated.detach().clone()
            )

        counts[name] = stop

        # Verify that this operation did not corrupt the
        # other request's KV page.
        for other, other_page in pages.items():
            if other == name:
                continue

            n = counts[other]
            expected_k = torch.zeros_like(
                k_cache[other_page]
            )
            expected_v = torch.zeros_like(
                v_cache[other_page]
            )

            expected_k[:n] = data[other]["k"][:n]
            expected_v[:n] = data[other]["v"][:n]

            check_close(
                k_cache[other_page],
                expected_k,
                f"{name} changed {other} key cache",
            )
            check_close(
                v_cache[other_page],
                expected_v,
                f"{name} changed {other} value cache",
            )

    torch.npu.synchronize()
    return k_cache, v_cache, outputs, counts


def main():
    data = make_data()
    pages = {"A": 1, "B": 3}

    schedule = ["A", "B"]
    remaining = {
        name: len(item["q"]) - item["prefill"]
        for name, item in data.items()
    }

    while any(remaining.values()):
        for name in ("B", "A"):
            if remaining[name]:
                schedule.append(name)
                remaining[name] -= 1

    k_cache, v_cache, combined, counts = execute(
        schedule, data, pages
    )

    for name in ("A", "B"):
        standalone_schedule = [name] * (
            1 + len(data[name]["q"]) - data[name]["prefill"]
        )

        _, _, standalone, _ = execute(
            standalone_schedule,
            data,
            {name: 1},
        )

        check_close(
            torch.stack(combined[name]),
            torch.stack(standalone[name]),
            f"{name}: interleaved vs standalone",
        )

        page = pages[name]
        length = counts[name]

        check_close(
            k_cache[page, :length],
            data[name]["k"],
            f"{name} final key cache",
        )
        check_close(
            v_cache[page, :length],
            data[name]["v"],
            f"{name} final value cache",
        )

        print(
            f"PASS Attention request {name}: "
            f"tokens={length}, page={page}",
            flush=True,
        )

    for page in (0, 2, 4):
        check_close(
            k_cache[page],
            torch.zeros_like(k_cache[page]),
            f"unused key page {page}",
        )
        check_close(
            v_cache[page],
            torch.zeros_like(v_cache[page]),
            f"unused value page {page}",
        )

    print(
        "PASS: paged MHA KV isolation, "
        "independent reference, gated output"
    )


if __name__ == "__main__":
    main()
