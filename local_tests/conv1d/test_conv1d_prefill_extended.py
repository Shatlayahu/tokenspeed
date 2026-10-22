import torch
import torch_npu
import torch.nn.functional as F

from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
)

DEVICE = "npu:0"
WIDTH = 4
torch.manual_seed(42)


def run_case(lengths, channels, dtype, history_mode):
    batch = len(lengths)
    total_tokens = sum(lengths)

    # 1. 输入与卷积权重
    x = torch.randn(
        total_tokens, channels, device=DEVICE, dtype=dtype
    ).T

    weight = torch.randn(
        channels, WIDTH, device=DEVICE, dtype=dtype
    ) * 0.1

    # 多个缓存槽位，0 号不使用
    state = torch.randn(
        batch + 2, channels, WIDTH - 1,
        device=DEVICE, dtype=dtype
    )

    original_state = state.clone()

    # 每个请求对应不同的缓存位置
    cache_indices = torch.arange(
        1, batch + 1, device=DEVICE, dtype=torch.int32
    )

    if history_mode == "none":
        flags = [False] * batch
    elif history_mode == "all":
        flags = [True] * batch
    else:
        flags = [i % 2 == 1 for i in range(batch)]

    has_initial_state = torch.tensor(
        flags, device=DEVICE, dtype=torch.bool
    )

    # 2. 构造 Prefill Metadata
    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    boundaries = torch.tensor(
        offsets, device=DEVICE, dtype=torch.int32
    )

    metadata = build_causal_conv1d_prefill_metadata(
        boundaries,
        lengths_cpu,
        CAUSAL_CONV1D_BLOCK_M,
    )

    # 3. 独立 CPU PyTorch 参考实现
    x_cpu = x.float().cpu()
    weight_cpu = weight.float().cpu()
    state_cpu = original_state.float().cpu()

    expected_output = torch.zeros_like(x_cpu)
    expected_state = original_state.cpu().clone()

    start = 0

    for b, length in enumerate(lengths):
        slot = b + 1

        if flags[b]:
            history = state_cpu[slot].clone()
        else:
            history = torch.zeros(
                channels, WIDTH - 1, dtype=torch.float32
            )

        for t in range(length):
            current = x_cpu[:, start + t]

            window = torch.cat(
                [history, current.unsqueeze(-1)], dim=-1
            )

            value = (window * weight_cpu).sum(dim=-1)
            value = F.silu(value)

            expected_output[:, start + t] = value

            # 更新最近 3 个卷积输入
            history = window[:, 1:].clone()

        expected_state[slot] = history.to(dtype)
        start += length

    # 4. 执行 Triton-Ascend Prefill
    actual_output = causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=None,
        conv_states=state,
        query_start_loc=boundaries,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=-1,
        validate_data=False,
        prefill_metadata=metadata,
    )

    torch.npu.synchronize()

    # 5. 数值比较
    actual_cpu = actual_output.float().cpu()
    expected_cpu = expected_output.to(dtype).float()

    error = (actual_cpu - expected_cpu).abs().max().item()

    atol = 0.05 if dtype == torch.bfloat16 else 1e-4
    rtol = 0.05 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(
        actual_cpu,
        expected_cpu,
        atol=atol,
        rtol=rtol,
    )

    # 6. Conv State 必须一致
    torch.testing.assert_close(
        state.cpu(),
        expected_state,
        atol=0,
        rtol=0,
    )

    print(
        f"PASSED | lengths={lengths}, "
        f"channels={channels}, dtype={dtype}, "
        f"history={history_mode}, "
        f"max_error={error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    cases = [
        ([1], 16),
        ([3], 16),
        ([5], 128),
        ([7, 8, 9], 16),
        ([1, 17, 3], 128),
        ([33, 2], 257),
    ]

    count = 0

    for dtype in [torch.float32, torch.bfloat16]:
        for lengths, channels in cases:
            for history_mode in ["none", "mixed", "all"]:
                run_case(
                    lengths,
                    channels,
                    dtype,
                    history_mode,
                )
                count += 1

    print(f"\nALL {count} PREFILL CASES PASSED")
