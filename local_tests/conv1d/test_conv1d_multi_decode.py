import torch
import torch_npu
import torch.nn.functional as F

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_update,
)


def run_case(batch, dim, dtype, activation):
    torch.manual_seed(42)

    device = "npu:0"
    width = 4
    steps = 10

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    # Kernel 使用的真实状态
    state = torch.randn(
        batch, dim, width - 1, device=device, dtype=dtype
    )

    # 参考实现使用独立状态
    ref_state = state.clone()

    for step in range(steps):
        x = torch.randn(batch, dim, device=device, dtype=dtype)

        # 保存原始输入，防止 Kernel 原地覆盖 x
        original_x = x.clone()

        # PyTorch 参考计算：使用 FP32 累加
        window = torch.cat(
            [ref_state, original_x.unsqueeze(-1)], dim=-1
        )

        expected = (
            window.float() * weight.float().unsqueeze(0)
        ).sum(dim=-1) + bias.float().unsqueeze(0)

        if activation == "silu":
            expected = F.silu(expected)

        expected = expected.to(dtype)
        ref_state = window[:, :, 1:].clone()

        # Ascend Triton Kernel
        actual = causal_conv1d_update(
            x=x,
            conv_state=state,
            weight=weight,
            bias=bias,
            activation=activation,
            parent_indices=None,
        )

        torch.npu.synchronize()

        # BF16 的允许误差比 FP32 大
        atol = 0.1 if dtype == torch.bfloat16 else 1e-4
        rtol = 0.1 if dtype == torch.bfloat16 else 1e-4

        torch.testing.assert_close(
            actual.float().cpu(),
            expected.float().cpu(),
            atol=atol,
            rtol=rtol,
            msg=f"Output mismatch at step {step}",
        )

        torch.testing.assert_close(
            state.cpu(),
            ref_state.cpu(),
            atol=0,
            rtol=0,
            msg=f"State mismatch at step {step}",
        )

    print(
        f"PASSED: batch={batch}, dim={dim}, "
        f"dtype={dtype}, activation={activation}, steps={steps}"
    )


if __name__ == "__main__":
    for dtype in [torch.float32, torch.bfloat16]:
        for batch, dim in [(1, 8), (2, 128), (4, 256)]:
            run_case(batch, dim, dtype, activation="silu")
