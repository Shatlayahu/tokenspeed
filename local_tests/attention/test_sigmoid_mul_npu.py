import torch
import torch_npu

from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

DEVICE = "npu:0"


def run_case(dtype, num_tokens, num_heads, head_dim, gate_layout):
    torch.manual_seed(42)

    hidden_dim = num_heads * head_dim

    # Attention 的输出必须是二维 contiguous Tensor
    x = torch.randn(
        num_tokens, hidden_dim,
        device=DEVICE, dtype=dtype
    ).contiguous()

    if gate_layout == "2d":
        gate = torch.randn(
            num_tokens, hidden_dim,
            device=DEVICE, dtype=dtype
        )
    else:
        # 模拟 Qwen3.5 的 q_gate 打包布局
        packed = torch.randn(
            num_tokens, num_heads * 2 * head_dim,
            device=DEVICE, dtype=dtype
        )

        heads = packed.view(
            num_tokens, num_heads, 2, head_dim
        )
        _, gate = heads.unbind(dim=2)

        assert gate.shape == (
            num_tokens, num_heads, head_dim
        )
        assert gate.stride(-1) == 1

    # 保存原始输入，避免被原地修改
    original_x = x.clone()
    original_ptr = x.data_ptr()

    # 独立 PyTorch 参考计算
    # FP32 中计算 Sigmoid，再转换到目标 dtype
    expected = (
        original_x.float()
        * torch.sigmoid(gate.reshape(num_tokens, -1).float())
    ).to(dtype)

    # 执行 Ascend Triton Kernel
    returned = sigmoid_mul(x, gate)

    torch.npu.synchronize()

    # 检查原地修改语义
    assert x.data_ptr() == original_ptr
    assert returned.data_ptr() == original_ptr

    actual = x.float().cpu()
    reference = expected.float().cpu()

    error = (actual - reference).abs().max().item()

    atol = 0.02 if dtype == torch.bfloat16 else 1e-4
    rtol = 0.02 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(
        actual,
        reference,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"PASSED | dtype={dtype}, "
        f"tokens={num_tokens}, "
        f"heads={num_heads}, head_dim={head_dim}, "
        f"gate={gate_layout}, "
        f"gate_stride={gate.stride()}, "
        f"max_error={error:.6f}",
        flush=True
    )


if __name__ == "__main__":
    count = 0

    for dtype in [torch.float32, torch.bfloat16]:
        for num_tokens in [1, 7]:
            for gate_layout in ["2d", "3d"]:
                run_case(
                    dtype=dtype,
                    num_tokens=num_tokens,
                    num_heads=8,
                    head_dim=256,
                    gate_layout=gate_layout,
                )
                count += 1

    print(f"\nALL {count} SIGMOID_MUL CASES PASSED")
