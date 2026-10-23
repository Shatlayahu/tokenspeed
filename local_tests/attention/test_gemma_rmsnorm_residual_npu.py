import torch
import torch_npu

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

DEVICE = "npu:0"
EPS = 1e-6


def assert_close(actual, expected, dtype, name):
    atol = rtol = 0.02 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=atol,
        rtol=rtol,
        msg=lambda m: f"{name}: {m}",
    )

    return (actual.float().cpu() - expected.float().cpu()).abs().max().item()


def run_residual_case(dtype, tokens, dim):
    torch.manual_seed(42)

    norm = GemmaRMSNorm(dim, eps=EPS).to(device=DEVICE, dtype=dtype)

    # 按照模型原本的权重加载流程
    with torch.no_grad():
        loaded_weight = torch.randn(dim, device=DEVICE, dtype=dtype) * 0.1
        norm._weight_loader(norm.weight, loaded_weight)

    x = torch.randn(tokens, dim, device=DEVICE, dtype=dtype)
    residual = torch.randn(tokens, dim, device=DEVICE, dtype=dtype)

    x_before = x.clone()
    residual_before = residual.clone()

    # FP32 数学参考
    summed_fp32 = x_before.float() + residual_before.float()
    variance = summed_fp32.square().mean(-1, keepdim=True)

    expected = (
        summed_fp32 * torch.rsqrt(variance + EPS) * (1.0 + norm.weight.float())
    ).to(dtype)

    # 实际调用会通过 TokenSpeed RMSNorm 接口
    # 进入 Ascend npu_add_rms_norm 路径
    actual, residual_out = norm(x, residual)
    torch.npu.synchronize()

    # 验证归一化输出
    output_error = assert_close(actual, expected, dtype, "normalized output")

    # 验证 residual 更新为 x + residual
    residual_expected = (x_before.float() + residual_before.float()).to(dtype)

    residual_error = assert_close(
        residual_out,
        residual_expected,
        dtype,
        "residual output",
    )

    # 检查返回的 residual 与原输入 residual 是否一致
    # （因为底层 NPU 实现使用 residual.copy_）
    assert (
        residual_out.data_ptr() == residual.data_ptr()
    ), "NPU residual should be updated in place"

    # 验证输入 x 未被意外修改
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)

    print(
        f"PASSED | dtype={dtype}, tokens={tokens}, "
        f"dim={dim}, output_error={output_error:.6f}, "
        f"residual_error={residual_error:.6f}",
        flush=True,
    )


def run_empty_case(dtype, dim):
    norm = GemmaRMSNorm(dim, eps=EPS).to(device=DEVICE, dtype=dtype)

    x = torch.empty((0, dim), device=DEVICE, dtype=dtype)
    residual = torch.empty_like(x)

    result = norm(x)
    assert result.shape == x.shape

    result, residual_out = norm(x, residual)
    assert result.shape == x.shape
    assert residual_out.shape == residual.shape

    print(
        f"PASSED | empty tokens | dtype={dtype}, dim={dim}",
        flush=True,
    )


if __name__ == "__main__":
    for dtype in [torch.float32, torch.bfloat16]:
        for tokens in [1, 7]:
            for dim in [128, 256]:
                run_residual_case(dtype, tokens, dim)

    for dtype in [torch.float32, torch.bfloat16]:
        run_empty_case(dtype, 128)

    print("\nALL 10 GEMMA RMSNORM RESIDUAL/EMPTY CASES PASSED")
