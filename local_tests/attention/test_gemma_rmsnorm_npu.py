
import torch
import torch_npu

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

DEVICE = "npu:0"


def run_case(dtype, tokens, dim):
    torch.manual_seed(42)

    norm = GemmaRMSNorm(dim, eps=1e-6).to(
        device=DEVICE, dtype=dtype
    )

    # 使用模型原有的权重加载函数，确保 gemma_weight 同步更新
    with torch.no_grad():
        loaded_weight = (
            torch.randn(dim, device=DEVICE, dtype=dtype) * 0.1
        )
        norm._weight_loader(norm.weight, loaded_weight)

    # 验证预计算的 Gemma 权重是正确的
    weight_error = (
        norm.gemma_weight.float()
        - (norm.weight.float() + 1.0)
    ).abs().max().item()

    print(
        f"gemma_weight max difference: {weight_error:.6f}",
        flush=True,
    )

    x = torch.randn(tokens, dim, device=DEVICE, dtype=dtype)

    # PyTorch FP32 参考计算（保留原有流程）
    xf = x.float()
    wf = norm.weight.float()

    expected = (
        xf
        * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        * (1.0 + wf)
    ).to(dtype)

    actual = norm(x)
    torch.npu.synchronize()

    atol = rtol = 0.02 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=atol,
        rtol=rtol,
    )

    error = (
        actual.float().cpu() - expected.float().cpu()
    ).abs().max().item()

    print(
        f"PASSED | dtype={dtype}, tokens={tokens}, "
        f"dim={dim}, max_error={error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    for dtype in [torch.float32, torch.bfloat16]:
        for tokens in [1, 7]:
            for dim in [128, 256]:
                run_case(dtype, tokens, dim)

    print("\nALL 8 GEMMA RMSNORM CASES PASSED")
