"""Numerical checks for the NPU add + Gemma RMSNorm model path."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm


@pytest.mark.parametrize("tokens", [1, 7])
@pytest.mark.parametrize("hidden_size", [128, 1024])
def test_add_norm_matches_fp32_reference(tokens: int, hidden_size: int) -> None:
    torch.manual_seed(11)
    norm = GemmaRMSNorm(hidden_size).to(device="npu", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden_size, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    total = x.float() + residual.float()
    expected = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)

    output, output_fp8, output_fp4, updated_residual = norm._add_norm(
        x,
        residual,
        fp8_scale=None,
        fp4_scale=None,
    )

    assert output_fp8 is None
    assert output_fp4 is None
    torch.testing.assert_close(output.float(), expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(updated_residual.float(), total, atol=2e-2, rtol=2e-2)
