"""Numerical checks for Qwen3.5 gated RMSNorm on NPU."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.attention.linear.layernorm_gated import RMSNorm


@pytest.mark.parametrize("tokens", [1, 7])
def test_gated_rmsnorm_matches_fp32_reference(tokens: int) -> None:
    torch.manual_seed(7)
    norm = RMSNorm(
        128,
        eps=1e-6,
        group_size=128,
        norm_before_gate=True,
        device="npu",
        dtype=torch.bfloat16,
    )
    x = torch.randn(tokens, 128, device="npu", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    output = norm(x, gate)

    x_float = x.float()
    gate_float = gate.float()
    expected = (
        x_float
        * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-6)
        * (gate_float * torch.sigmoid(gate_float))
    )
    torch.testing.assert_close(output.float(), expected, atol=3e-2, rtol=3e-2)
