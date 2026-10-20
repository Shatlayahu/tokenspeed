import torch
import torch_npu

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_update,
)

torch.manual_seed(42)

device = "npu:0"
batch = 1
dim = 8
width = 4

# 1. 构造输入
x = torch.randn(batch, dim, device=device, dtype=torch.float32)
weight = torch.randn(dim, width, device=device, dtype=torch.float32)

# 历史三个 token 的卷积输入
conv_state = torch.randn(
    batch, dim, width - 1,
    device=device, dtype=torch.float32
)

# 保存原始值，避免 in-place 修改影响参考计算
x_original = x.clone()
state_original = conv_state.clone()

# 2. PyTorch 参考计算
window = torch.cat(
    [state_original, x_original.unsqueeze(-1)],
    dim=-1,
)

expected_output = (
    window * weight.unsqueeze(0)
).sum(dim=-1)

expected_state = window[:, :, 1:].clone()

# 3. 调用 TokenSpeed Triton Kernel
actual_output = causal_conv1d_update(
    x=x,
    conv_state=conv_state,
    weight=weight,
    bias=None,
    activation=None,
    parent_indices=None,
)

torch.npu.synchronize()

# 4. 比较输出
torch.testing.assert_close(
    actual_output.cpu(),
    expected_output.cpu(),
    rtol=1e-4,
    atol=1e-4,
)

# 5. 比较状态
torch.testing.assert_close(
    conv_state.cpu(),
    expected_state.cpu(),
    rtol=0,
    atol=0,
)

print("Conv1D Decode output: PASSED")
print("Conv1D state update: PASSED")
