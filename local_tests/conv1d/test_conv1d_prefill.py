import torch
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
)

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
)

torch.manual_seed(42)

device = "npu:0"
length = 5
channels = 16
width = 4
dtype = torch.float32

# 1. 构造输入
# TokenSpeed Prefill 需要 (channels, total_tokens)
x = torch.randn(length, channels, device=device, dtype=dtype).T
weight = torch.randn(channels, width, device=device, dtype=dtype) * 0.1

# 分配两个缓存位置：0 未使用，1 属于当前请求
conv_states = torch.zeros(2, channels, width - 1, device=device, dtype=dtype)

# 保存原始输入和状态
original_x = x.clone()
original_states = conv_states.clone()

# 2. 构造 Prefill 元数据
lengths = torch.tensor([length], dtype=torch.int32)
boundaries = torch.tensor([0, length], device=device, dtype=torch.int32)

metadata = build_causal_conv1d_prefill_metadata(
    boundaries,
    lengths,
    CAUSAL_CONV1D_BLOCK_M,
)

cache_indices = torch.tensor([1], device=device, dtype=torch.int32)
has_initial_state = torch.tensor([False], device=device)

# 3. 独立 PyTorch 参考计算
# 每个 channel 单独进行宽度为 4 的因果卷积
reference_outputs = []

for t in range(length):
    window = torch.zeros(channels, width, device=device, dtype=torch.float32)

    for k in range(width):
        source_t = t - (width - 1) + k
        if source_t >= 0:
            window[:, k] = original_x[:, source_t].float()

    value = (window * weight.float()).sum(dim=-1)
    reference_outputs.append(F.silu(value))

expected_output = torch.stack(reference_outputs, dim=1).to(dtype)

# Prefill 完成后，保存最后 width-1 个卷积输入
expected_states = original_states.clone()
expected_states[1] = original_x[:, -(width - 1) :]

# 4. 调用 TokenSpeed Prefill Kernel
actual_output = causal_conv1d_fn(
    x=x,
    weight=weight,
    bias=None,
    conv_states=conv_states,
    query_start_loc=boundaries,
    cache_indices=cache_indices,
    has_initial_state=has_initial_state,
    activation="silu",
    pad_slot_id=-1,
    validate_data=False,
    prefill_metadata=metadata,
)

torch.npu.synchronize()

# 5. 检查输出
output_error = (actual_output.float() - expected_output.float()).abs().max().item()

print("Output max absolute error:", output_error)

torch.testing.assert_close(
    actual_output.cpu(),
    expected_output.cpu(),
    rtol=1e-4,
    atol=1e-4,
)
print("Conv1D Prefill output: PASSED")

# 6. 检查 Conv State
state_error = (conv_states.float() - expected_states.float()).abs().max().item()

print("State max absolute error:", state_error)

torch.testing.assert_close(
    conv_states.cpu(),
    expected_states.cpu(),
    rtol=0,
    atol=0,
)
print("Conv1D Prefill state update: PASSED")
