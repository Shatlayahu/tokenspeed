import torch
import torch_npu

from tokenspeed_kernel.ops.kvcache.triton import store_kv_cache

DEVICE = "npu:0"

def run(dtype):
    torch.manual_seed(42)

    tokens = 7
    heads = 2
    head_dim = 128
    slots_count = 16

    k = torch.randn(
        tokens, heads, head_dim, device=DEVICE, dtype=dtype
    )
    v = torch.randn_like(k)

    k_cache = torch.zeros(
        slots_count, heads, head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)

    slots = torch.tensor(
        [1, 3, 5, 7, 9, 11, 13],
        device=DEVICE, dtype=torch.int32
    )

    store_kv_cache(
        k, v, k_cache, v_cache, slots, enable_pdl=False
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        k_cache[slots].cpu(), k.cpu(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        v_cache[slots].cpu(), v.cpu(), atol=0, rtol=0
    )

    # 未写入的槽位必须保持原值
    unused = [0, 2, 4, 6, 8, 10, 12, 14, 15]
    assert torch.count_nonzero(k_cache[unused]).item() == 0
    assert torch.count_nonzero(v_cache[unused]).item() == 0

    print(f"PASSED | dtype={dtype}, tokens={tokens}")


for dtype in [torch.float32, torch.bfloat16]:
    run(dtype)

print("ALL KV CACHE WRITE CASES PASSED")
