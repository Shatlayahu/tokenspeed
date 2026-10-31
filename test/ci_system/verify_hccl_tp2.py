import os

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == 2, f"Expected 2 ranks, got {world_size}"

    torch.npu.set_device(local_rank)
    mapping = Mapping(rank=rank, world_size=world_size)

    try:
        pg_manager.init_distributed(
            mapping,
            distributed_init_method="env://",
            backend="hccl",
            timeout=120,
            device_id=None,
        )

        pg_manager.init_process_group(
            mapping.attn.tp_group,
            backend="hccl",
        )

        backend = HcclBackend()
        x = torch.tensor(
            [float(rank + 1)],
            dtype=torch.float32,
            device=f"npu:{local_rank}",
        )

        backend.all_reduce(x, mapping.attn.tp_group)
        torch.npu.synchronize()

        result = x.item()
        print(
            f"rank={rank}, device=npu:{local_rank}, "
            f"result={result}",
            flush=True,
        )
        assert result == 3.0, f"Expected 3.0, got {result}"
        print(f"PASS: rank={rank}", flush=True)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
