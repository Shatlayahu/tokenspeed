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

    assert world_size == 2

    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"

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
        group = mapping.attn.tp_group

        # Test 1: All Gather
        x = torch.tensor(
            [float(rank + 1)],
            dtype=torch.float32,
            device=device,
        )

        gathered = torch.empty(
            world_size,
            dtype=torch.float32,
            device=device,
        )

        backend.all_gather_single(gathered, x, group)
        torch.npu.synchronize()

        expected_gathered = torch.tensor(
            [1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )

        torch.testing.assert_close(gathered, expected_gathered)

        print(
            f"PASS: All Gather rank={rank}, "
            f"result={gathered.cpu().tolist()}",
            flush=True,
        )

        # Test 2: Reduce Scatter
        rs_input = torch.tensor(
            [float(rank + 1), float((rank + 1) * 10)],
            dtype=torch.float32,
            device=device,
        )

        rs_output = backend.reduce_scatter(rs_input, group)
        torch.npu.synchronize()

        expected = 3.0 if rank == 0 else 30.0

        assert abs(rs_output.item() - expected) < 1e-5

        print(
            f"PASS: Reduce Scatter rank={rank}, "
            f"result={rs_output.item()}",
            flush=True,
        )

        # Test 3: Token Reduce Scatter
        token_input = torch.tensor(
            [
                [float(rank + 1)],
                [float((rank + 1) * 10)],
            ],
            dtype=torch.float32,
            device=device,
        )

        token_output = backend.token_reduce_scatter(
            token_input,
            group,
            scattered_num_tokens=[1, 1],
        )

        torch.npu.synchronize()

        expected = 3.0 if rank == 0 else 30.0

        assert abs(token_output.item() - expected) < 1e-5

        print(
            f"PASS: Token Reduce Scatter rank={rank}, "
            f"result={token_output.item()}",
            flush=True,
        )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
