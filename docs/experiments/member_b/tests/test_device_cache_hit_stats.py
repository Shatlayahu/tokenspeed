"""Deterministic Host L2 capacity, eviction and cache-miss test.

Uses the real C++ Scheduler and simulated transfer ACKs.
Does not require NPU or modify the serving process.
"""

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.engine.scheduler_utils import (
    make_config,
    advance_scheduler,
    make_extend_result_event,
    cache_ops_from_plan,
)


def test_host_capacity_eviction_and_miss():
    PAGE = 128
    HOST_BLOCKS = 8

    groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=PAGE,
            total_pages=64,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=64,
            attn_kind=ts.AttnKind.State,
        ),
    ]

    scheduler = ts.Scheduler(make_config(
        num_device_pages=64,
        max_scheduled_tokens=1024,
        max_batch_size=1,
        prefix_granularity=PAGE,
        num_host_pages=HOST_BLOCKS + 1,
        disable_l2_cache=False,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        max_retracted_requests=0,
        l3_prefetch_min_pages=0,
        disable_prefix_cache=False,
        cache_groups=groups,
        prefix_replay_tokens=0,
    ))

    def stats(label):
        cached = scheduler.host_pool_cached_blocks()
        free = scheduler.host_pool_free_blocks()

        print(
            f"{label}: cached={cached}, free={free}, "
            f"capacity={HOST_BLOCKS}",
            flush=True,
        )

        assert cached <= HOST_BLOCKS
        assert free <= HOST_BLOCKS
        assert cached + free <= HOST_BLOCKS

        return cached

    def ack_writebacks(plan):
        events = []
        for op in cache_ops_from_plan(plan):
            if isinstance(op, ts.Cache.WriteBackOp):
                for op_id in op.op_ids:
                    event = ts.Cache.WriteBackDoneEvent()
                    event.op_id = int(op_id)
                    events.append(event)

        if events:
            advance_scheduler(scheduler, events)

        return len(events)

    def submit(request_id, tokens):
        req = ts.RequestSpec()
        req.request_id = request_id
        req.tokens = tokens
        req.max_new_tokens = 2
        scheduler.submit_requests([req])

    def cache_request(request_id, tokens, token_base):
        submit(request_id, tokens)

        prefill = scheduler.next_execution_plan()
        batches = [
            op for op in prefill.forward if op.request_ids
        ]
        assert len(batches) == 1
        assert list(batches[0].input_lengths) == [len(tokens)]

        advance_scheduler(
            scheduler,
            [make_extend_result_event(request_id, [token_base])],
        )

        decode = scheduler.next_execution_plan()
        advance_scheduler(
            scheduler,
            [make_extend_result_event(request_id, [token_base + 1])],
        )

        finish = ts.ForwardEvent.Finish()
        finish.request_id = request_id
        advance_scheduler(scheduler, [finish])

        ack_writebacks(decode)

        finish_plan = scheduler.next_execution_plan()
        ack_writebacks(finish_plan)

        assert scheduler.can_clear_cache()

        return stats(f"after {request_id}")


    base = list(range(1, 422))
    suffix = list(range(50001, 50011))

    # Cold request: Device L1 Miss
    cache_request("A", base, 9001)
    assert scheduler.device_cache_hits() == 0
    assert scheduler.device_cache_misses() == 1

    # Reuse A without clearing Device L1
    submit("A_hit", base + suffix)
    plan = scheduler.next_execution_plan()

    batches = [op for op in plan.forward if op.request_ids]
    assert len(batches) == 1

    batch = batches[0]
    prefix = int(list(batch.extend_prefix_lens)[0])
    compute = int(list(batch.input_lengths)[0])

    print("Device cached prefix:", prefix)
    print("Tokens to compute:", compute)
    print("Device hits:", scheduler.device_cache_hits())
    print("Device misses:", scheduler.device_cache_misses())
    print("Device evictions:", scheduler.device_cache_evictions())

    assert prefix == 421
    assert compute == 10
    assert scheduler.device_cache_hits() == 1
    assert scheduler.device_cache_misses() == 1
    assert scheduler.device_cache_evictions() == 0

    print("PASS: Device L1 Hit/Miss counters")


if __name__ == "__main__":
    test_host_capacity_eviction_and_miss()
