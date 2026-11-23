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
        scheduler.clear_l1_cache()

        return stats(f"after {request_id}")

    base = list(range(1, 422))
    suffix = list(range(50001, 50011))

    print("\n=== 1. Cache A ===")
    cached_a = cache_request("A", base, 9001)
    assert cached_a > 0

    print("\n=== 2. Fill Host with distinct prefixes ===")
    pressure_count = 0
    for i in range(1, 9):
        pressure = list(range(10000 * i, 10000 * i + 421))
        cache_request(f"P{i}", pressure, 9001 + i * 100)
        pressure_count += 1

    print("\n=== 3. Revisit A after Host pressure ===")
    submit("A_again", base + suffix)

    plan = scheduler.next_execution_plan()
    batches = [
        op for op in plan.forward if op.request_ids
    ]
    assert len(batches) == 1

    batch = batches[0]
    prefix = int(list(batch.extend_prefix_lens)[0])
    to_compute = int(list(batch.input_lengths)[0])

    loadbacks = [
        op for op in cache_ops_from_plan(plan)
        if isinstance(op, ts.Cache.LoadBackOp)
    ]

    print("A original prefix:", len(base))
    print("A restored prefix:", prefix)
    print("A tokens to compute:", to_compute)
    print("LoadBack operations:", len(loadbacks))
    print("Host pressure requests:", pressure_count)

    assert prefix + to_compute == len(base) + len(suffix)
    assert prefix < len(base), (
        "A still has its entire original prefix cached; "
        "Host eviction was not demonstrated"
    )

    if prefix == 0:
        assert not loadbacks
        print("PASS: Full cache miss and complete Prefill")
    else:
        print("PASS: Partial cache miss and recomputation")

    # Each request first-admits once; L1 is explicitly cleared
    # between requests, so all ten initial admissions miss Device L1.
    print("Device hits:", scheduler.device_cache_hits())
    print("Device misses:", scheduler.device_cache_misses())
    print("Device evictions:", scheduler.device_cache_evictions())

    assert scheduler.device_cache_hits() == 0
    assert scheduler.device_cache_misses() == 10

    # Explicit L1 clears must not inflate capacity-eviction counts.
    # This test does NOT yet establish that pressure evictions increment.
    stats("final")
    print("PASS: Host capacity bounded; old prefix evicted; miss handled")


if __name__ == "__main__":
    test_host_capacity_eviction_and_miss()
