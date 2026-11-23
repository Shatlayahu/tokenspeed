"""Ascend integration tests for the compact Host cache executor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest
from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest


def _replicated_contract(layout, *, num_lcm_blocks=None):
    """An arena contract for ``layout`` with every group replicated (shard 1)."""
    if num_lcm_blocks is None:
        num_lcm_blocks = layout.num_lcm_blocks
    specs = tuple(
        SimpleNamespace(group_id=group.group_id, shard_count=1)
        for group in layout.groups
    )
    return SimpleNamespace(
        group_specs=specs,
        virtual_block_counts={
            group.group_id: 1 + num_lcm_blocks * group.cache_blocks_per_lcm_block
            for group in layout.groups
        },
        num_lcm_blocks=num_lcm_blocks,
        token_capacity=num_lcm_blocks,
    )


def _synthetic_arena(layout):
    contract = _replicated_contract(layout)
    return SimpleNamespace(
        cache_group_specs=contract.group_specs, runtime_contract=contract
    )


class _SyntheticPool:
    def __init__(self, layout, arena=None):
        self._layout = layout
        if arena is None:
            arena = _synthetic_arena(layout)
        self.arena = arena

    def cache_transfer_layout(self):
        return self._layout

    def register_layerwise_load_tracker(self, tracker):
        self.load_tracker = tracker


NO_POOL = RetractionPoolRequest(
    host_gb=0.0, ratio=0.0, max_retracted_requests=0, tail_lcm_blocks_per_request=0
)


class NPUCompactLayoutRoundTripTest(unittest.TestCase):
    """Real Ascend NPU round-trip tests for the Host cache executor."""

    def setUp(self):
        try:
            import torch
            import torch_npu

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs NPU runtime dependencies: {exc}")

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("requires an Ascend NPU")

        torch.npu.set_device(0)

        self.torch = torch
        self.executor_module = executor_module
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def test_npu_executor_round_trip_restores_compact_layout_byte_exactly(self):
        torch = self.torch

        first = torch.full(
            (128,), 0xCC, dtype=torch.uint8, device="npu:0"
        )
        second = torch.full(
            (128,), 0xCC, dtype=torch.uint8, device="npu:0"
        )

        layout = self.CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=2,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                    fields=(
                        self.CacheField("layer.1.state", 0, 64, 10, 5),
                    ),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        pool = _SyntheticPool(layout)

        with patch.object(
            self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0
        ):
            executor = self.executor_module.HostCacheExecutor(
                pool,
                draft_pool=None,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
            )

        self.addCleanup(executor.shutdown)

        full_k_one = torch.tensor(
            [0x11, 0x12, 0x13, 0x14], dtype=torch.uint8
        )
        full_v_one = torch.tensor(
            [0x21, 0x22, 0x23, 0x24, 0x25, 0x26],
            dtype=torch.uint8,
        )
        full_k_four = torch.tensor(
            [0x41, 0x42, 0x43, 0x44], dtype=torch.uint8
        )
        full_v_four = torch.tensor(
            [0x51, 0x52, 0x53, 0x54, 0x55, 0x56],
            dtype=torch.uint8,
        )
        state_three = torch.tensor(
            [0x71, 0x72, 0x73, 0x74, 0x75],
            dtype=torch.uint8,
        )

        first[16:20].copy_(full_k_one)
        second[28:34].copy_(full_v_one)
        first[40:44].copy_(full_k_four)
        second[64:70].copy_(full_v_four)
        first[94:99].copy_(state_three)

        torch.npu.synchronize()

        # D2H: NPU -> pinned Host.
        executor._start_writing(
            [7],
            [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
            backup_pages=[],
            lane=executor._pinned_write_lane,
            prerequisite_stream=torch.npu.current_stream(),
        )

        executor.write_stream.synchronize()

        write_results = executor.poll_results()
        self.assertEqual(
            [int(event.op_id) for event in write_results], [7]
        )

        # Destroy original Device data.
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.npu.synchronize()

        # H2D: pinned Host -> NPU.
        load_index = executor._start_loading(
            [9],
            [(0, 2, 1), (0, 5, 4), (1, 4, 3)],
            prerequisite_stream=torch.npu.current_stream(),
        )

        self.assertIsNotNone(load_index)

        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)

        torch.npu.synchronize()

        load_results = executor.poll_results()
        self.assertEqual(
            [int(event.op_id) for event in load_results], [9]
        )

        expected_first = torch.full(
            (128,), 0xEE, dtype=torch.uint8
        )
        expected_second = torch.full(
            (128,), 0xEE, dtype=torch.uint8
        )

        expected_first[24:28].copy_(full_k_one)
        expected_second[40:46].copy_(full_v_one)
        expected_first[48:52].copy_(full_k_four)
        expected_second[76:82].copy_(full_v_four)
        expected_first[104:109].copy_(state_three)

        self.assertTrue(torch.equal(first.cpu(), expected_first))
        self.assertTrue(torch.equal(second.cpu(), expected_second))

    def test_npu_writeback_ack_waits_for_dma_completion(self):
        torch = self.torch

        first = torch.full((128,), 0xCC, dtype=torch.uint8, device="npu:0")
        second = torch.full((128,), 0xCC, dtype=torch.uint8, device="npu:0")

        layout = self.CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=2,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                    fields=(
                        self.CacheField("layer.1.state", 0, 64, 10, 5),
                    ),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        pool = _SyntheticPool(layout)

        with patch.object(
            self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0
        ):
            executor = self.executor_module.HostCacheExecutor(
                pool,
                draft_pool=None,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
            )

        self.addCleanup(executor.shutdown)

        first[16:20] = torch.tensor(
            [0x11, 0x12, 0x13, 0x14], dtype=torch.uint8
        )
        second[28:34] = torch.tensor(
            [0x21, 0x22, 0x23, 0x24, 0x25, 0x26],
            dtype=torch.uint8,
        )
        first[40:44] = torch.tensor(
            [0x41, 0x42, 0x43, 0x44], dtype=torch.uint8
        )
        second[64:70] = torch.tensor(
            [0x51, 0x52, 0x53, 0x54, 0x55, 0x56],
            dtype=torch.uint8,
        )
        first[94:99] = torch.tensor(
            [0x71, 0x72, 0x73, 0x74, 0x75],
            dtype=torch.uint8,
        )

        torch.npu.synchronize()

        prerequisite_stream = torch.npu.Stream()
        scratch = torch.zeros(
            (4096, 4096), dtype=torch.float32, device="npu:0"
        )
        prerequisite_done = torch.npu.Event()

        with torch.npu.stream(prerequisite_stream):
            for _ in range(2048):
                scratch.add_(1.0)
            prerequisite_done.record(prerequisite_stream)

        self.assertFalse(
            prerequisite_done.query(),
            "Prerequisite workload finished before WriteBack submission.",
        )

        executor._start_writing(
            [11],
            [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
            backup_pages=[],
            lane=executor._pinned_write_lane,
            prerequisite_stream=prerequisite_stream,
        )

        early_results = executor.poll_results()

        self.assertFalse(
            prerequisite_done.query(),
            "Prerequisite completed during submission or polling; "
            "the test did not observe a pending write-back.",
        )

        self.assertEqual(
            early_results,
            [],
            "WriteBack ACK arrived before prerequisite/D2H completion.",
        )

        prerequisite_stream.synchronize()
        self.assertTrue(prerequisite_done.query())

        executor.write_stream.synchronize()

        write_results = executor.poll_results()

        self.assertEqual(len(write_results), 1)
        self.assertEqual(
            type(write_results[0]).__name__,
            "WriteBackDoneEvent",
        )
        self.assertEqual(int(write_results[0].op_id), 11)

        self.assertEqual(executor.poll_results(), [])

    def test_npu_writeback_ack_releases_real_scheduler_blocks(self):
        import tokenspeed_scheduler as ts

        from tokenspeed.runtime.engine.scheduler_utils import (
            advance_scheduler,
            cache_ops_from_plan,
            make_config,
            make_extend_result_event,
        )

        torch = self.torch

        # A. Construct the real Scheduler with the same shared LCM pool.
        groups = [
            ts.CacheGroupConfig(
                group_id="full",
                block_granularity=2,
                total_pages=64,
                attn_kind=ts.AttnKind.Full,
            ),
            ts.CacheGroupConfig(
                group_id="swa",
                block_granularity=2,
                total_pages=64,
                attn_kind=ts.AttnKind.SlidingWindow,
                sliding_window_tokens=4,
            ),
        ]

        scheduler = ts.Scheduler(
            make_config(
                num_device_pages=64,
                max_scheduled_tokens=64,
                max_batch_size=8,
                prefix_granularity=2,
                num_host_pages=7,
                disable_l2_cache=False,
                enable_l3_storage=False,
                role="fused",
                num_snapshot_pages=1,
                max_retracted_requests=0,
                l3_prefetch_min_pages=0,
                disable_prefix_cache=True,
                cache_groups=groups,
            )
        )

        def available():
            return {
                group: scheduler.cache_group_available_pages(group)
                for group in ("full", "swa")
            }

        free_initial = available()

        device_capacity = scheduler.empty_lcm_blocks()
        host_capacity = scheduler.host_pool_free_blocks()

        def log_capacity(stage):
            device_empty = scheduler.empty_lcm_blocks()
            host_free = scheduler.host_pool_free_blocks()
            print(f"[cache-capacity] {stage} device_used={device_capacity - device_empty} device_empty={device_empty} device_available={scheduler.available_lcm_blocks()} host_used={host_capacity - host_free} host_free={host_free} host_cached={scheduler.host_pool_cached_blocks()}", flush=True)

        log_capacity("initial")

        self.assertEqual(
            free_initial["full"],
            free_initial["swa"],
            "Both group queries should expose the shared LCM pool count.",
        )

        # B. Submit r1 and complete prefill.
        spec = ts.RequestSpec()
        spec.request_id = "r1"
        spec.tokens = list(range(1, 9))
        spec.max_new_tokens = 1

        scheduler.submit_requests([spec])
        scheduler.next_execution_plan()

        advance_scheduler(
            scheduler,
            [make_extend_result_event("r1", [9001])],
        )

        # C. Capture the actual six-page WriteBack.
        finalize = scheduler.next_execution_plan()
        ops = cache_ops_from_plan(finalize)

        self.assertEqual(len(ops), 1)

        wb = ops[0]
        op_id = int(wb.op_ids[0])

        group_ids = list(wb.group_ids[0])
        src_pages = list(wb.src_pages[0])
        dst_pages = list(wb.dst_pages[0])

        self.assertEqual(group_ids, [0, 0, 0, 0, 1, 1])
        self.assertEqual(src_pages, [1, 2, 3, 4, 8, 9])
        self.assertEqual(dst_pages, [1, 2, 3, 4, 5, 6])
        self.assertEqual(list(wb.source_pinned), [True])
        self.assertEqual(len(src_pages), 6)

        # Only submit a decode completion if r1 was actually scheduled.
        scheduled_ids = [
            request_id
            for forward in finalize.forward
            for request_id in forward.request_ids
        ]

        self.assertIn(
            "r1",
            scheduled_ids,
            "Expected r1 to have a scheduled forward in the finalize plan.",
        )

        advance_scheduler(
            scheduler,
            [make_extend_result_event("r1", [9002])],
        )

        finish = ts.ForwardEvent.Finish()
        finish.request_id = "r1"
        advance_scheduler(scheduler, [finish])

        finish_plan = scheduler.next_execution_plan()
        finish_ops = cache_ops_from_plan(finish_plan)

        self.assertFalse(
            any(type(op).__name__ == "WriteBackOp" for op in finish_ops),
            "Finish must not emit another WriteBack.",
        )

        free_before_ack = available()
        print(f"Initial: {free_initial}, before ACK: {free_before_ack}")

        # D. Construct matching Ascend transfer layout.
        first = torch.full(
            (256,), 0xCC, dtype=torch.uint8, device="npu:0"
        )
        second = torch.full(
            (256,), 0xCC, dtype=torch.uint8, device="npu:0"
        )

        layout = self.CacheTransferLayout(
            num_lcm_blocks=64,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=1,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="swa",
                    cache_blocks_per_lcm_block=1,
                    fields=(
                        self.CacheField("layer.1.state", 0, 64, 10, 5),
                    ),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        pool = _SyntheticPool(layout)

        with patch.object(
            self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0
        ):
            executor = self.executor_module.HostCacheExecutor(
                pool,
                draft_pool=None,
                l2_tier=True,
                host_ratio=7 / 64,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
            )

        self.addCleanup(executor.shutdown)

        storage = executor.host_storage
        self.assertIsNotNone(storage)
        self.assertEqual(storage.num_host_lcm_blocks, 7)

        # E. Fill the actual Device source blocks with identifiable bytes.
        expected = {}

        for group_id, src_id, dst_id in zip(
            group_ids, src_pages, dst_pages
        ):
            if group_id == 0:
                k_value = 0x10 + src_id
                v_value = 0x20 + src_id

                first_offset = 8 + src_id * 8
                second_offset = 16 + src_id * 12

                first[first_offset:first_offset + 4].fill_(k_value)
                second[second_offset:second_offset + 6].fill_(v_value)

                expected[(group_id, dst_id, 0)] = torch.full(
                    (4,), k_value, dtype=torch.uint8
                )
                expected[(group_id, dst_id, 1)] = torch.full(
                    (6,), v_value, dtype=torch.uint8
                )
            else:
                state_value = 0x70 + src_id
                state_offset = 64 + src_id * 10

                first[state_offset:state_offset + 5].fill_(state_value)

                expected[(group_id, dst_id, 0)] = torch.full(
                    (5,), state_value, dtype=torch.uint8
                )

        torch.npu.synchronize()

        transfers = [
            (int(group), int(src), int(dst))
            for group, src, dst in zip(
                group_ids, src_pages, dst_pages
            )
        ]

        # F. Create a real pending prerequisite on Ascend.
        prerequisite_stream = torch.npu.Stream()

        scratch = torch.zeros(
            (4096, 4096),
            dtype=torch.float32,
            device="npu:0",
        )

        prerequisite_done = torch.npu.Event()

        with torch.npu.stream(prerequisite_stream):
            for _ in range(2048):
                scratch.add_(1.0)
            prerequisite_done.record(prerequisite_stream)

        self.assertFalse(
            prerequisite_done.query(),
            "Prerequisite finished before submission; retry with a larger workload.",
        )

        executor._start_writing(
            [op_id],
            transfers,
            backup_pages=[],
            lane=executor._pinned_write_lane,
            prerequisite_stream=prerequisite_stream,
        )

        early_results = executor.poll_results()

        self.assertFalse(
            prerequisite_done.query(),
            "Pending prerequisite window disappeared during polling.",
        )

        self.assertEqual(
            early_results,
            [],
            "WriteBack ACK arrived before prerequisite/D2H completion.",
        )

        for group in ("full", "swa"):
            self.assertEqual(
                scheduler.cache_group_available_pages(group),
                free_before_ack[group],
                f"{group}: Scheduler blocks must remain pinned during DMA.",
            )

        # G. Complete D2H, but do NOT ACK Scheduler yet.
        prerequisite_stream.synchronize()
        self.assertTrue(prerequisite_done.query())

        executor.write_stream.synchronize()

        # H. Verify every Host destination field byte-for-byte.
        for (group_id, dst_id, field_index), expected_bytes in expected.items():
            base = storage.host_block_offset(group_id, dst_id)

            field_offset = storage.host_field_offsets[group_id][field_index]

            start = base + field_offset
            end = start + expected_bytes.numel()

            actual = storage.host_buffer[start:end]

            self.assertTrue(
                torch.equal(actual.cpu(), expected_bytes),
                (
                    f"Host data mismatch: group={group_id}, "
                    f"dst={dst_id}, field_index={field_index}"
                ),
            )

        # DMA completion alone must not release Scheduler blocks.
        for group in ("full", "swa"):
            self.assertEqual(
                scheduler.cache_group_available_pages(group),
                free_before_ack[group],
            )

        # I. Obtain the actual runtime ACK and deliver it to Scheduler.
        write_results = executor.poll_results()

        self.assertEqual(len(write_results), 1)
        self.assertEqual(
            type(write_results[0]).__name__,
            "WriteBackDoneEvent",
        )
        self.assertEqual(int(write_results[0].op_id), op_id)

        log_capacity("before_ack")
        lcm_before = scheduler.available_lcm_blocks()
        advance_scheduler(scheduler, write_results)
        lcm_after = scheduler.available_lcm_blocks()
        log_capacity("after_ack")
        print(f"Available LCM blocks: {lcm_before} -> {lcm_after}")

        free_after_ack = available()
        released = {
            group: free_after_ack[group] - free_before_ack[group]
            for group in ("full", "swa")
        }
        print(f"After ACK: {free_after_ack}, available-page deltas: {released}")
        self.assertEqual(
            lcm_after - lcm_before, 6,
            f"Expected six pages to become available after WriteBackDone({op_id})",
        )

        self.assertEqual(executor.poll_results(), [])
