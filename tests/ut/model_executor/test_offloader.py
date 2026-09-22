from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import (
    AscendPrefetchOffloader,
    AscendStaticBufferPool,
    ParamInfo,
    _is_using_nz_weight,
    _ModuleOffloader,
)
from vllm_ascend.worker.model_runner_v1 import (
    _net_offloaded_device_bytes,
    _reclaim_offloaded_device_memory,
    _warm_up_tp_communicator_for_prefetch,
)


def test_create_offloader_without_config_returns_noop():
    offloader = create_offloader(None)

    assert isinstance(offloader, NoopOffloader)


def test_create_offloader_for_non_prefetch_backend_returns_noop():
    offload_config = SimpleNamespace(
        offload_backend=None,
        prefetch=None,
    )

    offloader = create_offloader(offload_config)

    assert isinstance(offloader, NoopOffloader)


def test_is_using_nz_weight_handles_invalid_npu_format(monkeypatch):
    param = SimpleNamespace(
        data=SimpleNamespace(device=SimpleNamespace(type="npu")),
    )

    monkeypatch.setattr(
        "vllm_ascend.model_executor.offloader.prefetch.torch_npu.get_npu_format",
        lambda _: object(),
        raising=False,
    )

    assert not _is_using_nz_weight(param)


def test_ascend_prefetch_formats_pool_before_binding_and_prefetch():
    events = []
    param_info = ParamInfo(
        name="weight",
        shape=(2, 2),
        stride=(2, 1),
        dtype=torch.float16,
        use_nz_buffer=True,
    )
    module_offloader = MagicMock()
    module_offloader.device = torch.device("cpu")
    module_offloader.offloaded_bytes = 8
    module_offloader.get_param_infos.return_value = [param_info]
    module_offloader.sync_cpu_storage.side_effect = lambda: events.append("sync")
    module_offloader.assign_buffer_slot.side_effect = lambda *_: events.append("bind")
    module_offloader.post_init.side_effect = lambda: events.append("post_init")
    module_offloader.start_onload_to_static.side_effect = lambda: events.append("prefetch")

    buffer_pool = SimpleNamespace(total_bytes=4)
    offloader = AscendPrefetchOffloader.__new__(AscendPrefetchOffloader)
    offloader.module_offloaders = [module_offloader]
    offloader.prefetch_step = 1
    offloader.group_size = 1
    offloader.num_in_group = 1
    offloader.mode = "cpu"
    offloader.total_offloaded_bytes = 0
    offloader.buffer_pool = None

    with (
        patch(
            "vllm_ascend.model_executor.offloader.prefetch.AscendStaticBufferPool",
            return_value=buffer_pool,
        ),
        patch(
            "vllm_ascend.model_executor.offloader.prefetch._format_static_buffers_for_nz",
            side_effect=lambda *_: events.append("format"),
        ),
    ):
        offloader.post_init()

    assert events == ["sync", "format", "bind", "post_init", "prefetch"]
    assert offloader.total_offloaded_bytes == 8


def test_ascend_static_pool_allocates_only_reachable_key_slot_pairs():
    even_layer = ParamInfo(
        name="weight",
        shape=(2, 2),
        stride=(2, 1),
        dtype=torch.float16,
    )
    odd_layer = ParamInfo(
        name="weight",
        shape=(3, 3),
        stride=(3, 1),
        dtype=torch.float16,
    )

    pool = AscendStaticBufferPool(
        param_infos_by_slot=[[even_layer, even_layer], [odd_layer]],
        slot_capacity=2,
        device=torch.device("cpu"),
    )

    assert pool.total_bytes == even_layer.num_bytes + odd_layer.num_bytes
    assert pool.get_buffer(*even_layer.key, slot_idx=0).shape == (2, 2)
    assert pool.get_buffer(*odd_layer.key, slot_idx=1).shape == (3, 3)
    assert pool._buffers[even_layer.key][1] is None
    assert pool._buffers[odd_layer.key][0] is None


def test_module_offloader_reuses_compute_ready_event_in_eager_decode():
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    offloader._buffer_pool = object()
    offloader._eager_compute_ready_event = MagicMock()
    offloader._copy_done_event = MagicMock()
    offloader.copy_stream = MagicMock()
    offloader._param_offloaders = {}

    current_stream = MagicMock()
    stream_context = MagicMock()
    stream_context.__enter__.return_value = None
    stream_context.__exit__.return_value = False
    with (
        patch(
            "vllm_ascend.model_executor.offloader.prefetch.torch.cuda.current_stream",
            return_value=current_stream,
        ),
        patch(
            "vllm_ascend.model_executor.offloader.prefetch.torch.cuda.is_current_stream_capturing",
            return_value=False,
        ),
        patch(
            "vllm_ascend.model_executor.offloader.prefetch.torch.cuda.stream",
            return_value=stream_context,
        ),
        patch("vllm_ascend.model_executor.offloader.prefetch.torch.cuda.Event") as event_factory,
    ):
        offloader.start_onload_to_static()
        offloader.start_onload_to_static()

    event_factory.assert_not_called()
    assert current_stream.record_event.call_count == 2
    current_stream.record_event.assert_called_with(offloader._eager_compute_ready_event)
    assert offloader.copy_stream.wait_event.call_count == 2


def test_net_offloaded_device_bytes_subtracts_static_pool():
    offloader = SimpleNamespace(
        total_offloaded_bytes=4_441_000_000,
        buffer_pool=SimpleNamespace(total_bytes=252_700_000),
    )

    assert _net_offloaded_device_bytes(offloader) == 4_188_300_000


def test_net_offloaded_device_bytes_never_negative():
    offloader = SimpleNamespace(
        total_offloaded_bytes=10,
        buffer_pool=SimpleNamespace(total_bytes=20),
    )

    assert _net_offloaded_device_bytes(offloader) == 0


def test_reclaim_offloaded_device_memory_releases_cache_and_updates_usage():
    offloader = SimpleNamespace(
        total_offloaded_bytes=400,
        buffer_pool=SimpleNamespace(total_bytes=100),
    )

    with patch("vllm_ascend.worker.model_runner_v1.torch.npu.empty_cache") as empty_cache:
        resident_bytes, reclaimed_bytes = _reclaim_offloaded_device_memory(
            1_000,
            offloader,
        )

    assert (resident_bytes, reclaimed_bytes) == (700, 300)
    empty_cache.assert_called_once_with()


def test_warm_up_tp_communicator_before_prefetch_model_load():
    offloader = AscendPrefetchOffloader.__new__(AscendPrefetchOffloader)
    device = torch.device("npu:0")
    warmup_tensor = MagicMock()
    current_stream = MagicMock()
    tp_group = SimpleNamespace(world_size=4, device_group=object())

    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.get_tp_group",
            return_value=tp_group,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.torch.zeros",
            return_value=warmup_tensor,
        ) as zeros,
        patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce") as all_reduce,
        patch(
            "vllm_ascend.worker.model_runner_v1.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch("vllm_ascend.worker.model_runner_v1.torch.npu.empty_cache") as empty_cache,
    ):
        warmed_up = _warm_up_tp_communicator_for_prefetch(offloader, device)

    assert warmed_up
    zeros.assert_called_once_with(1, dtype=torch.int32, device=device)
    all_reduce.assert_called_once_with(warmup_tensor, group=tp_group.device_group)
    current_stream.synchronize.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_warm_up_tp_communicator_skips_non_prefetch_offloader():
    with patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce") as all_reduce:
        warmed_up = _warm_up_tp_communicator_for_prefetch(
            NoopOffloader(),
            torch.device("npu:0"),
        )

    assert not warmed_up
    all_reduce.assert_not_called()
