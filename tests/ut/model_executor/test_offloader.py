from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import (
    AscendPrefetchOffloader,
    ParamInfo,
    _is_using_nz_weight,
    _use_pageable_cpu_storage,
)
from vllm_ascend.worker.model_runner_v1 import (
    _net_offloaded_device_bytes,
    _reclaim_offloaded_device_memory,
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


def test_pageable_cpu_storage_only_disables_pinning_in_scope():
    import vllm.model_executor.offloader.prefetch as vllm_prefetch

    original_should_pin_memory = vllm_prefetch.should_pin_memory
    with _use_pageable_cpu_storage():
        assert vllm_prefetch.should_pin_memory() is False

    assert vllm_prefetch.should_pin_memory is original_should_pin_memory


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
            "vllm_ascend.model_executor.offloader.prefetch.StaticBufferPool",
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
