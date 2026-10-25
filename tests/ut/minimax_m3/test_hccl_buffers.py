# SPDX-License-Identifier: Apache-2.0
"""Validate communication memory controls using the actual config and options."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def modules():
    pytest.importorskip("torch_npu")
    # Runtime imports keep CPU-only collection independent of NPU discovery.
    from vllm_ascend import utils
    from vllm_ascend.ascend_config import AscendConfig

    return utils, AscendConfig


@pytest.mark.parametrize("size", [1, 32, 200, 512])
def test_hccl_buffer_config_reaches_group_options(modules, monkeypatch, size):
    utils, config_type = modules
    config = config_type(sparse_kv_offload_config=SimpleNamespace(enabled=False), hccl_buffer_size_mb=size)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: config)
    monkeypatch.setattr(utils, "calculate_dp_buffer_size", lambda: 50)
    for group in ("tp", "pp", "ep"):
        options = utils.create_hccl_pg_options(group)
        assert options.hccl_config == {"hccl_buffer_size": size, "group_name": group}
    assert utils.get_hccl_config_for_pg_options("mc2") is None
    assert utils.get_hccl_config_for_pg_options("dp") == {"hccl_buffer_size": 50}
    assert utils.get_hccl_config_for_pg_options("dynamic_eplb") == {"hccl_buffer_size": 100}
    assert utils.get_hccl_config_for_pg_options("sleep_lifecycle_anchor") == {"hccl_buffer_size": 1}


@pytest.mark.parametrize("size", [0, -1, True, "32", 32.5, None])
def test_hccl_buffer_rejects_invalid_capacity(modules, size):
    _, config_type = modules
    with pytest.raises(ValueError, match="positive integer"):
        config_type(sparse_kv_offload_config=SimpleNamespace(enabled=False), hccl_buffer_size_mb=size)


def test_hccl_default_and_preinitialization_keep_200_mb(modules, monkeypatch):
    utils, config_type = modules
    config = config_type(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    monkeypatch.setattr(utils, "get_ascend_config", lambda: config)
    assert utils.get_default_buffer_config() == {"hccl_buffer_size": 200}

    def uninitialized():
        raise RuntimeError("config not initialized")

    monkeypatch.setattr(utils, "get_ascend_config", uninitialized)
    assert utils.get_default_buffer_config() == {"hccl_buffer_size": 200}
