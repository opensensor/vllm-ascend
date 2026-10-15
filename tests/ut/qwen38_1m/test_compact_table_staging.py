# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp.resident_candidates.qfn_compact_table_staging import CompactTableStaging, build_remap
from vllm_ascend._310p.host_staging import PinnedHostStaging


def test_staging_orders_host_reuse_after_dma_and_keeps_global_table():
    events = []
    completion = SimpleNamespace(
        query=lambda: False,
        synchronize=lambda: events.append("wait"),
        record=lambda: events.append("submitted"),
    )
    destination = torch.zeros((4, 7), dtype=torch.int32)
    table = SimpleNamespace(is_mamba_group=True, block_table=SimpleNamespace(gpu=destination))
    runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=[table])))
    resource = CompactTableStaging(
        runner,
        stage_factory=lambda shape, dtype: PinnedHostStaging(
            shape, dtype, pin_memory=False, event_factory=lambda: completion
        ),
    )
    global_table = np.arange(7, dtype=np.int64).reshape(1, 7)
    original = global_table.copy()
    resource.copy(runner, 0, global_table, destination[:1])
    resource.copy(runner, 0, global_table + 10, destination[:1])
    assert events == ["submitted", "wait", "submitted"]
    assert np.array_equal(global_table, original)
    assert torch.equal(destination[0], torch.arange(7) + 10)
    assert resource.copies == 2
    with pytest.raises(RuntimeError, match="another worker"):
        resource.copy(object(), 0, global_table, destination[:1])


def test_unreviewed_runner_is_rejected():
    with pytest.raises(RuntimeError, match="exact qualified"):
        build_remap(lambda: None, None, source="def unreviewed(): pass\n")
