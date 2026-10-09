# SPDX-License-Identifier: Apache-2.0
"""Capacity views share prefill storage without retaining per-chunk buffers."""

import pytest
import torch

from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE


def native(capacity=None):
    op = NativeFusedMoE.__new__(NativeFusedMoE)
    op.scratch = {}
    op.device = torch.device("cpu")
    op.activation_bits = 4
    op.fp16_route_workspace = True
    op.route_packed_down = True
    op.route_compact_down_scales = True
    op.raw_hidden_scales = True
    op.raw_input_scales = True
    op.configure_prefill_scratch(capacity)
    return op


def geometry(rows, experts=3):
    return FusedGeometry(rows, 2, experts, 256, 256, 3, 3, 4)


@pytest.mark.parametrize("invalid", [0, 16, 1.5, True, "1280", 10**9])
def test_invalid_capacity_rejected(invalid):
    with pytest.raises(ValueError, match="bounded integer"):
        native(invalid)


def test_default_retains_only_640_row_scratch():
    op = native()
    assert op.shared_scratch(geometry(639)) is None
    assert op.shared_scratch(geometry(1280)) is None
    assert op.shared_scratch(geometry(640)) is not None
    assert len(op.scratch) == 1


def test_partial_chunks_share_capacity_and_retain_independent_descriptors():
    op = native(1280)
    full = op.shared_scratch(geometry(1280))
    for rows in (17, 33, 640, 1280, 129, 1279):
        views = op.shared_scratch(geometry(rows))
        expected_shapes = [value.shape for value in op.allocate_scratch(geometry(rows))]
        assert [value.shape for value in views] == expected_shapes
        assert all(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() for a, b in zip(views, full))
    assert len(op.scratch) == 1
    assert op.shared_scratch(geometry(1281)) is None
    assert op.shared_scratch(geometry(8)) is None
    with pytest.raises(ValueError, match="retire graphs"):
        op.configure_prefill_scratch(2560)


def test_routed_packing_keeps_expert_partitions_separate():
    op = native(1280)
    left = op.shared_scratch(geometry(640, experts=3))
    right = op.shared_scratch(geometry(640, experts=4))
    assert len(op.scratch) == 2
    assert left[3].untyped_storage().data_ptr() != right[3].untyped_storage().data_ptr()
