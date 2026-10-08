# SPDX-License-Identifier: Apache-2.0
"""Tiled normalization descriptors, matrix isolation and pre-launch admission."""

import struct

import pytest
import torch

from tests.ut.glm_perf.test_sinkhorn_native import pre_fixture
from tools.glm_perf.resident_candidates.mhc_sinkhorn_normalize import wrap_pre
from tools.glm_perf.sinkhorn_native import reference_normalize
from tools.glm_perf.sinkhorn_tiled import (
    MATRIX_ELEMENTS,
    TILE_ROWS,
    SinkhornTiled,
    descriptor,
    gather_indices,
    reduction_order,
)


def test_prepared_gathers_preserve_each_matrix_and_both_axis_orders():
    indices = gather_indices().long() // 4
    assert indices.shape == (8, TILE_ROWS * MATRIX_ELEMENTS)
    matrix = torch.arange(TILE_ROWS * MATRIX_ELEMENTS).reshape(TILE_ROWS, 4, 4)
    gathered = matrix.flatten()[indices].reshape(8, TILE_ROWS, 4, 4)
    for component in range(4):
        assert torch.equal(gathered[component], matrix[:, :, component : component + 1].expand(-1, -1, 4))
        assert torch.equal(gathered[component + 4], matrix[:, component : component + 1, :].expand(-1, 4, -1))
    assert torch.equal(indices // MATRIX_ELEMENTS, torch.arange(TILE_ROWS).repeat_interleave(16).expand(8, -1))


@pytest.mark.parametrize("rows", [0, 641, True, 1.5])
@pytest.mark.parametrize("cached", [False, True])
def test_invalid_rows_rejected_even_when_equal_to_a_prepared_key(rows, cached):
    native = SinkhornTiled.__new__(SinkhornTiled)
    native.iterations, native.epsilon, native.order = 20, 1e-6, 2
    native.configs = {1: None} if cached else {}
    with pytest.raises(ValueError, match="rows"):
        native.prepare_counts([rows])


@pytest.mark.parametrize(
    "iterations,epsilon,order",
    [
        (0, 1e-6, 2),
        (65, 1e-6, 2),
        (True, 1e-6, 2),
        (20, float("nan"), 2),
        (20, -1, 2),
        (20, 2, 2),
        (20, 1e-6, True),
        (20, 1e-6, 3),
    ],
)
def test_invalid_arithmetic_contract(iterations, epsilon, order):
    with pytest.raises(ValueError):
        descriptor(1, iterations, epsilon, order)


def test_epsilon_bits_and_single_prepared_backing_are_preserved():
    native = SinkhornTiled.__new__(SinkhornTiled)
    native.iterations, native.epsilon, native.order = 20, 1e-6, 2
    native.device, native.configs = torch.device("cpu"), {}
    native.prepare_counts([1, 2, 640, 2])
    assert len(native.configs) == 3
    bits = struct.unpack("<I", struct.pack("<f", 1e-6))[0]
    for rows, config in native.configs.items():
        assert config.tolist() == [rows, 20, bits, 2]
    assert len({v.untyped_storage().data_ptr() for v in native.configs.values()}) == 1
    originals = dict(native.configs)
    native.prepare_counts([1, 2])
    assert all(native.configs[k] is v for k, v in originals.items())


def test_shape_dependent_order_is_prepared_before_capture():
    native = SinkhornTiled.__new__(SinkhornTiled)
    native.iterations, native.epsilon, native.order = 20, 1e-6, None
    native.device, native.configs = torch.device("cpu"), {}
    native.prepare_counts([1, 8, 127, 128, 129, 640])
    assert [native.configs[n][-1].item() for n in (1, 8, 127, 128, 129, 640)] == [2, 2, 2, 1, 1, 1]
    assert reduction_order(127) == 2 and reduction_order(128) == 1


def test_only_explicitly_qualified_prefill_shapes_use_fused_normalization():
    calls = []

    def normalize(mix):
        calls.append(mix.shape[0])
        return reference_normalize(mix, 20, 1e-6)

    fused = wrap_pre(pre_fixture, normalize, qualified_rows=(1, 2, 128, 640))
    for rows in (2, 127, 128, 639, 640):
        value = torch.randn(rows, 4, 4)
        assert torch.equal(fused(value, 20, 1e-6), pre_fixture(value, 20, 1e-6))
    assert calls == [2, 128, 640]


@pytest.mark.parametrize("rows", [(), (True,), (0,), (-1,), (1.5,)])
def test_invalid_qualified_row_contract_is_rejected(rows):
    with pytest.raises(ValueError, match="qualified"):
        wrap_pre(pre_fixture, None, qualified_rows=rows)


@pytest.mark.parametrize(
    "mix", [torch.ones(2, 4, 4).half(), torch.ones(4, 4), torch.ones(2, 4, 3), torch.ones(2, 4, 4).transpose(1, 2)]
)
def test_invalid_inputs_never_reach_native_launch(mix):
    native = SinkhornTiled.__new__(SinkhornTiled)
    native.device, native.configs = torch.device("cpu"), {2: None}
    native.launch = lambda *args: pytest.fail("unqualified input submitted")
    with pytest.raises(ValueError):
        native(mix)
