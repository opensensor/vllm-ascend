# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.qwen4exp.native_gate40_probe import route_ids
from tools.qwen4exp.resident_candidates import gate_up40


def test_native_candidate_requires_successful_transaction_resources():
    for resources in (None, {}, {"other_version": {}}):
        with pytest.raises(RuntimeError, match="qualify"):
            gate_up40.replacements(resources)


@pytest.mark.parametrize(
    "routes,dtype,shape,selected",
    [
        (30, torch.int32, (1280, 1280), False),
        (31, torch.int32, (1280, 1280), True),
        (40, torch.int32, (1280, 1280), True),
        (41, torch.int32, (1280, 1280), False),
        (60, torch.int32, (2560, 320), False),
        (32, torch.int64, (1280, 1280), False),
    ],
)
def test_only_extended_gate_up_uses_native_resource(monkeypatch, routes, dtype, shape, selected):
    direct = Mock(return_value="direct")
    fallback = Mock(return_value="qualified")
    monkeypatch.setattr(torch.ops._C_ascend, "npu_qwen_w4_a8_int4_matmul_310", fallback, raising=False)
    resources = {gate_up40.NATIVE_RESOURCE: {"stream40": direct}}
    function = next(iter(gate_up40.replacements(resources).values()))
    bank = SimpleNamespace(
        backend=gate_up40.NATIVE_INT4_BACKEND,
        weight=SimpleNamespace(shape=(128, *shape)),
        weight_scale="scale",
        weight_offset="offset",
        weight_sum="sum",
    )
    arguments = ("low", "high", "xs", "sums")
    ids = torch.empty(routes, dtype=dtype)
    result = function(bank, arguments, ids)
    assert result == ("direct" if selected else "qualified")
    called, unused = (direct, fallback) if selected else (fallback, direct)
    called.assert_called_once_with(*arguments, bank.weight, "scale", "offset", "sum", ids)
    unused.assert_not_called()


@pytest.mark.parametrize(
    "pattern", ["mixed", "invalid", "concentrated", "all_local", "peer_high", "boundary16", "boundary17"]
)
def test_probe_route_patterns_have_valid_shape_and_cover_fallback(pattern):
    ids = route_ids(40, 4, pattern, "cpu")
    assert ids.shape == (40,) and ids.dtype == torch.int32
    assert bool(((ids >= -1) & (ids < 4)).all()) if pattern != "peer_high" else bool((ids > 4).all())
    if pattern.startswith("boundary"):
        assert int((ids == 0).sum()) == int(pattern[-2:])
