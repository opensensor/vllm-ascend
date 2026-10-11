# SPDX-License-Identifier: Apache-2.0
"""Admission and graph-address ownership gates for the experimental INT8 partition."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.native_hybrid_prefill import NativeHybridPrefill


def operands(rows=129):
    inputs, outputs, experts, groups = 640, 2560, 2, 5
    bank = SimpleNamespace(
        weight=torch.empty(experts, outputs, inputs // 2, dtype=torch.int8),
        weight_scale=torch.empty(experts, outputs, groups, dtype=torch.float16),
        weight_offset=torch.empty(experts, outputs, groups, dtype=torch.float16),
        weight_sum=torch.empty(experts, outputs, groups, dtype=torch.float16),
    )
    prepared = (
        torch.empty(rows, inputs // 2, dtype=torch.int8),
        torch.empty(rows, inputs // 2, dtype=torch.int8),
        torch.empty(rows, groups, 8, dtype=torch.float32),
        torch.empty(rows, groups, 8, dtype=torch.float32),
    )
    return bank, prepared, torch.tensor([1, rows], dtype=torch.int64)


@pytest.mark.parametrize("unpack", [False, True])
def test_disjoint_partitions_share_output_and_original_weights_without_host_counts(monkeypatch, unpack):
    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    launches = []
    resource = NativeHybridPrefill(
        "small",
        "large",
        lambda kernel, args, blocks: launches.append((kernel, args, blocks)),
        torch.device("cpu"),
        "unpack" if unpack else None,
    )
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("host count synchronization"))
    bank, prepared, ends = operands()
    output = resource(bank, prepared, ends)
    assert [entry[0] for entry in launches] == (["small", "unpack", "large"] if unpack else ["small", "large"])
    small, large = launches[0][1], launches[-1][1]
    assert small[-2] is output and large[-2] is output
    assert small[4] is bank.weight and large[3] is bank.weight
    assert small[8] is ends and large[6] is ends
    assert large[-1].tolist() == [129, 640, 2560, 2, 128]
    assert all(entry[2] == 8 for entry in launches)
    if unpack:
        assert large[0].dtype == torch.int8 and large[0].shape == (129, 640)
        assert launches[1][1][-2] is large[0]
    else:
        assert large[0] is prepared[0]
    first_config = large[-1]
    resource(bank, prepared, ends)
    assert launches[-1][1][-1] is first_config


@pytest.mark.parametrize("rows", [0, 3, 30, 128, 25601])
def test_decode_and_oversized_shapes_never_launch(rows):
    resource = NativeHybridPrefill(None, None, lambda *args: pytest.fail("unsafe launch"), torch.device("cpu"))
    with pytest.raises(ValueError, match="geometry/dtype"):
        resource(*operands(rows))


def test_routed_ids_are_rejected_before_any_launch():
    resource = NativeHybridPrefill(None, None, lambda *args: pytest.fail("routed launch"), torch.device("cpu"))
    bank, prepared, ends = operands()
    with pytest.raises(ValueError, match="geometry/dtype"):
        resource(bank, prepared, ends.int())
