# SPDX-License-Identifier: Apache-2.0
"""Regression for direct drafts inside a padded FULL context."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
namespace = {}
exec((ROOT / "vllm_ascend/models/qwen4_exp/ops/qsa_metadata.py").read_text(), namespace)
packed_qsa_query_lens = namespace["packed_qsa_query_lens"]


@pytest.mark.parametrize(
    "lengths,rows,expected",
    [
        ([1, 1, 1], 1, [1, 0, 0]),
        ([1] * 18, 6, [1] * 6 + [0] * 12),
        ([3, 3], 6, [3, 3]),
        ([3, 3, 3], 5, [3, 2, 0]),
        ([0, 2, 0, 1], 3, [0, 2, 0, 1]),
        ([1] * 18, 18, [1] * 18),
        ([3], 0, [0]),
    ],
)
def test_boundaries_cover_actual_rows_and_keep_request_indices(lengths, rows, expected):
    tensor = torch.tensor(lengths, dtype=torch.int32)
    before = tensor.clone()
    assert packed_qsa_query_lens(SimpleNamespace(query_lens_cpu=tensor), rows) == expected
    assert torch.equal(tensor, before)
    assert sum(expected) == rows
    assert len(expected) == len(lengths)


@pytest.mark.parametrize("lengths,rows", [([1], 2), ([-1, 2], 1), ([1], -1)])
def test_invalid_geometry_is_rejected(lengths, rows):
    with pytest.raises(ValueError, match="cover the packed input"):
        packed_qsa_query_lens(SimpleNamespace(query_lens_cpu=torch.tensor(lengths)), rows)


def test_missing_host_boundaries_preserve_backend_fallback():
    assert packed_qsa_query_lens(SimpleNamespace(), 3) is None


def test_device_boundaries_never_copy_to_host():
    class DeviceLengths:
        device = SimpleNamespace(type="npu")

        def tolist(self):
            raise AssertionError("unexpected device readback")

    with pytest.raises(ValueError, match="CPU"):
        packed_qsa_query_lens(SimpleNamespace(query_lens_cpu=DeviceLengths()), 1)


def test_model_selection_and_attention_use_their_consumed_row_count():
    tree = ast.parse((ROOT / "vllm_ascend/models/qwen4_exp/model.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "packed_qsa_query_lens"
    ]
    assert len(calls) == 3
    assert all(isinstance(call.args[1], ast.Name) and call.args[1].id == "seq_len" for call in calls)
