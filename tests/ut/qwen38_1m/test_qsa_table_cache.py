# SPDX-License-Identifier: Apache-2.0
"""Admission boundaries for the exact-address Qwen QSA candidate."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.qsa_table_cache import tiling_values


def inputs(tokens=515, requests=2, width=4096):
    q = torch.empty(tokens, 6, 256, dtype=torch.float16)
    k = torch.empty(2, 16, 64, 16, dtype=torch.float16)
    selection = SimpleNamespace(
        group_indices=torch.empty(tokens, 512, dtype=torch.int32),
        group_counts=torch.empty(tokens, dtype=torch.int32),
        tail_starts=torch.empty(tokens, dtype=torch.int32),
        tail_counts=torch.empty(tokens, dtype=torch.int32),
    )
    return [
        q,
        k,
        torch.empty_like(k),
        selection,
        torch.empty(requests, width, dtype=torch.int32),
        torch.empty(requests + 1, dtype=torch.int32),
        None,
        4,
    ]


@pytest.mark.parametrize("requests", [1, 2, 4])
@pytest.mark.parametrize("width", [8, 2048, 4096])
def test_tiling_retains_native_task_and_address_geometry(requests, width):
    values = tiling_values(*inputs(requests=requests, width=width))
    assert values == (515, 6, 1, 6, 1, 256, 64, 16, width, 512, requests, 65, 515, 1 << 20)


@pytest.mark.parametrize("tokens", [1, 3, 6, 9, 12, 13, 64, 127, 2049])
def test_decode_graphs_and_large_batches_keep_original(tokens):
    assert tiling_values(*inputs(tokens=tokens)) is None


@pytest.mark.parametrize("requests,width", [(5, 4096), (0, 4096), (2, 4097), (2, 4104), (2, 7)])
def test_unsupported_tables_keep_original(requests, width):
    assert tiling_values(*inputs(requests=requests, width=width)) is None


def test_dtype_stride_scale_and_metadata_must_match():
    args = inputs()
    args[0] = args[0].float()
    assert tiling_values(*args) is None
    args = inputs()
    args[4] = torch.empty(4096, 2, dtype=torch.int32).T
    assert tiling_values(*args) is None
    args = inputs()
    args[6] = 0.5
    assert tiling_values(*args) is None
    args = inputs()
    args[3].tail_counts = torch.empty(1, dtype=torch.int32)
    assert tiling_values(*args) is None


@pytest.mark.parametrize("width", [1, 16, 128, 512])
def test_live_indexer_int64_selections_are_normalized(monkeypatch, width):
    from tools.qwen4exp.qsa_table_cache import QSATableCache

    args = inputs()
    selection = args[3]
    selection.group_indices = torch.arange(515 * width, dtype=torch.int64).view(515, width)
    selection.group_counts = torch.full((515,), -width, dtype=torch.int64)
    selection.tail_starts = torch.zeros(515, dtype=torch.int64)
    selection.tail_counts = torch.full((515,), -1, dtype=torch.int64)
    observed = []

    def launch(kernel, tensors, blocks):
        observed.append((tensors, blocks))
        tensors[-2].zero_()

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda d: SimpleNamespace(npu_stream=7), is_current_stream_capturing=lambda: False
        ),
        raising=False,
    )
    monkeypatch.setattr(torch.ops, "qwen_checkpoint_preparation_v1", SimpleNamespace(launch=launch))

    def fallback(*a, **kw):
        raise AssertionError("live int64 indexer inputs must be admitted")

    owner = QSATableCache(object(), fallback, args[0].device)
    owner(*args[:6])
    tensors, blocks = observed[0]
    assert blocks == 8 and owner.calls == 1 and owner.fallbacks == 0
    for actual, expected in zip(
        tensors[3:7], (selection.group_indices, selection.group_counts, selection.tail_starts, selection.tail_counts)
    ):
        assert actual.dtype == torch.int32 and torch.equal(actual.long(), expected)
    assert tensors[-1][9] == width
