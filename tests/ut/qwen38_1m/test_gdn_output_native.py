# SPDX-License-Identifier: Apache-2.0
"""Direct GDN launch admission, metadata ownership and output lifetime."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.gdn_output_native import GDNOutputNative, metadata_key


def operands(tokens=64):
    return (
        torch.zeros(1, 4, tokens, 128, dtype=torch.float16),
        torch.zeros(1, 4, tokens, 128, dtype=torch.float16),
        torch.zeros(1, 12, tokens, 128, dtype=torch.float16),
        torch.zeros(1, 12, tokens // 64, 128, 128, dtype=torch.float16),
        torch.zeros(1, 12, tokens, dtype=torch.float32),
    )


def key(values, cu=None, chunks=None, size=64, transpose=False):
    q, k, v, h, g = values
    return metadata_key(q, k, v, h, 128**-0.5, g, cu, chunks, size, transpose)


@pytest.mark.parametrize("tokens", [64, 128, 512, 2048])
def test_admits_padded_tp4_shapes(tokens):
    assert key(operands(tokens))[:2] == (tokens, 128**-0.5)


def test_varlen_chunk_plan_is_checked_without_tensor_readback():
    values = operands(192)
    assert key(values, [0, 64, 192], [0, 0, 1, 0, 1, 1]) is not None
    assert key(values, [0, 64, 192], [0, 0, 0, 1, 0, 2]) is None
    assert key(values, [0, 64, 128], [0, 0, 1, 0]) is None
    assert key(values, [0, 0, 192], [1, 0, 1, 1, 1, 2]) is None
    assert key(values, torch.tensor([0, 192]), []) is None


@pytest.mark.parametrize("change", ["dtype", "stride", "heads", "tail", "large", "state", "chunk", "transpose", "g"])
def test_rejects_unqualified_geometry(change):
    values = list(operands(65 if change == "tail" else 2112 if change == "large" else 64))
    if change == "dtype":
        values[0] = values[0].float()
    if change == "stride":
        values[0] = torch.empty(1, 4, 64, 256, dtype=torch.float16)[..., ::2]
    if change == "heads":
        values[0] = values[0][:, :3]
    if change == "state":
        values[3] = values[3][:, :, :, :64]
    if change == "g":
        values[4] = None
    assert key(values, size=128 if change == "chunk" else 64, transpose=change == "transpose") is None


def make_owner(monkeypatch):
    state = SimpleNamespace(capturing=False, stream=7, calls=[])
    runtime = SimpleNamespace(
        is_current_stream_capturing=lambda: state.capturing,
        current_stream=lambda device: SimpleNamespace(npu_stream=state.stream),
    )
    monkeypatch.setattr(torch, "npu", runtime, raising=False)

    def launch(kernel, args, blocks):
        state.calls.append((args, blocks))
        args[7].copy_(args[2] + 1)

    monkeypatch.setattr(torch.ops.qwen_checkpoint_preparation_v1, "launch", launch, raising=False)

    def reference(q, k, v, h, scale, **kwargs):
        return v + 2

    owner = GDNOutputNative(object(), reference, torch.device("cpu"))
    return owner, state


def test_launch_reuses_metadata_and_owns_distinct_outputs(monkeypatch):
    owner, state = make_owner(monkeypatch)
    q, k, v, h, g = operands()
    first = owner(q, k, v, h, 128**-0.5, g=g)
    v.fill_(3)
    second = owner(q, k, v, h, 128**-0.5, g=g)
    assert torch.all(first == 1) and torch.all(second == 4)
    assert len(owner.records) == 1 and owner.calls == 2
    assert state.calls[0][0][9] is state.calls[1][0][9]
    assert state.calls[0][0][7] is not state.calls[1][0][7]
    assert all(call[1] == 8 for call in state.calls)


@pytest.mark.parametrize("reason", ["disabled", "capture", "stream", "gamma"])
def test_fallback_keeps_native_graphs_and_other_streams_untouched(monkeypatch, reason):
    owner, state = make_owner(monkeypatch)
    owner.enabled = reason != "disabled"
    state.capturing = reason == "capture"
    state.stream = 8 if reason == "stream" else 7
    q, k, v, h, g = operands()
    actual = owner(q, k, v, h, 128**-0.5, g=g, g_gamma=g if reason == "gamma" else None)
    assert torch.all(actual == 2)
    assert owner.fallbacks == 1 and not state.calls and not owner.records
