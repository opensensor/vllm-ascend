# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp import direct_hc_mean as candidate


def make_op(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    calls = []

    def launch(kernel, arguments, blocks):
        normalized, gate, output, tiling = arguments
        calls.append(tiling)
        assert blocks == 8
        rows, width = tiling.tolist()
        output.copy_((normalized.view(rows, 4, width) * gate.view(rows, 4, width)).mean(1).half())

    return candidate.DirectHCMean("unused", kernel=object(), launch=launch), calls


@torch.no_grad()
def test_geometry_tiling_reuse_and_input_storage_preserved(monkeypatch):
    op, calls = make_op(monkeypatch)
    normalized = torch.arange(3 * 2048).reshape(3, 2048).float() / 1024
    gate = torch.full((3, 2048), 0.5).half()
    saved = [t.clone() for t in (normalized, gate)]
    first = op(normalized, gate)
    second = op(normalized, gate)
    assert first.shape == (3, 512) and first.dtype == torch.float16
    assert calls[0] is calls[1]
    assert torch.equal(first, second)
    assert all(torch.equal(a, b) for a, b in zip((normalized, gate), saved))


@pytest.mark.parametrize("kind", ["rank", "width", "shape", "tokens", "dtype", "device", "contiguous"])
def test_invalid_geometry_rejected_before_launch(monkeypatch, kind):
    op, calls = make_op(monkeypatch)
    normalized = torch.ones(3, 2048).float()
    gate = torch.ones(3, 2048).half()
    if kind == "rank":
        normalized = normalized.unsqueeze(0)
    elif kind == "width":
        normalized = normalized[:, :2047].contiguous()
        gate = gate[:, :2047].contiguous()
    elif kind == "shape":
        gate = gate[:, :2044].contiguous()
    elif kind == "tokens":
        normalized, gate = normalized[:0], gate[:0]
    elif kind == "dtype":
        normalized = normalized.half()
    elif kind == "device":
        monkeypatch.setattr(candidate, "_supports_device", lambda device: False)
    elif kind == "contiguous":
        normalized = torch.ones(2048, 3).T
    with pytest.raises(ValueError):
        op(normalized, gate)
    assert not calls


def test_capture_requires_shape_warmup(monkeypatch):
    op, calls = make_op(monkeypatch)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="warm up"):
        op(torch.ones(3, 2048).float(), torch.ones(3, 2048).half())
    assert not calls
