# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp import direct_hc_residual as candidate


def make_op(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    calls = []

    def launch(kernel, arguments, blocks):
        calls.append(arguments[-1])
        hyper, block, injection, output, tiling = arguments
        assert blocks == 8 and injection.dtype == torch.float32
        tokens, width = tiling.tolist()
        output.copy_(
            (hyper.float().view(tokens, 4, width) + block.float()[:, None] * injection[:, :, None]).flatten(1).half()
        )

    return candidate.DirectHCResidual("unused", kernel=object(), launch=launch), calls


@torch.no_grad()
def test_geometry_tiling_reuse_and_inputs_preserved(monkeypatch):
    op, calls = make_op(monkeypatch)
    hyper = torch.ones(3, 2048).half()
    block = torch.full((3, 512), 0.5).half()
    injection = torch.arange(12).reshape(3, 4).half() / 4
    originals = [t.clone() for t in (hyper, block, injection)]
    first = op(hyper, block, injection)
    second = op(hyper, block, injection)
    assert calls[0] is calls[1]
    assert torch.equal(first, second) and first.data_ptr() != hyper.data_ptr()
    assert all(torch.equal(a, b) for a, b in zip((hyper, block, injection), originals))


@pytest.mark.parametrize("kind", ["rank", "width", "streams", "tokens", "dtype", "device", "contiguous"])
def test_invalid_geometry_rejected_before_launch(monkeypatch, kind):
    op, calls = make_op(monkeypatch)
    hyper = torch.ones(3, 2048).half()
    block = torch.ones(3, 512).half()
    injection = torch.ones(3, 4).half()
    if kind == "rank":
        block = block.unsqueeze(0)
    elif kind == "width":
        block = block[:, :511].contiguous()
    elif kind == "streams":
        injection = injection[:, :3].contiguous()
    elif kind == "tokens":
        block = block[:0]
    elif kind == "dtype":
        hyper = hyper.float()
    elif kind == "device":
        monkeypatch.setattr(candidate, "_supports_device", lambda device: False)
    elif kind == "contiguous":
        hyper = torch.ones(2048, 3).half().T
    with pytest.raises(ValueError):
        op(hyper, block, injection)
    assert not calls


def test_capture_requires_shape_warmup(monkeypatch):
    op, calls = make_op(monkeypatch)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="warm up"):
        op(torch.zeros(3, 2048).half(), torch.zeros(3, 512).half(), torch.ones(3, 4).half())
    assert not calls
