# SPDX-License-Identifier: Apache-2.0
"""Offline ABI and normalization-contract checks; NPU parity is separate."""

import struct

import pytest
import torch

from tools.glm_perf.resident_candidates.mhc_sinkhorn_normalize import wrap_pre
from tools.glm_perf.sinkhorn_native import normalization_config, reference_normalize


@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("iterations", [1, 20, 64])
@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_native_config_layout(rows, iterations, epsilon, order):
    packed = struct.pack("<4q", *normalization_config(rows, iterations, epsilon, order))
    assert struct.unpack_from("<q", packed, 0)[0] == rows
    assert struct.unpack_from("<q", packed, 8)[0] == iterations
    assert struct.unpack_from("<f", packed, 16)[0] == pytest.approx(epsilon)
    assert packed[20:24] == bytes(4)
    assert struct.unpack_from("<q", packed, 24)[0] == order


@pytest.mark.parametrize(
    "args",
    [
        (0, 20, 1e-6, 0),
        (9, 20, 1e-6, 0),
        (2, 0, 1e-6, 0),
        (2, 65, 1e-6, 0),
        (2, 20, -1, 0),
        (2, 20, float("nan"), 0),
        (2, 20, float("inf"), 0),
        (2, 20, 1e-6, 3),
    ],
)
def test_reject_unsupported_geometry(args):
    with pytest.raises(ValueError):
        normalization_config(*args)


@pytest.mark.parametrize("args", [(1.5, 20, 1e-6, 0), (2, 2.5, 1e-6, 0), (2, 20, 1e-6, False)])
def test_reject_fractional_or_boolean_config(args):
    with pytest.raises(ValueError, match="integers"):
        normalization_config(*args)


def test_first_iteration_normalizes_columns_only():
    mix = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [3.0, 6.0, 9.0, 12.0], [4.0, 8.0, 12.0, 16.0]]])
    expected = torch.tensor([[[0.1] * 4, [0.2] * 4, [0.3] * 4, [0.4] * 4]])
    torch.testing.assert_close(reference_normalize(mix, 1, 0), expected, atol=0, rtol=0)
    # The next row normalization makes this matrix uniform.
    torch.testing.assert_close(reference_normalize(mix, 2, 0), torch.full_like(mix, 0.25), atol=0, rtol=0)


def pre_fixture(comb_logits, sinkhorn_repeat=20, hc_sinkhorn_eps=1e-6):
    num_tokens, hc_mult, _ = comb_logits.shape
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    return comb_mix


@pytest.mark.parametrize(
    "rows,width,iterations,eps,uses_native",
    [
        (1, 4, 20, 1e-6, True),
        (2, 4, 20, 1e-6, True),
        (8, 4, 20, 1e-6, True),
        (0, 4, 20, 1e-6, False),
        (9, 4, 20, 1e-6, False),
        (640, 4, 20, 1e-6, False),
        (2, 3, 20, 1e-6, False),
        (2, 4, 1, 1e-6, False),
        (2, 4, 20, 1e-5, False),
    ],
)
def test_decode_dispatch_and_prefill_fallback(rows, width, iterations, eps, uses_native):
    calls = []

    def normalize(mix):
        calls.append(mix.shape)
        return reference_normalize(mix, 20, 1e-6)

    wrapped = wrap_pre(pre_fixture, normalize)
    logits = torch.randn(rows, width, width)
    expected = pre_fixture(logits, iterations, eps)
    torch.testing.assert_close(wrapped(logits, iterations, eps), expected, atol=0, rtol=0)
    assert bool(calls) == uses_native
    calls.clear()
    wrapped_again = wrap_pre(wrapped, normalize)
    torch.testing.assert_close(wrapped_again(logits, iterations, eps), expected, atol=0, rtol=0)
    assert len(calls) == int(uses_native)


def test_refuse_changed_upstream_block():
    with pytest.raises(ValueError, match="upstream mHC normalization changed"):
        wrap_pre(reference_normalize, lambda mix: mix)
