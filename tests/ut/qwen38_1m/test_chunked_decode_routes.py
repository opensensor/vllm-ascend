# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates.chunked_decode_routes import make_grouped


class Input:
    def __init__(self, tokens):
        self.value = torch.arange(tokens).reshape(tokens, 1).expand(tokens, 2560)
        self.shape = self.value.shape
        self.device = SimpleNamespace(type="npu")

    def __getitem__(self, key):
        return self.value[key]


@pytest.mark.parametrize("tokens", [13, 16, 18, 24])
@pytest.mark.parametrize("chunk", [6, 8, 12])
def test_chunks_preserve_token_weight_route_alignment(tokens, chunk):
    inputs = Input(tokens)
    weights = torch.arange(tokens).reshape(tokens, 1).expand(tokens, 10) * 7
    ids = weights + 3
    calls = []

    def routed(x, w, i):
        calls.append(x.shape[0])
        return x[:, :1] + w[:, :1] + i[:, :1]

    module = SimpleNamespace(
        native_int4=True,
        device_routing=True,
        top_k=10,
        grouped_activation="cann_swiglu_pack",
        max_routed_rows=128,
        _forward_routed=routed,
    )

    def fallback(*args):
        pytest.fail("admitted shape should use bounded routed chunks")

    output = make_grouped(fallback, chunk_tokens=chunk)(module, inputs, weights, ids)
    torch.testing.assert_close(output, inputs.value[:, :1] + weights[:, :1] + ids[:, :1])
    assert sum(calls) == tokens
    assert max(calls) <= chunk


@pytest.mark.parametrize("tokens", [1, 12, 25, 640])
def test_preserves_small_routed_and_prefill_dispatch(tokens):
    module = SimpleNamespace(
        native_int4=True, device_routing=True, top_k=10, grouped_activation="cann_swiglu_pack", max_routed_rows=128
    )
    marker = object()
    output = make_grouped(lambda *args: marker)(module, Input(tokens), None, None)
    assert output is marker


@pytest.mark.parametrize("activation", ["torch", "unknown"])
def test_different_activation_math_falls_back(activation):
    module = SimpleNamespace(
        native_int4=True, device_routing=True, top_k=10, grouped_activation=activation, max_routed_rows=128
    )
    marker = object()
    assert make_grouped(lambda *args: marker)(module, Input(16), None, None) is marker


@pytest.mark.parametrize("chunk", [True, 0, 4, 13, "8"])
def test_rejects_unqualified_chunk_size(chunk):
    with pytest.raises(ValueError):
        make_grouped(None, chunk_tokens=chunk)
