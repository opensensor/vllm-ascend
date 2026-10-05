# SPDX-License-Identifier: Apache-2.0
"""Gate operands must be initialized independently of any particular graph."""

from types import SimpleNamespace

import torch

from vllm_ascend.models.glm5next_w2.kda_310 import (
    _safe_gate,
    _safe_gate_for_layer,
    prepare_kda_gate_weights,
)
from vllm_ascend.models.glm5next_w2.model import _prepare_kda_gate_weights


def layer():
    return SimpleNamespace(
        A_log=torch.tensor([0.4, 1.2]),
        dt_bias=torch.tensor([-2.0, 0.3, -1.0, 0.5]),
        kda_lower_bound=-5.0,
    )


def expected(attn, raw):
    return _safe_gate(raw, attn.A_log, attn.dt_bias, attn.kda_lower_bound)


def test_forward_does_not_persist_graph_intermediates():
    attn = layer()
    raw = torch.arange(8).reshape(1, 2, 2, 2).float() / 8
    torch.testing.assert_close(_safe_gate_for_layer(attn, raw), expected(attn, raw))
    assert not hasattr(attn, "_kda_gate_weights")
    assert not hasattr(attn, "_kda_safe_gate_cache")


def test_stale_legacy_capture_cache_is_ignored():
    attn = layer()
    attn._kda_safe_gate_cache = (attn.A_log, attn.dt_bias, torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 2))
    raw = torch.zeros(1, 2, 2, 2)
    actual = _safe_gate_for_layer(attn, raw)
    torch.testing.assert_close(actual, expected(attn, raw))
    assert not torch.all(actual == -2.5)


def test_load_time_cache_handles_both_batch_sizes_and_weight_reload():
    attn = layer()
    prepare_kda_gate_weights(attn)
    scale = attn._kda_gate_weights[2]
    for tokens in (8, 2, 8):
        raw = torch.linspace(-1, 1, tokens * 4).reshape(1, tokens, 2, 2)
        torch.testing.assert_close(_safe_gate_for_layer(attn, raw), expected(attn, raw))
        assert attn._kda_gate_weights[2] is scale
    attn.A_log.add_(0.75)
    attn.dt_bias.sub_(0.25)
    prepare_kda_gate_weights(attn)
    torch.testing.assert_close(_safe_gate_for_layer(attn, raw), expected(attn, raw))
    assert attn._kda_gate_weights[2] is not scale


def test_replaced_parameter_uses_uncached_fallback():
    attn = layer()
    prepare_kda_gate_weights(attn)
    attn.A_log = torch.tensor([-0.5, 0.25])
    raw = torch.ones(1, 2, 2, 2)
    torch.testing.assert_close(_safe_gate_for_layer(attn, raw), expected(attn, raw))


def test_loader_prepares_only_kda_layers():
    attn = layer()
    mla = SimpleNamespace()
    _prepare_kda_gate_weights(
        [SimpleNamespace(layer_kind="kda", self_attn=attn), SimpleNamespace(layer_kind="dsa", self_attn=mla)]
    )
    assert hasattr(attn, "_kda_gate_weights")
    assert not hasattr(mla, "_kda_gate_weights")
