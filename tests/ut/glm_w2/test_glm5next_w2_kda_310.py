# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend.models.glm5next_w2.kda_310 import (
    _actual_lengths,
    _flatten_spec_state_indices,
    _safe_gate,
)
from vllm_ascend.models.glm5next_w2.model import (
    _rms_norm_gated_310,
    _with_fp16_recurrent_state_dtype,
)


def test_safe_gate_matches_glm_bounded_gate_formula():
    raw_gate = torch.tensor([[[[-1.0, 0.5], [2.0, -0.25]]]])
    a_log = torch.log(torch.tensor([0.5, 2.0]))
    dt_bias = torch.tensor([0.25, -0.5, 1.0, 0.75])

    actual = _safe_gate(raw_gate, a_log, dt_bias, lower_bound=-5.0)
    shifted = raw_gate + dt_bias.reshape(1, 1, 2, 2)
    expected = -5.0 * torch.sigmoid(torch.exp(a_log).reshape(1, 1, 2, 1) * shifted)

    torch.testing.assert_close(actual, expected)
    assert torch.all(actual < 0)
    assert torch.all(actual > -5)


def test_actual_lengths_and_spec_indices_preserve_token_order(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(_EXTRA_CTX=SimpleNamespace(capturing=False)),
    )
    cu_seqlens = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    state_indices = torch.tensor(
        [
            [7, 8, -1],
            [-1, -1, -1],
            [11, 12, 13],
        ],
        dtype=torch.int32,
    )

    lengths = _actual_lengths(cu_seqlens, num_sequences=3)
    flattened = _flatten_spec_state_indices(state_indices, lengths, total_tokens=5)

    assert lengths.tolist() == [2, 0, 3]
    assert flattened.tolist() == [7, 8, 11, 12, 13]


def test_recurrent_cache_dtype_preserves_all_three_conv_state_dtypes():
    dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float32)

    assert _with_fp16_recurrent_state_dtype(dtypes) == (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float16,
    )


def test_310p_output_norm_uses_native_rms_norm_and_sigmoid_gate(monkeypatch):
    x = torch.tensor([[[[1.0, -2.0], [3.0, 4.0]]]], dtype=torch.float16)
    gate = torch.tensor([[[0.25, -0.5], [1.0, -1.5]]], dtype=torch.float16)
    weight = torch.tensor([1.5, 0.5], dtype=torch.float16)
    eps = 1e-5
    calls = []

    def npu_rms_norm(value, affine, epsilon):
        calls.append((value, affine, epsilon))
        value_fp32 = value.float()
        normalized = value_fp32 * torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + epsilon)
        return (normalized * affine.float()).to(value.dtype), None

    monkeypatch.setitem(
        __import__("sys").modules,
        "torch_npu",
        SimpleNamespace(npu_rms_norm=npu_rms_norm),
    )
    norm = SimpleNamespace(weight=weight, bias=None, eps=eps, activation="sigmoid")

    actual = _rms_norm_gated_310(x, gate, norm)
    x_fp32 = x.float()
    expected_norm = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    expected = expected_norm * weight.float() * torch.sigmoid(gate.float())

    torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-3)
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is weight
    assert calls[0][2] == eps


def test_310p_path_uses_per_channel_gate_and_paged_state_ops():
    source = (Path(__file__).parents[3] / "vllm_ascend" / "models" / "glm5next_w2" / "kda_310.py").read_text()

    assert "npu_recurrent_gated_delta_rule_310" in source
    assert "gk=gk" in source
    assert "g=None" in source
    assert "chunk_kda_fwd" in source
    assert "npu_causal_conv1d_310" in source
    assert ".item()" not in source
