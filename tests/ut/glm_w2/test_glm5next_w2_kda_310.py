# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch

from vllm_ascend.models.glm5next_w2.kda_310 import (
    _actual_lengths,
    _flatten_spec_state_indices,
    _safe_gate,
)
from vllm_ascend.models.glm5next_w2.model import (
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


def test_actual_lengths_and_spec_indices_preserve_token_order():
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


def test_310p_path_uses_per_channel_gate_and_paged_state_ops():
    source = (Path(__file__).parents[3] / "vllm_ascend" / "models" / "glm5next_w2" / "kda_310.py").read_text()

    assert "npu_recurrent_gated_delta_rule_310" in source
    assert "gk=gk" in source
    assert "g=None" in source
    assert "chunk_kda_fwd" in source
    assert "npu_causal_conv1d_310" in source
    assert ".item()" not in source
