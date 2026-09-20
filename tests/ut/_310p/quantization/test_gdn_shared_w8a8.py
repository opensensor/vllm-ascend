# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest
import torch

from vllm_ascend._310p.quantization.methods.gdn_shared_w8a8 import (
    AscendGDNW8A8LinearMethod310,
    is_gdn_quantized_projection,
    quantize_weight_per_output_channel,
)

MODULE = "vllm_ascend._310p.quantization.methods.gdn_shared_w8a8"


@pytest.mark.parametrize(
    "suffix",
    [
        "in_proj_qkvz",
        "in_proj_qkv",
        "in_proj_z",
    ],
)
def test_recognizes_gdn_input_projection_layouts(suffix: str) -> None:
    assert is_gdn_quantized_projection(f"model.layers.2.linear_attn.{suffix}")
    assert not is_gdn_quantized_projection(f"model.layers.2.self_attn.{suffix}")


@pytest.mark.parametrize("suffix", ["in_proj_ba", "in_proj_b", "in_proj_a", "out_proj"])
def test_keeps_gdn_gate_projections_unquantized(suffix: str) -> None:
    assert not is_gdn_quantized_projection(f"model.layers.2.linear_attn.{suffix}")


def test_weight_quantization_is_per_output_channel_and_zero_safe() -> None:
    weight = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [-0.5, 0.25, 0.5],
        ],
        dtype=torch.float16,
    )

    quantized, scale = quantize_weight_per_output_channel(weight)

    assert quantized.dtype == torch.int8
    assert torch.equal(quantized[0], torch.tensor([-127, 0, 127], dtype=torch.int8))
    assert torch.equal(quantized[1], torch.zeros(3, dtype=torch.int8))
    assert scale[1] == 1
    reconstructed = quantized.float() * scale.unsqueeze(1)
    assert torch.allclose(reconstructed, weight.float(), atol=0.005, rtol=0)


def test_process_weights_installs_native_dynamic_layout() -> None:
    method = AscendGDNW8A8LinearMethod310()
    layer = torch.nn.Module()
    original = torch.tensor([[-2.0, 2.0], [-1.0, 0.0]], dtype=torch.float16)
    layer.register_parameter("weight", torch.nn.Parameter(original, requires_grad=False))

    with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
        method.process_weights_after_loading(layer)

    assert layer.weight.shape == (2, 2)
    assert torch.equal(
        layer.weight,
        torch.tensor([[-127, -127], [127, 0]], dtype=torch.int8),
    )
    assert layer.weight_scale.dtype == torch.float32
    assert torch.allclose(layer.weight_scale, torch.tensor([2 / 127, 1 / 127]))


def _quantized_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(3, 4, dtype=torch.int8), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(4), requires_grad=False),
    )
    return layer


def test_apply_uses_native_dynamic_quantization() -> None:
    method = AscendGDNW8A8LinearMethod310()
    layer = _quantized_layer()
    hidden_states = torch.ones(2, 3, dtype=torch.float16)
    quantized_hidden_states = torch.ones(2, 3, dtype=torch.int8)
    pertoken_scale = torch.ones(2, dtype=torch.float32)
    expected = torch.full((2, 4), 3.0, dtype=torch.float16)

    with (
        patch(
            f"{MODULE}.torch_npu.npu_dynamic_quant",
            return_value=(quantized_hidden_states, pertoken_scale),
        ) as dynamic_quant,
        patch(
            f"{MODULE}.torch_npu.npu_quant_matmul",
            return_value=expected,
        ) as quant_matmul,
    ):
        output = method.apply(layer, hidden_states)

    dynamic_quant.assert_called_once_with(hidden_states)
    quant_matmul.assert_called_once()
    call = quant_matmul.call_args
    assert call.args[0] is quantized_hidden_states
    assert torch.equal(call.args[1], layer.weight.data)
    assert call.args[2] is layer.weight_scale
    assert call.kwargs["pertoken_scale"] is pertoken_scale
    assert call.kwargs["bias"] is None
    assert call.kwargs["output_dtype"] is torch.float16
    assert output is expected


def test_apply_restores_rank_three_shape() -> None:
    method = AscendGDNW8A8LinearMethod310()
    layer = _quantized_layer()
    hidden_states = torch.ones(2, 1, 3, dtype=torch.float16)

    with (
        patch(
            f"{MODULE}.torch_npu.npu_dynamic_quant",
            return_value=(torch.ones(2, 1, 3, dtype=torch.int8), torch.ones(2, 1)),
        ),
        patch(
            f"{MODULE}.torch_npu.npu_quant_matmul",
            return_value=torch.ones(2, 4, dtype=torch.float16),
        ) as quant_matmul,
    ):
        output = method.apply(layer, hidden_states)

    assert quant_matmul.call_args.args[0].shape == (2, 3)
    assert quant_matmul.call_args.kwargs["pertoken_scale"].shape == (2,)
    assert output.shape == (2, 1, 4)


def test_apply_rejects_bias() -> None:
    method = AscendGDNW8A8LinearMethod310()
    with pytest.raises(ValueError, match="do not support bias"):
        method.apply(torch.nn.Module(), torch.ones(1, 2), torch.ones(2))
