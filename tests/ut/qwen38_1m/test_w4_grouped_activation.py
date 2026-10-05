# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host gates for the opt-in Qwen native-W4 grouped SwiGLU pack path."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.qwen38_1m.test_w4_moe import config
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, w4_config
from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND


def model_config(backend=NATIVE_INT4_BACKEND, activation="cann_swiglu_pack"):
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(
        backend=backend,
        group_size=128,
        grouped_activation=activation,
    )
    if backend == NATIVE_INT4_BACKEND:
        cfg.ascend_expert_quantization["activation_quantization"] = "int8_per_group"
    return cfg


def test_grouped_swiglu_pack_requires_native_int4():
    assert w4_config(model_config())["grouped_activation"] == "cann_swiglu_pack"
    with pytest.raises(ValueError, match="grouped SwiGLU pack requires native INT4"):
        w4_config(model_config(backend="cube_310_grouped"))
    with pytest.raises(ValueError, match="grouped_activation"):
        w4_config(model_config(activation="unknown"))


def test_grouped_builtin_swiglu_requires_native_int4():
    assert w4_config(model_config(activation="cann_builtin_fp16"))["grouped_activation"] == "cann_builtin_fp16"
    with pytest.raises(ValueError, match="built-in SwiGLU requires native INT4"):
        w4_config(model_config(backend="cube_310_grouped", activation="cann_builtin_fp16"))


def test_grouped_swiglu_pack_feeds_native_down_projection_without_repacking():
    layer = W4SparseMoE(
        config=model_config(),
        dtype_policy=Qwen4ExpDtypePolicy(),
        expert_sharding=(1, 2),
    )
    tokens = 44  # top-3 expansion exceeds the routed operator's 128-row limit
    routes = tokens * layer.top_k
    inputs = torch.zeros(tokens, 256, dtype=torch.float16)
    weights = torch.full((tokens, layer.top_k), 1 / layer.top_k)
    ids = torch.full((tokens, layer.top_k), layer.expert_offset, dtype=torch.int64)
    packed = (torch.zeros(routes, 256, dtype=torch.float16),)
    gate_up = torch.zeros(routes, 512, dtype=torch.float16)
    with (
        patch("vllm_ascend.models.qwen4_exp.w4_moe.pack_activation_device", side_effect=lambda x: (x,)) as pack,
        patch.object(layer.projections["gate_up_proj"], "native_linear", return_value=gate_up) as gate_projection,
        patch("vllm_ascend.models.qwen4_exp.w4_moe.swiglu_pack_activation_device", return_value=packed) as fuse,
        patch.object(
            layer.projections["down_proj"], "native_linear", return_value=torch.zeros(routes, 256).half()
        ) as down_projection,
        patch.object(layer.projections["down_proj"], "grouped_linear", side_effect=AssertionError("repacked")),
    ):
        output = layer._forward_grouped(inputs, weights, ids)
    assert output.shape == (tokens, 256)
    assert torch.count_nonzero(output) == 0
    pack.assert_called_once()
    gate_projection.assert_called_once()
    fuse.assert_called_once_with(gate_up)
    down_projection.assert_called_once()
    assert down_projection.call_args.args[0] is packed


def test_grouped_builtin_swiglu_feeds_existing_down_projection():
    layer = W4SparseMoE(
        config=model_config(activation="cann_builtin_fp16"),
        dtype_policy=Qwen4ExpDtypePolicy(),
        expert_sharding=(1, 2),
    )
    tokens = 44
    routes = tokens * layer.top_k
    inputs = torch.zeros(tokens, 256, dtype=torch.float16)
    weights = torch.full((tokens, layer.top_k), 1 / layer.top_k)
    ids = torch.full((tokens, layer.top_k), layer.expert_offset, dtype=torch.int64)
    gate_up = torch.zeros(routes, 512, dtype=torch.float16)
    activation = torch.zeros(routes, 256, dtype=torch.float16)
    swiglu = Mock(return_value=activation)
    fake_swiglu = SimpleNamespace(npu_swiglu=swiglu)
    with (
        patch.dict(sys.modules, {"torch_npu": fake_swiglu}),
        patch("vllm_ascend.models.qwen4_exp.w4_moe.pack_activation_device", side_effect=lambda x: (x,)),
        patch.object(layer.projections["gate_up_proj"], "native_linear", return_value=gate_up),
        patch.object(
            layer.projections["down_proj"], "grouped_linear", return_value=torch.zeros_like(activation)
        ) as down,
        patch.object(layer.projections["down_proj"], "native_linear", side_effect=AssertionError("packed path")),
    ):
        output = layer._forward_grouped(inputs, weights, ids)
    assert output.shape == (tokens, 256)
    assert torch.count_nonzero(output) == 0
    swiglu.assert_called_once_with(gate_up, dim=-1)
    assert down.call_args.args[0] is activation


def test_prefill_swiglu_benchmark_plans_selected_chunk_without_npu(tmp_path):
    output = tmp_path / "prefill-swiglu.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.qwen4exp.benchmark_w4_prefill_swiglu_pack_310",
            "--dry-run",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan == {
        "rows": [5120, 15360, 20480],
        "width": 640,
        "trace_rows": 15360,
        "trace_capture": False,
        "npu_used": False,
    }
    assert not output.exists()
