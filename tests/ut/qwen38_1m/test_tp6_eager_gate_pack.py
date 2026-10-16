# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import tp6_eager_gate_pack as binding


@pytest.fixture
def metadata(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    device = type("Device", (), {"type": "npu"})()

    def tensor(shape, dtype):
        return SimpleNamespace(device=device, ndim=len(shape), shape=shape, dtype=dtype, is_contiguous=lambda: True)

    module = SimpleNamespace(
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        native_int4=True,
        top_k=10,
        num_local_experts=85,
        max_routed_rows=128,
        grouped_activation="cann_builtin_fp16",
        grouped_finalize="cann_v2",
    )
    resource = SimpleNamespace(failed=False, _configs={})
    return (
        module,
        tensor((65, 2560), torch.float16),
        tensor((65, 10), torch.float32),
        tensor((65, 10), torch.int64),
        resource,
    )


def test_admission_prevents_capture_training_and_descriptor_exhaustion(metadata, monkeypatch):
    module, inputs, weights, ids, resource = metadata
    assert not binding.admitted(*metadata)
    with torch.inference_mode():
        assert binding.admitted(*metadata)
        resource._configs = {i: None for i in range(binding.MAX_CONFIGS)}
        assert not binding.admitted(*metadata)
        key = ((650, 85, 1280, 2560, 8, 0, 1), inputs.device)
        resource._configs[key] = object()
        assert binding.admitted(*metadata)
        monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
        assert not binding.admitted(*metadata)
    resource.failed = True
    with pytest.raises(RuntimeError, match="failed native"):
        binding.admitted(*metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_int4", False),
        ("top_k", 8),
        ("grouped_finalize", "torch"),
        ("grouped_activation", "torch"),
        ("num_local_experts", 128),
        ("max_routed_rows", 1000),
    ],
)
def test_unsupported_composition_uses_retained_path(metadata, field, value):
    setattr(metadata[0], field, value)
    with torch.inference_mode():
        assert not binding.admitted(*metadata)


def loaded_chunk(namespace):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/w4_moe.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "W4SparseMoE")
    function = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_grouped_chunk"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def test_replacement_retains_dispatch_down_projection_and_finalizer(monkeypatch):
    trace = []
    rows, hidden = 130, 2560
    output = torch.ones(rows, hidden)
    dispatch = SimpleNamespace(
        token_indices=torch.arange(13).repeat_interleave(10),
        order=torch.arange(rows),
        inverse_order=torch.arange(rows),
        route_weights=torch.ones(rows, 1),
        group_list=torch.full((85,), rows),
    )
    down = SimpleNamespace(native_linear=lambda packed, ends: trace.append("down") or output)
    namespace = dict(
        torch=torch,
        build_grouped_expert_dispatch=lambda *args, **kwargs: dispatch,
        pack_activation_device=lambda inputs: (inputs,) * 4,
        finalize_grouped_routes=lambda *args: trace.append("finalizer") or args[0],
    )
    original = loaded_chunk(namespace)
    module = SimpleNamespace(
        compute_dtype=torch.float32,
        num_local_experts=85,
        expert_offset=0,
        grouped_route_count_mode="test",
        native_int4=True,
        top_k=10,
        max_routed_rows=128,
        grouped_finalize="cann_v2",
        projections={"gate_up_proj": object(), "down_proj": down},
    )

    def resource(bank, packed, ends):
        trace.append("fused")
        assert packed[0].shape == (rows, hidden)
        assert ends is dispatch.group_list
        return packed

    resource.failed = False
    monkeypatch.setattr(binding, "admitted", lambda *args: True)
    state = dict(mode="enabled", eligible_calls=0)
    chunk = binding.make_chunk(original, resource, state)
    result = chunk(module, torch.zeros(13, hidden), torch.ones(13, 10), torch.zeros(13, 10, dtype=torch.int64))
    assert result is output and trace == ["fused", "down", "finalizer"]
    assert state["eligible_calls"] == 1


def test_unqualified_loaded_method_is_not_replaced():
    def chunk():
        pass

    with pytest.raises(ValueError, match="unqualified grouped"):
        binding.make_chunk(chunk, None, {})
