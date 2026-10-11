# SPDX-License-Identifier: Apache-2.0
"""Admission and graph lifetime checks for explicitly qualified local math."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.native_channel_int8 import NativeLocalChannelSwiGLU
from tools.qwen4exp.native_route_control import NativeFixedRouteDispatch, NativeOwnedFinalize


@pytest.fixture
def native_cpu(monkeypatch):
    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("device data read on host"))
    launches = []
    return lambda kernel, args, blocks: launches.append((args, blocks)), launches


def test_fused_activation_retains_device_boundaries_and_graph_descriptor(native_cpu, monkeypatch):
    launch, calls = native_cpu
    operation = NativeLocalChannelSwiGLU(None, launch)
    inputs = torch.ones(60, 1280, dtype=torch.float16)
    ends = torch.zeros(85, dtype=torch.int64)
    codes, scales = operation(inputs, ends)
    args, blocks = calls[-1]
    assert args[0] is inputs and args[1] is ends and blocks == 8
    assert codes.shape == (60, 640) and codes.dtype == torch.int8
    assert scales.shape == (60, 8) and scales.dtype == torch.float32
    assert args[-1].tolist() == [60, 640, 85]
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: True)
    ends.fill_(60)
    operation(inputs, ends)
    assert calls[-1][0][-1] is args[-1]
    with pytest.raises(RuntimeError, match="prewarm"):
        operation(inputs[:30], ends)


@pytest.mark.parametrize("bad", ["dtype", "width", "empty", "ends_dtype", "ends_empty", "strided"])
def test_fused_activation_rejects_invalid_operands(native_cpu, bad):
    launch, calls = native_cpu
    inputs, ends = torch.ones(60, 1280, dtype=torch.float16), torch.zeros(85, dtype=torch.int64)
    if bad == "dtype":
        inputs = inputs.float()
    elif bad == "width":
        inputs = inputs[:, :1278].contiguous()
    elif bad == "empty":
        inputs = inputs[:0]
    elif bad == "ends_dtype":
        ends = ends.int()
    elif bad == "ends_empty":
        ends = ends[:0]
    else:
        inputs = inputs[:, ::2]
    with pytest.raises(ValueError):
        NativeLocalChannelSwiGLU(None, launch)(inputs, ends)
    assert not calls


def test_fixed_dispatch_keeps_live_ids_and_geometry_only_descriptor(native_cpu):
    launch, calls = native_cpu
    operation = NativeFixedRouteDispatch(None, launch)
    ids = torch.zeros(6, 10, dtype=torch.int64)
    tokens, inverse, ends = operation(ids, num_local_experts=85, expert_offset=170)
    args, blocks = calls[-1]
    assert args[0] is ids and blocks == 1
    assert tokens.shape == inverse.shape == (60,)
    assert tokens.dtype == inverse.dtype == torch.int32
    assert ends.shape == (85,) and ends.dtype == torch.int64
    assert args[-1].tolist() == [6, 10, 85, 170]
    ids.fill_(511)
    operation(ids, num_local_experts=85, expert_offset=170)
    assert calls[-1][0][-1] is args[-1]


@pytest.mark.parametrize(
    "shape,dtype,experts,offset",
    [
        ((7, 10), torch.int64, 85, 0),
        ((6, 11), torch.int64, 85, 0),
        ((3, 10), torch.int32, 85, 0),
        ((0, 10), torch.int64, 85, 0),
        ((3, 10), torch.int64, 129, 0),
        ((3, 10), torch.int64, 85, -1),
    ],
)
def test_dispatch_rejects_unsupported_capacity(native_cpu, shape, dtype, experts, offset):
    launch, calls = native_cpu
    with pytest.raises(ValueError):
        NativeFixedRouteDispatch(None, launch)(
            torch.zeros(shape, dtype=dtype), num_local_experts=experts, expert_offset=offset
        )
    assert not calls


def test_finalize_passes_peer_ownership_without_count_read(native_cpu):
    launch, calls = native_cpu
    routed = torch.full((30, 2560), float("nan"), dtype=torch.float16)
    weights = torch.ones(3, 10, dtype=torch.float32)
    inverse, ends = torch.zeros(30, dtype=torch.int32), torch.zeros(85, dtype=torch.int64)
    result = NativeOwnedFinalize(None, launch)(routed, weights, inverse, ends)
    assert result.shape == (3, 2560) and result.dtype == torch.float32
    args, blocks = calls[-1]
    assert args[:4] == [routed, weights, inverse, ends] and blocks == 8
    assert args[-1].tolist() == [3, 10, 2560, 85]


@pytest.mark.parametrize("failure", ["local", "peer", None])
def test_qualification_failure_is_joined_before_any_rank_captures(tmp_path, monkeypatch, failure):
    source = Path("tools/qwen4exp/resident_candidates/channel_local_math.py").read_text()
    tree = ast.parse(source)
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_replacements")
    capture = next(node for node in factory.body if isinstance(node, ast.FunctionDef) and node.name == "capture")
    events = []

    class Layer:
        pass

    layer = Layer()
    runner = SimpleNamespace(model=SimpleNamespace(runnable=SimpleNamespace(modules=lambda: [layer])))
    group = SimpleNamespace(world_size=6, cpu_group=object())
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(get_tp_group=lambda: group))

    def qualify(*args):
        events.append("qualify")
        if failure == "local":
            raise ValueError("mismatched real outputs")
        return {"passed": True}

    def join(errors, error, **kwargs):
        assert kwargs["group"] is group.cpu_group
        events.append("join")
        errors[:] = [error, "peer rejected" if failure == "peer" else None, None, None, None, None]

    scope = {
        "w4_moe": SimpleNamespace(W4SparseMoE=Layer),
        "qualify_real_layer": qualify,
        "old_forward": object(),
        "forward": object(),
        "torch": SimpleNamespace(distributed=SimpleNamespace(all_gather_object=join, get_rank=lambda: 0)),
        "json": json,
        "build": tmp_path,
        "previous_capture": lambda *args, **kwargs: events.append("capture"),
    }
    exec(compile(ast.Module(body=[capture], type_ignores=[]), "<capture-gate>", "exec"), scope)
    if failure:
        with pytest.raises(RuntimeError, match="before graph capture"):
            scope["capture"](runner)
        assert events == ["qualify", "join"]
    else:
        scope["capture"](runner)
        assert events == ["qualify", "join", "capture"]
    assert (tmp_path / "real-layer-rank-0.json").exists()
