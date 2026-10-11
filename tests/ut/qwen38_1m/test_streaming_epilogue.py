# SPDX-License-Identifier: Apache-2.0
"""Complete CPU handoff math and faulted dependency tests; no NPU execution."""

import ast
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.qwen4exp import streaming_epilogue as epilogue

ROOT = Path(__file__).resolve().parents[3]


def pack_reference(hidden):
    groups = hidden.float().reshape(hidden.shape[0], 5, 128)
    maximum = groups.abs().amax(-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    quant = (groups / scale.unsqueeze(-1)).round().clamp(-127, 127)
    high = torch.floor(quant / 16)
    low = quant - 16 * high - 8

    def packed(values):
        unsigned = torch.where(values < 0, values + 16, values)
        return (unsigned[..., ::2] + 16 * unsigned[..., 1::2]).to(torch.int8).reshape(hidden.shape[0], 320).contiguous()

    return (
        packed(low),
        packed(high),
        scale.unsqueeze(-1).expand(-1, -1, 8).contiguous(),
        quant.sum(-1).unsqueeze(-1).expand(-1, -1, 8).contiguous(),
    )


def independent_pack(hidden):
    values = hidden.numpy().astype(np.float32).reshape(hidden.shape[0], 5, 128)
    maximum = np.max(np.abs(values), axis=-1)
    scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(127))
    quant = np.clip(np.rint(values / scale[..., None]), -127, 127).astype(np.int32)
    high = np.floor_divide(quant, 16)
    low = quant - high * 16 - 8

    def packed(code):
        return ((code[..., ::2] & 15) | ((code[..., 1::2] & 15) << 4)).astype(np.uint8).view(np.int8).reshape(-1, 320)

    return (
        torch.from_numpy(packed(low)),
        torch.from_numpy(packed(high)),
        torch.from_numpy(np.repeat(scale[..., None], 8, axis=-1)),
        torch.from_numpy(np.repeat(quant.sum(-1).astype(np.float32)[..., None], 8, axis=-1)),
    )


def unpack(prepared):
    def signed(packed):
        unsigned = packed.to(torch.int32) & 255
        codes = torch.stack((unsigned & 15, unsigned >> 4), -1).flatten(-2)
        return torch.where(codes >= 8, codes - 16, codes)

    return signed(prepared[0]) + 16 * signed(prepared[1]) + 8


def projection_integer_reference(quant, scales, weight, live_rows):
    # Integer dots are exact. Each G128 correction is accumulated in baseline
    # order; shape-dependent floating GEMM rounding cannot mask window errors.
    output = torch.zeros((quant.shape[0], weight.shape[1]), dtype=torch.float32)
    for group in range(5):
        first = group * 128
        dot = quant[:live_rows, first : first + 128] @ weight[first : first + 128]
        output[:live_rows] += dot.float() * scales[:live_rows, group, :1] * 0.001
    return output.half()


def combine_reference(routed, dispatch, weights):
    # CPU surrogate of the selected CANN-v2 boundaries: input/scale FP16,
    # stable original route order, combined FP16 then widen to FP32. This is
    # not a claim that CPU models reproduce the native finalizer implementation.
    ordered = routed.index_select(0, dispatch.inverse_order).reshape(weights.shape[0], weights.shape[1], -1)
    rounded_weights = weights.half().float()
    output = torch.zeros((weights.shape[0], routed.shape[1]), dtype=torch.float32)
    for route in range(weights.shape[1]):
        output += ordered[:, route].float() * rounded_weights[:, route, None]
    return output.half().float()


def fixtures(tokens=2, top_k=3, live_rows=5, broken=None, fail=None):
    rows = tokens * top_k
    generator = torch.Generator().manual_seed(14)
    projected = (torch.randn(rows, 1280, generator=generator) * 0.5).half()
    projected[live_rows:] = 0
    weights = torch.randn(tokens, top_k, generator=generator) * 0.2
    inverse = torch.arange(rows - 1, -1, -1)
    dispatch = SimpleNamespace(inverse_order=inverse)
    ends = torch.tensor([live_rows], dtype=torch.int64)
    weight = ((torch.arange(640, dtype=torch.int32)[:, None] * 3 + torch.arange(2560)[None, :]) % 16 - 8).int()
    calls, pack_values = [], []

    def activation(value):
        calls.append("activation")
        gate, up = value.chunk(2, -1)
        hidden = (F.silu(gate) * up).half().contiguous()
        return hidden.float() if broken == "activation" else hidden

    def pack(hidden, same_ends):
        assert same_ends is ends
        calls.append("pack")
        value = list(pack_reference(hidden))
        pack_values.append(tuple(value))
        if broken == "packed":
            value[0] = value[0].float()
        return value

    def columns(bank, prepared, same_ends, first, count):
        assert bank is weight and same_ends is ends
        calls.append(("columns", first, count))
        output = projection_integer_reference(
            unpack(prepared), prepared[2], bank[:, first * 128 : (first + count) * 128], live_rows
        )
        return output[:, :-1] if broken == "columns" else output

    def finalize(routed, same_dispatch, same_weights):
        assert same_dispatch is dispatch and same_weights is weights
        calls.append("finalize")
        result = combine_reference(routed, dispatch, weights)
        return result.half() if broken == "finalize" else result

    def complete(stage, tensor):
        calls.append(stage)
        if stage == fail:
            raise RuntimeError("dependency failure")
        return ("CPU surrogate ordered", stage)

    return projected, dispatch, weights, ends, weight, activation, pack, columns, finalize, complete, calls, pack_values


def run(values, plan=None):
    projected, dispatch, weights, ends, bank, activation, pack, columns, finalize, complete, _, _ = values
    return epilogue.run_streaming_epilogue(
        projected,
        dispatch,
        weights,
        ends,
        bank,
        activation=activation,
        pack=pack,
        columns=columns,
        finalize=finalize,
        complete=complete,
        plan=plan,
    )


@pytest.mark.parametrize("tokens,top_k,live_rows", [(1, 1, 1), (2, 3, 5), (2, 3, 0), (2, 3, 6)])
@pytest.mark.parametrize("tiles_per_window", [1, 4, 8])
def test_full_pipeline_exact_independent_packed_bytes_and_all_columns(tokens, top_k, live_rows, tiles_per_window):
    values = fixtures(tokens, top_k, live_rows)
    projected, dispatch, weights, _, bank, _, _, _, _, _, calls, packed = values
    gate, up = projected.chunk(2, -1)
    hidden = (F.silu(gate) * up).half().contiguous()
    expected_pack = independent_pack(hidden)
    expected_routed = projection_integer_reference(unpack(expected_pack), expected_pack[2], bank, live_rows)
    expected = combine_reference(expected_routed, dispatch, weights)
    result = run(values, epilogue.WindowPlan(tiles_per_window=tiles_per_window))
    assert torch.equal(result.output, expected)
    assert len(packed) == 1 and calls.count("pack") == 1 and calls.count("activation") == 1
    for left, right in zip(packed[0], expected_pack):
        assert torch.equal(left, right)
    assert result.boundary_epochs[-1] == "complete_output"
    assert result.output.shape == (tokens, 2560) and result.output.dtype == torch.float32
    bounds = dict(result.logical_buffer_bounds)
    assert bounds["routed_window_fp16"] <= tokens * top_k * 1024 * 2
    expected_windows = [("columns", window.first_tile, window.tile_count) for window in result.windows]
    assert [call for call in calls if isinstance(call, tuple)] == expected_windows


def test_default_8_8_4_windows_preserve_column_ownership():
    plan = epilogue.WindowPlan()
    assert [(window.first_tile, window.tile_count) for window in plan.windows] == [(0, 8), (8, 8), (16, 4)]
    assert [(window.first_column, window.columns) for window in plan.windows] == [(0, 1024), (1024, 1024), (2048, 512)]


def test_growing_shrinking_and_all_peer_never_return_stale_columns():
    for live_rows in (6, 1, 0, 5):
        values = fixtures(live_rows=live_rows)
        result = run(values)
        if live_rows == 0:
            assert torch.count_nonzero(result.output) == 0
        else:
            assert torch.count_nonzero(result.output) > 0


def test_empty_skips_every_callback_and_has_zero_output_rows():
    values = fixtures(tokens=0, top_k=3, live_rows=0)
    result = run(values)
    assert result.output.shape == (0, 2560) and result.windows == () and result.boundary_epochs == ()
    assert values[-2] == []


@pytest.mark.parametrize("broken", ["activation", "packed", "columns", "finalize"])
def test_callback_dtype_shape_failures_do_not_expose_partial_output(broken):
    with pytest.raises(ValueError):
        run(fixtures(broken=broken))


@pytest.mark.parametrize(
    "stage,forbidden",
    [
        ("projected_gate_up", "activation"),
        ("builtin_activation", "pack"),
        ("packed_hidden", ("columns", 0, 8)),
        ("routed_window_0", "finalize"),
        ("finalized_window_0", "stored_window_0"),
        ("stored_window_0", ("columns", 8, 8)),
    ],
)
def test_failed_dependency_prevents_consumers_and_slot_reuse(stage, forbidden):
    values = fixtures(fail=stage)
    with pytest.raises(RuntimeError, match="dependency failure"):
        run(values)
    assert forbidden not in values[-2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiles_per_window": 0},
        {"tiles_per_window": 9},
        {"output_columns": 257},
        {"output_columns": 2688},
        {"activation_policy": "fp32_swiglu"},
        {"finalizer_policy": "torch"},
    ],
)
def test_unqualified_arithmetic_and_invalid_windows_rejected(kwargs):
    with pytest.raises(ValueError):
        epilogue.WindowPlan(**kwargs)


def test_logical_maximum_route_window_allocation_reduced_without_speed_claim():
    bounds = dict(epilogue.epilogue_buffer_bounds(25600, 2560, epilogue.WindowPlan()))
    assert bounds["routed_window_fp16"] == 52_428_800
    assert bounds["routed_window_fp16"] < 25600 * 2560 * 2
    assert bounds["projected_gate_up_fp16"] == 65_536_000
    assert bounds["builtin_activation_fp16"] == 32_768_000
    assert bounds["packed_hidden_scale_sum"] == 8_192_000


def test_native_column_bounds_compile_on_cpu(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "columns.cpp"
    source.write_text(r"""
#include "qwen_streaming_epilogue.h"
using namespace qwen_streaming;
static_assert(ColumnWindowContract{0,8}.Valid(2560));
static_assert(ColumnWindowContract{8,8}.FirstColumn()==1024);
static_assert(ColumnWindowContract{16,4}.Columns()==512);
static_assert(ColumnWindowContract{0,8}.RoutedBytes(25600)==52428800);
static_assert(!ColumnWindowContract{19,2}.Valid(2560));
static_assert(!ColumnWindowContract{0,9}.Valid(2560));
static_assert(!ColumnWindowContract{0,0}.Valid(2560));
static_assert(!ColumnWindowContract{0,8}.Valid(2559));
static_assert(ColumnWindowContract{0,8}.GateColumn(639)==639);
static_assert(ColumnWindowContract{0,8}.UpColumn(639)==1279);
static_assert(ColumnWindowContract{0,8}.OutputElement(2,1023)==3071);
int main(){return 0;}
""")
    executable = tmp_path / "check"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT / "tools/qwen4exp"), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True)


def test_no_tensor_value_controls_or_precision_substitution():
    path = ROOT / "tools/qwen4exp/streaming_epilogue.py"
    tree = ast.parse(path.read_text())
    forbidden = {"cpu", "item", "tolist", "nonzero", "numpy", "synchronize", "silu", "sigmoid"}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in forbidden]
    assert "torch_npu" not in path.read_text()


def test_cancellation_sensitive_original_route_order_is_preserved():
    routed = torch.tensor([[32768.0], [0.001], [-32768.0], [0.001]], dtype=torch.float16)
    weights = torch.ones((1, 4), dtype=torch.float32)
    identity = SimpleNamespace(inverse_order=torch.arange(4))
    reordered = SimpleNamespace(inverse_order=torch.tensor([0, 2, 1, 3]))
    # A different addition order is observably different even after FP16 final
    # rounding. Windowing must reuse dispatch, not construct a new route order.
    original = combine_reference(routed, identity, weights)
    changed = combine_reference(routed, reordered, weights)
    assert not torch.equal(original, changed)


@pytest.mark.parametrize(
    "corruption", ["projected_dtype", "projected_shape", "ends_dtype", "inverse_rows", "callbacks"]
)
def test_initial_contract_failures_submit_no_pipeline_stage(corruption):
    values = list(fixtures())
    if corruption == "projected_dtype":
        values[0] = values[0].float()
    elif corruption == "projected_shape":
        values[0] = values[0][:, :-1].contiguous()
    elif corruption == "ends_dtype":
        values[3] = values[3].int()
    elif corruption == "inverse_rows":
        values[1] = SimpleNamespace(inverse_order=values[1].inverse_order[:-1])
    else:
        values[9] = None
    with pytest.raises(ValueError):
        run(values)
    assert values[-2] == []


def test_windowing_exposes_extra_launches_and_unchanged_down_payload():
    value = epilogue.epilogue_logical_cost(25600, 2560)
    assert value["projection_calls"] == value["finalizer_calls"] == 3
    assert value["routed_down_write_bytes"] == value["routed_finalizer_read_bytes"] == 131072000
    assert value["combined_window_store_read_write_bytes"] == 52428800
    assert value["finalizer_cast_write_bytes"] == 460800
    assert not value["measured_bus_bytes"] and value["predicted_speed_multiplier"] is None


def test_v2_windows_bound_bulk_storage_with_two_finalizers():
    from tools.qwen4exp.streaming_epilogue import WindowPlan, epilogue_logical_cost

    plan = WindowPlan(tile_columns=160)
    assert [(w.first_column, w.columns) for w in plan.windows] == [(0, 1280), (1280, 1280)]
    cost = epilogue_logical_cost(25600, 2560, plan)
    assert cost["projection_calls"] == cost["finalizer_calls"] == 2
    assert cost["logical_buffer_bounds"]["routed_window_fp16"] == 25600 * 1280 * 2
    assert not cost["measured_bus_bytes"] and not cost["hardware_validated"]
