# SPDX-License-Identifier: Apache-2.0
"""Execute native vector bodies on CPU; no device/event/performance claims."""

import ast
import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tools.qwen4exp import native_prefill
from tools.qwen4exp.native_prefill import NativeLocalRouteGather, NativeWY
from tools.qwen4exp.speculation_sweep import compare, profiles
from tools.qwen4exp.stage_local_swiglu import source as swiglu_source
from vllm_ascend.core.qwen_prefill_pacing import PrefillPacingConfig, PrefillPacingPolicy
from vllm_ascend.models.qwen4_exp import model

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cpu_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen-prefill-cpu")
    wrapper = directory / "cpu.cpp"
    wrapper.write_text(
        '#include "kernel_operator.h"\n'
        f'#include "{ROOT / "tools/qwen4exp/native_wy.cpp"}"\n'
        f'#include "{ROOT / "tools/qwen4exp/native_route_gather.cpp"}"\n'
        'extern "C" void cpu_wy(void* k, void* v, void* gram, void* g, void* beta, void* w, void* u, void* config) {\n'
        "for (int i=0;i<8;++i) { AscendC::blockIndex=i; qwen_fused_wy_v1(k,v,gram,g,beta,w,u,config); }}\n"
        'extern "C" void cpu_routes(void* l, void* h, void* s, void* t, void* rows, void* ends,\n'
        "void* ol, void* oh, void* os, void* out_total, void* config) {\n"
        "for (int i=0;i<8;++i) { AscendC::blockIndex=i;\n"
        "qwen_local_route_gather_v1(l,h,s,t,rows,ends,ol,oh,os,out_total,config); }}\n"
    )
    library = directory / "cpu.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-ffp-contract=off",
            f"-I{ROOT / 'tests/ut/qwen38_1m/prefill_cpu_stubs'}",
            str(wrapper),
            "-o",
            str(library),
        ],
        check=True,
    )
    value = ctypes.CDLL(str(library))
    value.cpu_wy.argtypes = [ctypes.c_void_p] * 8
    value.cpu_routes.argtypes = [ctypes.c_void_p] * 11
    value.cpu_wy.restype = value.cpu_routes.restype = None
    return value


def launch_for(function):
    def launch(kernel, tensors, blocks):
        assert blocks == 8
        function(*[tensor.data_ptr() for tensor in tensors])

    return launch


def wy_reference(q, k, v, g, beta):
    path = ROOT / "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py"
    names = {
        "_expand_qk_to_v_heads",
        "_upper_incl_diag_mask",
        "_strictly_lower_decay",
        "_inv_small_unit_lower",
        "_inv_unit_lower_triangular",
        "_inv_unit_lower_recursive",
        "_ut_transform",
        "_compute_kernel_inputs_from_torch_wy",
    }
    nodes = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = {
        "torch": torch,
        "_WY_GROUPED_GRAM": True,
        "_UT_USE_BLOCKED_INVERSE": True,
        "_UT_INVERSE_BLOCK": 8,
        "_DECAY_MASK_CACHE": {},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["_compute_kernel_inputs_from_torch_wy"](q, k, v, g, beta, 64)


@pytest.mark.parametrize("key_heads,value_heads", [(1, 1), (4, 12), (16, 48)])
@pytest.mark.parametrize("seed", [0, 13, 42])
@pytest.mark.parametrize("beta_dtype", [torch.float16, torch.float32])
def test_native_wy_body_and_layout_match_blocked_reference(
    monkeypatch, cpu_library, key_heads, value_heads, seed, beta_dtype
):
    monkeypatch.setattr(native_prefill, "_on_npu", lambda d: True)
    monkeypatch.setattr(native_prefill, "_capturing", lambda d: False)
    torch.manual_seed(seed)
    batch = 2 if seed == 42 and key_heads == 4 else 1
    q = F.normalize(torch.randn(batch, 128, key_heads, 128), dim=-1).half()
    k = F.normalize(torch.randn_like(q.float()), dim=-1).half()
    v = torch.randn(batch, 128, value_heads, 128).half()
    g = F.logsigmoid(torch.randn(batch, 128, value_heads))
    beta = torch.rand_like(g).to(beta_dtype)
    native = NativeWY(None, launch_for(cpu_library.cpu_wy))
    actual = native(q, k, v, g, beta, 64)
    expected = wy_reference(q, k, v, g, beta)
    for index, (want, got) in enumerate(zip(expected, actual)):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert torch.isfinite(got).all()
        if index in (0, 1, 4):
            assert torch.equal(got, want)
        else:
            # FP32 substitution and blocked inverse change summation order.
            # This is a component gate, not a full-model quality tolerance.
            torch.testing.assert_close(got.float(), want.float(), rtol=0.002, atol=0.0005)
            cosine = F.cosine_similarity(got.double().flatten(), want.double().flatten(), dim=0)
            assert cosine > 0.999999


@pytest.mark.parametrize("active", [0, 1, 7, 19, 40])
def test_route_gather_only_reads_local_prefix_and_preserves_quantized_bytes(monkeypatch, cpu_library, active):
    monkeypatch.setattr(native_prefill, "_on_npu", lambda d: True)
    monkeypatch.setattr(native_prefill, "_capturing", lambda d: False)
    rows, routes, width = 4, 40, 1280
    low, high = (torch.randint(-128, 128, (rows, width), dtype=torch.int8) for _ in range(2))
    scale, total = (torch.randn(rows, width // 64, 8) for _ in range(2))
    indices = torch.arange(routes, dtype=torch.int32).remainder(rows)
    # Invalid indices in inactive peer rows prove the kernel does not consume
    # them, rather than copying and masking the result later.
    indices[active:] = 1 << 29
    ends = torch.tensor([0, active // 2, active], dtype=torch.int64)

    def launch(kernel, tensors, blocks):
        for tensor in tensors[6:10]:
            tensor.fill_(57)
        launch_for(cpu_library.cpu_routes)(kernel, tensors, blocks)

    resource = NativeLocalRouteGather(None, launch)
    prepared = (low, high, scale, total)
    actual = resource(prepared, indices, ends)
    for got, operand in zip(actual, prepared):
        assert torch.equal(got[:active], operand.index_select(0, indices[:active].long()))
        assert (got[active:] == 57).all(), "inactive capacity must never be written"


def test_paged_native_prefill_is_explicit_and_rejects_unused_gather_options():
    config = SimpleNamespace(ascend_qsa_prefill={"backend": "paged_native"})
    assert model._qsa_prefill_policy(config)[0] == "paged_native"
    assert model._qsa_prefill_policy(config)[2] is False
    assert model._qsa_prefill_policy(SimpleNamespace())[0] == "batched_gather"
    for field in ("query_tile", "parallel_gather"):
        config.ascend_qsa_prefill[field] = 8 if field == "query_tile" else False
        with pytest.raises(ValueError, match="does not use"):
            model._qsa_prefill_policy(config)
        del config.ascend_qsa_prefill[field]


def test_swiglu_staging_keeps_quantizer_body_unchanged_and_adds_device_end():
    original = (ROOT / "csrc/gmm/qwen_w4_a8_swiglu_pack_v310/op_kernel/qwen_w4_a8_swiglu_pack_v310.cpp").read_text()
    staged = swiglu_source(ROOT)
    assert original[: original.index('extern "C"')] == staged[: staged.index('extern "C"')]
    assert "ends.GetValue(c[1] - 1)" in staged
    assert "activeRows, c[0]" in staged


@pytest.mark.parametrize(
    "options",
    [
        {"alignment": 0},
        {"min_tokens": True},
        {"initial_tokens": 129},
        {"target_step_ms": float("nan")},
        {"smoothing": 0},
        {"max_tokens": 128},
    ],
)
def test_pacing_rejects_invalid_config(options):
    with pytest.raises(ValueError):
        PrefillPacingConfig(**options)


def test_pacing_adapts_clamps_aligns_and_ignores_invalid_observations():
    policy = PrefillPacingPolicy(PrefillPacingConfig())
    assert policy.budget(2560) == 256
    policy.observe(256, 2000)
    assert policy.budget(2560) == 128
    before = policy.ms_per_token
    policy.observe(0, 200)
    policy.observe(256, float("nan"))
    assert policy.ms_per_token == before
    fast = PrefillPacingPolicy(PrefillPacingConfig(smoothing=1))
    fast.observe(2560, 100)
    assert fast.budget(2560) == 2560
    assert fast.budget(64) == 64


def records(repeats=3):
    return [
        {
            "draft_length": depth,
            "concurrency": 1,
            "repeat": repeat,
            "request": {"model": f"arm{depth}", "max_tokens": 1024, "temperature": 0},
            "completion_tokens": 1024,
            "decode_seconds": 30 + 10 * depth,
            "ttft_seconds": 2,
            "sustained_seconds": 600,
            "max_core_c": 92,
            "energy_joules": None,
            "quality_pass": True,
            "image_pass": True,
            "thermal_shutdown": False,
            "thermal_policy_pass": True,
        }
        for depth in (0, 1, 2)
        for repeat in range(repeats)
    ]


def test_speculation_profiles_never_start_and_keep_graphs_consistent():
    plan = profiles()
    assert plan["autostart"] is False and plan["preserve_image_enabled"] is True
    for depth, arm in enumerate(plan["arms"]):
        assert arm["compilation_config"]["cudagraph_capture_sizes"] == [depth + 1, 3 * (depth + 1)]
    assert plan["arms"][0]["speculative_config"] is None


def test_speculation_requires_matched_repeats_sustained_and_thermal_quality():
    assert compare(records())["comparisons"][0]["recommended_draft_length"] == 0
    assert compare(records(1))["comparisons"][0]["recommended_draft_length"] is None
    rows = records()
    for row in rows:
        row["thermal_policy_pass"] = False
    assert compare(rows)["comparisons"][0]["recommended_draft_length"] is None
    with pytest.raises(ValueError, match="matched"):
        compare(records()[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        compare(records() + records()[:1])
    with pytest.raises(ValueError, match="matched"):
        rows = records()
        rows[0]["request"]["temperature"] = 0.7
        compare(rows)


def test_speculation_prefers_wall_throughput_including_holds_and_all_completed_rounds():
    rows = records()
    for row in rows:
        row.update(decode_rounds=2, completion_tokens=2048, wall_seconds=100 - 20 * row["draft_length"])
    assert compare(rows)["comparisons"][0]["recommended_draft_length"] == 2
    rows[0].pop("wall_seconds")
    with pytest.raises(ValueError, match="timing basis"):
        compare(rows)


@pytest.mark.parametrize("seed", [3, 15])
def test_native_wy_recurrent_outputs_and_fp32_state_over_four_chunks(monkeypatch, cpu_library, seed):
    monkeypatch.setattr(native_prefill, "_on_npu", lambda d: True)
    monkeypatch.setattr(native_prefill, "_capturing", lambda d: False)
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(1, 256, 4, 128), dim=-1).half()
    k = F.normalize(torch.randn_like(q.float()), dim=-1).half()
    v = torch.randn(1, 256, 12, 128).half()
    g = F.logsigmoid(torch.randn(1, 256, 12))
    beta = torch.rand_like(g)
    # Exercise zero padding after a real partial final chunk.
    for tensor in (q, k, v, g, beta):
        tensor[:, -19:] = 0
    native = NativeWY(None, launch_for(cpu_library.cpu_wy))
    candidate = native(q, k, v, g, beta, 64)
    reference = wy_reference(q, k, v, g, beta)
    initial = torch.randn(1, 12, 128, 128) * 0.01

    def recur(inputs):
        query, key, w, u, cumulative = inputs
        query, key = (tensor.repeat_interleave(3, dim=1).float() for tensor in (query, key))
        state = initial.clone()
        outputs = []
        for start in range(0, 256, 64):
            stop = start + 64
            decay = cumulative[:, :, start:stop]
            keys, queries = key[:, :, start:stop], query[:, :, start:stop] / (128**0.5)
            updated = u[:, :, start:stop].float() - w[:, :, start:stop].float() @ state
            causal = (queries @ keys.transpose(-1, -2)) * (
                decay[..., :, None] - decay[..., None, :]
            ).tril().exp().tril()
            outputs.append((queries @ state) * decay.exp()[..., None] + causal @ updated)
            state = (
                state * decay[..., -1:].exp()[..., None]
                + (keys * (decay[..., -1:] - decay).exp()[..., None]).transpose(-1, -2) @ updated
            )
        return torch.cat(outputs, dim=2), state

    for actual, expected in zip(recur(candidate), recur(reference)):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.001)
        assert F.cosine_similarity(actual.double().flatten(), expected.double().flatten(), dim=0) > 0.99999


def test_native_config_reuse_capture_and_dtype_guards(monkeypatch):
    resource = NativeLocalRouteGather(None, None)
    device = torch.device("cpu")
    first = resource.config((128, 16, 4), device)
    monkeypatch.setattr(native_prefill, "_capturing", lambda d: True)
    assert resource.config((128, 16, 4), device) is first
    with pytest.raises(RuntimeError, match="prewarm"):
        resource.config((128, 16, 5), device)
    q = torch.zeros(1, 64, 4, 128).half()
    v = torch.zeros(1, 64, 12, 128).half()
    g = torch.zeros(1, 64, 12)
    with pytest.raises(ValueError, match="dtype"):
        NativeWY(None, None)(q, q, v, g.half(), g, 64)
    with pytest.raises(ValueError, match="one NPU"):
        NativeWY(None, None)(q, q, v, g, g, 64)


@pytest.mark.parametrize("existed", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_route_binding_restores_owner_on_success_and_failure(existed, fail):
    path = ROOT / "tools/qwen4exp/resident_candidates/local_routes.py"
    factory = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    factory.body = [n for n in factory.body if not isinstance(n, ast.ImportFrom)]
    resource = SimpleNamespace(gather=Mock(), swiglu=Mock())

    def original(self, *args):
        self._qwen_local_routes.gather()
        if fail:
            raise RuntimeError("projection failed")
        return "result"

    scope = {
        "SimpleNamespace": SimpleNamespace,
        "RESOURCE_NAME": "qwen_prefill_v1",
        "W4SparseMoE": SimpleNamespace(_forward_grouped_chunk=original),
    }
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(path), "exec"), scope)
    forward = next(
        iter(
            scope["replacements"](
                {"qwen_prefill_v1": {"route_gather": resource.gather, "local_swiglu": resource.swiglu}}
            ).values()
        )
    )
    owner = SimpleNamespace(native_int4=True, grouped_activation="cann_swiglu_pack")
    if existed:
        owner._qwen_local_routes = None
    if fail:
        with pytest.raises(RuntimeError, match="projection failed"):
            forward(owner, None, None, None)
    else:
        assert forward(owner, None, None, None) == "result"
    assert hasattr(owner, "_qwen_local_routes") is existed
    if existed:
        assert owner._qwen_local_routes is None
    resource.gather.assert_called_once()


def test_failed_mtp_arm_invalidates_selection_even_after_successful_records():
    with pytest.raises(ValueError, match="failed arm"):
        compare(records() + [{"arm_failed": True}])


def test_mtp_hard_thermal_limit_cannot_be_overridden_by_receipt_flag():
    rows = records()
    for row in rows:
        row["max_core_c"] = 96
    assert compare(rows)["comparisons"][0]["recommended_draft_length"] is None
