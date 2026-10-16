# SPDX-License-Identifier: Apache-2.0
"""Portable contracts/fallback tests; these do not execute AscendC kernels."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.mhc_probe import compare, summarize
from tools.sidecard.npu_mhc import (
    NUMERICAL_CONTRACT,
    NativeMHC,
    admissible,
    ordered_post_reference,
    post_reference,
    pre_reference,
    prepare_weights,
    validate_eps,
)


@pytest.mark.parametrize("rows", [1, 4, 32])
@pytest.mark.parametrize("sign", [-1, 1])
def test_post_range_orientation_and_negative_gates(rows, sign):
    # Binary fractions and integers avoid accidental order-dependent FP32 loss.
    residual = sign * (torch.arange(4 * 4096).reshape(1, 4, 4096).float() + 100000)
    residual = residual.expand(rows, -1, -1).contiguous()
    matrix = torch.tensor([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0.5], [0.25, 0, 0, 0]])
    comb = matrix.expand(rows, -1, -1).contiguous()
    x = torch.full((rows, 4096), 8, dtype=torch.float16)
    post = torch.tensor([-1, 0, 2, -0.5])[None, :, None].expand(rows, -1, -1).contiguous()
    got = ordered_post_reference(x, residual, post, comb, torch)
    wanted = (
        torch.stack((residual[:, 3] * 0.25 - 8, residual[:, 0], -residual[:, 1] + 16, residual[:, 2] * 0.5 - 4), 1)
        .bfloat16()
        .float()
    )
    assert torch.equal(got, wanted)
    assert torch.equal(got, post_reference(x, residual, post, comb, torch))
    assert got.dtype == torch.float32 and got.shape == (rows, 4, 4096)
    assert bool(got.isfinite().all()) and bool((got.abs() > 65504).any())


@pytest.mark.parametrize("rows", [1, 4, 32])
def test_pre_all_fields_preserve_bf16_range_and_norm(rows):
    residual = torch.tensor([100000, -90000, 80000, -70000], dtype=torch.float32)[None, :, None]
    residual = residual.expand(rows, -1, 4096).contiguous()
    fn = torch.zeros(24, 16384)
    scale = torch.tensor([-2, 3, -4], dtype=torch.float32)
    base = torch.linspace(-2, 2, 24)
    norm = torch.linspace(0.5, 1.5, 4096).half()
    post, logits, hidden = pre_reference(residual, fn, scale, base, norm, 1e-6, 1e-6, torch)
    pre = torch.sigmoid(base[:4]) + 1e-6
    rounded = (residual * pre[None, :, None]).sum(1).bfloat16().float()
    wanted = (rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6) * norm.float()).half()
    assert torch.equal(hidden, wanted)
    assert torch.equal(logits, base[8:].reshape(1, 4, 4).expand(rows, -1, -1))
    assert torch.equal(post, (2 * torch.sigmoid(base[4:8])).bfloat16().float()[None, :, None].expand(rows, -1, -1))
    assert post.dtype == logits.dtype == torch.float32 and hidden.dtype == torch.float16
    assert bool(hidden.isfinite().all())


def test_pre_changed_projection_matrix_affects_the_correct_logit_and_gate():
    residual = torch.tensor([1, -2, 3, -4], dtype=torch.float32)[None, :, None].expand(1, 4, 4096).contiguous()
    fn = torch.zeros(24, 16384)
    fn[0, 0], fn[4, 4096], fn[8, 8192], fn[9, 12288] = 1, -1, 2, -2
    scale = torch.tensor([-0.5, 0.25, 2], dtype=torch.float32)
    base = torch.linspace(-0.25, 0.25, 24)
    norm = torch.ones(4096).half()
    post, logits, hidden = pre_reference(residual, fn, scale, base, norm, 1e-6, 1e-6, torch)
    inverse = torch.rsqrt(residual.square().mean() + 1e-6)
    expected_logits = base[8:].clone()
    expected_logits[0] += 6 * inverse * scale[2]
    expected_logits[1] += 8 * inverse * scale[2]
    assert torch.equal(logits.flatten(), expected_logits)
    expected_post = 2 * torch.sigmoid(base[4:8] + torch.tensor([2, 0, 0, 0]) * inverse * scale[1])
    assert torch.equal(post.flatten(), expected_post.bfloat16().float())
    assert bool(hidden.isfinite().all())


@pytest.mark.parametrize("eps", [0, -1, True, float("nan"), float("inf"), 1e40, 1e-80])
def test_reject_invalid_float_attributes(eps):
    with pytest.raises(ValueError):
        validate_eps(eps, 1e-6)


def test_fixed_contract_catches_each_changed_output():
    expected = (torch.ones(1, 4, 1), torch.eye(4)[None], torch.ones(1, 4096).half(), torch.zeros(1, 4, 4))
    assert compare(expected, expected, torch)["passed"]
    for index in range(4):
        changed = [value.clone() for value in expected]
        changed[index].flatten()[0] += 1
        assert not compare(tuple(changed), expected, torch)["passed"]
    changed = [value.clone() for value in expected]
    changed[0].flatten()[0] = float("nan")
    assert not compare(tuple(changed), expected, torch)["passed"]
    assert NUMERICAL_CONTRACT["raw_logits"] == {"atol": 2e-4, "rtol": 2e-5}


def fake_residual(rows=1, *, device="npu", contiguous=True, width=4096):
    return SimpleNamespace(
        device=SimpleNamespace(type=device),
        shape=(rows, 4, width),
        dtype=torch.float32,
        is_contiguous=lambda: contiguous,
    )


@pytest.mark.parametrize(
    "residual",
    [fake_residual(device="cuda"), fake_residual(rows=4), fake_residual(contiguous=False), fake_residual(width=128)],
)
def test_unqualified_unsupported_shapes_use_explicit_fallback(residual):
    def unexpected(*args):
        raise AssertionError("unqualified operator was called")

    policy = NativeMHC(unexpected, unexpected, unexpected, pre_rows=frozenset({1}), post_rows=frozenset({1}))
    assert policy.pre(residual, (), 1e-6, 1e-6, 20, torch, lambda: "eager-pre") == "eager-pre"
    assert policy.post(None, residual, None, None, lambda: "eager-post") == "eager-post"


def test_no_rows_enabled_by_default_and_device_errors_propagate():
    def fail(*args):
        raise RuntimeError("issued device operation failed")

    policy = NativeMHC(fail, fail, fail)
    assert policy.post(None, fake_residual(), None, None, lambda: "eager") == "eager"
    policy = NativeMHC(fail, fail, fail, post_rows=frozenset({1}))
    with pytest.raises(RuntimeError, match="issued device operation"):
        policy.post(None, fake_residual(), None, None, lambda: "unsafe-retry")


def test_pre_retains_qualified_sinkhorn_and_rounds_comb_only_after_it():
    post, logits, hidden = torch.ones(1, 4, 1), torch.arange(16).reshape(1, 4, 4).float(), torch.ones(1, 4096).half()
    calls = []

    def sinkhorn(value, iterations, eps):
        calls.append((value, iterations, eps))
        return value / 13

    policy = NativeMHC(lambda *args: (post, logits, hidden), lambda *args: None, sinkhorn, pre_rows=frozenset({1}))
    result = policy.pre(fake_residual(), (), 1e-6, 1e-6, 20, torch, lambda: None)
    assert result[0] is post and result[2] is hidden
    assert torch.equal(result[1], (logits / 13).bfloat16().float())
    assert len(calls) == 1 and calls[0][0] is logits and calls[0][1:] == (20, 1e-6)


@pytest.mark.parametrize("rows", [{1}, frozenset({True}), frozenset({0}), frozenset({33})])
def test_policy_requires_immutable_validated_row_sets(rows):
    with pytest.raises(ValueError):
        NativeMHC(lambda: None, lambda: None, lambda: None, pre_rows=rows)


@pytest.mark.parametrize(
    "kind,norm_name", [("attn", "input_layernorm.weight"), ("ffn", "post_attention_layernorm.weight")]
)
def test_prepare_weights_keeps_norm_dtype_and_separates_mixer(kind, norm_name):
    weights = {
        f"hc_{kind}_fn": torch.ones(24, 16384).half(),
        f"hc_{kind}_scale": torch.ones(3).half(),
        f"hc_{kind}_base": torch.ones(24).half(),
        norm_name: torch.ones(4096).half(),
    }
    prepared = prepare_weights(weights, kind, norm_name)
    assert all(value.dtype == torch.float32 and value.is_contiguous() for value in prepared[:3])
    assert prepared[3].dtype == torch.float16
    assert weights[f"hc_{kind}_fn"].dtype == torch.float16
    assert admissible(fake_residual(), frozenset({1}))


def test_prepare_weights_reuses_existing_contiguous_fp32_residency():
    weights = {
        "hc_attn_fn": torch.ones(24, 16384),
        "hc_attn_scale": torch.ones(3),
        "hc_attn_base": torch.ones(24),
        "input_layernorm.weight": torch.ones(4096).half(),
    }
    prepared = prepare_weights(weights, "attn", "input_layernorm.weight")
    for actual, expected in zip(prepared, weights.values(), strict=True):
        assert actual is expected


@pytest.mark.parametrize("enabled", [(), ("PRE",), ("POST",), ("PRE", "POST")])
@pytest.mark.parametrize("discovery", ["folder", "direct", "direct-first", "folder-first"])
def test_cmake_discovery_keeps_candidates_default_off(tmp_path, enabled, discovery):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake unavailable; build-option integration test needs CMake")
    root = Path(__file__).resolve().parents[3]
    pre = root / "csrc/attention/sidecard_mhc_pre_v310"
    post = root / "csrc/gmm/sidecard_mhc_post_v310"
    post_entries = {
        "folder": ((post, "post"),),
        "direct": ((post / "op_host", "post-direct"),),
        "direct-first": ((post / "op_host", "post-direct"), (post, "post")),
        "folder-first": ((post, "post"), (post / "op_host", "post-direct")),
    }
    post_discovery = "".join(
        f'add_subdirectory("{source.as_posix()}" {binary})\n' for source, binary in post_entries[discovery]
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(MhcBuildPolicy LANGUAGES NONE)\n"
        "set(ASCEND_OP_NAME all)\n"
        "set(BUILD_OPS_RTY_KERNEL ON)\n"
        "set_property(GLOBAL PROPERTY registered_candidates 0)\n"
        "function(add_op_to_compiled_list)\n"
        "  get_property(count GLOBAL PROPERTY registered_candidates)\n"
        '  math(EXPR count "${count} + 1")\n'
        "  set_property(GLOBAL PROPERTY registered_candidates ${count})\n"
        "endfunction()\n"
        "function(add_ops_compile_options)\nendfunction()\n"
        f'add_subdirectory("{pre.as_posix()}" pre)\n'
        + post_discovery
        + "get_property(count GLOBAL PROPERTY registered_candidates)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/registered-count.txt" "${count}")\n'
    )
    subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            *(f"-DSIDECARD_BUILD_MHC_{name}_V310=ON" for name in enabled),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "build/registered-count.txt").read_text() == str(len(enabled))


def test_lazy_import_and_help_without_device_sdk():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys; import tools.sidecard.npu_mhc; "
        "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.mhc_probe", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout and "--output" in result.stdout and "--rank" in result.stdout


@pytest.mark.parametrize("operation", ["pre", "post"])
def test_component_gate_qualifies_each_requested_mixer_without_promoting_failed_rows(operation):
    report = {
        "cases": [
            {
                "rows": rows,
                "pre": {"passed": operation == "pre" and rows == 1},
                "post_passed": operation == "post" and rows == 1,
            }
            for rows in (1, 4, 32)
        ],
        "measurements": [
            {
                "rows": rows,
                **{
                    f"{kind}_{phase}": {"median_host_microseconds": 1 if kind == "candidate" else 2}
                    for kind in ("candidate", "baseline")
                    for phase in ("pre", "post")
                },
            }
            for rows in (1, 4, 32)
        ],
    }
    requested = {"pre": [], "post": []}
    requested[operation] = [1]
    summarize(report, requested)
    assert report["passed"]
    assert report[f"numerically_eligible_{operation}_rows"] == [1]
    assert report[f"probe_eligible_{operation}_rows"] == [1]
    requested[operation] = [1, 4]
    summarize(report, requested)
    assert not report["passed"]


def test_component_gate_never_qualifies_missing_row_or_measurement_evidence():
    report = {"cases": [{"rows": 1, "pre": {"passed": True}, "post_passed": True}], "measurements": []}
    summarize(report, {"pre": [2], "post": []})
    assert not report["passed"]
    assert report["probe_eligible_pre_rows"] == []
