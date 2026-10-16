# SPDX-License-Identifier: Apache-2.0
"""Offline numerical/contract tests; these do not execute the AscendC kernel."""

import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.kda_probe import assigned_device, configuration
from tools.sidecard.layer_math import recurrent_kda
from tools.sidecard.npu_kda import BOUNDS, HEAD_DIM, OPERATOR_NAME, native, reference, resolve_operator, validate_inputs


def fixture(heads=16, seed=71):
    generator = torch.Generator().manual_seed(seed)
    q, k, v = [torch.randn(1, heads, HEAD_DIM, generator=generator).half() for _ in range(3)]
    gate = -torch.rand(1, heads, HEAD_DIM, generator=generator) * 4
    beta = torch.rand(1, heads, generator=generator)
    state = torch.randn(heads, HEAD_DIM, HEAD_DIM, generator=generator) * 0.1
    return q, k, v, gate, beta, state


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
def test_portable_reference_matches_runtime_and_updates_nonzero_carry(heads):
    *inputs, state = fixture(heads)
    expected_state = state.clone()
    original = state.clone()
    actual = reference(*inputs, state, torch)
    expected = recurrent_kda(*inputs, expected_state, torch)
    assert torch.equal(actual, expected)
    assert torch.equal(state, expected_state)
    assert not torch.equal(state, original)
    assert actual.dtype == torch.float16 and actual.shape == (1, heads, HEAD_DIM)
    assert state.dtype == torch.float32


@pytest.mark.parametrize("scale", [0.0, 1e-5, 16.0])
def test_range_and_normalization_boundaries(scale):
    q, k, v, gate, beta, state = fixture()
    q, k = q * scale, k * scale
    expected_state = state.clone()
    actual = reference(q, k, v, gate, beta, state, torch)
    expected = recurrent_kda(q, k, v, gate, beta, expected_state, torch)
    assert torch.equal(actual, expected)
    assert torch.equal(state, expected_state)
    assert bool(torch.isfinite(state).all())


def test_changed_inputs_and_sequential_carry_match_runtime():
    *_, state = fixture()
    expected_state = state.clone()
    for step in range(12):
        q, k, v, gate, beta, _ = fixture(seed=step + 19)
        actual = reference(q, k, v, gate, beta, state, torch)
        expected = recurrent_kda(q, k, v, gate, beta, expected_state, torch)
        assert torch.equal(actual, expected)
        assert torch.equal(state, expected_state)


def test_premature_fp16_normalization_changes_carry():
    q, k, v, gate, beta, state = fixture()
    expected_state = state.clone()
    reference(q, k, v, gate, beta, expected_state, torch)
    # The old native signature cannot represent either these normalized keys or
    # beta exactly. This regression test detects calling it through a downcast.
    normalized_key = k[0].float() * torch.rsqrt(k[0].float().square().sum(-1, keepdim=True) + 1e-6)
    rounded_key = normalized_key.half().float()
    state.mul_(gate[0].exp()[:, None, :])
    innovation = (v[0].float() - (state * rounded_key[:, None, :]).sum(-1)) * beta[0].half().float()[:, None]
    state.add_(innovation[:, :, None] * rounded_key[:, None, :])
    assert not torch.equal(state, expected_state)


@pytest.mark.parametrize(
    "change", ["rows", "heads", "key_dim", "beta_half", "gate_half", "state_half", "state_layout", "stride"]
)
def test_invalid_geometry_and_dtype_rejected(change):
    fields = list(fixture())
    if change == "rows":
        fields[0] = fields[0].expand(2, -1, -1)
    elif change == "heads":
        fields[0] = fields[0][:, :7]
    elif change == "key_dim":
        fields[1] = fields[1][:, :, :64]
    elif change in {"beta_half", "gate_half", "state_half"}:
        index = {"beta_half": 4, "gate_half": 3, "state_half": 5}[change]
        fields[index] = fields[index].half()
    elif change == "state_layout":
        fields[5] = fields[5].permute(1, 0, 2).contiguous()
    else:
        fields[0] = fields[0].transpose(-1, -2).contiguous().transpose(-1, -2)
    with pytest.raises(ValueError):
        validate_inputs(*fields, torch)


def test_native_never_silently_runs_cpu_or_downcasts():
    with pytest.raises(ValueError, match="NPU inputs"):
        native(*fixture(), torch)


def test_resolver_uses_existing_custom_op_namespace():
    expected = object()
    fake_torch = SimpleNamespace(
        ops=SimpleNamespace(
            _C_ascend=SimpleNamespace(**{OPERATOR_NAME: expected}),
            vllm_ascend=SimpleNamespace(**{OPERATOR_NAME: object()}),
        )
    )
    assert resolve_operator(fake_torch) is expected


@pytest.mark.parametrize("operations", [SimpleNamespace(), SimpleNamespace(_C_ascend=SimpleNamespace())])
def test_resolver_reports_missing_private_registration(operations):
    with pytest.raises(RuntimeError, match="_C_ascend::sidecard_kda_decode_310"):
        resolve_operator(SimpleNamespace(ops=operations))


def test_fp32_carry_retains_values_above_fp16_range():
    q, k, v, gate, beta, state = fixture()
    q.zero_()
    beta.zero_()
    gate.zero_()
    state.fill_(70000)
    actual = reference(q, k, v, gate, beta, state, torch)
    assert torch.equal(state, torch.full_like(state, 70000))
    assert bool((actual == 0).all())


def test_gate_alias_with_mutable_carry_is_rejected():
    q, k, v, _, beta, state = fixture()
    gate = state.flatten()[: q.numel()].view(q.shape)
    with pytest.raises(ValueError, match="overlap mutable carry"):
        validate_inputs(q, k, v, gate, beta, state, torch)


def test_inputs_on_other_device_are_rejected_before_pointer_access():
    q, k, v, gate, beta, state = fixture()
    with pytest.raises(ValueError, match="same device"):
        validate_inputs(q, k, v, gate.to("meta"), beta, state, torch)


def test_bounds_are_fixed_and_recurrence_has_no_fp16_carry_tolerance():
    assert BOUNDS.state_atol == 5e-5
    assert BOUNDS.output_atol == 2e-3
    assert BOUNDS.sequential_state_atol == 1e-3


def test_probe_requires_authenticated_library_and_assigned_chips(tmp_path):
    library = tmp_path / "candidate.so"
    library.write_bytes(b"offline fixture, never loaded")
    config = {
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "physical_chips": [2, 3],
        "heads": [16, 32],
    }
    assert configuration(config)[2] == [16, 32]
    for update in (
        {"physical_chips": [0, 1]},
        {"library_sha256": "0" * 64},
        {"repeats": 1},
        {"heads": [7]},
        {"sequence": False},
    ):
        with pytest.raises(ValueError):
            configuration(config | update)


def test_probe_base_library_is_authenticated_as_a_pair(tmp_path):
    addon, base = tmp_path / "addon.so", tmp_path / "base.so"
    addon.write_bytes(b"addon fixture")
    base.write_bytes(b"base fixture")
    config = {
        "library": str(addon),
        "library_sha256": hashlib.sha256(addon.read_bytes()).hexdigest(),
        "physical_chips": [2, 3],
        "base_library": str(base),
        "base_library_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
    }
    configuration(config)
    for update in ({"base_library_sha256": "0" * 64}, {"base_library": "relative.so"}):
        with pytest.raises(ValueError, match="base library"):
            configuration(config | update)
    del config["base_library"]
    with pytest.raises(ValueError, match="base library"):
        configuration(config)


def test_probe_uses_logical_ordinals_for_the_assigned_pair(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    assert [assigned_device(rank) for rank in (0, 1)] == ["npu:0", "npu:1"]
    for rank in (-1, 2, True):
        with pytest.raises(ValueError, match="leased"):
            assigned_device(rank)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="leased"):
        assigned_device(0)


def test_help_without_accelerator_imports():
    script = """
import runpy, sys
sys.argv = ['kda_probe', '--help']
try:
    runpy.run_module('tools.sidecard.kda_probe', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'torch_npu' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--rank" in result.stdout
