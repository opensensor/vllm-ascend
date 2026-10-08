# SPDX-License-Identifier: Apache-2.0
"""Deferred complete MoE gates must be meaningful and require explicit devices."""

import pytest
import torch

from tools.glm_perf.fused_weight_layout import unpack_cube
from tools.glm_perf.w4_storage_probe import compare, run, synthetic_weights, variants


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("hidden,intermediate", [(256, 256), (512, 256)])
def test_variants_preserve_both_projections_and_account_for_device_bytes(bits, hidden, intermediate):
    codes, scales = synthetic_weights(bits, hidden, intermediate)
    pairs = variants(codes, scales)
    assert set(pairs) == {"baseline", "gate_up", "down", "both"}
    for name, pair in pairs.items():
        for stage, (actual, old, k) in enumerate(zip(pair, codes, (hidden, intermediate))):
            selected = name == "both" or name == ("gate_up" if stage == 0 else "down")
            assert actual.shape[-1] == k * (4 if selected else bits) // 8
            assert torch.equal(unpack_cube(actual, k), unpack_cube(old, k))
    assert pairs["gate_up"][1] is codes[1] and pairs["down"][0] is codes[0]


def test_mixed_existing_w4_bank_is_not_repromoted():
    codes, scales = synthetic_weights(3)
    codes = variants(codes, scales)["gate_up"]
    assert set(variants(codes, scales)) == {"baseline", "down"}
    with pytest.raises(ValueError, match="no W2/W3"):
        variants(variants(codes, scales)["down"], scales)


def test_no_gate_can_import_backend_or_read_checkpoint_without_explicit_device_opt_in(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run(tmp_path / "missing", tmp_path / "result.json", checkpoint=tmp_path / "missing-model")
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "kwargs", [{"repeats": 0}, {"tokens": (641,)}, {"activations": (16,)}, {"checkpoint": "missing"}, {"layer": 3}]
)
def test_invalid_gate_configuration_fails_before_device_access(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(tmp_path / "missing", tmp_path / "result.json", allow_device_gate=True, **kwargs)


@pytest.mark.parametrize(
    "candidate",
    [torch.ones(2, 256) + 1e-6, torch.zeros(2, 256), torch.ones(2, 256).half(), torch.full((2, 256), float("nan"))],
)
def test_arithmetic_gate_rejects_drift_invalid_dtype_and_nonfinite(candidate):
    with pytest.raises(AssertionError):
        compare([torch.ones(2, 256), candidate])


def test_active_routes_and_empty_routes_have_separate_strict_oracles():
    compare([torch.ones(2, 256), torch.ones(2, 256)])
    compare([torch.zeros(2, 256), torch.zeros(2, 256)], expect_zero=True)
    with pytest.raises(AssertionError, match="only zero"):
        compare([torch.zeros(2, 256), torch.zeros(2, 256)])
    with pytest.raises(AssertionError, match="stale"):
        compare([torch.ones(2, 256), torch.ones(2, 256)], expect_zero=True)
    with pytest.raises(AssertionError, match="bits"):
        compare([torch.zeros(2, 256), -torch.zeros(2, 256)], expect_zero=True)
