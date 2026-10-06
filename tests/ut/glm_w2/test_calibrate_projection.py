# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.deepseek_w2.w2_format import unpack_codes
from tools.glm_w2.calibrate_projection import calibrate, reconstruct


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_activation_calibration_protects_observed_features_and_keeps_packing(bits):
    weight = torch.full((32, 64), 0.8)
    weight[:, 0] = 32
    weight[:, 32] = -32
    generator = torch.Generator().manual_seed(71)
    calibration = torch.randn(64, 64, generator=generator)
    validation = torch.randn(32, 64, generator=generator)
    calibration[:, (0, 32)] = 0
    validation[:, (0, 32)] = 0
    original = weight.clone()
    tensors, report = calibrate(weight, calibration, validation, bits, fractions=(1.0, 0.5, 0.25, 0.125, 0.0625))
    assert torch.equal(weight, original)
    assert tensors["codes"].shape == (32, 64 * bits // 8)
    assert tensors["codes"].dtype == torch.uint8
    assert tensors["scale"].shape == (1, 2)
    assert tensors["scale"].dtype == torch.float32
    assert report["candidate_validation"]["output_mse"] < report["baseline_validation"]["output_mse"]
    assert all(block["selected_loss"] <= block["baseline_loss"] for block in report["blocks"])
    if bits == 3:
        raw = tensors["codes"].int().reshape(32, -1, 3)
        words = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16)
        fields = torch.stack([(words >> (3 * field)) & 7 for field in range(8)], -1)
        decoded = torch.where(fields >= 4, fields - 8, fields).to(torch.int8).reshape(32, 64)
    else:
        decoded = unpack_codes(tensors["codes"], 64, bits)
    assert torch.isfinite(reconstruct(decoded, tensors["scale"])).all()
    assert report["full_model_quality"] == "not_evaluated"


def test_identity_candidate_preserves_baseline_objective():
    generator = torch.Generator().manual_seed(12)
    weight = torch.randn(64, 64, generator=generator)
    calibration = torch.randn(8, 64, generator=generator)
    _, report = calibrate(weight, calibration, calibration[:2].clone(), 3, fractions=(1.0,))
    assert report["baseline_validation"] == report["candidate_validation"]
    assert all(block["fraction"] == 1 for block in report["blocks"])


@pytest.mark.parametrize("bad", ("shape", "empty", "nan", "bits", "fraction"))
def test_invalid_calibration_input_fails_before_producing_artifact(bad):
    weight, calibration, validation = torch.ones(32, 32), torch.ones(2, 32), torch.ones(2, 32)
    bits, fractions = 3, (1.0, 0.5)
    if bad == "shape":
        weight = torch.ones(33, 32)
    elif bad == "empty":
        validation = torch.empty(0, 32)
    elif bad == "nan":
        calibration[0, 0] = float("nan")
    elif bad == "bits":
        bits = True
    elif bad == "fraction":
        fractions = (0.5,)
    with pytest.raises(ValueError):
        calibrate(weight, calibration, validation, bits, fractions)
