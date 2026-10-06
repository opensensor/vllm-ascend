# SPDX-License-Identifier: Apache-2.0
"""Activation-calibrated scale search in the existing signed block-code format.

This is a blockwise reconstruction surrogate, not GPTQ error feedback and not
a full-model quality gate. Its output is an isolated projection artifact, never
an in-place checkpoint conversion. Calibration and held-out activations are
explicitly separate inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tools.deepseek_w2.w2_format import pack_codes, quantize_weight

BLOCK = 32
SCALE_FRACTIONS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25)


def reconstruct(codes, scales):
    expanded = scales.half().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (codes.half() * expanded).float()


def validate_inputs(weight, calibration, validation, bits):
    if type(bits) is not int or bits not in (2, 3, 4):
        raise ValueError("supported signed code widths are 2, 3, and 4")
    if weight.ndim != 2 or any(size <= 0 or size % BLOCK for size in weight.shape):
        raise ValueError("weight must have positive dimensions divisible by 32")
    for name, tensor in (("weight", weight), ("calibration", calibration), ("validation", validation)):
        if tensor.device.type != "cpu" or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite floating-point CPU data")
    for name, tensor in (("calibration", calibration), ("validation", validation)):
        if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != weight.shape[1]:
            raise ValueError(f"{name} must be nonempty [samples,K] matching weight K")


def projection_metrics(weight, reconstructed, activations):
    target = activations.double() @ weight.double().T
    actual = activations.double() @ reconstructed.double().T
    error = (target - actual).square().mean().item()
    energy = target.square().mean().item()
    return {"output_mse": error, "relative_output_mse": error / max(energy, torch.finfo(torch.float64).tiny)}


def block_loss(block, covariance, candidate_codes, candidate_scale):
    restored = (candidate_codes.half() * candidate_scale.half()).double()
    error = block - restored
    return ((error @ covariance) * error).sum().item() / BLOCK


def pack_artifact(codes, bits):
    if bits != 3:
        return pack_codes(codes, bits).contiguous()
    # Eight consecutive signed W3 codes form one 24-bit word. Keep this
    # explicit so calibration also works before the separate W3 loader patch.
    groups = (codes.int() & 7).reshape(*codes.shape[:-1], -1, 8)
    words = torch.zeros_like(groups[..., 0])
    for field in range(8):
        words |= groups[..., field] << (3 * field)
    return (
        torch.stack([(words >> (8 * byte)) & 255 for byte in range(3)], -1)
        .to(torch.uint8)
        .reshape(*codes.shape[:-1], codes.shape[-1] * 3 // 8)
        .contiguous()
    )


def calibrate(weight, calibration, validation, bits=3, fractions=SCALE_FRACTIONS):
    validate_inputs(weight, calibration, validation, bits)
    if not fractions or fractions[0] != 1.0 or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("scale fractions must begin with 1 and lie in (0,1]")
    baseline_codes, baseline_scales = quantize_weight(weight, bits, method="minmax")
    baseline_scales = baseline_scales.float()
    codes, scales = baseline_codes.clone(), baseline_scales.clone()
    block_scores = []
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for first_k in range(0, weight.shape[1], BLOCK):
        x = calibration[:, first_k : first_k + BLOCK].double()
        covariance = x.T @ x / x.shape[0]
        for first_n in range(0, weight.shape[0], BLOCK):
            block = weight[first_n : first_n + BLOCK, first_k : first_k + BLOCK].double()
            index = (first_n // BLOCK, first_k // BLOCK)
            best_codes = baseline_codes[first_n : first_n + BLOCK, first_k : first_k + BLOCK]
            best_scale = baseline_scales[index]

            baseline_loss = best_loss = block_loss(block, covariance, best_codes, best_scale)
            selected_fraction = 1.0
            for fraction in fractions[1:]:
                candidate_scale = baseline_scales[index] * fraction
                candidate_codes = (block / candidate_scale.double()).round().clamp(lower, upper).to(torch.int8)
                candidate_loss = block_loss(block, covariance, candidate_codes, candidate_scale)
                if candidate_loss < best_loss:
                    best_loss, best_codes, best_scale = candidate_loss, candidate_codes, candidate_scale
                    selected_fraction = fraction
            codes[first_n : first_n + BLOCK, first_k : first_k + BLOCK] = best_codes
            scales[index] = best_scale
            block_scores.append(
                {
                    "n_block": index[0],
                    "k_block": index[1],
                    "fraction": selected_fraction,
                    "baseline_loss": baseline_loss,
                    "selected_loss": best_loss,
                }
            )
    baseline = reconstruct(baseline_codes, baseline_scales)
    candidate = reconstruct(codes, scales)
    report = {
        "bits": bits,
        "shape": list(weight.shape),
        "block_shape": [BLOCK, BLOCK],
        "objective": "sum of per-K-block activation reconstruction losses; cross-block cancellation excluded",
        "calibration_samples": calibration.shape[0],
        "validation_samples": validation.shape[0],
        "baseline_calibration": projection_metrics(weight, baseline, calibration),
        "candidate_calibration": projection_metrics(weight, candidate, calibration),
        "baseline_validation": projection_metrics(weight, baseline, validation),
        "candidate_validation": projection_metrics(weight, candidate, validation),
        "blocks": block_scores,
        "full_model_quality": "not_evaluated",
    }
    return {"codes": pack_artifact(codes, bits), "scale": scales.contiguous()}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="safetensors containing weight, calibration, validation"
    )
    parser.add_argument("--output", required=True, type=Path, help="new isolated projection artifact")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=3)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("preserve existing projection artifacts and reports")
    source = load_file(str(args.input))
    tensors, report = calibrate(source["weight"], source["calibration"], source["validation"], args.bits)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={"format": "signed_block32", "method": "activation_scale_search"})
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": str(args.output), "report": str(report_path)}))


if __name__ == "__main__":
    main()
