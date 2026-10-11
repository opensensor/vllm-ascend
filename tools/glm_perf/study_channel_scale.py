# SPDX-License-Identifier: Apache-2.0
"""Real FP8 weight error and synthetic-activation projection study, CPU only."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .channel_scale_checkpoint import file_sha256, read_fp8_projection
from .channel_scale_math import dequantize_weights, project, quantize_weights
from .fused_weight_layout import unpack_cube

SAMPLE_CHANNELS = 128


def relative_error(candidate, reference):
    denominator = reference.double().square().mean().sqrt()
    return float((candidate.double() - reference.double()).square().mean().sqrt() / denominator.clamp_min(1e-30))


def legacy_weights(root, layer, expert, name, channels, k):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{name}_proj"
    parts = []
    for suffix in ("_codes", "_scale"):
        key = prefix + suffix
        with safe_open(str(root / index[key]), framework="pt", device="cpu") as handle:
            parts.append(handle.get_tensor(key).clone())
    packed, scales = parts
    codes = unpack_cube(packed[None].contiguous(), k)[0].index_select(0, channels)
    return codes.float() * scales.float().index_select(0, channels // 32).repeat_interleave(32, dim=1)


def block32_activation_reference(x, weight):
    grouped = x.float().reshape(x.shape[0], -1, 32)
    scales = grouped.abs().amax(-1) / 7
    scales[scales == 0] = 1
    reconstructed = (grouped / scales[..., None]).round().clamp(-7, 7) * scales[..., None]
    return reconstructed.flatten(1) @ weight.T


def study(source, output, layers=(3, 14, 33), expert=0, rows=16, legacy_model=None):
    source, output = Path(source), Path(output)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    report = {
        "source": str(source),
        "source_index_sha256": file_sha256(index_path),
        "activation_data": "seeded Gaussian and synthetic channel outliers; NOT model calibration",
        "channels": "128 evenly spaced output channels per original FP8 projection",
        "qualification": "CPU sensitivity study only; no full-model quality or speed measurement",
        "results": [],
        "legacy": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(20261010)
    for layer in layers:
        for name in ("gate", "up", "down"):
            original = read_fp8_projection(source, index, layer, expert, name)
            channels = torch.linspace(0, original.shape[0] - 1, SAMPLE_CHANNELS).long()
            weights = original.index_select(0, channels)
            k = weights.shape[-1]
            del original
            inputs = torch.randn(rows, k, generator=rng).half()
            outliers = inputs.clone()
            outliers[:, ::127] *= 16
            references = {"gaussian": inputs.float() @ weights.T, "outliers": outliers.float() @ weights.T}
            if legacy_model is not None:
                old = legacy_weights(legacy_model, layer, expert, name, channels, k)
                report["legacy"].append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "projection": name,
                        "weight_nrmse": relative_error(old, weights),
                        "gaussian_projection_nrmse": relative_error(
                            block32_activation_reference(inputs, old), references["gaussian"]
                        ),
                        "outliers_projection_nrmse": relative_error(
                            block32_activation_reference(outliers, old), references["outliers"]
                        ),
                        "contract": "disk codes/scales, CPU A4 block32; NPU rounding unqualified",
                    }
                )
            for group in dict.fromkeys((256, 512, 1024, k)):
                for clipping in (1.0, 0.9, 0.8):
                    bank = quantize_weights(weights[None], group, clipping=clipping)
                    weight_error = relative_error(dequantize_weights(bank)[0], weights)
                    for bits in (4, 8):
                        entry = {
                            "layer": layer,
                            "expert": expert,
                            "projection": name,
                            "group": group,
                            "clipping": clipping,
                            "activation_bits": bits,
                            "weight_nrmse": weight_error,
                        }
                        for distribution, x in (("gaussian", inputs), ("outliers", outliers)):
                            entry[distribution + "_projection_nrmse"] = relative_error(
                                project(x, bank, 0, bits), references[distribution]
                            )
                        report["results"].append(entry)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"layer": layer, "projection": name, "candidates": len(report["results"])}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 14, 33])
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--legacy-model", type=Path)
    args = parser.parse_args()
    study(args.source, args.output, args.layers, args.expert, legacy_model=args.legacy_model)


if __name__ == "__main__":
    main()
