# SPDX-License-Identifier: Apache-2.0
"""CPU-only ModelSlim INT4 IR bridge to M3 block32 storage; not calibration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from tools.deepseek_w2.w2_format import pack_codes, unpack_codes
from tools.minimax_m3.checkpoint import read_manifest
from vllm_ascend._310p.ops.minimax_m3 import unpack_w4

BLOCK = 32
SAMPLE_ROWS = 64


def quantize_modelslim(weight):
    # Optional dependency, isolated to this offline tool.
    from msmodelslim.ir.api import calculate_qparam, quantize
    from msmodelslim.ir.qal import QDType, QScope, QStorage

    if weight.device.type != "cpu" or weight.ndim != 2 or any(d % BLOCK for d in weight.shape):
        raise ValueError("Expected CPU matrix with 32-aligned dimensions")
    if not torch.isfinite(weight).all():
        raise ValueError("Nonfinite weights")
    rows, cols = weight.shape
    blocks = weight.float().reshape(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK).permute(0, 2, 1, 3)
    flat = blocks.reshape(-1, BLOCK * BLOCK)
    params = calculate_qparam(
        flat.amin(-1, keepdim=True), flat.amax(-1, keepdim=True), QDType.INT4, QScope.PER_TOKEN, True
    )
    # PER_TOKEN here means a synthetic row containing one 32x32 weight block.
    # Store the scale before code selection so the runtime represents the same quantizer.
    scale = params.ext["scale"].half()
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Scales not representable in FP16")
    params.ext["scale"] = scale.float()
    if torch.count_nonzero(params.ext["offset"]):
        raise ValueError("M3 block-W4 cannot represent nonzero offsets")
    quant = quantize(QStorage(QDType.FLOAT, flat), params).value.to(torch.int8)
    codes = quant.reshape(rows // BLOCK, cols // BLOCK, BLOCK, BLOCK).permute(0, 2, 1, 3).reshape(rows, cols)
    packed = pack_codes(codes, 4)
    torch.testing.assert_close(unpack_codes(packed, cols, 4), codes, atol=0, rtol=0)
    return packed, scale.reshape(rows // BLOCK, cols // BLOCK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("--modelslim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.modelslim_root.resolve()))
    torch.set_num_threads(2)
    index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
    manifest = read_manifest(args.export)
    results = []
    for layer in (3, 31, 59):
        for projection in ("w1", "w2", "w3"):
            name = f"language_model.model.layers.{layer}.block_sparse_moe.experts.0.{projection}.weight"
            with safe_open(args.source / index[name], framework="pt", device="cpu") as handle:
                weight = handle.get_slice(name)[:SAMPLE_ROWS].float()
            packed, scales = quantize_modelslim(weight)
            restored = unpack_w4(packed, scales)
            record = manifest["weights"][name]
            with safe_open(args.export / record["file"], framework="pt", device="cpu") as handle:
                existing = unpack_w4(
                    handle.get_slice("codes")[:SAMPLE_ROWS], handle.get_slice("scales")[: SAMPLE_ROWS // BLOCK]
                )
            denom = weight.square().mean().sqrt()
            results.append(
                dict(
                    name=name,
                    shape=list(weight.shape),
                    modelslim_minmax_nrmse=((restored - weight).square().mean().sqrt() / denom).item(),
                    existing_mse_nrmse=((existing - weight).square().mean().sqrt() / denom).item(),
                    packed_roundtrip_exact=True,
                )
            )
    report = dict(
        hardware_validated=False,
        calibrated=False,
        source=str(args.source.resolve()),
        modelslim_revision=subprocess.check_output(
            ["git", "-C", str(args.modelslim_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        method="ModelSlim symmetric minmax on 32x32 weight blocks; FP16 scales; signed W4 bridge",
        limitation="Nine 64-row expert slices, not full matrices or model quality",
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
