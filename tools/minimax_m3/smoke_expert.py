# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight CPU gate: export, reload, and execute one M3 expert."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tools.minimax_m3.checkpoint import load_expert_banks
from tools.minimax_m3.export_w4 import export_tensor, local_file
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts, clipped_swiglu


def smoke_expert(source, destination, *, layer=3, expert=0):
    destination.mkdir(parents=True, exist_ok=True)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    started = time.monotonic()
    records = {}
    for projection in ("w1", "w2", "w3"):
        name = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"
        with safe_open(local_file(source, index[name]), framework="pt", device="cpu") as handle:
            record = export_tensor(
                handle.get_slice(name), name, destination / f"{projection}.safetensors", method="mse"
            )
        record["name"] = name
        records[name] = record
    intermediate, hidden = next(record["source_shape"] for name, record in records.items() if ".w1." in name)
    # A one-expert local slice, retaining its original global identity.
    bank = PackedM3Experts(
        hidden, intermediate, expert + 1, expert + 1, expert, 1, prefix=f"model.layers.{layer}.block_sparse_moe.experts"
    )
    loaded = load_expert_banks(bank, destination, {"weights": records})
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, hidden, generator=generator).half()
    router = torch.full((2, expert + 1), -10.0)
    router[:, expert] = 10.0
    actual = bank(x, router)
    if not torch.isfinite(actual).all():
        raise ValueError("Real expert CPU output contains nonfinite values")
    projections = {}
    for name, record in records.items():
        projection = name.split(".")[-2]
        with safe_open(local_file(source, index[name]), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(name).float()
        if projection in ("w1", "w3"):
            projections[projection] = (x.float() @ weight.T).half()
    act = clipped_swiglu(projections["w1"], projections["w3"]).half()
    down_name = next(name for name in records if ".w2." in name)
    with safe_open(local_file(source, index[down_name]), framework="pt", device="cpu") as handle:
        reference = (act.float() @ handle.get_tensor(down_name).float().T).half().float() * 2
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
    return {
        "scope": (
            "One real expert; CPU export/load/MLP only. Random inputs, not activation calibration or model accuracy."
        ),
        "source": str(source),
        "output": str(destination),
        "elapsed_seconds": time.monotonic() - started,
        "loaded_buffers": loaded,
        "output_shape": list(actual.shape),
        "output_finite": True,
        "output_l2": float(actual.norm()),
        "source_mlp_output_cosine": float(cosine),
        "source_mlp_output_nrmse": float((actual - reference).norm() / reference.norm()),
        "projection_contract": "FP16 input and dequantized weights, FP32 accumulation oracle, FP16 projection output",
        "weights": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = smoke_expert(args.source, args.destination, layer=args.layer, expert=args.expert)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "weights"}))


if __name__ == "__main__":
    main()
