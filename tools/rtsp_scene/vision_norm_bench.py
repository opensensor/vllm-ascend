"""Checkpoint-backed parity and paired graph timing for 310P vision normalization."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from tools.rtsp_scene.kernel_compare import compare_case
from tools.rtsp_scene.vision_norm import VISION_WIDTH, VisionZeroAddNorm

DEFAULT_ROWS = (1760, 2640, 3520, 3688)
NORM_EPSILON = 1e-6


def load_norm(model, device):
    # Checkpoint access belongs to this standalone benchmark, not serving.
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    norm = nn.LayerNorm(VISION_WIDTH, eps=NORM_EPSILON, dtype=torch.float16, device=device)
    for attribute in ("weight", "bias"):
        matches = [name for name in index if name.endswith(f"visual.blocks.0.norm1.{attribute}")]
        if len(matches) != 1:
            raise ValueError("Provide the complete Qwen3-VL vision normalization checkpoint")
        name = matches[0]
        with safe_open(model / index[name], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(name)
        getattr(norm, attribute).data.copy_(value.to(device=device, dtype=torch.float16))
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.device < 0 or min(*args.rows, args.iterations, args.repeats) <= 0:
        parser.error("Use a fresh output, nonnegative device and positive row/iteration/repeat counts")
    # Hardware imports are isolated from CPU test collection.
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{args.device}")
    with torch.inference_mode():
        reference = load_norm(args.model, device)
        zero = torch.zeros(max(args.rows), 1, VISION_WIDTH, dtype=torch.float16, device=device)
        candidate = VisionZeroAddNorm(reference, zero, torch_npu.npu_add_layer_norm)
        report = {
            "scope": "Isolated loaded vision norm, not end-to-end serving or stream capacity",
            "device": args.device,
            "zero_bytes": zero.numel() * zero.element_size(),
            "records": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for rows in args.rows:
            x = torch.empty(rows, 1, VISION_WIDTH, dtype=torch.float16, device=device)
            result = compare_case(
                reference, (reference, candidate), x, iterations=args.iterations, repeats=args.repeats
            )
            result["rows"] = rows
            report["records"].append(result)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result, allow_nan=False), flush=True)
            if not result["passed"] or not result.get("graph_passed", False):
                raise RuntimeError("Vision normalization parity or changed-input graph gate failed")


if __name__ == "__main__":
    main()
