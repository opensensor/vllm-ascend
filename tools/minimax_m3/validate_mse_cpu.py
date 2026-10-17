# SPDX-License-Identifier: Apache-2.0
"""Compare compiled conversion bytes with a previously recorded Torch probe."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tools.minimax_m3.export_w4 import local_file
from tools.minimax_m3.probe_w4 import quantize_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if reference["method"] != "mse":
        raise ValueError("Expected an MSE reference probe")
    torch.set_num_threads(args.threads)
    results = []
    for record in reference["results"]:
        start = time.perf_counter()
        source, codes, scales = (hashlib.sha256() for _ in range(3))
        with safe_open(local_file(args.source, record["shard"]), framework="pt", device="cpu") as handle:
            view = handle.get_slice(record["tensor"])
            for row in range(0, view.get_shape()[0], reference["chunk_rows"]):
                weight = view[row : row + reference["chunk_rows"]].contiguous()
                source.update(weight.view(torch.uint8).numpy().tobytes())
                packed, stored, _ = quantize_chunk(weight, "mse", mse_backend="numba")
                codes.update(packed.numpy().tobytes())
                scales.update(stored.view(torch.uint8).numpy().tobytes())
        result = {
            "tensor": record["tensor"],
            "seconds": time.perf_counter() - start,
            "source_equal": source.hexdigest() == record["source_tensor_sha256"],
            "packed_equal": codes.hexdigest() == record["packed_codes_sha256"],
            "scales_equal": scales.hexdigest() == record["fp16_scales_sha256"],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not all(row["source_equal"] and row["packed_equal"] and row["scales_equal"] for row in results):
        raise ValueError("Compiled MSE output differs from the reference probe")


if __name__ == "__main__":
    main()
