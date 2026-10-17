# SPDX-License-Identifier: Apache-2.0
"""Validate an offline export and print (never execute) its initial serve command."""

import argparse
import shlex
from pathlib import Path

from tools.minimax_m3.checkpoint import read_manifest


def serve_command(model_path, *, memory_utilization=0.95):
    """Existing upstream PP partition setting; no new Ascend environment knobs."""
    if not 0 < memory_utilization <= 1:
        raise ValueError("memory utilization must be in (0, 1]")
    return [
        "env",
        "VLLM_PP_LAYER_PARTITION=21,20,19",
        "vllm",
        "serve",
        str(model_path),
        "--served-model-name",
        "minimax-m3-w4",
        "--load-format",
        "minimax_m3_w4",
        "--dtype",
        "half",
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "3",
        "--enable-expert-parallel",
        "--enforce-eager",
        "--block-size",
        "128",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "128",
        "--gpu-memory-utilization",
        str(memory_utilization),
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--port",
        "8000",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--memory-utilization", type=float, default=0.95)
    args = parser.parse_args()
    folder = args.model_path.resolve()
    manifest = read_manifest(folder)
    print(f"Complete manifest: {manifest['tensor_count']} source tensors; payloads are checked at load time.")
    print("Hardware-unvalidated candidate. Run only in a separately authorized NPU window, from /workspace.")
    print("Budget gate: estimate 41.89 GiB on the busiest rank at 8K/one sequence, including a 3 GiB allowance.")
    print("Check serving_budget against actual per-device capacity before using this memory fraction.")
    print(shlex.join(serve_command(folder, memory_utilization=args.memory_utilization)))


if __name__ == "__main__":
    main()
