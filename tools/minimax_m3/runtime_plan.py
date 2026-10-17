# SPDX-License-Identifier: Apache-2.0
"""Validate an offline export and print (never execute) its initial serve command."""

import argparse
import shlex
from pathlib import Path

from tools.minimax_m3.checkpoint import read_manifest


def serve_command(model_path):
    """Existing upstream PP partition setting; no new Ascend environment knobs."""
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
        "0.85",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--port",
        "8000",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    folder = args.model_path.resolve()
    manifest = read_manifest(folder)
    print(f"Complete manifest: {manifest['tensor_count']} source tensors; payloads are checked at load time.")
    print("Hardware-unvalidated candidate. Run only in a separately authorized NPU window, from /workspace.")
    print(shlex.join(serve_command(folder)))


if __name__ == "__main__":
    main()
