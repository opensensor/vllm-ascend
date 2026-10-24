# SPDX-License-Identifier: Apache-2.0
"""Qualify a versioned native manifest in an isolated NPU process first."""

import argparse
import json
from pathlib import Path

import torch

from tools.glm_perf.resident_native import NativeSession


def operator_exists(name):
    try:
        torch._C._dispatch_find_schema_or_throw(name, "")
        return True
    except RuntimeError:
        return False


def main():
    import torch_npu  # noqa: F401 -- CLI-only device registration.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    session = NativeSession()
    prepared = session.prepare(json.loads(args.manifest.read_text()), operator_exists)
    with torch.inference_mode():
        result = session.load(prepared["native_digest"], torch.ops.load_library, operator_exists, torch.npu.synchronize)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
