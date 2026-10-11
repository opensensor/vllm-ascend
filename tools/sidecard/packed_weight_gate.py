# SPDX-License-Identifier: Apache-2.0
"""Compare direct CUDA Cube conversion against independent host packing."""

import argparse
import ctypes
import json
from pathlib import Path

from .device_backend import CUDA_NIBBLE_PLANES, encode_cuda_weights
from .hardware_probe import DEVICE_TO_HOST, HOST_TO_DEVICE, CudaCopies
from .native_expert_probe import digest_file

SHAPES = ((2, 256, 512), (1, 4096, 4096), (1, 4096, 2048))
WEIGHT_BITS = (2, 3, 4)


def run(library, output):
    import numpy as np
    import torch

    from tools.glm_perf.fused_weight_layout import pack_cube

    torch.set_num_threads(8)
    report = {"passed": False, "library_sha256": digest_file(library), "cases": []}
    function = ctypes.CDLL(str(library)).sidecard_oracle_pack_cube
    function.argtypes, function.restype = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 4, ctypes.c_int
    generator = np.random.default_rng(3106000)
    try:
        for shape in SHAPES:
            for bits in WEIGHT_BITS:
                sign = 1 << (bits - 1)
                values = generator.integers(-sign, sign, size=shape, dtype=np.int8)
                native = pack_cube(torch.from_numpy(values), bits).numpy().tobytes()
                expected = encode_cuda_weights(values, CUDA_NIBBLE_PLANES)
                cuda = CudaCopies(Path("/usr/local/cuda/lib64/libcudart.so"), len(native), len(expected))
                try:
                    source = ctypes.create_string_buffer(native, len(native))
                    cuda.check(cuda.library.cudaMemcpy(cuda.input, source, len(native), HOST_TO_DEVICE))
                    cuda.check(function(cuda.input, cuda.output, *shape, bits))
                    cuda.check(cuda.library.cudaDeviceSynchronize())
                    destination = ctypes.create_string_buffer(len(expected))
                    cuda.check(cuda.library.cudaMemcpy(destination, cuda.output, len(expected), DEVICE_TO_HOST))
                    actual = destination.raw
                    if actual != expected:
                        raise AssertionError("direct CUDA Cube conversion changed signed integer codes")
                    case = {
                        "shape": list(shape),
                        "weight_bits": bits,
                        "bitwise_equal": True,
                        "native_bytes": len(native),
                        "cuda_bytes": len(expected),
                    }
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
                finally:
                    cuda.close()
        report["passed"] = True
    finally:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.library, args.output)), flush=True)


if __name__ == "__main__":
    main()
