# SPDX-License-Identifier: Apache-2.0
"""Measure exact expert row boundaries in separate control/candidate processes."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("output must be new and repeats positive")
    import torch
    import torch_npu

    from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    cases = []
    for bits in (3, 4, 2):
        torch.manual_seed(310256 + bits)
        experts, n, k = 4, 4096, 2048 if bits == 2 else 4096
        codes = torch.randint(0, 256, (experts, n, k * bits // 8), dtype=torch.uint8)
        pack = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
        packed = torch.stack([pack(expert, k) for expert in codes]).view(torch.int8).npu()
        scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
        for rows in (128, 129, 192, 256, 257):
            x = torch.randn(experts * rows, k).half().npu()
            ends = (torch.arange(1, experts + 1, dtype=torch.int64) * rows).npu()
            for _ in range(3):
                output = op(x, packed, scales, ends)
            torch.npu.synchronize()
            digest = hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()
            samples = []
            for _ in range(args.repeats):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = op(x, packed, scales, ends)
                torch.npu.synchronize()
                samples.append(1000 * (time.perf_counter() - start))
            cases.append(
                {
                    "bits": bits,
                    "rows_per_expert": rows,
                    "output_sha256": digest,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
            del x, ends, output
        del codes, packed, scales
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(json.dumps({"cases": cases}))


if __name__ == "__main__":
    main()
