# SPDX-License-Identifier: Apache-2.0
"""Matched mHC post+round timings and incremental peak allocations; opt-in NPU use."""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", type=Path, help="Optional supplemental candidate binding")
    parser.add_argument("--tokens", type=int, nargs="+", default=[640, 1280, 2560])
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 3 or any(not 0 < rows <= 32768 for rows in args.tokens):
        parser.error("require repeats >= 3 and token counts in [1,32768]")

    # Hardware dependencies belong only in explicit benchmark execution.
    import torch
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    if args.extension:
        torch.ops.load_library(str(args.extension.resolve()))
    native = torch.ops._C_ascend.npu_glm_mhc_post_310

    def reference(x, residual, post, comb):
        return (torch.einsum("nij,nih->njh", comb, residual) + post * x.float().unsqueeze(1)).half().float()

    def streaming(x, residual, post, comb):
        output = residual[:, :1, :] * comb[:, 0, :, None]
        for index in range(1, 4):
            output.addcmul_(residual[:, index : index + 1, :], comb[:, index, :, None])
        output.add_(post * x.float().unsqueeze(1))
        return output.half().float()

    functions = {"einsum_round": reference, "streaming_round": streaming, "native_round": native}
    records = []
    for rows in args.tokens:
        torch.manual_seed(310)
        values = (
            torch.randn(rows, 4096, device="npu").half(),
            torch.randn(rows, 4, 4096, device="npu").half().float(),
            torch.randn(rows, 4, 1, device="npu").sigmoid().half().float(),
            torch.randn(rows, 4, 4, device="npu").softmax(-1).half().float(),
        )
        expected = reference(*values)
        parity = {}
        for name, function in functions.items():
            output = function(*values)
            torch.testing.assert_close(output, expected, rtol=1e-3, atol=2e-6)
            parity[name] = {
                "max_abs": (output - expected).abs().max().item(),
                "rounded_mismatches": (output != expected).sum().item(),
            }
            del output
        del expected
        timings = {name: [] for name in functions}
        peaks = {}
        for name, function in functions.items():
            for _ in range(3):
                function(*values)
            torch.npu.synchronize()
            before = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            output = function(*values)
            torch.npu.synchronize()
            peaks[name] = torch.npu.max_memory_allocated() - before
            del output
        names = list(functions)
        for repetition in range(args.repeats):
            # Rotate order so no implementation always runs first or last.
            for name in names[repetition % len(names) :] + names[: repetition % len(names)]:
                torch.npu.synchronize()
                start = time.perf_counter()
                output = functions[name](*values)
                torch.npu.synchronize()
                timings[name].append((time.perf_counter() - start) * 1000)
                del output
        records.append(
            {
                "tokens": rows,
                "width": 4096,
                "variants": {
                    name: {
                        "median_ms": statistics.median(timings[name]),
                        "samples_ms": timings[name],
                        "incremental_peak_bytes": peaks[name],
                        **parity[name],
                    }
                    for name in functions
                },
            }
        )
        del values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"scope": "operator only; no serving throughput claim", "results": records}, indent=2)
    )
    print(args.output)


if __name__ == "__main__":
    main()
