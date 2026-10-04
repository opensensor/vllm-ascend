# SPDX-License-Identifier: Apache-2.0
"""One-card gate for extending the fused native-W4 SwiGLU pack to prefill.

This tests only the activation epilogue, with synthetic FP16 gate/up rows.
It does not change the serving model or claim a full-layer speedup.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

SELECTED_PREFILL_ROWS = 15360
PREFILL_ROWS = (5120, SELECTED_PREFILL_ROWS, 20480)
ACTIVATION_WIDTH = 640


def timed_ms(function, iterations, trials, torch):
    for _ in range(2):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="new directory for large-shape baseline and fused traces")
    parser.add_argument("--rows", type=int, nargs="+", default=PREFILL_ROWS)
    parser.add_argument("--width", type=int, default=ACTIVATION_WIDTH)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if any(rows <= 0 or rows > max(PREFILL_ROWS) for rows in args.rows):
        parser.error(f"rows must be in [1, {max(PREFILL_ROWS)}]")
    if args.width < 256 or args.width > 2560 or args.width % 128:
        parser.error("width must be a multiple of 128 in [256, 2560]")
    if args.iterations <= 0 or args.trials <= 0:
        parser.error("iterations and trials must be positive")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": args.rows,
                    "width": args.width,
                    "trace_rows": SELECTED_PREFILL_ROWS if SELECTED_PREFILL_ROWS in args.rows else max(args.rows),
                    "trace_capture": args.trace_dir is not None,
                    "npu_used": False,
                }
            )
        )
        return
    if args.output is None or args.output.exists() or (args.trace_dir and args.trace_dir.exists()):
        parser.error("--output and --trace-dir must name new paths")

    import torch
    import torch.nn.functional as F
    import torch_npu

    from tools.qwen4exp.npu_profile import capture_npu_profile
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device, swiglu_pack_activation_device
    from vllm_ascend.utils import enable_custom_op

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires one Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True)

    trace_rows = SELECTED_PREFILL_ROWS if SELECTED_PREFILL_ROWS in args.rows else max(args.rows)
    with torch.inference_mode(), args.output.open("x") as output:
        for rows in args.rows:
            generator = torch.Generator().manual_seed(1000 + rows)
            gate_up = (torch.randn(rows, args.width * 2, generator=generator) * 1.5).half().npu()

            def reference(gate_up=gate_up):
                gate, up = gate_up.float().chunk(2, -1)
                return pack_activation_device((F.silu(gate) * up).half())

            def candidate(gate_up=gate_up):
                return swiglu_pack_activation_device(gate_up)

            expected = [value.cpu() for value in reference()]
            actual = [value.cpu() for value in candidate()]
            for value, target in zip(actual, expected):
                torch.testing.assert_close(value, target, rtol=0, atol=0)
            max_abs = [float((value.float() - target.float()).abs().max()) for value, target in zip(actual, expected)]

            samples = {"torch": [], "fused": []}
            methods = {"torch": reference, "fused": candidate}
            for trial in range(args.trials):
                for method in ("torch", "fused") if trial % 2 == 0 else ("fused", "torch"):
                    samples[method].extend(timed_ms(methods[method], args.iterations, 1, torch))

            trace_roots = None
            if args.trace_dir and rows == trace_rows:
                trace_roots = {method: str(args.trace_dir / method) for method in methods}
                for method, function in methods.items():
                    capture_npu_profile(function, Path(trace_roots[method]), torch, torch_npu)

            record = {
                "rows": rows,
                "width": args.width,
                "max_abs": max_abs,
                "timings_ms": {
                    method: {"median_ms": statistics.median(values), "trials_ms": values}
                    for method, values in samples.items()
                },
                "trace_roots": trace_roots,
                "scope": "synthetic_activation_epilogue_only",
                "model_ttft_measured": False,
            }
            line = json.dumps(record)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
