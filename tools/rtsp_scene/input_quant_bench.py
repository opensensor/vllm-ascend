"""Check codes/scales and paired complete projections for fused NZ input quantization."""

import argparse
import json
from pathlib import Path

import torch

from tools.rtsp_scene.dense_runtime import DenseProjection, KernelBackend
from tools.rtsp_scene.input_quant import INPUT_WIDTHS, SCALE_STRIDE, InputQuantizer
from tools.rtsp_scene.kernel_compare import MAGNITUDES, ReferenceLinear, compare_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--input-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or min(*args.rows, args.iterations, args.repeats) <= 0 or max(args.rows) > 8:
        parser.error("Use a fresh output and positive timing counts with rows in [1,8]")
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20261013)
    device = torch.device(f"npu:{args.device}")
    backend = KernelBackend(args.build)
    quantize = InputQuantizer(args.input_build, backend, device)
    report = {
        "device": args.device,
        "scope": "Isolated input quantization and complete projections",
        "codes": [],
        "layouts": [],
        "projections": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for inner in INPUT_WIDTHS:
            for rows in args.rows:
                x = torch.empty(rows, inner, dtype=torch.float16, device=device)
                for magnitude in MAGNITUDES:
                    x.normal_(0, magnitude)
                    codes, scales = torch_npu.npu_dynamic_quant(x)
                    actual, actual_scales = quantize(x)
                    record = {
                        "rows": rows,
                        "inner": inner,
                        "magnitude": magnitude,
                        "codes_equal": bool(torch.equal(codes.cpu(), actual.cpu())),
                        "scales_equal": bool(
                            torch.equal(scales.cpu(), actual_scales.reshape(rows, SCALE_STRIDE)[:, 0].cpu())
                        ),
                    }
                    report["codes"].append(record)
                    if not record["codes_equal"] or not record["scales_equal"]:
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        raise RuntimeError("Input quantization codes or scales differ from CANN")
                formatted = torch_npu.npu_format_cast(x, 29)
                expected, scales = torch_npu.npu_dynamic_quant(torch_npu.npu_format_cast(formatted, 2))
                actual, actual_scales = quantize(formatted)
                layout = {
                    "rows": rows,
                    "inner": inner,
                    "codes_equal": bool(torch.equal(expected.cpu(), actual.cpu())),
                    "scales_equal": bool(
                        torch.equal(scales.cpu(), actual_scales.reshape(rows, SCALE_STRIDE)[:, 0].cpu())
                    ),
                }
                report["layouts"].append(layout)
                if not layout["codes_equal"] or not layout["scales_equal"]:
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise RuntimeError("Non-ND fallback quantization differs from CANN")
        for name in ("qkv", "output", "gate_up", "down"):
            reference = ReferenceLinear(args.model, name, args.device, torch_npu)
            original = DenseProjection(reference, backend)
            candidate = DenseProjection(reference, backend, input_nz=True, scale_stride=SCALE_STRIDE)
            # Both paths read the identical packed checkpoint weights.
            candidate.packed_weight = original.packed_weight
            for rows in args.rows:
                x = torch.empty(rows, reference.weight.shape[0], dtype=torch.float16, device=device)

                def fused(x, candidate=candidate):
                    codes, scales = quantize(x)
                    return candidate.quantized(codes, scales)

                result = compare_case(reference, (original, fused), x, iterations=args.iterations, repeats=args.repeats)
                result.update({"rows": rows, "projection": name})
                report["projections"].append(result)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "graph_checks")}), flush=True)
                if not result["passed"] or not result.get("graph_passed", False):
                    raise RuntimeError("Fused input projection or changed-input graph gate failed")


if __name__ == "__main__":
    main()
