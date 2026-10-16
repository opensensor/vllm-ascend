"""Gate NZ SwiGLU/quantization and the complete TP-local FFN before serving."""

import argparse
import json
from pathlib import Path

from tools.rtsp_scene.dense_bench import checked_build, load_projection, measure_pair, pack_weight
from tools.rtsp_scene.dense_shapes import decode_rows, kernel_namespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    build = checked_build(args.build)
    if (
        args.output.exists()
        or min(*args.tokens, args.iterations, args.repeats) < 1
        or max(args.tokens) > decode_rows(build)
    ):
        parser.error("Use a fresh output and rows within the frozen activation row limit")
    import torch
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20261012)
    torch.ops.load_library(str((args.build / "dense-bridge.so").resolve()))
    namespace = kernel_namespace(build)
    kernel_class = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch
    act = kernel_class(str((args.build / "nz-swiglu.bin").resolve()), "rtsp_nz_swiglu")
    gate = kernel_class(str((args.build / "dense-k4096.bin").resolve()), "rtsp_dense_i8")
    down = kernel_class(str((args.build / "dense-k6144.bin").resolve()), "rtsp_dense_i8")
    wg, sg = load_projection(args.model, "gate_up", args.rank, 2)
    wd, sd = load_projection(args.model, "down", args.rank, 2)
    packed_g, packed_d = pack_weight(wg, build["columns"]).npu(), pack_weight(wd, build["columns"]).npu()
    wg = torch_npu.npu_format_cast(wg.npu(), 29).t()
    wd = torch_npu.npu_format_cast(wd.npu(), 29).t()
    sg, sd = sg.npu(), sd.npu()
    report = {
        "scope": "TP-local FFN; excludes norm, attention, head and HCCL",
        "build": build,
        "rank": args.rank,
        "activation": [],
        "ffn": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for rows in args.tokens:
        c = torch.tensor([rows, 1], dtype=torch.int64, device="npu")
        cg = torch.tensor([rows, 12288, 1, 0, 1], dtype=torch.int64, device="npu")
        cd = torch.tensor([rows, 4096, 0, 1, 8], dtype=torch.int64, device="npu")
        x = torch.randn(rows, 4096, dtype=torch.float16, device="npu") * 0.1
        z = torch.ops.npu.empty_with_format([rows, 12288], dtype=torch.float16, device=x.device, acl_format=29)
        q = torch.ops.npu.empty_with_format([rows, 6144], dtype=torch.int8, device=x.device, acl_format=29)
        # Initialize inactive NZ rows; activation kernel only writes active rows.
        q.zero_()
        scales = torch.empty(rows, 8, dtype=torch.float32, device="npu")
        out = torch.empty(rows, 4096, dtype=torch.float16, device="npu")
        debug = torch.empty(rows, 6144, dtype=torch.float16, device="npu")

        def activation(z=z, q=q, scales=scales, c=c, debug=debug):
            launch(act, [z, q, scales, c, debug], 8)

        def baseline_activation(z=z):
            return torch_npu.npu_dynamic_quant(torch_npu.npu_swiglu(torch_npu.npu_format_cast(z, 2)))

        for magnitude in (0.0, 0.0001, 0.01, 0.1, 1.0, 3.0):
            z.copy_(torch.randn(rows, 12288, dtype=torch.float16, device="npu") * magnitude)
            activation()
            actual = torch_npu.npu_format_cast(q, 2).cpu().int()
            expected, expected_scales = baseline_activation()
            expected_activation = torch_npu.npu_swiglu(torch_npu.npu_format_cast(z, 2)).cpu()
            actual_activation = debug.cpu()
            quant_only, _ = torch_npu.npu_dynamic_quant(debug)
            expected = expected.cpu().int()
            difference = (actual - expected).abs()
            scale_actual, scale_expected = scales[:, 0].cpu(), expected_scales.cpu()
            normalized = actual_activation.float() / torch.where(scale_expected == 0, 1, scale_expected)[:, None]
            safe_scale = torch.where(scale_expected == 0, 1, scale_expected)
            half_normalized = (actual_activation * safe_scale.reciprocal().half()[:, None]).float()
            half_away = half_normalized.sign() * (half_normalized.abs() + 0.5).floor()
            maximum = actual_activation.abs().amax(-1)
            half_divide = (actual_activation / torch.where(maximum == 0, 1, maximum)[:, None] * 127).float()
            half_divide_away = half_divide.sign() * (half_divide.abs() + 0.5).floor()
            mismatch = (actual != expected).nonzero()[:8]
            record = {
                "rows": rows,
                "magnitude": magnitude,
                "max_code_error": int(difference.max()),
                "different_codes": int((difference != 0).sum()),
                "different_activation_values": int((actual_activation != expected_activation).sum()),
                "max_activation_error": float((actual_activation - expected_activation).abs().max()),
                "different_quant_only_codes": int((actual != quant_only.cpu().int()).sum()),
                "quant_diagnostic": [
                    {
                        "normalized": float(normalized[r, j]),
                        "custom": int(actual[r, j]),
                        "reference": int(expected[r, j]),
                    }
                    for r, j in mismatch.tolist()
                ],
                "cpu_rint_vs_reference": int((normalized.round().int() != expected).sum()),
                "cpu_half_rint_vs_reference": int((normalized.half().float().round().int() != expected).sum()),
                "cpu_half_multiply_rint_vs_reference": int((half_normalized.round().int() != expected).sum()),
                "cpu_half_multiply_away_vs_reference": int((half_away.int() != expected).sum()),
                "cpu_half_divide_away_vs_reference": int((half_divide_away.int() != expected).sum()),
                "scale_max_error": float((scale_actual - scale_expected).abs().max()),
                "scales_actual": scale_actual.tolist(),
                "scales_reference": scale_expected.tolist(),
            }
            record["passed"] = int(difference.max()) == 0 and bool(
                torch.allclose(scale_actual, scale_expected, rtol=0.001, atol=1e-7)
            )
            report["activation"].append(record)
            save()
            print(json.dumps(record), flush=True)
        if not all(record["passed"] for record in report["activation"] if record["rows"] == rows):
            continue
        c[1] = 0  # Disable the ND diagnostic write during FFN and timing runs.

        def custom_ffn(x=x, z=z, cg=cg, activation=activation, q=q, scales=scales, out=out, cd=cd):
            qx, xs = torch_npu.npu_dynamic_quant(x)
            launch(gate, [qx, packed_g, sg, xs, z, cg], 8)
            activation()
            launch(down, [q, packed_d, sd, scales, out, cd], 8)
            return out

        def reference_ffn(x=x):
            qx, xs = torch_npu.npu_dynamic_quant(x)
            reference_z = torch_npu.npu_quant_matmul(qx, wg, sg, pertoken_scale=xs, output_dtype=torch.float16)
            reference_q, reference_s = torch_npu.npu_dynamic_quant(torch_npu.npu_swiglu(reference_z))
            return torch_npu.npu_quant_matmul(
                reference_q, wd, sd, pertoken_scale=reference_s, output_dtype=torch.float16
            )

        record = {"rows": rows, "checks": []}
        for trial in range(3):
            x.normal_(0, 0.1 * (trial + 1))
            actual, expected = custom_ffn().cpu().float(), reference_ffn().cpu().float()
            finite = bool(torch.isfinite(actual).all())
            relative = (
                float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
                if finite
                else None
            )
            record["checks"].append(
                {
                    "finite": finite,
                    "relative_l2_error": relative,
                    "max_abs_error": float((actual - expected).abs().max()) if finite else None,
                    "passed": finite and bool(torch.equal(actual, expected)),
                }
            )
        record["passed"] = all(check["passed"] for check in record["checks"])
        if record["passed"]:
            record["activation_reference"], record["activation_custom"] = measure_pair(
                baseline_activation, activation, torch.npu.synchronize, args.iterations, args.repeats
            )
            record["eager_reference"], record["eager_custom"] = measure_pair(
                reference_ffn, custom_ffn, torch.npu.synchronize, args.iterations, args.repeats
            )
            custom_graph, reference_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
            with torch.npu.graph(custom_graph):
                custom_ffn()
            with torch.npu.graph(reference_graph):
                reference_out = reference_ffn()
            x.normal_(0, 0.15)
            custom_graph.replay()
            reference_graph.replay()
            a, e = out.cpu().float(), reference_out.cpu().float()
            record["graph_changed_input_relative_l2"] = float(
                torch.linalg.vector_norm(a - e) / torch.linalg.vector_norm(e)
            )
            record["graph_changed_input_passed"] = bool(torch.isfinite(a).all()) and bool(torch.equal(a, e))
            if record["graph_changed_input_passed"]:
                record["graph_reference"], record["graph_custom"] = measure_pair(
                    reference_graph.replay, custom_graph.replay, torch.npu.synchronize, args.iterations, args.repeats
                )
            del custom_graph, reference_graph
        report["ffn"].append(record)
        save()
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
