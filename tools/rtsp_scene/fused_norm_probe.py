"""Gate native residual/RMSNorm/dynamic-quant fusion on the pinned 310P runtime."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--custom-build", type=Path)
    parser.add_argument("--matrix-build", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh private output file")
    # Initialize the NPU only in the isolated hardware probe process.
    import torch
    import torch_npu
    from safetensors import safe_open

    from tools.rtsp_scene.build_norm_quant import checked_build
    from tools.rtsp_scene.dense_bench import measure_pair
    from tools.rtsp_scene.dense_runtime import KernelBackend
    from tools.rtsp_scene.dense_shapes import kernel_namespace

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    weights = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    backend = None
    if args.custom_build:
        if args.matrix_build is None:
            parser.error("The custom candidate requires a frozen matrix bridge")
        checked_build(args.custom_build)
        backend = KernelBackend(args.matrix_build)
        kernel_class = getattr(torch.classes, kernel_namespace(backend.manifest)).Kernel
        kernel = kernel_class(str(args.custom_build / "norm-quant.bin"), "rtsp_norm_quant")
    records = []
    with torch.inference_mode():
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            key = next(k for k in weights if "layers.0." in k and k.endswith(suffix))
            with safe_open(args.model / weights[key], framework="pt") as file:
                gamma = file.get_tensor(key).to(device="npu", dtype=torch.float16)
            cases = (
                (rows, magnitude)
                for rows in ((1, 2, 4, 8) if backend else (1, 2, 4, 8, 464, 984))
                for magnitude in ((0, 0.0001, 0.1, 1, 3) if backend else (0.1,))
            )
            for rows, magnitude in cases:
                x = torch.randn(rows, gamma.numel(), dtype=torch.float16, device="npu") * magnitude
                residual = torch.randn_like(x) * magnitude
                config = torch.tensor([rows], dtype=torch.int64, device="npu")

                def reference(x=x, residual=residual, gamma=gamma):
                    norm, _, total = torch_npu.npu_add_rms_norm(x, residual, gamma, 1e-6)
                    q, scales = torch_npu.npu_dynamic_quant(norm)
                    return q, scales, total

                def candidate(x=x, residual=residual, gamma=gamma, config=config):
                    if backend is None:
                        return torch_npu.npu_add_rms_norm_dynamic_quant(x, residual, gamma, epsilon=1e-6)
                    norm, total = torch.empty_like(x), torch.empty_like(x)
                    q = backend.empty(x.shape[0], x.shape[1], torch.int8, x.device, True)
                    scales = torch.empty(x.shape[0], 8, dtype=torch.float32, device=x.device)
                    backend.launch(kernel, [x, residual, gamma, norm, total, q, scales, config])
                    return q, scales[:, 0], total, norm

                record = {
                    "module": key,
                    "rows": rows,
                    "magnitude": magnitude,
                    "device": args.device,
                    "implementation": "custom" if backend else "native",
                }
                try:
                    expected = reference()
                    actual = candidate()
                    torch.npu.synchronize()
                    record["outputs"] = [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in actual]
                    # Match every reference output by shape, dtype and exact values;
                    # output positions differ between the native operator versions.
                    equal = [
                        any(t.dtype == r.dtype and t.shape == r.shape and torch.equal(t.cpu(), r.cpu()) for t in actual)
                        for r in expected
                    ]
                    record.update({"supported": True, "reference_outputs_equal": equal, "passed": all(equal)})
                    if backend:
                        norm = torch_npu.npu_add_rms_norm(x, residual, gamma, 1e-6)[0]
                        record["norm_equal"] = torch.equal(norm.cpu(), actual[3].cpu())
                        record["norm_max_abs_error"] = float((norm.cpu() - actual[3].cpu()).abs().max())
                        record["passed"] = record["passed"] and record["norm_equal"]
                    if record["passed"]:
                        left, right = measure_pair(reference, candidate, torch.npu.synchronize, 100, 8)
                        record.update(
                            {
                                "control": left,
                                "candidate": right,
                                "reduction_percent": 100 * (1 - right["median_us"] / left["median_us"]),
                            }
                        )
                except Exception as error:
                    record.update({"supported": False, "passed": False, "error": str(error)[:2200]})
                records.append(record)
                args.output.write_text(json.dumps(records, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in record.items() if k not in ("error", "control", "candidate")}),
                    flush=True,
                )
                if not record.get("supported"):
                    return


if __name__ == "__main__":
    main()
