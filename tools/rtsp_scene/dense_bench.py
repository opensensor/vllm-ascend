"""Correctness-gated narrow INT8 projection experiments on actual TP shards."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from tools.rtsp_scene.dense_shapes import (
    DENSE_OUTPUT_TILES,
    PROJECTIONS,
    decode_rows,
    kernel_namespace,
    shard_slices,
    validate_cube_tile,
)


def pack_weight(weight, columns: int):
    """Pack [N,K] into output-tile-major [N/tile,K/32,tile/16,16,32]."""
    rows, inner = weight.shape
    if columns not in DENSE_OUTPUT_TILES or rows % columns or inner % 32:
        raise ValueError("Weight does not fit the frozen Cube tile")
    return weight.reshape(rows // columns, columns // 16, 16, inner // 32, 32).permute(0, 3, 1, 2, 4).contiguous()


def load_projection(model: Path, projection: str, rank: int, size: int, layer: int = 0):
    # Checkpoint imports remain outside the serving process.
    import torch
    from safetensors import safe_open

    if type(layer) is not int or layer < 0:
        raise ValueError("Checkpoint layer must be a nonnegative integer")
    prefix = f"model.language_model.layers.{layer}."
    groups = {
        "qkv": [prefix + "self_attn." + item for item in ("q_proj", "k_proj", "v_proj")],
        "output": [prefix + "self_attn.o_proj"],
        "gate_up": [prefix + "mlp." + item for item in ("gate_proj", "up_proj")],
        "down": [prefix + "mlp.down_proj"],
    }
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    weights, scales = [], []
    for name in groups[projection]:
        with safe_open(model / index[name + ".weight"], framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(name + ".weight")
        with safe_open(model / index[name + ".weight_scale"], framework="pt", device="cpu") as shard:
            scale = shard.get_tensor(name + ".weight_scale").flatten()
        row_slice, column_slice = shard_slices(projection, *weight.shape, rank, size)
        weights.append(weight[row_slice, column_slice])
        # Row-parallel shards retain the ORIGINAL scale; never requantize partial K.
        scales.append(scale[row_slice])
    return torch.cat(weights).contiguous(), torch.cat(scales).float().contiguous()


def measure(function, synchronize, iterations: int, repeats: int):
    for _ in range(5):
        function()
    synchronize()
    trials = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        synchronize()
        trials.append((time.perf_counter() - start) * 1e6 / iterations)
    return {"median_us": statistics.median(trials), "trials_us": trials}


def checked_build(directory: Path) -> dict:
    build = json.loads((directory / "build.json").read_text())
    validate_cube_tile(build["columns"], build["inner_tile"])
    kernel_namespace(build)
    decode_rows(build)
    for name in ("dense-bridge.so", "dense-k2048.bin", "dense-k4096.bin", "dense-k6144.bin", "nz-swiglu.bin"):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != build["sha256"].get(name):
            raise ValueError(f"Native build fingerprint mismatch: {name}")
    return build


def measure_pair(reference, custom, synchronize, iterations: int, repeats: int) -> tuple[dict, dict]:
    functions = (reference, custom)
    for function in functions:
        for _ in range(5):
            function()
    synchronize()
    trials = ([], [])
    for repeat in range(repeats):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                functions[index]()
            synchronize()
            trials[index].append((time.perf_counter() - start) * 1e6 / iterations)
    return tuple({"median_us": statistics.median(values), "trials_us": values} for values in trials)


def main() -> None:
    """Build projection inputs and compare reference and candidate kernel timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS, nargs="+", default=list(PROJECTIONS))
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(*args.tokens, args.iterations, args.repeats) <= 0 or max(args.tokens) > 16:
        parser.error("Use a fresh output file and token counts in [1,16]")
    import torch
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20261012)
    build = checked_build(args.build)
    torch.ops.load_library(str((args.build / "dense-bridge.so").resolve()))
    namespace = kernel_namespace(build)
    kernel_class = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch
    report = {
        "scope": "layer 0 checkpoint TP shard; isolated microbenchmark, not service throughput",
        "build": build,
        "rank": args.rank,
        "size": args.size,
        "records": [],
        "timing": {
            "iterations": args.iterations,
            "repeats": args.repeats,
            "order": "alternating baseline/candidate; warm resident weights; includes replay dispatch",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for projection in args.projection:
        weight_cpu, scales_cpu = load_projection(args.model, projection, args.rank, args.size)
        columns, inner = weight_cpu.shape
        weight = torch_npu.npu_format_cast(weight_cpu.npu(), 29).t()
        packed = pack_weight(weight_cpu, build["columns"]).npu()
        scale = scales_cpu.npu()
        kernel = kernel_class(str((args.build / f"dense-k{inner}.bin").resolve()), "rtsp_dense_i8")
        for rows in args.tokens:
            x = torch.randn(rows, inner, dtype=torch.float16, device="npu") * 0.1
            qx, xs = torch_npu.npu_dynamic_quant(x)

            def reference(qx=qx, weight=weight, scale=scale, xs=xs):
                return torch_npu.npu_quant_matmul(qx, weight, scale, pertoken_scale=xs, output_dtype=torch.float16)

            for nz in (False, True):
                config = torch.tensor([rows, columns, int(nz), 0, 1], dtype=torch.int64, device="npu")
                out = torch.ops.npu.empty_with_format(
                    [rows, columns], dtype=torch.float16, device=x.device, acl_format=29 if nz else 2
                )

                def custom(kernel=kernel, qx=qx, packed=packed, scale=scale, xs=xs, out=out, config=config):
                    launch(kernel, [qx, packed, scale, xs, out, config], 8)
                    return out

                record = {
                    "projection": projection,
                    "rows": rows,
                    "weight_shape": [columns, inner],
                    "nz_output": nz,
                    "checks": [],
                }
                for trial in range(3):
                    x.normal_(0, 0.1 * (trial + 1))
                    new_qx, new_xs = torch_npu.npu_dynamic_quant(x)
                    qx.copy_(new_qx)
                    xs.copy_(new_xs)
                    expected = reference().cpu().float()
                    actual = torch_npu.npu_format_cast(custom(), 2).cpu().float()
                    finite = bool(torch.isfinite(actual).all())
                    error = (actual - expected).abs()
                    maximum = float(error.max()) if finite else None
                    relative = (
                        float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
                        if finite
                        else None
                    )
                    passed = finite and bool(torch.equal(actual, expected))
                    record["checks"].append(
                        {
                            "finite": finite,
                            "max_absolute_error": maximum,
                            "relative_l2_error": relative,
                            "passed": passed,
                        }
                    )
                    if not passed:
                        record["diagnostic"] = {
                            "expected": expected[0, :32].tolist(),
                            "actual": actual[0, :32].tolist(),
                        }
                        break
                record["passed"] = all(check["passed"] for check in record["checks"])
                if record["passed"]:
                    record["eager_reference"], record["eager_custom"] = measure_pair(
                        reference, custom, torch.npu.synchronize, args.iterations, args.repeats
                    )
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        custom()
                    x.normal_(0, 0.15)
                    new_qx, new_xs = torch_npu.npu_dynamic_quant(x)
                    qx.copy_(new_qx)
                    xs.copy_(new_xs)
                    graph.replay()
                    actual = torch_npu.npu_format_cast(out, 2).cpu().float()
                    expected = reference().cpu().float()
                    record["graph_changed_input_passed"] = bool(torch.equal(actual, expected))
                    if record["graph_changed_input_passed"]:
                        baseline_graph = torch.npu.NPUGraph()
                        with torch.npu.graph(baseline_graph):
                            reference()
                        record["graph_reference"], record["graph_custom"] = measure_pair(
                            baseline_graph.replay, graph.replay, torch.npu.synchronize, args.iterations, args.repeats
                        )
                    del graph
                report["records"].append(record)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
