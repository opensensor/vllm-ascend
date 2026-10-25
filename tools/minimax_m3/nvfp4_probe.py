# SPDX-License-Identifier: Apache-2.0
"""Prepare and measure a bounded source-preserving real-expert fixture."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tools.minimax_m3.int8_native import GroupedW4Kernel as Int8Kernel
from tools.minimax_m3.mixed_checkpoint import dequant_nvfp4
from tools.minimax_m3.nvfp4 import pack_source, unpack_source
from tools.minimax_m3.nvfp4_native import GroupedW4Kernel, validate_kernel
from vllm_ascend._310p.ops.minimax_m3 import clipped_swiglu


def prepare(source, inputs, output, layer):
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    records = torch.load(inputs, weights_only=False, map_location="cpu")
    ids = records[0]["ends"].tolist()
    fixture = {"layer": layer, "expert_ids": ids, "input": records[0]["input"], "banks": {}, "expected": {}}
    expected = {}
    for projection in ("w1", "w3", "w2"):
        banks, results = [], []
        for row, expert in enumerate(ids):
            base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}"
            tensors = []
            for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale"):
                name = base + suffix
                with safe_open(source / index[name], framework="pt", device="cpu") as handle:
                    tensors.append(handle.get_tensor(name))
            bank = pack_source(*tensors)
            reference = dequant_nvfp4(*tensors).half()
            if not torch.equal(unpack_source(*bank), reference):
                raise AssertionError("Source-preserving fixture changed decoded FP16 weights")
            x = (
                fixture["input"][row : row + 1]
                if projection != "w2"
                else clipped_swiglu(expected["w1"][row : row + 1], expected["w3"][row : row + 1]).half()
            )
            results.append((x.float() @ reference.float().T).half())
            banks.append(bank)
        fixture["banks"][projection] = tuple(torch.stack(t) for t in zip(*banks))
        expected[projection] = torch.cat(results)
    fixture["expected"] = expected
    torch.save(fixture, output)
    return {"layer": layer, "experts": ids, "decoded_weights_exact": True, "fixture_bytes": output.stat().st_size}


def compare(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {
        "nrmse": float((actual - expected).norm() / expected.norm().clamp_min(1e-30)),
        "max_abs": float((actual - expected).abs().max()),
        "finite": bool(actual.isfinite().all()),
    }


def graph_check(kernel, x, codes, scales, routes, lookup, iterations):
    """Exercise changing device inputs/routes without recapture or host copies."""
    fixed_x, fixed_routes = x.clone(), routes.clone()
    kernel(fixed_x, codes, scales, fixed_routes, lookup)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = kernel(fixed_x, codes, scales, fixed_routes, lookup)
    checks = []
    for factor in (0.5, -1.25):
        fixed_x.copy_(x * factor)
        fixed_routes.copy_(routes.flip(0))
        graph.replay()
        actual = captured.cpu()
        expected = kernel(fixed_x, codes, scales, fixed_routes, lookup).cpu()
        checks.append(torch.equal(actual, expected))
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.npu.synchronize()
    return {"changing_inputs_exact": all(checks), "milliseconds": (time.perf_counter() - started) * 1000 / iterations}


def measure_w4(manifest, banks_path, fixture_path, iterations):
    """Optional matched current-W4 baseline; includes its INT8 compute error."""
    spec = json.loads(manifest.read_text())
    for library in spec["libraries"]:
        torch.ops.load_library(library["path"])
    kernel = Int8Kernel(spec["name"], str(manifest.parent / "grouped_w4.bin"), spec["name"] + "_kernel")
    fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
    banks = torch.load(banks_path, weights_only=True, map_location="cpu")
    x = fixture["input"].npu()
    routes = torch.arange(x.shape[0], dtype=torch.int32, device=x.device)
    results, cases = {}, {}
    for projection in ("w1", "w3", "w2"):
        codes, scales = [t.npu() for t in banks[projection]]
        input_ = x if projection != "w2" else clipped_swiglu(results["w1"], results["w3"]).half()
        results[projection] = kernel(input_, codes, scales, routes)
        case = compare(results[projection].cpu(), fixture["expected"][projection])
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            kernel(input_, codes, scales, routes)
        torch.npu.synchronize()
        case["milliseconds"] = (time.perf_counter() - started) * 1000 / iterations
        cases[projection] = case
    return cases


def measure(manifest, fixture_path, device, iterations):
    # Only this explicitly invoked device probe imports torch_npu.
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    spec = json.loads(manifest.read_text())
    for library in spec["libraries"]:
        torch.ops.load_library(library["path"])
    kernel = GroupedW4Kernel(spec["name"], str(manifest.parent / "grouped_w4.bin"), spec["name"] + "_kernel")
    synthetic = validate_kernel(kernel)
    if not synthetic["passed"]:
        return {"passed": False, "synthetic": synthetic}
    fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
    x = fixture["input"].npu()
    routes = torch.arange(x.shape[0], dtype=torch.int32, device=x.device)
    outputs, cases = {}, {}
    for projection in ("w1", "w3", "w2"):
        codes, scales, lookup = [t.npu() for t in fixture["banks"][projection]]
        input_ = x if projection != "w2" else clipped_swiglu(outputs["w1"], outputs["w3"]).half()
        output = kernel(input_, codes, scales, routes, lookup)
        outputs[projection] = output
        case = compare(output.cpu(), fixture["expected"][projection])
        case["per_expert"] = [compare(a, b) for a, b in zip(output.cpu(), fixture["expected"][projection])]
        # Compact routing and grouped routing must refer to the same experts.
        ends = torch.arange(1, x.shape[0] + 1, dtype=torch.int64, device=x.device)
        grouped = kernel(input_, codes, scales, ends, lookup)
        case["compact_grouped_exact"] = torch.equal(grouped, output)
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            kernel(input_, codes, scales, routes, lookup)
        torch.npu.synchronize()
        case["milliseconds"] = (time.perf_counter() - started) * 1000 / iterations
        case["graph"] = graph_check(kernel, input_, codes, scales, routes, lookup, iterations)
        cases[projection] = case
        del codes, scales, lookup
    return {
        "passed": all(
            c["finite"] and c["nrmse"] < 0.003 and c["compact_grouped_exact"] and c["graph"]["changing_inputs_exact"]
            for c in cases.values()
        ),
        "synthetic": synthetic,
        "real": cases,
        "layer": fixture["layer"],
        "expert_ids": fixture["expert_ids"],
        "activation_quantization": "FP16; source FP4 activation quantization is not reproduced",
        "loop_resolution_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "measure"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--w4-manifest", type=Path)
    parser.add_argument("--w4-banks", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if (args.w4_manifest is None) != (args.w4_banks is None):
        parser.error("provide both w4-manifest and w4-banks")
    torch.set_num_threads(8)
    with torch.inference_mode():
        if args.action == "prepare":
            if args.source is None or args.inputs is None:
                parser.error("prepare requires source and inputs")
            result = prepare(args.source, args.inputs, args.fixture, args.layer)
        else:
            if args.manifest is None:
                parser.error("measure requires manifest")
            result = measure(args.manifest, args.fixture, args.device, args.iterations)
            if args.w4_manifest is not None:
                result["w4_int8_baseline"] = measure_w4(args.w4_manifest, args.w4_banks, args.fixture, args.iterations)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
