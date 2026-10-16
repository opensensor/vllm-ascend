"""Pair frozen native builds in one process, with real weights and mutable graphs."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from tools.rtsp_scene.dense_bench import checked_build, load_projection, measure_pair
from tools.rtsp_scene.dense_runtime import NZMLP, DecodeLinearMethod, DenseProjection, KernelBackend
from tools.rtsp_scene.dense_shapes import kernel_namespace

MAGNITUDES = (0.0, 0.0001, 0.1, 1.0, 3.0)


def check_build_pair(control: Path, candidate: Path) -> tuple[dict, dict]:
    manifests = checked_build(control), checked_build(candidate)
    if kernel_namespace(manifests[0]) == kernel_namespace(manifests[1]):
        raise ValueError("Paired bridges must use different frozen Torch namespaces")
    return manifests


class ReferenceLinear(nn.Module):
    def __init__(self, model, projection, rank, npu, layer=0):
        super().__init__()
        weight, scale = load_projection(model, projection, rank, 2, layer=layer)
        self.register_buffer("weight", npu.npu_format_cast(weight.npu(), 29).t())
        self.register_buffer("weight_scale", scale.npu())
        self.bias = None
        self.tp_size = 2
        self.input_is_parallel = True
        self.reduce_results = False
        self.quant = npu.npu_dynamic_quant
        self.matmul = npu.npu_quant_matmul

    def forward(self, x):
        q, scales = self.quant(x)
        return self.matmul(q, self.weight, self.weight_scale, pertoken_scale=scales, output_dtype=torch.float16)

    def apply(self, layer, x, bias=None):
        return self(x)


class ReferenceFFN(nn.Module):
    def __init__(self, model, rank, npu, layer=0):
        super().__init__()
        self.gate_up_proj = ReferenceLinear(model, "gate_up", rank, npu, layer)
        self.down_proj = ReferenceLinear(model, "down", rank, npu, layer)
        self.activation = npu.npu_swiglu

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_up_proj(x)))


def identity(x):
    return x


def compare_case(reference, functions, x, *, iterations, repeats, check_only=False):
    record = {"checks": []}
    for magnitude in MAGNITUDES:
        x.normal_(0, magnitude)
        expected = reference(x).cpu()
        actual = [function(x).cpu() for function in functions]
        finite = [bool(torch.isfinite(value).all()) for value in actual]
        record["checks"].append(
            {
                "magnitude": magnitude,
                "bitwise_equal": [bool(torch.equal(value, expected)) for value in actual],
                "reference_finite": bool(torch.isfinite(expected).all()),
                "finite": finite,
                "max_abs_error": [
                    float((value.float() - expected.float()).abs().max())
                    if ok and torch.isfinite(expected).all()
                    else None
                    for value, ok in zip(actual, finite)
                ],
            }
        )
    record["passed"] = all(all(c["bitwise_equal"]) and all(c["finite"]) for c in record["checks"])
    if not record["passed"] or check_only:
        return record
    graphs, outputs = [], []
    for function in functions:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
            output = function(x)
        graphs.append(graph)
        outputs.append(output)
    record["graph_checks"] = []
    for magnitude in (0.01, 0.3, 2.0):
        x.normal_(0, magnitude)
        expected = reference(x).cpu()
        for graph in graphs:
            graph.replay()
        actual = [output.cpu() for output in outputs]
        record["graph_checks"].append(
            {"magnitude": magnitude, "bitwise_equal": [bool(torch.equal(value, expected)) for value in actual]}
        )
    record["graph_passed"] = all(all(c["bitwise_equal"]) for c in record["graph_checks"])
    if record["graph_passed"]:
        record["graph_control"], record["graph_candidate"] = measure_pair(
            graphs[0].replay, graphs[1].replay, torch.npu.synchronize, iterations, repeats
        )
        record["reduction_percent"] = 100 * (
            1 - record["graph_candidate"]["median_us"] / record["graph_control"]["median_us"]
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0, help="Physical scene chip for this probe")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 9, 16])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--shared-device", action="store_true", help="Label timings as contended diagnostics")
    parser.add_argument("--layers", nargs="+", type=int, default=[0])
    parser.add_argument("--check-only", action="store_true", help="Check all requested weights without graph timings")
    args = parser.parse_args()
    if (
        args.output.exists()
        or min(*args.tokens, args.iterations, args.repeats) < 1
        or max(args.tokens) > 16
        or min(args.layers) < 0
    ):
        parser.error("Use a fresh output, positive timing counts and token rows in [1,16]")
    manifests = check_build_pair(args.control, args.candidate)
    # NPU runtime imports are restricted to the standalone initialized worker.
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20261013)
    backends = KernelBackend(args.control), KernelBackend(args.candidate)
    report = {
        "scope": "real requested TP shard weights; excludes HCCL, attention, head and vision; not serving throughput",
        "rank": args.rank,
        "device": args.device,
        "shared_device_with_live_logger": args.shared_device,
        "check_only": args.check_only,
        "timing": {"iterations": args.iterations, "repeats": args.repeats, "order": "alternating paired graphs"},
        "builds": manifests,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layer, name in ((layer, name) for layer in args.layers for name in ("qkv", "output", "ffn")):
        reference = (
            ReferenceFFN(args.model, args.rank, torch_npu, layer)
            if name == "ffn"
            else ReferenceLinear(args.model, name, args.rank, torch_npu, layer)
        )
        if name == "ffn":
            functions = tuple(NZMLP(reference, backend, identity) for backend in backends)
            inner = 4096
        else:
            projections = tuple(DenseProjection(reference, backend) for backend in backends)
            methods = tuple(DecodeLinearMethod(reference, projection) for projection in projections)
            functions = tuple(lambda x, method=method, layer=reference: method.apply(layer, x) for method in methods)
            inner = reference.weight.shape[0]
        for rows in args.tokens:
            x = torch.randn(rows, inner, dtype=torch.float16, device="npu")
            record = compare_case(
                reference,
                functions,
                x,
                iterations=args.iterations,
                repeats=args.repeats,
                check_only=args.check_only,
            )
            record.update({"projection": name, "rows": rows, "layer": layer})
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k not in ("checks", "graph_checks")}), flush=True)
        del functions, reference
    if not all(r["passed"] and (args.check_only or r.get("graph_passed", False)) for r in report["records"]):
        raise SystemExit("Frozen build comparison failed exact numerical parity")


if __name__ == "__main__":
    main()
