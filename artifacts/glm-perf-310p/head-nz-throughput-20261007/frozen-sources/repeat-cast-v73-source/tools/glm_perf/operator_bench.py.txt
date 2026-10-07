# SPDX-License-Identifier: Apache-2.0
"""Isolated GLM 310P grouped projection benchmark.

Run ``measure`` once per OPP package in separate processes, with the same seed
and options. ``compare`` reads the two result files without loading either OPP.
These are operator latencies, not serving throughput.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

EXPERTS = 72
OUTPUT_TILE = 16
INPUT_TILE = 128
WORKSPACE_OUTPUT_TILE = 128
OPERATOR = "npu_w2_grouped_blocked_dequant_matmul_310"
OPP_OPERATOR_NAME = "w2groupedblockeddequantmatmulv310"
DEFAULT_WARMUP = 4
DEFAULT_REPEATS = 20
DEFAULT_SEED = 20260930
MAX_GROUPED_ROUTES = 5120
DEFAULT_PREFILL_ROWS = 416
DEFAULT_ROUTE_PATTERNS = ("distributed", "repeated", "singleton", "mixed", "zero_local", "peer_owned")
EXTRA_ROUTE_PATTERNS = ("uniform_all_experts", "uniform_all_experts_quarter")


@dataclass(frozen=True)
class Case:
    projection: str
    bits: int
    output_width: int
    input_width: int
    routed_rows: int
    route_pattern: str

    @property
    def key(self) -> str:
        return f"{self.projection}_{self.routed_rows}_{self.route_pattern}"


def cases(
    prefill_rows: int = DEFAULT_PREFILL_ROWS,
    *,
    row_sizes: tuple[int, ...] | None = None,
    route_patterns: tuple[str, ...] | None = None,
) -> list[Case]:
    """Cover decode and prefill, including empty, singleton, and repeated groups."""
    if not 32 < prefill_rows <= MAX_GROUPED_ROUTES:
        raise ValueError(f"prefill_rows must be 33..{MAX_GROUPED_ROUTES} (grouped operator route bound)")
    selected_rows = (8, 32, prefill_rows) if row_sizes is None else row_sizes
    if not selected_rows or any(not 0 < row <= MAX_GROUPED_ROUTES for row in selected_rows):
        raise ValueError(f"row_sizes must be 1..{MAX_GROUPED_ROUTES}")
    patterns = DEFAULT_ROUTE_PATTERNS if route_patterns is None else route_patterns
    if not patterns or any(pattern not in (*DEFAULT_ROUTE_PATTERNS, *EXTRA_ROUTE_PATTERNS) for pattern in patterns):
        raise ValueError("route_patterns contains an unknown route pattern")
    shapes = (("w4_gate_up", 4, 4096, 4096), ("w2_down", 2, 4096, 2048))
    return [
        Case(name, bits, output, input_width, row_count, pattern)
        for name, bits, output, input_width in shapes
        for row_count in selected_rows
        for pattern in patterns
    ]


def route_counts(case: Case) -> list[int]:
    """Counts are local expert routes; peer-owned rows are omitted."""
    counts = [0] * EXPERTS
    if case.route_pattern == "zero_local":
        return counts
    if case.route_pattern == "singleton":
        for expert in range(min(EXPERTS, case.routed_rows)):
            counts[expert] = 1
        return counts
    if case.route_pattern == "mixed":
        counts[0], counts[17], counts[71] = 1, case.routed_rows - 2, 1
        return counts
    if case.route_pattern in EXTRA_ROUTE_PATTERNS:
        local_rows = case.routed_rows // 4 if case.route_pattern == "uniform_all_experts_quarter" else case.routed_rows
        for row in range(local_rows):
            counts[row % EXPERTS] += 1
        return counts
    local_rows = case.routed_rows // 2 if case.route_pattern == "peer_owned" else case.routed_rows
    if case.route_pattern == "repeated":
        counts[17] = local_rows
    elif case.route_pattern in ("distributed", "peer_owned"):
        active = (0, 17) if case.routed_rows == 8 else (0, 8, 17, 26, 35, 44, 53, 71)
        for row in range(local_rows):
            counts[active[row % len(active)]] += 1
    else:
        raise ValueError(f"unknown route pattern: {case.route_pattern}")
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_package(opp_root: Path, binary: Path, sources: list[Path], layout: str) -> dict:
    root = opp_root.resolve(strict=True)
    binary = binary.resolve(strict=True)
    if not binary.is_file() or not binary.is_relative_to(root):
        raise ValueError("--opp-binary must be a file inside --opp-root")
    configured = [Path(path).resolve() for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if path]
    if configured != [root]:
        raise ValueError("ASCEND_CUSTOM_OPP_PATH must contain only --opp-root for an isolated measurement")
    relative_binary = binary.relative_to(root)
    binary_identity = "".join(char for char in str(relative_binary).lower() if char.isalnum())
    if OPP_OPERATOR_NAME not in binary_identity or binary.suffix != ".o":
        raise ValueError("--opp-binary must be a grouped W2/W4 compiled operator binary")
    sidecar = binary.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    code_inputs = [entry for entry in metadata["supportInfo"]["inputs"] if entry["name"] == "codes"]
    expected_dtype = {"canonical": "uint8", "nzpacked": "int8"}[layout]
    if len(code_inputs) != 1 or code_inputs[0]["dtype"] != expected_dtype:
        raise ValueError(f"--opp-binary metadata does not match {layout} code dtype {expected_dtype}")
    if not sources:
        raise ValueError("at least one --source is required")
    return {
        "opp_root": str(root),
        "opp_search_path": [str(path) for path in configured],
        "operator": OPERATOR,
        "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "binary_metadata": {"path": str(sidecar), "sha256": sha256_file(sidecar), "code_dtype": expected_dtype},
        "sources": [{"path": str(source.resolve(strict=True)), "sha256": sha256_file(source)} for source in sources],
    }


def summarize_timings(samples_ms: list[float]) -> dict:
    if not samples_ms or any(sample < 0 for sample in samples_ms):
        raise ValueError("timing samples must be nonempty and nonnegative")
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def compare_outputs(reference, candidate, *, atol: float, rtol: float) -> dict:
    """Compare CPU tensors; exact mode uses bitwise equality, including signed zero."""
    import torch

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("output shape or dtype differs")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("output contains non-finite values")
    if atol == 0 and rtol == 0:
        equal = torch.equal(reference.view(torch.uint8), candidate.view(torch.uint8))
    else:
        equal = torch.allclose(reference, candidate, atol=atol, rtol=rtol)
    delta = (reference.float() - candidate.float()).abs()
    return {
        "passed": bool(equal),
        "output_dtype": str(reference.dtype),
        "max_abs_error": float(delta.max()) if delta.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
        "comparison": "bitwise" if atol == rtol == 0 else "bounded_reduction_order",
    }


def nzpacked_codes(codes, k: int, bits: int):
    """Losslessly reorder each packed 16 x 128 tile into Cube NZ order."""
    import torch

    experts, n, packed_k = codes.shape
    codes_per_byte = 8 // bits
    if n % OUTPUT_TILE or k % INPUT_TILE or packed_k != k // codes_per_byte:
        raise ValueError("packed code geometry does not fit the NZ tile")
    n_tiles, k_tiles = n // OUTPUT_TILE, k // INPUT_TILE
    tile_bytes = OUTPUT_TILE * INPUT_TILE // codes_per_byte
    rows = codes.view(experts, n_tiles, OUTPUT_TILE, k_tiles, INPUT_TILE // codes_per_byte)
    rows = rows.permute(0, 1, 3, 2, 4)
    fields = torch.stack([(rows >> (bits * field)) & ((1 << bits) - 1) for field in range(codes_per_byte)], dim=-1)
    nz = fields.reshape(experts, n_tiles, k_tiles, OUTPUT_TILE, INPUT_TILE)
    nz = nz.permute(0, 1, 2, 4, 3).reshape(experts, n_tiles, k_tiles, codes_per_byte, tile_bytes)
    out = nz[..., 0, :].clone()
    for field in range(1, codes_per_byte):
        out |= nz[..., field, :] << (bits * field)
    return out.reshape(experts, n, packed_k).contiguous()


def measure_case(case: Case, args, op) -> tuple[dict, object]:
    import torch

    generator = torch.Generator().manual_seed(args.seed + case.bits + case.routed_rows)
    codes = torch.randint(
        0,
        256,
        (EXPERTS, case.output_width, case.input_width // (8 // case.bits)),
        generator=generator,
        dtype=torch.uint8,
    )
    logical_code_hash = hashlib.sha256(codes.numpy().tobytes()).hexdigest()
    if args.layout == "nzpacked":
        codes = nzpacked_codes(codes, case.input_width, case.bits).view(torch.int8)
    code_hash = hashlib.sha256(codes.numpy().tobytes()).hexdigest()
    scales = torch.rand(EXPERTS, case.output_width // 32, case.input_width // 32, generator=generator) * 0.02 + 0.005
    inputs = torch.randn(case.routed_rows, case.input_width, generator=generator).half()
    scale_hash = hashlib.sha256(scales.numpy().tobytes()).hexdigest()
    input_hash = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    scales, inputs = scales.npu(), inputs.npu()
    counts = route_counts(case)
    ends = torch.tensor([sum(counts[: index + 1]) for index in range(EXPERTS)], dtype=torch.int64, device="npu")
    codes = codes.npu()
    for _ in range(args.warmup):
        output = op(inputs, codes, scales, ends)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    before_bytes = torch.npu.memory_allocated()
    samples = []
    for _ in range(args.repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        output = op(inputs, codes, scales, ends)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    peak_delta = max(0, torch.npu.max_memory_allocated() - before_bytes)
    output = output.cpu()
    if not torch.isfinite(output).all():
        raise RuntimeError(f"non-finite output: {case.key}")
    record = {
        **asdict(case),
        "key": case.key,
        "local_rows": sum(counts),
        "expert_counts": counts,
        "logical_code_sha256": logical_code_hash,
        "packed_code_sha256": code_hash,
        "scale_sha256": scale_hash,
        "input_sha256": input_hash,
        "packed_code_dtype": str(codes.dtype),
        "output_dtype": str(output.dtype),
        "output_shape": list(output.shape),
        "operator_latency": summarize_timings(samples),
        "operator_workspace_bytes": None,
        "operator_workspace_note": "ACL tiling workspace is not exposed by this PyTorch wrapper",
        "operator_workspace_formula": (
            "GetLibApiWorkSpaceSize() + min(GetCoreNumAic(), output_width / 128) * 128 * input_width * sizeof(uint16_t)"
        ),
        "workspace_tile_bytes_per_core": WORKSPACE_OUTPUT_TILE * case.input_width * 2,
        "allocator_peak_delta_bytes": peak_delta,
        "allocator_peak_delta_note": (
            "PyTorch allocator peak delta includes output allocations and may omit ACL workspace"
        ),
    }
    return record, output


def measure(args) -> None:
    import torch
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("warmup must be >=1 and repeats >=2")
    package = describe_package(args.opp_root, args.opp_binary, args.source, args.layout)
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("measure requires Ascend 310P")
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    op = getattr(torch.ops._C_ascend, OPERATOR)
    records, outputs = [], {}
    for case in cases(args.prefill_rows, row_sizes=args.rows, route_patterns=args.route_patterns):
        record, output = measure_case(case, args, op)
        records.append(record)
        outputs[case.key] = output
        print(json.dumps({"case": case.key, "median_ms": record["operator_latency"]["median_ms"]}), flush=True)
    if sha256_file(Path(package["binary"]["path"])) != package["binary"]["sha256"]:
        raise RuntimeError("OPP binary changed during measurement")
    if sha256_file(Path(package["binary_metadata"]["path"])) != package["binary_metadata"]["sha256"]:
        raise RuntimeError("OPP binary metadata changed during measurement")
    payload = {
        "schema_version": 1,
        "kind": "isolated_operator_measurement",
        "label": args.label,
        "package": package,
        "layout": args.layout,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "device": args.device,
        "torch": str(torch.__version__),
        "torch_npu": torch_npu.__version__,
        "records": records,
        "outputs": outputs,
    }
    torch.save(payload, args.output)


def compare(args) -> None:
    import torch
    from torch.torch_version import TorchVersion

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("parity bounds must be nonnegative")
    # Older measurements serialized torch.__version__ as TorchVersion rather
    # than str. These files are locally generated by this harness; allow just
    # that benign value type while keeping weights-only loading enabled.
    with torch.serialization.safe_globals([TorchVersion]):
        baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
        candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        if payload.get("schema_version") != 1 or payload.get("kind") != "isolated_operator_measurement":
            raise ValueError(f"{name} has unsupported measurement schema")
    if baseline["package"]["binary"]["sha256"] == candidate["package"]["binary"]["sha256"]:
        raise ValueError("baseline and candidate use the same OPP binary hash")
    if (
        baseline["seed"] != candidate["seed"]
        or baseline["warmup"] != candidate["warmup"]
        or baseline["repeats"] != candidate["repeats"]
    ):
        raise ValueError("seed, warmup, and repeat counts must match")
    baseline_records = {record["key"]: record for record in baseline["records"]}
    candidate_records = {record["key"]: record for record in candidate["records"]}
    if baseline_records.keys() != candidate_records.keys():
        raise ValueError("case sets differ")
    results = []
    for key in sorted(baseline_records):
        before, after = baseline_records[key], candidate_records[key]
        for field in (
            "projection",
            "bits",
            "output_width",
            "input_width",
            "routed_rows",
            "route_pattern",
            "expert_counts",
            "local_rows",
            "output_dtype",
            "logical_code_sha256",
            "scale_sha256",
            "input_sha256",
        ):
            if before[field] != after[field]:
                raise ValueError(f"case {key}: {field} differs")
        if baseline["layout"] == candidate["layout"] and before["packed_code_sha256"] != after["packed_code_sha256"]:
            raise ValueError(f"case {key}: packed codes differ")
        parity = compare_outputs(baseline["outputs"][key], candidate["outputs"][key], atol=args.atol, rtol=args.rtol)
        baseline_ms = before["operator_latency"]["median_ms"]
        candidate_ms = after["operator_latency"]["median_ms"]
        results.append(
            {
                "key": key,
                "parity": parity,
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "speedup_fraction": (baseline_ms - candidate_ms) / baseline_ms,
            }
        )
    summary = {
        "schema_version": 1,
        "kind": "isolated_operator_comparison",
        "metric_scope": "operator latency only; serving throughput must be measured separately",
        "baseline": {"path": str(args.baseline), "package": baseline["package"], "layout": baseline["layout"]},
        "candidate": {"path": str(args.candidate), "package": candidate["package"], "layout": candidate["layout"]},
        "parity_bounds": {"atol": args.atol, "rtol": args.rtol},
        "results": results,
        "all_parity_passed": all(result["parity"]["passed"] for result in results),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"all_parity_passed": summary["all_parity_passed"], "results": len(results)}))
    if not summary["all_parity_passed"]:
        raise SystemExit("output parity failed; see comparison JSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure_parser = sub.add_parser("measure", help="run one OPP package on 310P")
    measure_parser.add_argument("--opp-root", type=Path, required=True, help="entry in ASCEND_CUSTOM_OPP_PATH")
    measure_parser.add_argument("--opp-binary", type=Path, required=True, help="actual operator binary inside OPP root")
    measure_parser.add_argument(
        "--source", type=Path, action="append", required=True, help="kernel source file; repeatable"
    )
    measure_parser.add_argument("--layout", choices=("canonical", "nzpacked"), required=True)
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    measure_parser.add_argument("--device", type=int, default=0)
    measure_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    measure_parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    measure_parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    measure_parser.add_argument("--prefill-rows", type=int, default=DEFAULT_PREFILL_ROWS)
    measure_parser.add_argument("--rows", type=int, nargs="+", help="explicit routed-row sizes for a focused sweep")
    measure_parser.add_argument(
        "--route-patterns",
        nargs="+",
        choices=(*DEFAULT_ROUTE_PATTERNS, *EXTRA_ROUTE_PATTERNS),
        help="restrict the measurement to selected route patterns",
    )
    measure_parser.set_defaults(func=measure)
    compare_parser = sub.add_parser("compare", help="compare outputs from separate measure processes")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--atol", type=float, default=0.0, help="predeclared absolute reduction-order bound")
    compare_parser.add_argument("--rtol", type=float, default=0.0, help="predeclared relative reduction-order bound")
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
