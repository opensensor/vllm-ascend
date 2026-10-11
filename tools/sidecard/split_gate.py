# SPDX-License-Identifier: Apache-2.0
"""Real selected-expert CUDA/Ascend packet gate; no attention/model serving."""

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

from .cuda_combine import CudaCombiner
from .device_backend import prepare_cpu_banks
from .external_runtime import ExternalWorkerPool, WorkerLaunch
from .native_expert_probe import digest_file
from .placement import ASCEND_OWNERS, CUDA_OWNER, ExpertPlacement
from .reference import ExpertReply, combine, dispatch
from .wire import PacketLimits, WireIdentity, decode_packet, encode_packet

GATE_EXPERTS = 8
GATE_WIDTH = 4096
GATE_TOKENS = (1, 4, 17)
GATE_ASSIGNMENT = (CUDA_OWNER, "npu:0", "npu:1", "npu:2", "npu:3", CUDA_OWNER, "npu:0", "npu:3")
RELATIVE_TOLERANCE = 0.002
ABSOLUTE_TOLERANCE = 0.002


def gate_placement(layer):
    if type(layer) is not int or not 3 <= layer < 45:
        raise ValueError("split gate requires one main-model expert layer")
    return ExpertPlacement((CUDA_OWNER, *ASCEND_OWNERS), (GATE_ASSIGNMENT,), first_moe_layer=layer)


def prepare_golden(checkpoint, output, layers):
    """Create CPU native-contract expectations in a separate preparation process."""
    if not layers or len(layers) > 3 or len(set(layers)) != len(layers):
        raise ValueError("split golden requires one to three distinct layers")
    for layer in layers:
        gate_placement(layer)
    output.mkdir(parents=True, exist_ok=False)
    identity_sha = digest_file(checkpoint / "native-layout.json")

    import torch

    from tools.glm_perf.fused_moe_probe import projection_reference
    from tools.glm_perf.fused_weight_layout import unpack_cube

    manifest = {"checkpoint_root": str(checkpoint.resolve()), "checkpoint_sha256": identity_sha, "layers": []}
    for layer in layers:
        placement = gate_placement(layer)
        identity = WireIdentity.for_placement(placement, identity_sha)
        limits = PacketLimits(GATE_WIDTH, max(GATE_TOKENS), max(GATE_TOKENS) * GATE_EXPERTS, GATE_EXPERTS, GATE_EXPERTS)
        bank = prepare_cpu_banks(checkpoint, tuple((layer, expert) for expert in range(GATE_EXPERTS)), identity_sha)[
            layer
        ]
        gate, gs, down, ds = bank["tensors"]
        gate, down = unpack_cube(gate, 4096), unpack_cube(down, 2048)
        cases = []
        generator = torch.Generator().manual_seed(5813 + layer)
        for rows in GATE_TOKENS:
            base = torch.randn(rows, GATE_WIDTH, generator=generator).half()
            for step, multiplier in enumerate((1.0, 0.75)):
                x = (base * multiplier).half()
                hidden = tuple(tuple(row) for row in x.float().tolist())
                # Rotate original route slots; include a retained zero weight.
                ids = tuple(
                    tuple((slot + token + step) % GATE_EXPERTS for slot in range(GATE_EXPERTS)) for token in range(rows)
                )
                weights = tuple(
                    tuple(
                        0.0 if slot == (token + step) % GATE_EXPERTS else (slot + 1) / 32
                        for slot in range(GATE_EXPERTS)
                    )
                    for token in range(rows)
                )
                shared = tuple(
                    tuple(((column + token) % 7 - 3) / 128 for column in range(GATE_WIDTH)) for token in range(rows)
                )
                raw = []
                for expert in range(GATE_EXPERTS):
                    gu = projection_reference(x, gate[expert], gs[expert].float(), bank["plan"]["activation_bits"])
                    g, u = gu.chunk(2, -1)
                    activated = (torch.nn.functional.silu(g.float()) * u.float()).clamp(-65504, 65504).half()
                    result = projection_reference(
                        activated, down[expert], ds[expert].float(), bank["plan"]["activation_bits"]
                    ).float()
                    raw.append(tuple(tuple(row) for row in result.tolist()))
                expected_replies = []
                requests = dispatch(placement, "golden", layer, hidden, ids, weights)
                case_prefix = f"layer-{layer}-rows-{rows}-step-{step}"
                paths = {}
                for request in requests:
                    reply = ExpertReply(
                        "golden",
                        layer,
                        request.owner,
                        request.routes,
                        tuple(raw[route.expert][route.token] for route in request.routes),
                    )
                    filename = case_prefix + f"-owner-{placement.owners.index(request.owner)}.bin"
                    frame = encode_packet(reply, identity, limits)
                    (output / filename).write_bytes(frame)
                    paths[request.owner] = {"file": filename, "sha256": digest_file(output / filename)}
                    expected_replies.append(reply)
                expected = combine(placement, "golden", layer, hidden, ids, weights, expected_replies, shared, 2.5)
                filename = case_prefix + "-input.bin"
                (output / filename).write_bytes(x.numpy().tobytes())
                combined = case_prefix + "-combined.bin"
                (output / combined).write_bytes(b"".join(struct.pack("<4096f", *row) for row in expected))
                cases.append(
                    {
                        "rows": rows,
                        "step": step,
                        "hidden_file": filename,
                        "hidden_sha256": digest_file(output / filename),
                        "ids": ids,
                        "weights": weights,
                        "shared": shared,
                        "routed_scale": 2.5,
                        "raw_replies": paths,
                        "combined_file": combined,
                        "combined_sha256": digest_file(output / combined),
                    }
                )
        manifest["layers"].append({"layer": layer, "cases": cases})
    (output / "golden.json").write_text(json.dumps(manifest) + "\n")
    return {
        "golden_root": str(output),
        "golden_sha256": digest_file(output / "golden.json"),
        "checkpoint_sha256": identity_sha,
        "hardware_validated": False,
        "cases": sum(len(layer["cases"]) for layer in manifest["layers"]),
    }


def compare_rows(actual, expected):
    if len(actual) != len(expected) or any(len(a) != len(e) for a, e in zip(actual, expected, strict=True)):
        raise ValueError("split gate output shape mismatch")
    maximum, squares, norm, count, equal = 0.0, 0.0, 0.0, 0, True
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for value, reference in zip(actual_row, expected_row, strict=True):
            if not math.isfinite(value) or not math.isfinite(reference):
                raise ValueError("nonfinite split gate result")
            delta = abs(value - reference)
            if delta > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(reference):
                raise AssertionError(f"split gate parity failed: actual={value}, expected={reference}")
            maximum = max(maximum, delta)
            squares += delta * delta
            norm += reference * reference
            count += 1
            equal &= struct.pack("<f", value) == struct.pack("<f", reference)
    return {
        "max_absolute_error": maximum,
        "implementation_nrmse": math.sqrt(squares / max(norm, 1e-16)),
        "elements": count,
        "bitwise_equal": equal,
    }


def read_checked(root, filename, expected_sha):
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or path.stat().st_size > 16 * 1024**2:
        raise ValueError("unsafe split gate fixture path or size")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("split gate fixture checksum mismatch")
    return data


def run(golden, expected_golden_sha, npu_python, cuda_python, library, cuda_runtime, logs):
    raw = read_checked(golden, "golden.json", expected_golden_sha)
    manifest = json.loads(raw)
    checkpoint = Path(manifest["checkpoint_root"])
    if digest_file(checkpoint / "native-layout.json") != manifest["checkpoint_sha256"]:
        raise ValueError("split gate source differs from golden checkpoint")
    logs.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "eight selected real GLM experts, CUDA reduction/correctness kernel and four Ascend workers",
        "golden_sha256": expected_golden_sha,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "cuda_library_sha256": digest_file(library),
        "full_glm_inference_validated": False,
        "cuda_attention_validated": False,
        "coordinator_reduction": "cuda_ordered_fp32",
        "capacity_admitted": False,
        "cases": [],
        "workers": [],
        "reduction_contract_gates": [],
    }
    for layer_record in manifest["layers"]:
        layer = layer_record["layer"]
        placement = gate_placement(layer)
        limits = PacketLimits(GATE_WIDTH, max(GATE_TOKENS), max(GATE_TOKENS) * GATE_EXPERTS, GATE_EXPERTS, GATE_EXPERTS)
        identity = WireIdentity.for_placement(placement, manifest["checkpoint_sha256"])
        launches = {
            owner: WorkerLaunch(
                npu_python, {"kind": "ascend_native_gate", "checkpoint_root": str(checkpoint), "device_index": index}
            )
            for index, owner in enumerate(ASCEND_OWNERS)
        }
        launches[CUDA_OWNER] = WorkerLaunch(
            cuda_python,
            {
                "kind": "cuda_oracle_gate",
                "checkpoint_root": str(checkpoint),
                "library": str(library.resolve()),
                "library_sha256": report["cuda_library_sha256"],
                "cuda_runtime": str(cuda_runtime),
            },
        )
        with (
            CudaCombiner(library, report["cuda_library_sha256"], cuda_runtime, max(GATE_TOKENS)) as combiner,
            ExternalWorkerPool(
                placement, limits, identity, launches, logs / f"layer-{layer}", combiner=combiner
            ) as pool,
        ):
            for name, values, reduction_weights in (
                ("cancellation_order", (1e8, 1.0, -1e8), (1.0, 1.0, 1.0)),
                ("separate_multiply_add", (-1.0, 1 - 2**-23), (1.0, 1 + 2**-23)),
            ):
                ids = (tuple(range(len(values))),)
                hidden = ((0.0,) * GATE_WIDTH,)
                shared = ((0.0,) * GATE_WIDTH,)
                requests = dispatch(placement, name, layer, hidden, ids, (reduction_weights,))
                replies = tuple(
                    ExpertReply(
                        name,
                        layer,
                        request.owner,
                        request.routes,
                        tuple((values[route.slot],) * GATE_WIDTH for route in request.routes),
                    )
                    for request in reversed(requests)
                )
                expected = combine(placement, name, layer, hidden, ids, (reduction_weights,), replies, shared, 2.5)
                actual = combiner(placement, name, layer, hidden, ids, (reduction_weights,), replies, shared, 2.5)
                parity = compare_rows(actual, expected)
                if not parity["bitwise_equal"]:
                    raise AssertionError("CUDA reduction changed the FP32 arithmetic contract")
                report["reduction_contract_gates"].append({"layer": layer, "kind": name, **parity})
            report["workers"].append({"layer": layer, "pids": {owner: p.pid for owner, p in pool.processes.items()}})
            for case in layer_record["cases"]:
                body = read_checked(golden, case["hidden_file"], case["hidden_sha256"])
                hidden = tuple(struct.unpack_from("<4096e", body, row * GATE_WIDTH * 2) for row in range(case["rows"]))
                reference = read_checked(golden, case["combined_file"], case["combined_sha256"])
                expected = tuple(
                    struct.unpack_from("<4096f", reference, row * GATE_WIDTH * 4) for row in range(case["rows"])
                )
                execution_id = f"raw-check:{layer}:{case['rows']}:{case['step']}"
                requests = dispatch(placement, execution_id, layer, hidden, case["ids"], case["weights"])
                owner_errors = {}
                for request in requests:
                    receipt = case["raw_replies"][request.owner]
                    expected_reply = decode_packet(
                        read_checked(golden, receipt["file"], receipt["sha256"]), identity, limits
                    )
                    actual_reply = pool.peers[request.owner].exchange(request)
                    if actual_reply.routes != expected_reply.routes:
                        raise ValueError("split gate changed route-slot identity")
                    owner_errors[request.owner] = compare_rows(actual_reply.outputs, expected_reply.outputs)
                started = time.perf_counter()
                result = pool.execute(layer, hidden, case["ids"], case["weights"], case["shared"], case["routed_scale"])
                elapsed = (time.perf_counter() - started) * 1000
                errors = compare_rows(result, expected)
                repeated = pool.execute(
                    layer, hidden, case["ids"], case["weights"], case["shared"], case["routed_scale"]
                )
                if repeated != result:
                    raise AssertionError("split gate changed under repeated input")
                report["cases"].append(
                    {
                        "layer": layer,
                        "rows": case["rows"],
                        "step": case["step"],
                        "raw_owner_errors": owner_errors,
                        "combined_errors": errors,
                        "validated_packet_round_trip_ms": elapsed,
                        "repeat_equal": True,
                    }
                )
    report["split_expert_hardware_validated"] = True
    report["workers_exited_cleanly"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--layers", type=int, nargs="+", default=[3])
    execute = sub.add_parser("run")
    execute.add_argument("--golden", type=Path, required=True)
    execute.add_argument("--golden-sha256", required=True)
    execute.add_argument("--npu-python", type=Path, required=True)
    execute.add_argument("--cuda-python", type=Path, default=Path(sys.executable))
    execute.add_argument("--library", type=Path, required=True)
    execute.add_argument("--cuda-runtime", type=Path, default=Path("/usr/local/cuda/lib64/libcudart.so"))
    execute.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    result = (
        prepare_golden(args.checkpoint, args.output, args.layers)
        if args.command == "prepare"
        else run(
            args.golden,
            args.golden_sha256,
            args.npu_python,
            args.cuda_python,
            args.library,
            args.cuda_runtime,
            args.logs,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
