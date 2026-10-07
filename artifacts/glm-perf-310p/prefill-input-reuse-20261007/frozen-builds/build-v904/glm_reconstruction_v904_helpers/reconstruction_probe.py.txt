# SPDX-License-Identifier: Apache-2.0
"""Native W4 component gates and paired timing, independent of model serving."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from .glm_int4 import (
    GroupedProjectionGeometry,
    NativeW4Projection,
    activation_limbs,
    block_reference,
    pack_nz_codes,
    packed_weight_bits,
    signed_nibbles,
    unpack_canonical_codes,
)
from .reconstruction_native import PROFILES, ProjectionGeometry, ReconstructionProjection


def digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def component_cases(full_shapes=False):
    shapes = ((128, 256), (256, 512))
    if full_shapes:
        shapes += ((2048, 4096), (4096, 4096), (4096, 2048))
    return tuple(ProjectionGeometry(rows, 3, n, k) for n, k in shapes for rows in (1, 2, 8, 16, 32, 64))


def host_inputs(geometry):
    generator = torch.Generator().manual_seed(310 + geometry.rows + geometry.n + geometry.k)
    inputs = torch.randn(geometry.rows, geometry.k, generator=generator).half()
    signed = torch.randint(-8, 8, (geometry.experts, geometry.n, geometry.k), generator=generator, dtype=torch.int8)
    scales = torch.rand(geometry.experts, geometry.n // 32, geometry.k // 32, generator=generator) * 0.01 + 0.001
    active = max(1, geometry.rows - 1)
    ends = torch.tensor([0, max(1, active // 2), active], dtype=torch.int64)
    return inputs, signed, scales, ends


def verify_activation_pack(inputs_cpu, limbs):
    low, high, scales, quant = activation_limbs(inputs_cpu)
    actual_low, actual_high, actual_scale = (tensor.cpu() for tensor in limbs)
    shape = low.shape
    # Upper 32 padded values are deliberately ignored by zero weight lanes.
    unpacked_low = signed_nibbles(actual_low[..., :16]).reshape(shape)
    unpacked_high = signed_nibbles(actual_high[..., :16]).reshape(shape)
    if not torch.equal(low.to(torch.int8), unpacked_low) or not torch.equal(high.to(torch.int8), unpacked_high):
        raise AssertionError("native activation limbs differ from independent INT8 reference")
    if not torch.equal(actual_scale, scales[..., None].expand_as(actual_scale)):
        raise AssertionError("native activation scales differ from CPU reference")
    if not torch.equal(unpacked_low.int() + 16 * unpacked_high.int() + 8, quant):
        raise AssertionError("native activation limbs do not reconstruct INT8 exactly")


def time_pair(operations, repeats):
    samples = {name: [] for name in operations}
    for operation in operations.values():
        operation()
    torch.npu.synchronize()
    names = tuple(operations)
    for repetition in range(repeats):
        for name in names if repetition % 2 == 0 else names[::-1]:
            torch.npu.synchronize()
            start = time.perf_counter()
            operations[name]()
            torch.npu.synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
    return {name: {"median_ms": statistics.median(values), "samples_ms": values} for name, values in samples.items()}


def activation_corner_gate(projection):
    inputs = torch.zeros(8, 256, dtype=torch.float16)
    inputs[1] = 2**-24  # Smallest positive FP16 subnormal.
    inputs[2] = -(2**-24)
    inputs[3] = torch.finfo(torch.float16).max
    inputs[4] = -torch.finfo(torch.float16).max
    inputs[5] = torch.arange(256).remainder(255).sub(127).half()
    inputs[6] = inputs[5].flip(0)
    inputs[7, 31::32] = 127
    inputs[7, :31] = 0.5  # Ties use round-to-nearest-even.
    verify_activation_pack(inputs, projection.pack(inputs.npu()))
    return {"zero_subnormal_extreme_and_rounding_ties": True, "passed": True}


def gate_case(projection, geometry, baseline=None, repeats=5, bits=4):
    inputs_cpu, signed, scales_cpu, ends_cpu = host_inputs(geometry)
    if bits < 4:
        signed = (signed.int() % (1 << bits) - (1 << (bits - 1))).to(torch.int8)
    codes_cpu = pack_nz_codes(signed, bits)
    inputs, codes, scales, ends = (tensor.npu() for tensor in (inputs_cpu, codes_cpu, scales_cpu, ends_cpu))
    limbs = projection.pack(inputs)
    verify_activation_pack(inputs_cpu, limbs)
    actual = projection.project(inputs, codes, scales, ends, limbs).cpu()
    unsigned = projection.project(inputs, codes.view(torch.uint8), scales, ends, limbs).cpu()
    if not torch.equal(actual, unsigned):
        raise AssertionError("packed signed and unsigned byte aliases must produce identical outputs")
    expected = torch.zeros(geometry.rows, geometry.n, dtype=torch.float16)
    first = 0
    for expert, end in enumerate(ends_cpu.tolist()):
        if end > first:
            expected[first:end] = block_reference(inputs_cpu[first:end], signed[expert], scales_cpu[expert])
        first = end
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    if not torch.isfinite(actual).all() or torch.count_nonzero(actual[first:]):
        raise AssertionError("native projection has nonfinite results or nonzero peer rows")
    operations = {
        "native_prepared": lambda: projection.project(inputs, codes, scales, ends, limbs),
        "native_with_pack": lambda: projection(inputs, codes, scales, ends),
    }
    if baseline is not None:
        operations["fp16_grouped"] = lambda: baseline(inputs, codes, scales, ends)
    timing = time_pair(operations, repeats)
    if baseline is not None:
        control = baseline(inputs, codes, scales, ends).cpu()
        drift = (actual.float() - control.float()).abs()
        drift_report = {"max_abs": drift.max().item(), "mean_abs": drift.mean().item()}
    else:
        drift_report = None
    return {
        "geometry": geometry.__dict__,
        "weight_bits": bits,
        "unsigned_byte_alias_exact": True,
        "activation_pack_exact": True,
        "native_reference_passed": True,
        "output_sha256": digest(actual),
        "input_sha256": digest(inputs_cpu),
        "codes_sha256": digest(codes_cpu),
        "scales_sha256": digest(scales_cpu),
        "fp16_baseline_drift": drift_report,
        "timing": timing,
    }


def graph_gate(projection, geometry, bits=4, baseline=None):
    """Replay twice with changed buffers, including changed route boundaries."""
    inputs_cpu, signed, scales_cpu, ends_cpu = host_inputs(geometry)
    if bits < 4:
        signed = (signed.int() % (1 << bits) - (1 << (bits - 1))).to(torch.int8)
    inputs, codes, scales, ends = (t.npu() for t in (inputs_cpu, pack_nz_codes(signed, bits), scales_cpu, ends_cpu))
    projection(inputs, codes, scales, ends)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = projection(inputs, codes, scales, ends)
    for replay in range(2):
        changed = inputs_cpu * (replay + 2)
        boundaries = torch.tensor([1, 1, geometry.rows - 1], dtype=torch.int64)
        inputs.copy_(changed.npu())
        ends.copy_(boundaries.npu())
        graph.replay()
        torch.npu.synchronize()
        if baseline is not None:
            expected = baseline(inputs, codes, scales, ends).cpu()
        else:
            expected = torch.zeros(geometry.rows, geometry.n, dtype=torch.float16)
            first = 0
            for expert, end in enumerate(boundaries.tolist()):
                if end > first:
                    expected[first:end] = block_reference(changed[first:end], signed[expert], scales_cpu[expert])
                first = end
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.002, atol=0.002)
    return {
        "geometry": geometry.__dict__,
        "weight_bits": bits,
        "changed_input_and_routes": True,
        "replays": 2,
        "passed": True,
    }


def w3_gate(projections, geometry, baseline, repeats=3):
    inputs_cpu, signed, scales_cpu, ends_cpu = host_inputs(geometry)
    signed = (signed.int() % 8 - 4).to(torch.int8)
    inputs, codes, scales, ends = (t.npu() for t in (inputs_cpu, pack_nz_codes(signed, 3), scales_cpu, ends_cpu))
    control = baseline(inputs, codes, scales, ends).cpu()
    operations = {profile: lambda p=p: p(inputs, codes, scales, ends) for profile, p in projections.items()}
    operations["fp16_grouped"] = lambda: baseline(inputs, codes, scales, ends)
    exact = {}
    for profile, projection in projections.items():
        actual = projection(inputs, codes, scales, ends).cpu()
        torch.testing.assert_close(actual, control, rtol=0.002, atol=0.002)
        exact[profile] = torch.equal(actual, control)
    return {
        "geometry": geometry.__dict__,
        "passed": True,
        "bitwise_equal": exact,
        "timing": time_pair(operations, repeats),
    }


def real_weight_gate(projection, checkpoint, prefix, baseline, repeats=3):
    # Read only one expert; no serving tensor or checkpoint is rewritten.
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = []
    for suffix in ("_codes", "_scale"):
        name = prefix + suffix
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
            tensors.append(shard.get_tensor(name))
    canonical, scale = tensors
    n, k = scale.shape[0] * 32, scale.shape[1] * 32
    bits = packed_weight_bits(k, canonical.shape[1])
    signed = unpack_canonical_codes(canonical, k)
    geometry = ProjectionGeometry(2, 1, n, k)
    inputs_cpu = torch.randn(2, k, generator=torch.Generator().manual_seed(310)).half()
    inputs, codes, scales, ends = (
        t.npu() for t in (inputs_cpu, pack_nz_codes(signed[None], bits), scale.float()[None], torch.tensor([2]))
    )
    actual = projection(inputs, codes, scales, ends).cpu()
    expected = block_reference(inputs_cpu, signed, scale.float())
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    control = baseline(inputs, codes, scales, ends).cpu()
    drift = (actual.float() - control.float()).abs()
    return {
        "tensor_prefix": prefix,
        "weight_bits": bits,
        "geometry": geometry.__dict__,
        "native_reference_passed": True,
        "canonical_codes_sha256": digest(canonical),
        "scales_sha256": digest(scale),
        "fp16_baseline_drift": {"max_abs": drift.max().item(), "mean_abs": drift.mean().item()},
        "timing": time_pair(
            {
                "native_with_pack": lambda: projection(inputs, codes, scales, ends),
                "fp16_grouped": lambda: baseline(inputs, codes, scales, ends),
            },
            repeats,
        ),
    }


def moe_pipeline_gate(projection, bits):
    """Gate routing -> fused gate/up -> SwiGLU -> down -> FP32 combine."""
    generator = torch.Generator().manual_seed(2300 + bits)
    inputs_cpu = torch.randn(2, 256, generator=generator).half()
    signed_gate = torch.randint(
        -(1 << (bits - 1)), 1 << (bits - 1), (3, 512, 256), generator=generator, dtype=torch.int8
    )
    signed_down = torch.randint(
        -(1 << (bits - 1)), 1 << (bits - 1), (3, 256, 256), generator=generator, dtype=torch.int8
    )
    scale_gate = torch.rand(3, 16, 8, generator=generator) * 0.01 + 0.001
    scale_down = torch.rand(3, 8, 8, generator=generator) * 0.01 + 0.001
    inputs = inputs_cpu.npu()
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).npu()
    weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]]).float().npu()
    gate_codes, down_codes = (pack_nz_codes(t, bits).npu() for t in (signed_gate, signed_down))
    gate_scales, down_scales = scale_gate.npu(), scale_down.npu()
    swiglu = torch.ops._C_ascend.npu_w2_swiglu_310
    combine = torch.ops._C_ascend.npu_w2_route_combine_310

    def pipeline():
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        tokens = torch.arange(inputs.shape[0], device=inputs.device).repeat_interleave(ids.shape[1])
        sorted_x = inputs.index_select(0, tokens.index_select(0, order)).contiguous()
        ends = (flat[None, :] < torch.arange(1, 4, device=inputs.device)[:, None]).sum(-1).to(torch.int64).contiguous()
        inverse = torch.argsort(order).contiguous()
        gate_up = projection(sorted_x, gate_codes, gate_scales, ends)
        activation = swiglu(gate_up)
        routed = projection(activation, down_codes, down_scales, ends)
        return combine(routed, inverse, weights, ends)

    def reference():
        flat = ids.cpu().flatten()
        order = torch.argsort(flat, stable=True)
        tokens = torch.arange(2).repeat_interleave(2)[order]
        sorted_x = inputs.cpu()[tokens]
        ends = (flat[None, :] < torch.arange(1, 4)[:, None]).sum(-1).tolist()
        gate_up = torch.zeros(4, 512, dtype=torch.float16)
        routed = torch.zeros(4, 256, dtype=torch.float16)
        first = 0
        for expert, end in enumerate(ends):
            if end > first:
                gate_up[first:end] = block_reference(sorted_x[first:end], signed_gate[expert], scale_gate[expert])
            first = end
        gate, up = gate_up.chunk(2, -1)
        activation = (torch.nn.functional.silu(gate.float()) * up.float()).half()
        first = 0
        for expert, end in enumerate(ends):
            if end > first:
                routed[first:end] = block_reference(activation[first:end], signed_down[expert], scale_down[expert])
            first = end
        return (routed[torch.argsort(order)].reshape(2, 2, 256).float() * weights.cpu()[..., None]).sum(1)

    actual = pipeline()
    torch.testing.assert_close(actual.cpu(), reference(), rtol=0.002, atol=0.002)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = pipeline()
    inputs.copy_((inputs_cpu * 1.5).npu())
    ids.copy_(torch.tensor([[2, 0], [2, 2]], dtype=torch.int64).npu())
    weights.copy_(torch.tensor([[0.2, 0.8], [0.3, 0.7]]).float().npu())
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), reference(), rtol=0.002, atol=0.002)
    return {
        "weight_bits": bits,
        "passed": True,
        "changed_activation_routes_and_weights": True,
        "stages": ["routing", "activation_pack", "gate_up", "swiglu", "down", "route_combine"],
        "graph_replays": 1,
    }


def run(
    build_dir,
    output,
    device=0,
    full_shapes=False,
    binding=None,
    repeats=5,
    include_w3=False,
    checkpoint=None,
    prefixes=(),
    support_libraries=(),
):
    # Lazy import is intentional: CPU case planning/tests never initialize NPU.
    import torch_npu

    if output.exists() or repeats < 2 or (include_w3 or checkpoint) and binding is None:
        raise ValueError("output must be new and repeats at least two")
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    options = json.loads((build_dir / "provenance.json").read_text()).get("_build", {"version": 1})
    version = options["version"]
    namespace = f"glm_reconstruction_v{version}"
    bridge_name = f"glm_reconstruction_bridge_v{version}.so"
    torch.ops.load_library(str(build_dir / bridge_name))
    baseline = None
    if binding is not None:
        torch.ops.load_library(str(binding))
        baseline = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    for library in support_libraries:
        torch.ops.load_library(str(library))
    cases = component_cases(full_shapes)
    if options.get("all_bits"):
        cases += tuple(
            GroupedProjectionGeometry(rows, 3, n, k)
            for n, k in ((128, 256), (256, 512))
            for rows in (3, 128, 256, 1024)
        )
    real_cases = tuple(
        ProjectionGeometry(2, 1, n, k) for n, k in ((2048, 4096), (4096, 4096), (4096, 2048), (2048, 2048))
    )
    projection = NativeW4Projection(
        build_dir / "glm_w4a8_pack.bin",
        build_dir / "glm_w4a8_matmul.bin",
        (*cases, *real_cases),
        namespace=namespace,
        tile_pipeline=options.get("tile_pipeline", False),
        output_columns=options.get("output_columns", 16),
        all_bits=options.get("all_bits", False),
    )
    binary_names = ("glm_w4a8_pack.bin", "glm_w4a8_matmul.bin", bridge_name)
    if include_w3:
        binary_names += ("reconstruction_kernel.bin",)
    payload = {
        "schema_version": 1,
        "complete": False,
        "device": device,
        "build_options": options,
        "binaries": {name: hashlib.sha256((build_dir / name).read_bytes()).hexdigest() for name in binary_names},
        "baseline_binding_sha256": hashlib.sha256(binding.read_bytes()).hexdigest() if binding else None,
        "support_libraries": [
            {"path": str(path.resolve(strict=True)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in support_libraries
        ],
        "records": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload["activation_corner_gate"] = activation_corner_gate(projection)
        weight_bits = (2, 3, 4) if options.get("all_bits") else (4,)
        for bits in weight_bits:
            for geometry in cases:
                record = gate_case(projection, geometry, baseline, repeats, bits)
                payload["records"].append(record)
                print(json.dumps(record), flush=True)
        payload["graph_records"] = [
            graph_gate(projection, ProjectionGeometry(8, 3, 128, 256), bits) for bits in weight_bits
        ]
        if options.get("all_bits"):
            fresh = GroupedProjectionGeometry(5, 3, 128, 256)
            payload["lazy_metadata_records"] = [
                gate_case(projection, fresh, baseline, repeats, bits) for bits in weight_bits
            ]
            payload["prefill_graph_records"] = [
                graph_gate(projection, GroupedProjectionGeometry(128, 3, 128, 256), bits) for bits in weight_bits
            ]
        if include_w3:
            projections = {
                p: ReconstructionProjection(build_dir / "reconstruction_kernel.bin", p, cases, namespace=namespace)
                for p in PROFILES
            }
            payload["w3_records"] = [w3_gate(projections, g, baseline, repeats) for g in cases]
            payload["w3_graph_records"] = [
                {"profile": p, **graph_gate(projection, ProjectionGeometry(8, 3, 128, 256), 3, baseline)}
                for p, projection in projections.items()
            ]
        if checkpoint:
            if not prefixes:
                raise ValueError("real checkpoint gates require explicit W4 tensor prefixes")
            payload["real_weight_records"] = [
                real_weight_gate(projection, checkpoint, p, baseline, repeats) for p in prefixes
            ]
        if options.get("all_bits") and baseline is not None:
            payload["moe_pipeline_records"] = [moe_pipeline_gate(projection, bits) for bits in weight_bits]
        payload["complete"] = True
    except Exception as error:
        payload["error"] = str(error)
        raise
    finally:
        output.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--full-shapes", action="store_true")
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-w3", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tensor-prefix", action="append", default=[])
    parser.add_argument("--support-library", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(
        args.build_dir,
        args.output,
        args.device,
        args.full_shapes,
        args.binding,
        args.repeats,
        args.include_w3,
        args.checkpoint,
        args.tensor_prefix,
        args.support_library,
    )


if __name__ == "__main__":
    main()
