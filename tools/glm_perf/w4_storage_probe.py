# SPDX-License-Identifier: Apache-2.0
"""Paired complete native MoE gate for lossless W4 storage; device use is opt-in."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open

from .fused_moe_profile import frozen_helper
from .fused_weight_layout import pack_cube, tensor_digest
from .glm_int4 import BLOCK, packed_weight_bits
from .native_checkpoint import INDEX, file_digest, kernel_assets, write_json
from .reconstruction_probe import time_pair
from .w4_storage_checkpoint import plan, promote

VARIANTS = ("gate_up", "down", "both")
TOKEN_COUNTS = (2, 8, 17, 640)
TOP_K = 8


def synthetic_weights(bits, hidden=256, intermediate=256, experts=3):
    generator = torch.Generator().manual_seed(310 + bits)
    codes, scales = [], []
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        signed = torch.randint(
            -(1 << (bits - 1)), 1 << (bits - 1), (experts, n, k), dtype=torch.int8, generator=generator
        )
        codes.append(pack_cube(signed, bits))
        scales.append(torch.rand(experts, n // BLOCK, k // BLOCK, generator=generator) * 0.01 + 0.001)
    return tuple(codes), tuple(scales)


def checkpoint_weights(source, layer, first_expert=0, experts=3):
    """Read original native expert bytes/scales, with authoritative index selection.

    This is a full-dimensional operator fixture using a small contiguous expert
    subset. It is not a full-rank or model inference qualification.
    """
    inventory = plan(source, extra_bytes_per_rank=0)
    if (
        layer not in inventory["layers"]
        or experts < 2
        or first_expert < 0
        or first_expert + experts > inventory["num_experts"]
    ):
        raise ValueError("select a routed layer and at least two available experts")
    per_rank = inventory["num_experts"] // inventory["world_size"]
    if first_expert // per_rank != (first_expert + experts - 1) // per_rank:
        raise ValueError("operator fixture must stay within one resident expert rank")
    index = json.loads((source / INDEX).read_text())["weight_map"]
    codes, scales = [], []
    for expert in range(first_expert, first_expert + experts):
        row = {}
        for projection in ("gate", "up", "down"):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
            for kind in ("codes", "scale"):
                name = prefix + "_" + kind
                with safe_open(str(source / index[name]), framework="pt", device="cpu") as handle:
                    row[projection, kind] = handle.get_tensor(name).clone()
        codes.append((torch.cat((row["gate", "codes"], row["up", "codes"])), row["down", "codes"]))
        scales.append((torch.cat((row["gate", "scale"], row["up", "scale"])), row["down", "scale"]))
    banks = tuple(torch.stack([row[stage] for row in codes]) for stage in (0, 1))
    scale_banks = tuple(torch.stack([row[stage] for row in scales]) for stage in (0, 1))
    return banks, scale_banks


def variants(codes, scales):
    widths = tuple(scale.shape[-1] * BLOCK for scale in scales)
    bits = tuple(packed_weight_bits(k, code.shape[-1]) for code, k in zip(codes, widths))
    output = {"baseline": codes}
    promoted = tuple(promote(code, k) if width < 4 else code for code, k, width in zip(codes, widths, bits))
    for name in VARIANTS:
        selected = (name in ("gate_up", "both"), name in ("down", "both"))
        if any(enabled and width == 4 for enabled, width in zip(selected, bits)):
            continue
        output[name] = tuple(new if enabled else old for old, new, enabled in zip(codes, promoted, selected))
    if len(output) == 1:
        raise ValueError("fixture has no W2/W3 reconstruction to remove")
    return output


def compare(results, *, expect_zero=False):
    values = [value.cpu() for value in results]
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values):
        raise AssertionError("MoE output must be finite FP32")
    if any(not torch.equal(values[0].view(torch.int32), value.view(torch.int32)) for value in values[1:]):
        raise AssertionError("lossless W4 storage changed full MoE output bits")
    if expect_zero:
        if any(torch.count_nonzero(value) for value in values):
            raise AssertionError("empty/peer routes retained stale results")
    elif not torch.count_nonzero(values[0]):
        raise AssertionError("active MoE routes produced only zero results")


def gate_case(helper, build, options, codes_cpu, scales_cpu, tokens, activation, repeats):
    """Capture separate complete pipelines, then mutate every replay input.

    Separate native objects own separate scratch/output allocations. The same
    frozen build handles each packing. All CPU widening/copying happens outside
    graph capture and timed replay; quantization and reduction remain inside.
    """
    packed = variants(codes_cpu, scales_cpu)
    experts, hidden = codes_cpu[0].shape[0], scales_cpu[0].shape[-1] * BLOCK
    intermediate = scales_cpu[1].shape[-1] * BLOCK
    natives = {
        name: helper.NativeFusedMoE(
            build,
            namespace=options["namespace"],
            activation_bits=activation,
            prepared_weight_layout=True,
            weight_decode_lut=options.get("weight_decode_lut", False),
            fp16_route_workspace=options.get("fp16_route_workspace", False),
        )
        for name in packed
    }
    generator = torch.Generator().manual_seed(tokens * 100 + activation)
    x = torch.randn(tokens, hidden, generator=generator).half().npu()
    ids_cpu = torch.randint(0, experts + 1, (tokens, TOP_K), dtype=torch.int64, generator=generator)
    ids_cpu[:, 0] = 0
    ids_cpu[:, 1] = experts  # every fixture includes peer routes
    ids = ids_cpu.npu()
    weights = torch.rand(tokens, TOP_K, generator=generator).npu()
    weights[:, 2] = 0
    scales = tuple(
        (
            value.half()
            if options.get("fp16_weight_scales")
            else value.half().float()
            if options.get("prerounded_weight_scales")
            else value
        ).npu()
        for value in scales_cpu
    )
    device_codes = {name: tuple(value.npu() for value in pair) for name, pair in packed.items()}
    operations = {
        name: partial(native, x, pair[0], scales[0], pair[1], scales[1], weights, ids)
        for name, native in natives.items()
        for pair in (device_codes[name],)
    }
    graphs, results = {}, []
    for name, operation in operations.items():
        operation()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = operation()
        graphs[name] = graph
        results.append(result)
    checks = []
    for state in ("original", "changed_inputs_codes_and_scales", "all_peer", "zero_weights"):
        if state == "changed_inputs_codes_and_scales":
            x.mul_(1.25)
            weights.mul_(0.75)
            ids.copy_(ids_cpu.roll(1, 0).npu())
            changed = variants(tuple(value.roll(1, 0) for value in codes_cpu), scales_cpu)
            for name, pair in changed.items():
                for destination, value in zip(device_codes[name], pair):
                    destination.copy_(value.npu())
            for scale in scales:
                scale.copy_(scale.roll(1, 0))
        elif state == "all_peer":
            ids.fill_(experts)
        elif state == "zero_weights":
            ids.copy_(ids_cpu.npu())
            weights.zero_()
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        compare(results, expect_zero=state in ("all_peer", "zero_weights"))
        checks.append(state)
    # Restore a representative active workload before measuring replay.
    ids.copy_(ids_cpu.npu())
    weights.copy_(torch.rand(tokens, TOP_K, generator=generator).npu())
    for graph in graphs.values():
        graph.replay()
    torch.npu.synchronize()
    compare(results)
    timings = {}
    for name in graphs:
        if name != "baseline":
            timings[name] = time_pair(
                {"baseline": graphs["baseline"].replay, "candidate": graphs[name].replay}, repeats
            )
    return dict(
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        top_k=TOP_K,
        activation_bits=activation,
        source_bits=[
            packed_weight_bits(scale.shape[-1] * BLOCK, code.shape[-1]) for code, scale in zip(codes_cpu, scales_cpu)
        ],
        bitwise_equal=True,
        replay_checks=checks,
        timing=timings,
        additional_code_bytes={
            name: sum(value.numel() for value in pair) - sum(value.numel() for value in codes_cpu)
            for name, pair in packed.items()
        },
    )


def run(
    build,
    output,
    *,
    allow_device_gate=False,
    device=0,
    repeats=5,
    checkpoint=None,
    layer=None,
    first_expert=0,
    experts=3,
    tokens=TOKEN_COUNTS,
    activations=(4, 8),
):
    if not allow_device_gate:
        raise ValueError("full MoE hardware gate requires explicit --allow-device-gate")
    if type(repeats) is not int or repeats < 1 or not tokens or any(value not in TOKEN_COUNTS for value in tokens):
        raise ValueError("require positive repeats and supported decode/prefill token counts")
    if not activations or any(value not in (4, 8) for value in activations):
        raise ValueError("activation widths must be 4/8")
    if (checkpoint is None) != (layer is None):
        raise ValueError("real expert fixture requires both checkpoint and layer")
    provenance, _ = kernel_assets(build)
    options = provenance["_build"]
    fixtures = (
        [(bits, *synthetic_weights(bits)) for bits in (2, 3)]
        if checkpoint is None
        else [("real_experts", *checkpoint_weights(checkpoint, layer, first_expert, experts))]
    )
    # Reject all-W4 fixtures before importing the device backend or loading code.
    for _, codes, scales in fixtures:
        variants(codes, scales)
    import torch_npu  # noqa: F401 -- explicit hardware gate only.

    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    helper, _ = frozen_helper(build)
    records = [
        dict(fixture=label, **gate_case(helper, build, options, codes, scales, count, activation, repeats))
        for label, codes, scales in fixtures
        for activation in activations
        for count in tokens
    ]
    report = dict(
        complete=True,
        feature="lossless_w4_storage",
        provenance=provenance,
        source_index_sha256=file_digest(checkpoint / INDEX) if checkpoint is not None else None,
        layer=layer,
        first_expert=first_expert,
        real_expert_codes=checkpoint is not None,
        real_model_evaluated=False,
        cube_operation_count_changed=False,
        fixture_sha256=[
            dict(
                label=label,
                codes=[tensor_digest(value) for value in codes],
                scales=[tensor_digest(value) for value in scales],
            )
            for label, codes, scales in fixtures
        ],
        records=records,
    )
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-device-gate", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--first-expert", type=int, default=0)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_COUNTS)
    parser.add_argument("--activations", type=int, nargs="+", default=(4, 8))
    args = parser.parse_args()
    print(
        run(
            args.build,
            args.output,
            allow_device_gate=args.allow_device_gate,
            device=args.device,
            repeats=args.repeats,
            checkpoint=args.checkpoint,
            layer=args.layer,
            first_expert=args.first_expert,
            experts=args.experts,
            tokens=tuple(args.tokens),
            activations=tuple(args.activations),
        )["complete"]
    )


if __name__ == "__main__":
    main()
