# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-bank operator parity, timing and changed-input graph replay gate.

Only --plan-only is permitted during offline work. Tensor implementation parity
does not qualify full-model coding/reasoning quality or serving throughput.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .channel_scale_checkpoint import load_partition
from .channel_scale_layout import logical_bank
from .channel_scale_math import fused_moe_reference
from .channel_scale_runtime import (
    DECODE_ROWS,
    Geometry,
    NativeChannelScaleMoE,
    ScratchArena,
    scratch_shapes,
    shape_bytes,
    verify_bundle,
)


def plan(overlay, bundle, layer, first, stop, tokens, decode_bits=8, prefill_bits=8):
    manifest = verify_bundle(bundle)
    gate, down = load_partition(overlay, layer, first, stop, host_budget_bytes=32 * 1024**3)
    e, twice_i, h = gate.shape
    geometry = Geometry(
        e,
        h,
        twice_i // 2,
        gate.group_size,
        down.group_size,
        manifest.get("fused_gate_up", False),
        manifest.get("fused_decode", False),
        manifest.get("fused_prefill", False),
    )
    return (
        gate,
        down,
        {
            "namespace": manifest["namespace"],
            "abi_version": manifest["abi_version"],
            "cube_rows": manifest.get("cube_rows", 32),
            "layout": manifest["layout"],
            "bank_shape": [e, h, twice_i // 2],
            "scratch_bytes": {
                str(t): shape_bytes(scratch_shapes(t, 8, geometry, decode_bits if t <= DECODE_ROWS else prefill_bits))
                for t in tokens
            },
            "decode_bits": decode_bits,
            "prefill_bits": prefill_bits,
            "reciprocal_quant": manifest.get("reciprocal_quant", False),
            "integer_reconstruct": manifest.get("integer_reconstruct", False),
            "cache_source_rows": manifest.get("cache_source_rows", False),
            "status": "CPU admission only",
        },
    )


def implementation_error(actual, expected):
    return float(
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-8)
    )


def run(overlay, bundle, layer, first, stop, tokens, device_index=0, samples=5, decode_bits=8, prefill_bits=8):
    # Backend/device initialization is intentionally confined to this explicit
    # hardware gate, never import time or offline admission.
    import torch_npu  # noqa: F401

    gate, down, report = plan(overlay, bundle, layer, first, stop, tokens, decode_bits, prefill_bits)
    device = torch.device("npu", device_index)
    torch.npu.set_device(device)
    arena = ScratchArena(512 * 1024**2)
    native = NativeChannelScaleMoE(
        bundle,
        gate,
        down,
        device=device,
        arena=arena,
        expert_offset=first,
        max_tokens=max(tokens),
        decode_bits=decode_bits,
        prefill_bits=prefill_bits,
    )
    references = tuple(logical_bank(b) for b in (gate, down))
    report["cases"] = []
    generator = torch.Generator().manual_seed(1010)
    for count in tokens:
        bits = decode_bits if count <= DECODE_ROWS else prefill_bits
        x = torch.randn(count, gate.shape[-1], generator=generator).half()
        ids = torch.randint(first, stop, (count, 8), generator=generator)
        if count in (2, 3):
            # Seven local routes/token force 14 and 21 rows into one expert:
            # A8 must actually execute padded physical M32 and M48 operations.
            ids.fill_(first)
        ids[:, -1] = stop + 10  # exercise masked peer suffix, not unwritten route rows
        weights = torch.rand(count, 8, generator=generator)
        weights /= weights.sum(-1, keepdim=True)
        expected = fused_moe_reference(
            x, *references, weights, ids, expert_offset=first, bits=bits, reciprocal=report["reciprocal_quant"]
        )
        dx, dw, di = (v.to(device) for v in (x, weights, ids))
        result = native(dx, dw, di)
        torch.npu.synchronize()
        error = implementation_error(result.cpu(), expected)
        if not torch.isfinite(result).all().item() or error > 0.02:
            raise AssertionError(f"real-bank implementation parity failed: T={count}, NRMSE={error}")
        durations = []
        for _ in range(samples):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            result = native(dx, dw, di)
            end.record()
            torch.npu.synchronize()
            durations.append(float(start.elapsed_time(end)))
        replay_error = None
        if count <= 4:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = native(dx, dw, di)
            dx.copy_((x * 0.75).to(device))
            # Different route order and values must actually reach the replay.
            changed_ids = ids.flip(1).contiguous()
            di.copy_(changed_ids.to(device))
            dw.copy_((weights * 0.5).to(device))
            graph.replay()
            torch.npu.synchronize()
            expected = fused_moe_reference(
                x * 0.75,
                *references,
                weights * 0.5,
                changed_ids,
                expert_offset=first,
                bits=bits,
                reciprocal=report["reciprocal_quant"],
            )
            replay_error = implementation_error(captured.cpu(), expected)
            if replay_error > 0.02:
                raise AssertionError(f"changed-input graph replay parity failed: T={count}, NRMSE={replay_error}")
            # Release each capture before the next shape and before unloading
            # the native owner, rather than retaining capture resources.
            del graph, captured
        report["cases"].append(
            {
                "tokens": count,
                "implementation_nrmse": error,
                "changed_input_graph_nrmse": replay_error,
                "median_device_ms": statistics.median(durations),
                "device_samples_ms": durations,
                "arena_live_bytes": arena.allocated_bytes,
                "physical_cube_rows_from_routes": sorted(
                    {
                        (
                            (
                                min(rows - offset, report["cube_rows"] // (2 if bits == 8 else 1))
                                * (2 if bits == 8 else 1)
                                + 15
                            )
                            // 16
                        )
                        * 16
                        for expert in range(first, stop)
                        for rows in [int(((ids == expert) & (weights != 0)).sum())]
                        for offset in range(0, rows, report["cube_rows"] // (2 if bits == 8 else 1))
                    }
                ),
            }
        )
    report["status"] = "real-bank operator implementation gate only; full-model quality/throughput pending"
    report["completed_at_unix"] = time.time()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--stop", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4, 17, 37, 640])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--decode-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--prefill-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        report = plan(
            args.overlay,
            args.bundle,
            args.layer,
            args.first,
            args.stop,
            args.tokens,
            args.decode_bits,
            args.prefill_bits,
        )[2]
    else:
        report = run(
            args.overlay,
            args.bundle,
            args.layer,
            args.first,
            args.stop,
            args.tokens,
            args.device,
            decode_bits=args.decode_bits,
            prefill_bits=args.prefill_bits,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
