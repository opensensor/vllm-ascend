# SPDX-License-Identifier: Apache-2.0
"""Explicit isolated real-weight gates. Default mode only checks build artifacts.

Never attach to a server, patch model classes, or claim serving qualification.
Use --run-npu only in a maintenance window with space for one diagnostic layer.
"""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.tp6_fusion_sources import digest

ROUTED_ROWS = (30, 60, 120)
GROUPED_ROWS = (180, 650, 6400, 20480)
OWNERSHIP = ("none", "mixed", "all")


def plan(quick=False):
    return dict(
        routed_rows=(30, 120) if quick else ROUTED_ROWS,
        grouped_rows=(180, 650) if quick else GROUPED_ROWS,
        quick=quick,
        ownership=OWNERSHIP,
        hc_rows=(3, 6, 12, 18),
        tp_size=6,
        required_ranks=list(range(6)),
        scopes=("packed bytes and metadata", "downstream projection", "finalization", "HC mix and combine"),
        graph_qualified=False,
        serving_qualified=False,
        hardware_validated=False,
    )


def exact(expected, actual):
    import torch

    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(exact(a, b) for a, b in zip(expected, actual))
    return (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and torch.equal(expected.cpu().contiguous().view(torch.uint8), actual.cpu().contiguous().view(torch.uint8))
    )


def run(args, report, save, owner):
    # Backend/model imports are intentionally confined to this explicit hardware entry.
    import torch
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing
    from tools.qwen4exp.hc_checkpoint import load_hyperconnection
    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from tools.qwen4exp.tp6_fusions import MODES, load_resources
    from tools.qwen4exp.tp6_operand_reuse import make_mix
    from vllm_ascend.models.qwen4_exp.model import _format_eager_linear_weights_npu
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
        NATIVE_INT4_BACKEND,
        pack_activation_device,
        swiglu_pack_activation_device,
    )
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    resources = load_resources(args.build, args.bridge, args.bridge_sha256, owner=owner)
    generator = torch.Generator().manual_seed(161026)

    def record(kind, passed, functions=None, **metadata):
        row = dict(kind=kind, passed=passed, **metadata)
        # Timing is contingent on this case's exact gate, never a tolerance check.
        if passed and functions:
            for function in functions.values():
                function()
            row["timing"] = paired_timing(functions, args.iterations, args.repeats)
        report["records"].append(row)
        report["passed"] &= passed
        save()

    with torch.inference_mode():
        layer = load_layer(args.model, args.layer, args.rank, 6, NATIVE_INT4_BACKEND)
        bank, down = layer.projections["gate_up_proj"], layer.projections["down_proj"]
        experts = bank.weight.shape[0]
        report["experts"] = experts
        for routed, cases in ((True, report["plan"]["routed_rows"]), (False, report["plan"]["grouped_rows"])):
            for rows in cases:
                source_rows = rows // 10 if routed else rows
                for ownership in OWNERSHIP:
                    for pattern in ("random", "zero", "changed"):
                        x = torch.randn(source_rows, 2560, generator=generator).half()
                        if pattern == "zero":
                            x.zero_()
                        elif pattern == "changed":
                            x.mul_(0.01)
                        prepared = pack_activation_device(x.npu())
                        active = {"none": 0, "mixed": rows // 3, "all": rows}[ownership]
                        if routed:
                            ends_cpu = torch.arange(rows, dtype=torch.int32) % experts
                            ends_cpu[active::2] = -1
                            ends_cpu[active + 1 :: 2] = experts
                        else:
                            ends_cpu = torch.tensor([(i + 1) * active // experts for i in range(experts)])
                        ends = ends_cpu.npu()
                        baseline = lambda bank=bank, prepared=prepared, ends=ends: swiglu_pack_activation_device(
                            bank.native_linear(prepared, ends)
                        )
                        candidate = lambda bank=bank, prepared=prepared, ends=ends: resources["gate_pack"](
                            bank, prepared, ends
                        )
                        expected, actual = baseline(), candidate()
                        passed = exact(expected, actual)
                        # The unchanged native down projection is an independent consumer.
                        downstream = exact(down.native_linear(expected, ends), down.native_linear(actual, ends))
                        record(
                            "gate_pack",
                            passed and downstream,
                            {"retained": baseline, "fused": candidate} if pattern == "random" else None,
                            rows=rows,
                            routed=routed,
                            ownership=ownership,
                            pattern=pattern,
                            packed_exact=passed,
                            downstream_exact=downstream,
                        )
                        del expected, actual
                # Avoid keeping the last closure's packed inputs across cases.
                del baseline, candidate, prepared, ends
        del bank, down, layer
        torch.npu.synchronize()
        for rows in report["plan"]["grouped_rows"]:
            for ownership in OWNERSHIP:
                active = {"none": 0, "mixed": rows // 3, "all": rows}[ownership]
                source = torch.randn(rows, 2560, generator=generator).half()
                source[active:] = 0
                routed = source.npu()
                inverse = torch.randperm(rows, generator=generator).npu()
                order = torch.argsort(inverse)
                weights = torch.rand(rows, 1, generator=generator).npu()
                ends = torch.tensor([(i + 1) * active // experts for i in range(experts)]).npu()
                baseline = lambda routed=routed, weights=weights, order=order, inverse=inverse: (
                    (routed.float() * weights.index_select(0, order))
                    .index_select(0, inverse)
                    .reshape(-1, 10, 2560)
                    .sum(1)
                )
                expected = baseline()
                for mode in MODES:
                    candidate = lambda mode=mode, routed=routed, inverse=inverse, weights=weights, ends=ends: resources[
                        mode
                    ](routed, inverse, weights, ends)
                    actual = candidate()
                    passed = exact(expected, actual)
                    record(
                        "finalize",
                        passed,
                        {"retained": baseline, mode: candidate},
                        rows=rows,
                        ownership=ownership,
                        mode=mode,
                    )
                    # Verify local-aware variants never consume a stale peer tail.
                    routed[active:] = float("nan")
                    record(
                        "finalize_poisoned_tail",
                        exact(expected, candidate()),
                        rows=rows,
                        ownership=ownership,
                        mode=mode,
                    )
                    routed[active:] = 0
                del routed, expected, actual, baseline, candidate
        hc, keys = load_hyperconnection(args.model, args.hc_prefix)
        hc = hc.npu().eval()
        _format_eager_linear_weights_npu(hc)
        hc.prepare_norm_affine()
        replacement = make_mix(type(hc).mix)
        report["hc_keys"] = keys
        for rows in (3, 6, 12, 18):
            for pattern in ("random", "zero", "changed"):
                value = torch.randn(rows, 10240, generator=generator).half().npu()
                block = torch.randn(rows, 2560, generator=generator).half().npu()
                if pattern == "zero":
                    value.zero_()
                elif pattern == "changed":
                    value.mul_(0.01)

                def baseline(value=value, block=block):
                    mixed, residual = hc.mix(value)
                    return mixed, hc.combine(block, residual)

                def candidate(value=value, block=block):
                    mixed, residual = replacement(hc, value)
                    return mixed, hc.combine(block, residual)

                record(
                    "operand_reuse",
                    exact(baseline(), candidate()),
                    {"retained": baseline, "reuse": candidate} if pattern == "random" else None,
                    rows=rows,
                    pattern=pattern,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run-npu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="bounded first gate; excludes large prefill shapes")
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-sha256")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hc-prefix")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, choices=range(6), default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from tools.qwen4exp.build_tp6_fusions import verify_build

    manifest = verify_build(args.build)
    report = dict(
        plan=plan(args.quick),
        build=manifest,
        rank=args.rank,
        layer=args.layer,
        records=[],
        passed=False,
        hardware_validated=False,
        serving_qualified=False,
        server_modified=False,
    )
    if not args.run_npu:
        print(json.dumps(report, indent=2))
        return 0
    if not all((args.bridge, args.bridge_sha256, args.model, args.hc_prefix, args.output)):
        parser.error("hardware run requires bridge + SHA256, model, hc-prefix and a new output path")
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be positive")
    if digest(args.bridge.read_bytes()) != args.bridge_sha256:
        parser.error("bridge SHA256 mismatch")
    # Reserve evidence before any backend/device initialization.
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    owner = {}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["passed"] = True
        run(args, report, save, owner)
        report["completed"] = True
        report["hardware_executed"] = True
        report["operator_matrix_passed"] = report["passed"]
    except BaseException as error:
        report["passed"] = False
        report["error"] = repr(error)
        save()
        raise
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
