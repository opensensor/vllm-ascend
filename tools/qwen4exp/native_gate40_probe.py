# SPDX-License-Identifier: Apache-2.0
"""Small native registration/parity gates and optional real-bank comparisons."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.direct_gate_up import DirectGateUp
from tools.qwen4exp.native_gate40_spec import BRIDGE_NAMESPACE, SYMBOL


def prepare_resources(paths):
    kernel_class = getattr(torch.classes, BRIDGE_NAMESPACE).Kernel
    launch = getattr(torch.ops, BRIDGE_NAMESPACE).launch
    return {name: DirectGateUp(kernel_class(path, SYMBOL), launch) for name, path in paths.items()}


def synthetic_bank():
    # Host-only weight preparation. Device code consumes the same packed bank
    # as the ordinary qualified operator, including per-group correction sums.
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_native_metadata, pack_native_weight

    generator = torch.Generator().manual_seed(1115)
    codes = torch.randint(-128, 128, (4, 1280, 1280), dtype=torch.int8, generator=generator)
    packed, sums = zip(*(pack_native_weight(expert) for expert in codes))
    scales = torch.rand(4, 1280, 20, generator=generator).half() * 0.01 + 0.005
    offsets = torch.randint(-3, 4, scales.shape, generator=generator).half()
    return tuple(
        value.npu()
        for value in (
            torch.stack(packed),
            torch.stack([pack_native_metadata(value) for value in scales]),
            torch.stack([pack_native_metadata(value) for value in offsets]),
            torch.stack(sums),
        )
    )


def route_ids(rows, experts, pattern, device):
    if pattern == "mixed":
        # TP4-like ownership: one quarter local, three quarters peer rows.
        slots = torch.arange(rows, dtype=torch.int32)
        return torch.where(slots % 4 == 0, slots // 4 % experts, -1).to(device)
    if pattern == "all_local":
        return (torch.arange(rows, dtype=torch.int32) % experts).to(device)
    if pattern == "invalid":
        return torch.full((rows,), -1, dtype=torch.int32, device=device)
    if pattern == "peer_high":
        return torch.full((rows,), experts + 1, dtype=torch.int32, device=device)
    if pattern == "concentrated":
        return torch.zeros(rows, dtype=torch.int32, device=device)
    if pattern in ("boundary16", "boundary17"):
        ids = torch.ones(rows, dtype=torch.int32)
        ids[: int(pattern[-2:])] = 0
        return ids.to(device)
    raise ValueError("unknown route pattern")


@torch.no_grad()
def validate_resources(resources, small=True, bank=None, repeats=7, on_record=None):
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device

    bank = synthetic_bank() if bank is None else bank
    reference = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    shapes = ((4, 40),) if small else ((3, 30), (4, 40), (6, 60), (9, 90), (12, 120), (31, 31), (41, 41), (128, 128))
    patterns = (
        ("mixed", "concentrated", "invalid")
        if small
        else ("mixed", "concentrated", "invalid", "all_local", "peer_high", "boundary16", "boundary17")
    )
    records = []
    for tokens, rows in shapes:
        for pattern in patterns:
            ids = route_ids(rows, bank[0].shape[0], pattern, bank[0].device)
            gates = []
            generator = torch.Generator().manual_seed(1125)
            prepared = None
            for _ in range(2 if small else 3):
                x = (torch.randn(tokens, 2560, generator=generator) * 0.1).half().npu()
                updated = pack_activation_device(x)
                if prepared is None:
                    prepared = updated
                else:
                    for target, source in zip(prepared, updated):
                        target.copy_(source)
                expected = reference(*prepared, *bank, ids)
                for name, resource in resources.items():
                    result = compare_outputs(expected, resource(*prepared, *bank, ids))
                    gates.append({"profile": name, "outputs": result})
                    if not gate_passed(result):
                        raise RuntimeError(
                            f"native parity failed: {name} tokens={tokens} routes={rows} {pattern}: {result}"
                        )
            record = {"tokens": tokens, "routes": rows, "routing": pattern, "eager_gates": gates}
            if not small:
                from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing

                functions = {"qualified_op": lambda prepared=prepared, ids=ids: reference(*prepared, *bank, ids)}
                functions.update(
                    {
                        name: lambda resource=resource, prepared=prepared, ids=ids: resource(*prepared, *bank, ids)
                        for name, resource in resources.items()
                    }
                )
                graphs = {name: capture(function) for name, function in functions.items()}
                record["graph_gates"] = []
                # Change both activation limbs/metadata and routing values
                # after capture, keeping static graph tensor storage intact.
                for index in range(3):
                    x = (torch.randn(tokens, 2560, generator=generator) * 0.1).half().npu()
                    for target, source in zip(prepared, pack_activation_device(x)):
                        target.copy_(source)
                    ids.copy_(route_ids(rows, bank[0].shape[0], patterns[index], ids.device))
                    for graph, _ in graphs.values():
                        graph.replay()
                    torch.npu.synchronize()
                    expected = functions["qualified_op"]()
                    for name, (_, output) in graphs.items():
                        result = compare_outputs(expected, output)
                        record["graph_gates"].append({"profile": name, "outputs": result})
                        if not gate_passed(result):
                            raise RuntimeError(f"changed graph input parity failed: {name} {tokens}/{rows}")
                # Restore the named distribution before paired timings.
                ids.copy_(route_ids(rows, bank[0].shape[0], pattern, ids.device))
                record["graph"] = paired_timing(
                    {name: graph.replay for name, (graph, _) in graphs.items()}, 50, repeats
                )
                del graphs
            records.append(record)
            if on_record is not None:
                on_record(record)
    torch.npu.synchronize()
    return {"passed": True, "weights": "synthetic" if small else "provided_or_synthetic_bank", "records": records}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-component", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or (args.checkpoint and not args.full_component):
        parser.error("choose a new output, positive repeats; checkpoint requires full-component")
    import torch_npu

    from tools.glm_perf.resident_native import NativeSession
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    session = NativeSession()
    value = json.loads(args.manifest.read_text())
    exists = lambda op: hasattr(getattr(torch.ops, op.split("::")[0]), op.split("::")[1])
    prepared = session.prepare(value, exists)
    receipt = session.load(prepared["native_digest"], torch.ops.load_library, exists, torch.npu.synchronize)
    report = {"scope": "isolated_direct_native_gate_up_not_whole_model", "native_receipt": receipt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.full_component:
        bank = None
        if args.checkpoint:
            from tools.qwen4exp.profile_w4_layer_310 import load_layer
            from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND

            layer = load_layer(args.checkpoint, 0, 0, 4, NATIVE_INT4_BACKEND)
            projection = layer.projections["gate_up_proj"]
            bank = projection.weight, projection.weight_scale, projection.weight_offset, projection.weight_sum
        report["real_checkpoint"] = str(args.checkpoint) if args.checkpoint else None
        report["component"] = {"passed": False, "records": []}

        def on_record(record):
            report["component"]["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

        try:
            report["component"] = validate_resources(
                session.resources[value["name"]], small=False, bank=bank, repeats=args.repeats, on_record=on_record
            )
        except (RuntimeError, ValueError) as error:
            report["component"]["error"] = f"{type(error).__name__}: {error}"
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
