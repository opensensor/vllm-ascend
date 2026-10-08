# SPDX-License-Identifier: Apache-2.0
"""Explicit paired full-MoE replay and timing gate for deferred route columns."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import torch

from .reconstruction_probe import time_pair

FEATURES = ("native_route_columns", "prerounded_weight_scales")


def verify_pair(baseline, candidate, feature="native_route_columns"):
    if feature not in FEATURES:
        raise ValueError("unsupported full-MoE paired feature")
    reports = [json.loads((root / "provenance.json").read_text()) for root in (baseline, candidate)]
    old, new = (report["_build"] for report in reports)
    if old.get(feature, False) or new.get(feature) is not True:
        raise ValueError("pair requires a baseline without and candidate with " + feature)
    ignored = {"namespace", "version", "helper_package", feature}
    if any(old.get(key, False) != new.get(key, False) for key in old.keys() | new.keys() if key not in ignored):
        raise ValueError("route-column pair differs in another kernel schedule")
    for root, report in zip((baseline, candidate), reports):
        if report["_build"].get("fp16_route_workspace") is not True:
            raise ValueError("paired producer/reducer require FP16 workspace")
        for name in ("glm_fused_gate_up.bin", "glm_fused_down.bin", "glm_fused_reduce.bin", "glm_fused_pack.bin"):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != report[name]["binary_sha256"]:
                raise ValueError("paired native binary changed: " + name)
        package = root / report["_build"]["helper_package"]
        for name, expected in report["_helpers"].items():
            if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
                raise ValueError("paired native helper changed: " + name)
        bridge = root / f"glm_reconstruction_bridge_v{report['_build']['version']}.so"
        if hashlib.sha256(bridge.read_bytes()).hexdigest() != report["reconstruction_bridge.cpp"]["binary_sha256"]:
            raise ValueError("paired native bridge changed")
    return reports


def load(root, report, activation_bits):
    options = report["_build"]
    package = options["helper_package"]
    directory = root / package
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[package].__file__).resolve() != (directory / "__init__.py").resolve():
        raise ValueError("loaded helper belongs to another build")
    torch.ops.load_library(str(root / f"glm_reconstruction_bridge_v{options['version']}.so"))
    helper = importlib.import_module(package + ".glm_fused_moe")
    return helper.NativeFusedMoE(
        root,
        namespace=options["namespace"],
        activation_bits=activation_bits,
        prepared_weight_layout=options["prepared_weight_layout"],
        weight_decode_lut=options.get("weight_decode_lut", False),
        fp16_route_workspace=True,
    )


def run(baseline, candidate, output, *, allow_device_gate=False, device=0, repeats=5, feature="native_route_columns"):
    if not allow_device_gate:
        raise ValueError("full MoE hardware gate requires explicit --allow-device-gate")
    reports = verify_pair(baseline, candidate, feature)
    import torch_npu  # noqa: F401 -- explicit hardware gate only.

    torch.npu.set_device(device)
    records = []
    for activation in (4, 8):
        natives = [load(root, report, activation) for root, report in zip((baseline, candidate), reports)]
        for bits in (2, 3, 4):
            for tokens in (2, 8, 17, 640):
                generator = torch.Generator().manual_seed(tokens * 100 + bits)
                experts, width = 3, 256
                gate = torch.randint(
                    -(1 << (bits - 1)),
                    1 << (bits - 1),
                    (experts, 2 * width, width),
                    dtype=torch.int8,
                    generator=generator,
                )
                down = torch.randint(
                    -(1 << (bits - 1)), 1 << (bits - 1), (experts, width, width), dtype=torch.int8, generator=generator
                )
                codes = [natives[0].pack_weight_codes(value, bits).npu() for value in (gate, down)]
                scale_inputs = [
                    torch.rand(experts, n // 32, width // 32, generator=generator) * 0.01 + 0.001
                    for n in (2 * width, width)
                ]
                # Exercise half-rounding ties, subnormals and signed zero, not
                # just the ordinary random scales found in expert checkpoints.
                corners = torch.tensor([0.0, -0.0, 2**-25, 2**-24, 2**-6 + 2**-17, 2**-6 + 3 * 2**-17])
                scale_inputs[0].view(-1)[: corners.numel()] = corners
                scales = [
                    tuple(
                        (value.half().float() if getattr(native, "prerounded_weight_scales", False) else value).npu()
                        for value in scale_inputs
                    )
                    for native in natives
                ]
                x = torch.randn(tokens, width, generator=generator).half().npu()
                ids = torch.randint(0, experts + 1, (tokens, 8), dtype=torch.int64, generator=generator).npu()
                weights = torch.rand(tokens, 8, generator=generator).npu()
                weights[:, 0] = 0
                operations = {
                    name: partial(native, x, codes[0], pair_scales[0], codes[1], pair_scales[1], weights, ids)
                    for name, native, pair_scales in zip(("baseline", "candidate"), natives, scales)
                }
                results, graphs = [], []
                for operation in operations.values():
                    operation()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        result = operation()
                    graphs.append(graph)
                    results.append(result)
                for replay in range(2):
                    if replay:
                        x.mul_(1.25)
                        weights.mul_(0.75)
                        ids.fill_(1)
                    for graph in graphs:
                        graph.replay()
                    torch.npu.synchronize()
                    assert torch.equal(results[0].cpu().view(torch.int32), results[1].cpu().view(torch.int32)), (
                        "full MoE bits changed"
                    )
                timing = time_pair(dict(zip(operations, (graph.replay for graph in graphs))), repeats)
                records.append(
                    dict(
                        tokens=tokens,
                        weight_bits=bits,
                        activation_bits=activation,
                        bitwise_equal=True,
                        changed_replay=True,
                        timing=timing,
                    )
                )
    report = dict(complete=True, feature=feature, provenance=reports, records=records, real_model_evaluated=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-device-gate", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feature", choices=FEATURES, default="native_route_columns")
    args = parser.parse_args()
    print(
        run(
            args.baseline,
            args.candidate,
            args.output,
            allow_device_gate=args.allow_device_gate,
            device=args.device,
            repeats=args.repeats,
            feature=args.feature,
        )["complete"]
    )


if __name__ == "__main__":
    main()
