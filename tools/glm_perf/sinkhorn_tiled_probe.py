# SPDX-License-Identifier: Apache-2.0
"""Exact FP32, bounds and changing-replay gates before any serving admission."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 31, 64, 128, 640)
ITERATIONS = (1, 20, 64)
LOGIT_SCALES = (0, 0.1, 1, 10, 100)


def verify(build):
    provenance = json.loads((build / "provenance.json").read_text())
    for name, expected in provenance["assets"].items():
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != expected:
            raise ValueError("tiled normalization asset changed: " + name)
    return provenance


def load(build):
    provenance = verify(build)
    namespace = provenance["namespace"]
    torch.ops.load_library(str(build / f"glm_sinkhorn_tiled_bridge_v{provenance['version']}.so"))
    spec = importlib.util.spec_from_file_location(namespace + "_helper", build / "sinkhorn_tiled.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, provenance


def reference(mix, iterations, epsilon):
    mix = mix / (mix.sum(dim=-2, keepdim=True) + epsilon)
    for _ in range(iterations - 1):
        mix = mix / (mix.sum(dim=-1, keepdim=True) + epsilon)
        mix = mix / (mix.sum(dim=-2, keepdim=True) + epsilon)
    return mix


def run(build, output, *, allow_device_gate=False, device=0, order=None):
    if not allow_device_gate:
        raise ValueError("normalization device gate requires explicit selection")
    import torch_npu  # noqa: F401 -- register only after explicit hardware selection.

    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    helper, provenance = load(build)
    records = []
    generator = torch.Generator().manual_seed(968)
    for iterations in ITERATIONS:
        native = helper.SinkhornTiled(build, provenance["namespace"], iterations=iterations, order=order)
        native.prepare_counts(COUNTS)
        for rows in COUNTS:
            for scale in LOGIT_SCALES:
                prefix, suffix, elements = 16, 32, rows * 16
                input_backing = torch.full((prefix + elements + suffix,), -17.25, device=native.device)
                value = input_backing[prefix : prefix + elements].view(rows, 4, 4)
                output_backing = torch.full_like(input_backing, -29.5)
                result = output_backing[prefix : prefix + elements].view(rows, 4, 4)

                def invoke(native=native, rows=rows, value=value, result=result):
                    cores = min(helper.VECTOR_CORES, (rows + helper.TILE_ROWS - 1) // helper.TILE_ROWS)
                    native.launch(native.kernel, [value, result, native.configs[rows], native.indices], cores)

                def update(rows=rows, scale=scale, value=value, native=native):
                    logits = (torch.randn(rows, 4, 4, generator=generator) * scale).to(native.device)
                    value.copy_(torch.softmax(logits, dim=-1) + 1e-6)

                update()
                invoke()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    invoke()
                for replay in range(2):
                    if replay:
                        update()
                    before = input_backing.cpu().view(torch.int32)
                    expected = reference(value, iterations, 1e-6).cpu().view(torch.int32)
                    graph.replay()
                    torch.npu.synchronize()
                    actual = result.cpu().view(torch.int32)
                    if not torch.equal(actual, expected):
                        failed = dict(
                            rows=rows,
                            iterations=iterations,
                            scale=scale,
                            order=order,
                            mismatch_elements=int((actual != expected).sum()),
                        )
                        output.write_text(
                            json.dumps(
                                dict(complete=False, provenance=provenance, failure=failed, records=records), indent=2
                            )
                            + "\n"
                        )
                        raise AssertionError("tiled normalization changed FP32 bits: " + str(failed))
                    assert torch.equal(input_backing.cpu().view(torch.int32), before), "input or guards modified"
                    owned = output_backing.cpu()
                    assert torch.all(owned[:prefix] == -29.5) and torch.all(owned[-suffix:] == -29.5), (
                        "output guards modified"
                    )
                assert torch.equal(native(value).cpu().view(torch.int32), expected), "prepared wrapper changed bits"
                timing = None
                if iterations == 20 and scale == 1:
                    reference_graph = torch.npu.NPUGraph()
                    with torch.npu.graph(reference_graph):
                        reference_result = reference(value, iterations, 1e-6)
                    reference_graph.replay()
                    torch.npu.synchronize()
                    assert torch.equal(reference_result.cpu().view(torch.int32), expected)
                    samples = {"native": [], "reference": []}
                    for _ in range(10):
                        for name, replay_fn in (("reference", reference_graph.replay), ("native", graph.replay)):
                            torch.npu.synchronize()
                            began = time.perf_counter()
                            replay_fn()
                            torch.npu.synchronize()
                            samples[name].append((time.perf_counter() - began) * 1000)
                    timing = {
                        name: dict(median_ms=statistics.median(values), samples_ms=values)
                        for name, values in samples.items()
                    }
                records.append(
                    dict(
                        rows=rows,
                        iterations=iterations,
                        logit_scale=scale,
                        passed=True,
                        exact_fp32=True,
                        changed_replay=True,
                        guards_checked=True,
                        timing=timing,
                    )
                )
    report = dict(
        complete=True,
        provenance=provenance,
        records=records,
        order=order,
        row_orders={str(rows): helper.reduction_order(rows) for rows in COUNTS} if order is None else None,
        serving_evaluated=False,
        language_quality_evaluated=False,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--order", type=int, choices=(0, 1, 2), help="diagnostic override; serving requires shape order"
    )
    parser.add_argument("--allow-device-gate", action="store_true")
    args = parser.parse_args()
    result = run(
        args.build_dir, args.output, device=args.device, order=args.order, allow_device_gate=args.allow_device_gate
    )
    print(json.dumps(dict(complete=result["complete"], cases=len(result["records"]))))


if __name__ == "__main__":
    main()
