# SPDX-License-Identifier: Apache-2.0
"""Diagnostic beta row gate; this does not qualify or change full KDA serving."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .build_query_bf16_vector import digest

COUNTS = (1, 16, 63, 64, 65, 134, 640)
CHANNELS = (16, 32, 64, 128, 256)
BETA_VALUES = (0.0, 1.0, 0.1234567, -0.9876543, 1.0002, 0.00001)


def verify(build):
    provenance = json.loads((build / "provenance.json").read_text())
    for name, expected in provenance["assets"].items():
        if digest(build / name) != expected:
            raise ValueError("KDA beta scale asset changed: " + name)
    return provenance


def run(build, output, *, allow_device_gate=False, device=0):
    if not allow_device_gate:
        raise ValueError("KDA beta diagnostic requires explicit device selection")
    provenance = verify(build)
    import torch_npu  # noqa: F401 -- explicit device gate only.

    torch.set_num_threads(4)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    namespace = provenance["namespace"]
    torch.ops.load_library(str(build / f"glm_kda_beta_scale_bridge_v{provenance['version']}.so"))
    factory, launch = getattr(torch.classes, namespace).Kernel, getattr(torch.ops, namespace).launch
    kernels = {
        name: factory(str(build / f"kda_beta_{name}.bin"), "glm_kda_beta_scale_v1") for name in ("scalar", "vector")
    }
    records = []
    for rows in COUNTS:
        for channels in CHANNELS:
            for strided in (False, True):
                prefix, suffix = 16, 32
                source_stride, destination_stride = channels * (17 if strided else 1), channels + 16
                source_size = (rows - 1) * source_stride + channels
                generator = torch.Generator().manual_seed(rows * 1000 + channels)
                bits = torch.randint(0, 65536, (prefix + source_size + suffix,), generator=generator).short()
                # Finite FP16 values, including signed zeros and subnormals.
                bits = torch.where((bits.int() & 0x7C00) == 0x7C00, bits.int() & -1025, bits.int()).short()
                input_backing = bits.npu()
                source = input_backing[prefix : prefix + source_size].view(torch.float16)
                beta_backing = torch.full((16 + rows * 3 + 16,), -91.0).npu()
                beta = beta_backing[16 : 16 + rows * 3]
                config = torch.tensor((rows, channels, source_stride, destination_stride, 3), dtype=torch.int64).npu()
                output_banks = {
                    name: torch.full((prefix + rows * destination_stride + suffix,), -73.5, dtype=torch.float16).npu()
                    for name in kernels
                }
                operations, results, graphs = {}, {}, {}
                for name in kernels:
                    target = output_banks[name][prefix : prefix + rows * destination_stride]
                    results[name] = target.as_strided((rows, channels), (destination_stride, 1))

                    def invoke(name=name, target=target, source=source, beta=beta, config=config, rows=rows):
                        launch(kernels[name], [source, beta, target, config], min(8, rows))

                    operations[name] = invoke
                    invoke()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        invoke()
                    graphs[name] = graph
                for replay, beta_value in enumerate(BETA_VALUES):
                    if replay:
                        bits = bits.roll(109)
                        input_backing.copy_(bits)
                    beta[::3].fill_(beta_value)
                    before_beta = beta_backing.cpu()
                    for graph in graphs.values():
                        graph.replay()
                    torch.npu.synchronize()
                    scalar = results["scalar"].cpu().view(torch.int16)
                    vector = results["vector"].cpu().view(torch.int16)
                    if not torch.equal(scalar, vector):
                        failure = dict(
                            rows=rows,
                            channels=channels,
                            strided=strided,
                            beta=beta_value,
                            mismatches=int((scalar != vector).sum()),
                        )
                        output.write_text(
                            json.dumps(
                                dict(complete=False, provenance=provenance, failure=failure, records=records), indent=2
                            )
                        )
                        raise AssertionError("KDA beta vector changed scalar output bits: " + str(failure))
                    assert torch.equal(input_backing.cpu(), bits), "input or guards changed"
                    assert torch.equal(beta_backing.cpu(), before_beta), "beta or guards changed"
                    for bank in output_banks.values():
                        copied = bank.cpu()
                        assert torch.all(copied[:prefix] == -73.5) and torch.all(copied[-suffix:] == -73.5)
                        gaps = copied[prefix : prefix + rows * destination_stride].view(rows, destination_stride)[
                            :, channels:
                        ]
                        assert torch.all(gaps == -73.5), "write crossed a row boundary"
                timing = {}
                for name, graph in graphs.items():
                    samples = []
                    for _ in range(10):
                        torch.npu.synchronize()
                        began = time.perf_counter()
                        graph.replay()
                        torch.npu.synchronize()
                        samples.append((time.perf_counter() - began) * 1000)
                    timing[name] = statistics.median(samples)
                records.append(
                    dict(
                        rows=rows,
                        channels=channels,
                        strided=strided,
                        exact_bits=True,
                        changed_replay=True,
                        guards_checked=True,
                        timing_wall_ms=timing,
                    )
                )
    report = dict(
        complete=True,
        provenance=provenance,
        records=records,
        full_kda_evaluated=False,
        serving_evaluated=False,
        finite_fp16_inputs=True,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allow-device-gate", action="store_true")
    args = parser.parse_args()
    report = run(args.build_dir, args.output, device=args.device, allow_device_gate=args.allow_device_gate)
    print(json.dumps(dict(complete=report["complete"], cases=len(report["records"]))))


if __name__ == "__main__":
    main()
