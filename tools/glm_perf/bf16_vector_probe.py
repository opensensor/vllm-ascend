# SPDX-License-Identifier: Apache-2.0
"""Exhaustive BF16 words, FP32 boundaries, bounds and changed graph replay."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

COUNTS = (1, 2, 7, 8, 15, 16, 17, 31, 63, 64, 127, 128, 1023, 1024, 1025, 65536, 5 * 65536, 640 * 128, 640 * 32 * 128)
MODES = (0, 1, 4, 5)


def verify(build):
    provenance = json.loads((build / "provenance.json").read_text())
    for name, expected in provenance["assets"].items():
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != expected:
            raise ValueError("vector BF16 asset changed: " + name)
    return provenance


def load(build):
    provenance = verify(build)
    torch.ops.load_library(str(build / f"glm_bf16_vector_bridge_v{provenance['version']}.so"))
    spec = importlib.util.spec_from_file_location(provenance["namespace"] + "_helper", build / "bf16_vector.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, provenance


def input_bits(count, mode, replay):
    index = torch.arange(count, dtype=torch.int64)
    high = (index * (251 if replay else 109) + (65519 if replay else 17)) % 65536
    if mode == 1:
        return high.to(torch.int16)
    # Each BF16 word occurs with exact, below-tie, tie and above-tie low words.
    low = torch.tensor((0, 0x7FFF, 0x8000, 0x8001, 0xFFFF))[index % 5]
    return ((high << 16) | low).to(torch.int32)


def run(build, output, *, allow_device_gate=False, device=0):
    if not allow_device_gate:
        raise ValueError("vector BF16 hardware gate requires explicit selection")
    import torch_npu  # noqa: F401 -- explicit device gate only.

    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    helper, provenance = load(build)
    native = helper.NativeBf16Vector(build, provenance["namespace"])
    native.prepare_counts(COUNTS)
    records = []
    for mode in MODES:
        for count in COUNTS:
            source_dtype = torch.bfloat16 if mode == 1 else torch.float32
            storage_dtype = torch.int16 if mode == 1 else torch.int32
            output_dtype = {0: torch.bfloat16, 1: torch.float32, 4: torch.float32, 5: torch.float16}[mode]
            output_storage = torch.int32 if output_dtype == torch.float32 else torch.int16
            alignment = 32 // torch.empty((), dtype=output_dtype).element_size()
            padded = (count + alignment - 1) // alignment * alignment
            before = torch.full((16 + count + 32,), 1234, dtype=storage_dtype)
            before[16 : 16 + count] = input_bits(count, mode, 0)
            source_bank = before.to(native.device)
            source = source_bank[16 : 16 + count].view(source_dtype)
            output_bank = torch.full((16 + padded + 32,), 5678, dtype=output_storage, device=native.device)
            result = output_bank[16 : 16 + padded].view(output_dtype)

            def invoke(source=source, result=result, count=count, mode=mode):
                native.launch(
                    native.kernel,
                    [source, result, native.configs[count, mode]],
                    min(helper.VECTOR_CORES, (count + helper.TILE - 1) // helper.TILE),
                )

            invoke()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                invoke()
            for replay in range(2):
                if replay:
                    before[16 : 16 + count] = input_bits(count, mode, replay)
                    source_bank.copy_(before)
                expected = helper.reference(before[16 : 16 + count], mode)
                graph.replay()
                torch.npu.synchronize()
                actual = result[:count].cpu().view(output_storage)
                if not torch.equal(actual, expected):
                    mismatch = (actual != expected).nonzero().flatten()
                    failure = dict(
                        mode=mode, count=count, replay=replay, mismatches=mismatch.numel(), first=mismatch[:8].tolist()
                    )
                    output.write_text(
                        json.dumps(
                            dict(complete=False, provenance=provenance, failure=failure, records=records), indent=2
                        )
                    )
                    raise AssertionError("vector BF16 bits changed: " + str(failure))
                assert torch.equal(source_bank.cpu(), before), "input or guards changed"
                owned = output_bank.cpu()
                assert torch.all(owned[:16] == 5678) and torch.all(owned[16 + padded :] == 5678), (
                    "output guards changed"
                )
                assert not torch.any(owned[16 + count : 16 + padded]), "owned padding must be zero"
            wrapped = native.convert(source, output_dtype, mode)
            assert torch.equal(wrapped.cpu().view(output_storage), expected), "prepared wrapper changed bits"
            samples = []
            for _ in range(10):
                torch.npu.synchronize()
                began = time.perf_counter()
                graph.replay()
                torch.npu.synchronize()
                samples.append((time.perf_counter() - began) * 1000)
            records.append(
                dict(
                    mode=mode,
                    count=count,
                    passed=True,
                    exact_bits=True,
                    changed_replay=True,
                    guards_checked=True,
                    median_wall_ms=statistics.median(samples),
                )
            )
    report = dict(complete=True, provenance=provenance, records=records, serving_evaluated=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allow-device-gate", action="store_true")
    args = parser.parse_args()
    result = run(args.build_dir, args.output, device=args.device, allow_device_gate=args.allow_device_gate)
    print(json.dumps(dict(complete=result["complete"], cases=len(result["records"]))))


if __name__ == "__main__":
    main()
