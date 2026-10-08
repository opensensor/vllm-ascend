# SPDX-License-Identifier: Apache-2.0
"""Explicit future device gate; importing this module performs no NPU work."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

COUNTS = (1, 2, 15, 16, 17, 31, 1023, 1024, 1025, 8192, 32768, 65536, 640 * 32 * 128)


def verify(build):
    provenance = json.loads((build / "provenance.json").read_text())
    for name, expected in provenance["assets"].items():
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != expected:
            raise ValueError("query-vector build asset changed: " + name)
    bridge = Path(provenance["bridge"]["path"])
    if hashlib.sha256(bridge.read_bytes()).hexdigest() != provenance["bridge"]["sha256"]:
        raise ValueError("query-vector bridge changed")
    return provenance


def run(build, output, *, allow_device_gate=False, device=0):
    if not allow_device_gate:
        raise ValueError("hardware gates require explicit --allow-device-gate")
    provenance = verify(build)
    import torch_npu  # noqa: F401 -- lazy NPU registration only for the explicitly requested gate.

    torch.npu.set_device(device)
    torch.ops.load_library(provenance["bridge"]["path"])
    spec = importlib.util.spec_from_file_location("query_vector_frozen_gate", build / "query_bf16_vector.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native = helper.NativeQueryVector(build, provenance["namespace"])
    native.prepare_counts(COUNTS)
    records = []
    for count in COUNTS:
        # All 65,536 FP16 patterns occur when count >= 65,536, including both
        # signed canonical NaNs, infinities, zeros, subnormals and RNE ties.
        prefix, suffix = 16, 32
        initial = ((torch.arange(prefix + count + suffix, dtype=torch.int64) * 109 + 17) % 65536).to(torch.int16)
        backing = initial.to(native.device)
        bits = backing[prefix : prefix + count]
        value = bits.view(torch.float16)
        native(value)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = native(value)
        for replay in range(2):
            if replay:
                initial = ((torch.arange(initial.numel(), dtype=torch.int64) * 251 + 65519) % 65536).to(torch.int16)
                backing.copy_(initial)
            expected = helper.fp16_bits_to_bf16(initial[prefix : prefix + count])
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(result.cpu().view(torch.int16), expected), "vector BF16 conversion changed bits"
            assert torch.equal(backing.cpu(), initial), "conversion modified its input or guards"
            padded = (count + 15) // 16 * 16
            owned = result.as_strided((padded,), (1,)).view(torch.int16).cpu()
            assert not torch.any(owned[count:]), "owned DMA padding was not zero"
        records.append(dict(count=count, passed=True, changed_replay=True, input_guards=True, output_padding=True))
    report = dict(
        complete=True,
        provenance=provenance,
        records=records,
        quality_evaluated=False,
        nan_policy="signed canonical NaN matches the existing scalar converter",
        timing_evaluated=False,
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
    report = run(args.build_dir, args.output, allow_device_gate=args.allow_device_gate, device=args.device)
    print(json.dumps(dict(complete=report["complete"], cases=len(report["records"]))))


if __name__ == "__main__":
    main()
