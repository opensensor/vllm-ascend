# SPDX-License-Identifier: Apache-2.0
"""Bounded standalone NPU qualification; does not change serving workers."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(args.bridge))
    kernels = {
        name: torch.classes.glm_sinkhorn_v1.Kernel(str(root / binary), "glm_kda_score_probe_v1")
        for name, binary in (("baseline", "score-baseline.bin"), ("cached", "score-cached.bin"))
    }
    records = []
    result = {
        "device": args.device,
        "shape": [16, 64, 128],
        "blocks": 8,
        "scope": "raw Aqk/Akk scores only; not the full KDA pipeline or serving",
        "hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("score-baseline.bin", "score-cached.bin", "kda-score-probe.cpp")
        },
        "cases": records,
    }

    def save():
        (root / f"score-probe-device-{args.device}.json").write_text(json.dumps(result, indent=2) + "\n")

    for seed, decay in ((1, 0), (2, 0.01), (3, 0.1), (4, 1), (5, 5), (6, 20), (7, 0.3), (8, 0)):
        generator = torch.Generator().manual_seed(seed)
        q_cpu = (torch.randn(16, 64, 128, generator=generator) * 0.088).half()
        k_cpu = (torch.randn(16, 64, 128, generator=generator) * 0.088).half()
        g_cpu = (-torch.rand(16, 64, 128, generator=generator).cumsum(1) * decay).half()
        inputs = [tensor.to(f"npu:{args.device}") for tensor in (q_cpu, k_cpu, g_cpu)]
        outputs = {
            name: [torch.full((16, 64, 64), float("nan"), device=f"npu:{args.device}") for _ in range(2)]
            for name in kernels
        }

        def launch(name, inputs=inputs, outputs=outputs):
            torch.ops.glm_sinkhorn_v1.launch(kernels[name], inputs + outputs[name], 8)

        for name in kernels:
            launch(name)
        torch.npu.synchronize()
        actual = {name: [tensor.cpu() for tensor in pair] for name, pair in outputs.items()}
        mismatches = sum(
            int((a.view(torch.int32) != b.view(torch.int32)).sum())
            for a, b in zip(actual["baseline"], actual["cached"])
        )
        finite = all(bool(torch.isfinite(tensor).all()) for pair in actual.values() for tensor in pair)
        upper_zero = all(
            int(torch.triu(tensor, diagonal=1).count_nonzero()) == 0 for pair in actual.values() for tensor in pair
        )
        # Long decay legitimately underflows distant causal pairs. Diagonals
        # have zero gate difference and must still contain the expected dots.
        nonzero = all(int(tensor.count_nonzero()) >= 1000 for pair in actual.values() for tensor in pair)
        diagonal_reference = [(row * k_cpu).float().sum(-1) for row in (q_cpu, k_cpu)]
        diagonal_error = max(
            float((tensor.diagonal(dim1=-2, dim2=-1) - expected).abs().max())
            for tensor, expected in zip(actual["baseline"], diagonal_reference)
        )
        input_unchanged = all(torch.equal(device.cpu(), cpu) for device, cpu in zip(inputs, (q_cpu, k_cpu, g_cpu)))
        record = {
            "seed": seed,
            "decay": decay,
            "fp32_bit_mismatches": mismatches,
            "finite": finite,
            "upper_triangle_zero": upper_zero,
            "nonzero": nonzero,
            "inputs_unchanged": input_unchanged,
            "diagonal_reference_max_abs": diagonal_error,
        }
        if decay == 0:
            reference = [(row[:, :, None, :] * k_cpu[:, None, :, :]).float().sum(-1).tril() for row in (q_cpu, k_cpu)]
            record["zero_gate_reference_max_abs"] = max(
                float((a - b).abs().max()) for a, b in zip(actual["baseline"], reference)
            )
        records.append(record)
        save()
        print(json.dumps(record), flush=True)
        assert mismatches == 0 and finite and upper_zero and nonzero and input_unchanged, record
        assert record.get("zero_gate_reference_max_abs", 0) < 1e-5, record
        assert diagonal_error < 1e-5, record

    # Alternate launch order to reduce temporal bias. Each sample includes a
    # synchronization and three complete 16-head score launches.
    for name in kernels:
        launch(name)
    torch.npu.synchronize()
    samples = {name: [] for name in kernels}
    for trial in range(9):
        order = ("baseline", "cached") if trial % 2 == 0 else ("cached", "baseline")
        for name in order:
            start = time.perf_counter()
            for _ in range(3):
                launch(name)
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000 / 3)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    result["timings"] = {
        "samples_ms": samples,
        "median_ms": medians,
        "speedup": medians["baseline"] / medians["cached"],
    }
    result["passed"] = True
    save()
    print(json.dumps(result["timings"]), flush=True)


if __name__ == "__main__":
    main()
