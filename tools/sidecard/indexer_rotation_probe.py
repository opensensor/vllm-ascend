# SPDX-License-Identifier: Apache-2.0
"""Qualify and time fused CUDA indexer butterflies without loading a model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .layer_indexer import hadamard_reference

VECTOR_COUNTS = (1, 8, 32, 256, 1024)
SAMPLES = 5
ITERATIONS = 100


def run(output):
    """Compare complete output bytes, graph replay, and host-inclusive timings."""
    import torch

    from .cuda_layer_ops import hadamard

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    cases, timings = [], []

    def reference(x):
        return hadamard_reference(x, torch).bfloat16().half()

    with torch.inference_mode():
        for vectors in VECTOR_COUNTS:
            for scale in (0.0, 0.125, 1.0, 64.0, 4096.0):
                torch.manual_seed(21377 + vectors)
                values = (torch.randn(vectors, 128, device="cuda") * scale).bfloat16().float()
                actual, expected = hadamard(values), reference(values)
                actual_bytes, expected_bytes = actual.cpu().numpy().tobytes(), expected.cpu().numpy().tobytes()
                equal = actual_bytes == expected_bytes
                cases.append(dict(vectors=vectors, scale=scale, bitwise_equal=equal))
                if not equal:
                    raise AssertionError("fused Hadamard changed rounded output bytes")
            for name, operation in (("reference", reference), ("fused", hadamard)):
                for _ in range(5):
                    operation(values)
                torch.cuda.synchronize()
                samples = []
                for _ in range(SAMPLES):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    begin.record()
                    for _ in range(ITERATIONS):
                        operation(values)
                    end.record()
                    end.synchronize()
                    samples.append(
                        dict(
                            wall_microseconds=(time.perf_counter() - started) * 1e6 / ITERATIONS,
                            stream_microseconds=begin.elapsed_time(end) * 1000 / ITERATIONS,
                        )
                    )
                timings.append(dict(vectors=vectors, mode=name, samples=samples))
        graph_input = torch.randn(32, 128, device="cuda").bfloat16().float()
        hadamard(graph_input)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = hadamard(graph_input)
        for trial in range(4):
            graph_input.copy_(torch.randn_like(graph_input).bfloat16().float() * (trial + 1))
            graph.replay()
            actual = graph_output.cpu().numpy().tobytes()
            expected = reference(graph_input).cpu().numpy().tobytes()
            if actual != expected:
                raise AssertionError("fused graph replay changed rounded output bytes")
            cases.append(dict(graph_replay=trial, bitwise_equal=True, output_sha256=hashlib.sha256(actual).hexdigest()))
        for invalid in (torch.empty(1, 127, device="cuda"), torch.empty(1, 128, device="cuda", dtype=torch.float16)):
            try:
                hadamard(invalid)
            except ValueError:
                continue
            raise AssertionError("fused rotation admitted invalid input geometry")
    report = dict(
        passed=True,
        full_model_validated=False,
        scope="FP32 butterfly order, BF16 then FP16 rounding; stream times include launch gaps",
        cases=cases,
        timings=timings,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, cases=len(cases))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
