# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare one synthetic Qwen hyperconnection; never load a model/server."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.resident_candidates.shared_hc_operand import mix
from vllm_ascend.models.qwen4_exp.model import _GatedResidual

HIDDEN_SIZE = 2560
HC_COUNT = 4
LOWRANK = 320


def capture(function):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            function()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = function()
    return graph, output


def paired_timing(functions, iterations, repeats):
    samples = {name: [] for name in functions}
    events = {name: [] for name in functions}
    for trial in range(repeats):
        names = list(functions)
        if trial % 2:
            names.reverse()
        for name in names:
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000 / iterations)
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            for _ in range(iterations):
                functions[name]()
            end.record()
            end.synchronize()
            events[name].append(begin.elapsed_time(end) / iterations)
    return {
        "wall_median_ms": {name: statistics.median(values) for name, values in samples.items()},
        "event_median_ms": {name: statistics.median(values) for name, values in events.items()},
        "wall_samples_ms": samples,
        "event_samples_ms": events,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(1055)
    module = _GatedResidual(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN_SIZE,
        lowrank=LOWRANK,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
    ).npu()
    for parameter in module.parameters():
        parameter.copy_((torch.randn(parameter.shape) * 0.01).half().npu())
        if parameter.ndim == 2 and min(parameter.shape) >= 16:
            parameter.data = torch_npu.npu_format_cast(parameter.data, 29)
    module.prepare_norm_affine()
    records = []
    for tokens in (3, 6, 2560):
        values = torch.randn(tokens, HC_COUNT * HIDDEN_SIZE).half().npu()
        block = torch.randn(tokens, HIDDEN_SIZE).half().npu()

        def baseline(values=values, block=block):
            mixed, residual = module.mix(values)
            return mixed, module.combine(block, residual)

        def shared(values=values, block=block):
            mixed, residual = mix(module, values)
            return mixed, module.combine(block, residual)

        expected = baseline()
        actual = shared()
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
            assert bool(torch.isfinite(right).all().cpu())
        record = {
            "tokens": tokens,
            "hidden_size": HIDDEN_SIZE,
            "hc_count": HC_COUNT,
            "lowrank": LOWRANK,
            "scope": "synthetic_hyperconnection_mix_combine_not_whole_model",
            "bitwise_equal": True,
            "saved_normalized_bytes": {
                "baseline": tokens * HC_COUNT * HIDDEN_SIZE * 4,
                "shared": tokens * HC_COUNT * HIDDEN_SIZE * 2,
            },
            "output_sha256": hashlib.sha256(actual[1].cpu().numpy().tobytes()).hexdigest(),
        }
        for _ in range(3):
            baseline()
            shared()
        record["eager"] = paired_timing({"baseline": baseline, "shared": shared}, 2 if tokens == 2560 else 10, 7)
        if tokens < 2560:
            baseline_graph, expected = capture(baseline)
            shared_graph, actual = capture(shared)
            baseline_graph.replay()
            shared_graph.replay()
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
            record["graph"] = paired_timing({"baseline": baseline_graph.replay, "shared": shared_graph.replay}, 100, 7)
            del baseline_graph, shared_graph
        records.append(record)
        print(
            json.dumps({"tokens": tokens, "bitwise_equal": True, "timing": record["eager"]["wall_median_ms"]}),
            flush=True,
        )
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
