# SPDX-License-Identifier: Apache-2.0
"""Validate exact G128 INT8 Cube dots; this is not a model throughput test."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms


def pack_operands(activation, weight):
    if activation.shape != (16, 128) or weight.shape != (64, 128):
        raise ValueError("the probe supports M16/N64/K128 only")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("the probe requires signed INT8 operands")
    left = activation.reshape(16, 4, 32).permute(1, 0, 2).contiguous()
    right = weight.reshape(4, 16, 4, 32).permute(0, 2, 1, 3).contiguous()
    return left, right


def unpack_output(output):
    return output.reshape(4, 16, 16).permute(1, 0, 2).reshape(16, 64)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--time", action="store_true", help="requires idle NPUs; never infer serving speed from this")
    args = parser.parse_args()
    manifest = json.loads((args.build_dir / "provenance.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((args.build_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"probe digest mismatch: {name}")
    torch.ops.load_library(manifest["bridge"])
    records = []
    for device in args.devices:
        torch.npu.set_device(device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        kernel = torch.classes.qwen_int8_group_probe_v1.Kernel(manifest["binary"], "qwen_int8_group_probe_v1")
        output = torch.empty((4, 16, 16), dtype=torch.int32, device="npu")
        generator = torch.Generator().manual_seed(314)
        for pattern in ["random", "zero", "signed_extremes"]:
            x = torch.randint(-127, 128, (16, 128), generator=generator, dtype=torch.int8)
            w = torch.randint(-23, 24, (64, 128), generator=generator, dtype=torch.int8)
            if pattern == "zero":
                x.zero_()
            elif pattern == "signed_extremes":
                x.fill_(127)
                x[:, ::2].fill_(-127)
                w.fill_(23)
                w[::2].neg_()
            left, right = (tensor.npu() for tensor in pack_operands(x, w))

            def run(kernel=kernel, left=left, right=right, output=output):
                torch.ops.qwen_int8_group_probe_v1.launch(kernel, [left, right, output], 1)

            run()
            torch.npu.synchronize()
            expected = x.int() @ w.int().t()
            assert torch.equal(unpack_output(output.cpu()), expected), (device, pattern)
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                run()
            torch.npu.current_stream().wait_stream(stream)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                run()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(unpack_output(output.cpu()), expected)
            record = {"device": device, "pattern": pattern, "exact_int32": True, "graph_replay": True}
            if args.time:
                record["eager"] = timed_ms(run, 30, 5)
                record["graph"] = timed_ms(graph.replay, 30, 5)
            records.append(record)
            del graph
        del kernel, left, right, output
        torch.npu.empty_cache()
    args.output.write_text(json.dumps({"passed": True, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
