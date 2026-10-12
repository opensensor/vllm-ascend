# SPDX-License-Identifier: Apache-2.0
"""Qualify changed-input KDA attention graph replay with real TP2 weights."""

import argparse
import json
import os
from pathlib import Path

from .layer_math import TextLayer
from .layer_profile import measure
from .layer_weights import load_dense


def run(checkpoint, device, output):
    import torch
    import torch_npu

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("graph gate requires the assigned sidecard")
    torch.npu.set_device(int(device.split(":")[1]))
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    config = json.loads((checkpoint / "config.json").read_bytes())["text_config"]
    weights, _ = load_dense(checkpoint, (18,), device, torch, tp_rank=0)
    layer = TextLayer(18, weights[18], config, device, torch, None, 2048, tp_rank=0)
    torch.manual_seed(20261015)
    inputs = [torch.randn(1, 4096).half().to(device) for _ in range(8)]
    with torch.inference_mode():
        layer.reset()
        expected = []
        for x in inputs:
            expected.append((layer.kda(x).clone(), layer.state.clone(), layer.conv.clone()))
        layer.reset()
        static = inputs[0].clone()
        for _ in range(3):
            layer.kda(static)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = layer.kda(static)
        torch.npu.synchronize()
        layer.reset()
        checks = []
        for x, reference in zip(inputs, expected, strict=True):
            static.copy_(x)
            graph.replay()
            torch.npu.synchronize()
            values = (actual, layer.state, layer.conv)
            checks.append(
                {"bitwise_equal": [bool(torch.equal(a.cpu(), b.cpu())) for a, b in zip(values, reference, strict=True)]}
            )
        layer.reset()
        eager = measure(lambda: layer.kda(static), torch.npu.synchronize)
        layer.reset()
        replay = measure(graph.replay, torch.npu.synchronize)
    report = {
        "device": device,
        "passed": all(all(c["bitwise_equal"]) for c in checks),
        "checks": checks,
        "eager": eager,
        "replay": replay,
        "complete_model_speed_validated": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("KDA attention replay changed output or persistent state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("npu:0", "npu:1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.output)


if __name__ == "__main__":
    main()
