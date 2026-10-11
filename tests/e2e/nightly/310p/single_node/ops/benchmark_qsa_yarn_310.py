# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare 1M YaRN rotary math with CPU and replay graphs on six idle 310P chips."""

import argparse
import json
from pathlib import Path

import torch

from vllm_ascend.models.qwen4_exp.qsa import _mrope_interleaved_dims, partial_rope_cos_sin
from vllm_ascend.models.qwen4_exp.rope_scaling import QSARopeScaling

torch.npu.set_compile_mode(jit_compile=False)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
parameters = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}
policy = QSARopeScaling.from_parameters(parameters)
frequency_cpu = policy.inverse_frequencies(64, 10000000.0)
results = []
for device_index in range(6):
    torch.npu.set_device(device_index)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{device_index}")
    frequencies = frequency_cpu.to(device)
    axes_cpu = torch.tensor(_mrope_interleaved_dims([11, 11, 10]))
    axes = axes_cpu.to(device)
    host = torch.tensor([0, 262143, 262144, 524288, 1048575])
    host = torch.stack((host, host // 3, host // 7))
    image_positions = host.to(device)
    image_kwargs = dict(
        rotary_dim=64,
        base=10000000.0,
        dtype=torch.float32,
        rope_scaling=policy,
        inv_freq=frequencies,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        frequency_axes=axes,
    )
    image_tables = partial_rope_cos_sin(image_positions, **image_kwargs)
    image_angles = host.index_select(0, axes_cpu).t() * frequency_cpu
    for actual, expected in zip(image_tables, (image_angles.cos(), image_angles.sin())):
        expected = torch.cat((expected, expected), dim=-1) * policy.attention_factor
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
    raw_errors = []
    image_angles_npu = image_angles.to(device)
    for actual, expected in zip(
        (image_angles_npu.cos(), image_angles_npu.sin()), (image_angles.cos(), image_angles.sin())
    ):
        raw_errors.append(float((actual.cpu() - expected).abs().max()))
    host = host[0]
    positions = host.to(device)
    kwargs = dict(
        rotary_dim=64,
        base=10000000.0,
        dtype=torch.float32,
        rope_scaling=policy,
        inv_freq=frequencies,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        frequency_axes=axes,
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            partial_rope_cos_sin(positions, **kwargs)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        cos, sin = partial_rope_cos_sin(positions, **kwargs)
    errors = []
    for delta in [0, -137, 0]:
        current = (host + delta).clamp_min(0)
        positions.copy_(current)
        graph.replay()
        torch.npu.synchronize()
        angles = current[:, None] * frequency_cpu
        for actual, expected in zip((cos.cpu(), sin.cpu()), (angles.cos(), angles.sin())):
            expected = torch.cat((expected, expected), dim=-1) * policy.attention_factor
            errors.append(float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    results.append(
        {
            "device": device_index,
            "text_graph_replays": 3,
            "image_eager_passed": True,
            "max_abs_error": max(errors),
            "unreduced_trig_max_error": max(raw_errors),
            "frequency_bytes": frequencies.numel() * frequencies.element_size(),
        }
    )
    del graph, cos, sin, frequencies, positions, axes, image_positions, image_tables
    torch.npu.empty_cache()
args.output.write_text(json.dumps({"passed": True, "devices": results}, indent=2) + "\n")
print(json.dumps(results), flush=True)
