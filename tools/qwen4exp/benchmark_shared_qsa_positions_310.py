# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated changing-input 310P replay gate for shared QSA causal positions.

No model weights, server, RPC, or HCCL. Uses real graph segments and host-copy
callbacks; consumer clones stand in for twelve QSA layers. Timings describe
only causal-position staging and graph segmentation, not model throughput.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing
from tools.qwen4exp.resident_candidates import shared_qsa_positions as candidate
from vllm_ascend.models.qwen4_exp import model

QSA_LAYERS = 12
MTP_QUERY_TOKENS = 3


class PositionCapture(BreakableCUDAGraphCapture):
    """Use the NPU capture API without installing a global CUDA monkeypatch."""

    def _begin_segment(self):
        assert not self._capturing
        graph = torch.npu.NPUGraph()
        graph.capture_begin()
        self._current_graph = graph
        self._capturing = True


def metadata(lengths, queries):
    return SimpleNamespace(
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
        query_lens_cpu=torch.tensor(queries, dtype=torch.int32),
    )


def position_pass(function, context, rope):
    cache = {}
    return [
        function(boundaries, rope.shape[-1], rope.device, rope, prefix, cache).clone()
        for prefix, boundaries in context.attn_metadata.items()
    ]


def capture(function):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            function()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    captured = PositionCapture()
    with torch.npu.stream(stream), captured:
        outputs = function()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    return captured, outputs


def replay_on_stream(captured, stream):
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        captured.replay()
    torch.npu.current_stream().wait_stream(stream)


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
    records = []
    context = SimpleNamespace(attn_metadata={})
    prefixes = [f"qsa{layer}" for layer in range(QSA_LAYERS)]
    with (
        patch.object(model, "get_forward_context", return_value=context),
        patch.object(candidate, "get_forward_context", return_value=context),
    ):
        for requests in (1, 2, 3, 4):
            tokens = requests * MTP_QUERY_TOKENS
            boundaries = metadata([23410 + 100 * index for index in range(requests)], [3] * requests)
            context.attn_metadata = dict.fromkeys(prefixes, boundaries)
            rope = torch.stack((torch.arange(tokens) + 1000, torch.zeros(tokens), torch.ones(tokens))).long().npu()
            functions = {
                "baseline": lambda rope=rope: position_pass(model._step_qsa_logical_positions, context, rope),
                "shared": lambda rope=rope: position_pass(candidate.step_logical_positions, context, rope),
            }
            captured = {name: capture(function) for name, function in functions.items()}
            stream = torch.npu.Stream()
            replays = {
                name: lambda graph=graph, stream=stream: replay_on_stream(graph, stream)
                for name, (graph, outputs) in captured.items()
            }
            changes = [
                ([23413 + 100 * index for index in range(requests)], [3] * requests),
                ([262144 - 10 * index for index in range(requests)], [3] * requests),
            ]
            if requests > 1:
                # Test the staging contract for different packed boundaries
                # with the same total. This does not qualify attention replay
                # for a nonuniform scheduler batch.
                changes.append(([4096] * requests, [1, 5, *([3] * (requests - 2))]))
            for lengths, queries in changes:
                boundaries = metadata(lengths, queries)
                context.attn_metadata = dict.fromkeys(prefixes, boundaries)
                expected = model._host_qsa_logical_positions(boundaries, tokens)
                for name, replay in replays.items():
                    replay()
                    for output in captured[name][1]:
                        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            assert captured["baseline"][0].num_eager_breaks == QSA_LAYERS
            assert captured["shared"][0].num_eager_breaks == 1
            context.attn_metadata = dict.fromkeys(prefixes, metadata([23410] * requests, [MTP_QUERY_TOKENS] * requests))
            records.append(
                {
                    "requests": requests,
                    "tokens": tokens,
                    "scope": "causal_position_staging_only_not_model_throughput",
                    "bitwise_equal_changing_boundaries": True,
                    "eager_breaks": {name: graph.num_eager_breaks for name, (graph, _) in captured.items()},
                    "graph_segments": {name: graph.num_graphs for name, (graph, _) in captured.items()},
                    "graph_replay": paired_timing(replays, 30, 7),
                }
            )
            print(json.dumps(records[-1]), flush=True)
            del captured, replays
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
