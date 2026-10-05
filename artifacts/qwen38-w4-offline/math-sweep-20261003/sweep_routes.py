"""Isolate Qwen W4A8 packing and routed projection on one Ascend 310P.

This uses synthetic packed weights and varying G128 metadata. It leaves the
serving process alone and measures neither a full MoE layer nor model tokens/s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from tools.qwen4exp.profile_projection_dtypes_310 import timed
from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    pack_activation_device,
    pack_native_metadata,
    pack_native_weight,
    swiglu_pack_activation_device,
)
from vllm_ascend.utils import enable_custom_op

EXPERTS = 32
ROUTES_PER_TOKEN = 8
PROJECTIONS = {"gate_up": (2560, 1280), "down": (640, 2560)}


def make_bank(width: int, outputs: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    codes = torch.randint(-128, 128, (EXPERTS, outputs, width // 2), dtype=torch.int8, generator=generator)
    scales = (torch.rand(EXPERTS, outputs, width // 128, generator=generator) * 0.018 + 0.002).half()
    offsets = torch.randint(-8, 8, scales.shape, dtype=torch.int8, generator=generator)
    packed, sums = zip(*(pack_native_weight(codes[expert]) for expert in range(EXPERTS)))
    return (
        torch.stack(packed).npu(),
        torch.stack([pack_native_metadata(value) for value in scales]).npu(),
        torch.stack([pack_native_metadata(value).half() for value in offsets]).npu(),
        torch.stack(sums).npu(),
    )


def route_ids(rows: int, pattern: str) -> torch.Tensor:
    positions = torch.arange(rows, dtype=torch.int32)
    if pattern == "single":
        return torch.zeros_like(positions)
    if pattern == "four":
        return positions % 4
    if pattern == "tp4_sparse":
        return torch.where(positions % 4 == 0, (positions // 4) % EXPERTS, -1)
    if pattern == "distinct":
        return positions % EXPERTS
    raise ValueError(pattern)


def capture(function, unroll: int) -> tuple[torch.npu.NPUGraph, object]:
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            function()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(unroll):
            output = function()
    graph.replay()
    torch.npu.synchronize()
    return graph, output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-unroll", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or min(args.tokens + [args.iterations, args.repeats, args.graph_unroll]) <= 0:
        parser.error("require a fresh output and positive timings")
    if max(args.tokens) * ROUTES_PER_TOKEN > 128:
        parser.error("routed kernel supports at most 128 routes")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if "310" not in torch.npu.get_device_name(0) or not enable_custom_op():
        raise RuntimeError("requires Ascend 310P and Qwen custom operators")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    generator = torch.Generator().manual_seed(20261003)
    patterns = ("single", "four", "tp4_sparse", "distinct")

    with args.output.open("x") as output_file:
        for projection, (width, outputs) in PROJECTIONS.items():
            bank = make_bank(width, outputs, generator)
            for tokens in args.tokens:
                x = (torch.randn(tokens, width, generator=generator) * 0.1).half().npu()
                original_x = x.clone()
                packed = pack_activation_device(x)
                pack_graph, pack_output = capture(lambda x=x: pack_activation_device(x), args.graph_unroll)
                pack_ms = timed(pack_graph.replay, args.iterations, args.repeats, args.graph_unroll)
                del pack_graph, pack_output
                rows = tokens * ROUTES_PER_TOKEN
                if projection == "down":
                    gate_up = (torch.randn(rows, 2 * width, generator=generator) * 0.1).half().npu()

                    def swiglu_reference(gate_up=gate_up) -> tuple[torch.Tensor, ...]:
                        gate, up = gate_up.float().chunk(2, -1)
                        return pack_activation_device((F.silu(gate) * up).half())

                    fused_pack = swiglu_pack_activation_device(gate_up)
                    expected_pack = swiglu_reference()
                    for actual, expected in zip(fused_pack, expected_pack):
                        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                    swiglu_graph, _ = capture(
                        lambda gate_up=gate_up: swiglu_pack_activation_device(gate_up), args.graph_unroll
                    )
                    swiglu_ms = timed(swiglu_graph.replay, args.iterations, args.repeats, args.graph_unroll)
                    del swiglu_graph
                else:
                    swiglu_ms = None
                for pattern in patterns:
                    ids_cpu = route_ids(rows, pattern)
                    ids = ids_cpu.npu()
                    prepared = lambda packed=packed, bank=bank, ids=ids: op(*packed, *bank, ids)
                    complete = lambda x=x, bank=bank, ids=ids: op(*pack_activation_device(x), *bank, ids)
                    eager = prepared().cpu()
                    torch.testing.assert_close(complete().cpu(), eager, rtol=0, atol=0)
                    prepared_graph, prepared_output = capture(prepared, args.graph_unroll)
                    complete_graph, complete_output = capture(complete, args.graph_unroll)
                    torch.testing.assert_close(prepared_output.cpu(), eager, rtol=0, atol=0)
                    torch.testing.assert_close(complete_output.cpu(), eager, rtol=0, atol=0)
                    prepared_ms = timed(prepared_graph.replay, args.iterations, args.repeats, args.graph_unroll)
                    complete_ms = timed(complete_graph.replay, args.iterations, args.repeats, args.graph_unroll)
                    # The full graph must consume changing input and route IDs.
                    changed = x * 0.5
                    x.copy_(changed)
                    ids.fill_(-1)
                    complete_graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(complete_output.cpu(), torch.zeros_like(eager), rtol=0, atol=0)
                    x.copy_(original_x)
                    ids.copy_(ids_cpu)
                    del prepared_graph, complete_graph, prepared_output, complete_output
                    record = {
                        "projection": projection,
                        "tokens": tokens,
                        "routes": rows,
                        "pattern": pattern,
                        "local_routes": int(((ids_cpu >= 0) & (ids_cpu < EXPERTS)).sum()),
                        "distinct_local_experts": int(ids_cpu[ids_cpu >= 0].unique().numel()),
                        "pack_only": pack_ms,
                        "swiglu_pack_only": swiglu_ms,
                        "prepared_projection": prepared_ms,
                        "pack_and_projection": complete_ms,
                        "exact_prepared_full_graph_parity": True,
                        "changing_input_and_peer_zero_replay": True,
                        "scope": "synthetic_32_experts_no_router_collective_or_full_model",
                    }
                    output_file.write(json.dumps(record) + "\n")
                    output_file.flush()
                    print(projection, tokens, pattern, flush=True)
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
