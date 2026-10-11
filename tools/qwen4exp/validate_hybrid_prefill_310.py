# SPDX-License-Identifier: Apache-2.0
"""Real-weight boundary and changing-input graph gates for isolated experiments."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.qwen4exp.benchmark_channel_grouped_310 import channel_bank
from tools.qwen4exp.benchmark_grouped_prefill_310 import load_grouped_resources
from tools.qwen4exp.benchmark_int8_projection_probe_310 import server_activity
from tools.qwen4exp.native_channel_int8 import (
    NativeChannelActivation,
    NativeChannelInt8,
    NativeDirectChannelActivation,
    NativePackedChannelInt8,
)
from tools.qwen4exp.native_hybrid_prefill import signed_nibble_lookup

ROWS = 640


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-activation", action="store_true")
    parser.add_argument("--packed-weight", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.rank < 6:
        parser.error("new output and a rank in [0,5] required")
    before = server_activity("http://127.0.0.1:8001")
    if before[0]:
        raise RuntimeError("live traffic present")
    import torch_npu

    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND, pack_activation_device
    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(6)
    torch.npu.set_device(args.rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    layer = load_layer(args.model, args.layer, args.rank, 6, NATIVE_INT4_BACKEND)
    resources = load_grouped_resources(args.build, layer.gate.device, cache_metadata=True, paired_limbs=True)
    modes = ("hybrid", "hybrid_a8", "cached", "paired")
    manifest = json.loads((args.build / "provenance.json").read_text())
    kernel = getattr(torch.classes, manifest["namespace"]).Kernel
    launch = getattr(torch.ops, manifest["namespace"]).launch
    channel_activation = NativeChannelActivation(
        kernel(manifest["binaries"]["native_hybrid_prefill"]["path"], "qwen_channel_activation_v1"),
        launch,
        signed_nibble_lookup().to(layer.gate.device),
    )
    direct_activation = NativeDirectChannelActivation(
        kernel(manifest["binaries"]["native_hybrid_prefill"]["path"], "qwen_direct_channel_activation_v1"),
        launch,
        local_kernel=kernel(
            manifest["binaries"]["native_hybrid_prefill"]["path"], "qwen_direct_local_channel_activation_v1"
        )
        if args.local_activation
        else None,
    )
    packed_projection = (
        NativePackedChannelInt8(
            kernel(manifest["binaries"]["native_hybrid_prefill"]["path"], "qwen_packed_channel_int8_projection_v1"),
            launch,
            signed_nibble_lookup().to(layer.gate.device),
        )
        if args.packed_weight
        else None
    )
    generator = torch.Generator().manual_seed(987)
    cases = []
    for name, bank in layer.projections.items():
        inputs = bank.weight.shape[2] * 2
        x = torch.randn(ROWS, inputs, generator=generator).half().to(bank.weight.device)
        x[0].zero_()
        x[1].zero_()
        x[1, 0] = torch.finfo(torch.float16).smallest_normal
        direct_q, direct_s = direct_activation(x)
        values_direct = x.cpu().float()
        maximum_direct = values_direct.abs().amax(-1, keepdim=True)
        expected_direct_s = torch.where(maximum_direct == 0, torch.ones_like(maximum_direct), maximum_direct / 127)
        expected_direct_q = (values_direct / expected_direct_s).round().clamp(-127, 127).to(torch.int8)
        torch.testing.assert_close(direct_q.cpu(), expected_direct_q, rtol=0, atol=0)
        torch.testing.assert_close(direct_s.cpu(), expected_direct_s.expand(-1, 8), rtol=1e-7, atol=1e-10)
        cases.append({"projection": name, "direct_activation_exact": True})
        prepared = pack_activation_device(x)
        # Independent CPU recipe includes the qualified FP16 rounding boundary
        # between G128 dequantization and per-row dynamic quantization.
        actual_q, actual_s = channel_activation(prepared)
        low, high, group_scales, _ = [value.cpu() for value in prepared]

        def nibbles(packed):
            unsigned = packed.to(torch.int32) & 255
            lanes = torch.stack((unsigned & 15, unsigned >> 4), dim=-1).flatten(-2)
            return torch.where(lanes >= 8, lanes - 16, lanes)

        values = (nibbles(low) + 16 * nibbles(high) + 8).float().reshape(ROWS, -1, 128)
        values = (values * group_scales[:, :, :1]).half().float().reshape(ROWS, inputs)
        maximum = values.abs().amax(-1, keepdim=True)
        expected_s = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
        expected_q = (values / expected_s).round().clamp(-127, 127).to(torch.int8)
        torch.testing.assert_close(actual_q.cpu(), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_s.cpu(), expected_s.expand(-1, 8), rtol=1e-7, atol=1e-10)
        cases.append({"projection": name, "channel_activation_exact": True})
        subset = SimpleNamespace(
            weight=bank.weight[:2], weight_scale=bank.weight_scale[:2], weight_offset=bank.weight_offset[:2]
        )
        w8, w8_scales, _ = channel_bank(subset)
        raw_w8 = w8.cpu().reshape(2, bank.weight.shape[1] // 16, inputs // 128, 4, 16, 32)
        raw_w8 = raw_w8.permute(0, 1, 4, 2, 3, 5).reshape(2, bank.weight.shape[1], inputs).int()
        channel_ends = torch.tensor([127, 256], dtype=torch.int64, device=x.device)
        if args.local_activation:
            direct_activation(x, channel_ends)
        for symbol in (
            "qwen_channel_int8_projection_v1",
            "qwen_channel_int8_projection_m64_v1",
            "qwen_channel_int8_projection_m64_n64_v1",
        ):
            channel_projection = NativeChannelInt8(
                kernel(manifest["binaries"]["native_hybrid_prefill"]["path"], symbol),
                launch,
                column_tile=64 if "n64" in symbol else 160,
            )
            if symbol == "qwen_channel_int8_projection_v1":
                debug_integer = torch.zeros(ROWS, bank.weight.shape[1], dtype=torch.int32, device=x.device)
                debug_kernel = kernel(
                    manifest["binaries"]["native_hybrid_prefill"]["path"], "qwen_channel_int8_integer_debug_v1"
                )
                config = channel_projection.config((ROWS, inputs, bank.weight.shape[1], 2, 1), x.device)
                launch(debug_kernel, [actual_q, w8, w8_scales, actual_s, channel_ends, debug_integer, config], 8)
                actual_dot = debug_integer.cpu()
                expected_dot = torch.zeros_like(actual_dot)
                for expert, (begin, end) in enumerate(((0, 127), (127, 256))):
                    expected_dot[begin:end] = actual_q[begin:end].cpu().int() @ raw_w8[expert].t()
                print("checking integer", name, flush=True)
                torch.testing.assert_close(actual_dot, expected_dot, rtol=0, atol=0)
            eager_output = channel_projection(actual_q, w8, w8_scales, actual_s, channel_ends).cpu()
            eager_expected = torch.zeros_like(eager_output)
            eager_q, eager_s = actual_q.cpu().int(), actual_s.cpu()
            for expert, (begin, end) in enumerate(((0, 127), (127, 256))):
                dot = eager_q[begin:end] @ raw_w8[expert].t()
                eager_expected[begin:end] = (
                    dot.float() * w8_scales[expert].cpu().unsqueeze(0) * eager_s[begin:end, :1]
                ).half()
            print("checking eager", name, symbol, flush=True)
            torch.testing.assert_close(eager_output, eager_expected, rtol=0, atol=0)
            if packed_projection is not None:
                packed_output = packed_projection(actual_q, subset, w8_scales, actual_s, channel_ends)
                torch.testing.assert_close(packed_output.cpu(), eager_expected, rtol=0, atol=0)
            graph = torch.npu.NPUGraph()
            torch.npu.synchronize()
            with torch.npu.graph(graph):
                replay_q, replay_s = channel_activation(prepared)
                channel_output = channel_projection(replay_q, w8, w8_scales, replay_s, channel_ends)
                graph_direct_q, graph_direct_s = direct_activation(x)
                direct_output = channel_projection(graph_direct_q, w8, w8_scales, graph_direct_s, channel_ends)
                if packed_projection is not None:
                    packed_output = packed_projection(graph_direct_q, subset, w8_scales, graph_direct_s, channel_ends)
                if args.local_activation:
                    local_q, local_s = direct_activation(x, channel_ends)
                    local_output = channel_projection(local_q, w8, w8_scales, local_s, channel_ends)
            changed_x = torch.randn(ROWS, inputs, generator=generator).half().to(x.device)
            changed_x[0].zero_()
            x.copy_(changed_x)
            changed = pack_activation_device(torch.randn(ROWS, inputs, generator=generator).half().to(x.device))
            for target, source in zip(prepared, changed):
                target.copy_(source)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            expected = torch.zeros(ROWS, bank.weight.shape[1], dtype=torch.float16)
            qa, sx = replay_q.cpu().int(), replay_s.cpu()
            for expert, (begin, end) in enumerate(((0, 127), (127, 256))):
                dot = qa[begin:end] @ raw_w8[expert].t()
                expected[begin:end] = (dot.float() * w8_scales[expert].cpu().unsqueeze(0) * sx[begin:end, :1]).half()
            print("checking", name, symbol, flush=True)
            torch.testing.assert_close(channel_output.cpu(), expected, rtol=0, atol=0)
            assert torch.equal(channel_output.cpu().view(torch.uint8), expected.view(torch.uint8))
            direct_expected = torch.zeros_like(expected)
            graph_values = x.cpu().float()
            graph_maximum = graph_values.abs().amax(-1, keepdim=True)
            graph_expected_s = torch.where(graph_maximum == 0, torch.ones_like(graph_maximum), graph_maximum / 127)
            graph_expected_q = (graph_values / graph_expected_s).round().clamp(-127, 127).to(torch.int8)
            torch.testing.assert_close(graph_direct_q.cpu(), graph_expected_q, rtol=0, atol=0)
            torch.testing.assert_close(graph_direct_s.cpu(), graph_expected_s.expand(-1, 8), rtol=1e-7, atol=1e-10)
            for expert, (begin, end) in enumerate(((0, 127), (127, 256))):
                dot = graph_expected_q[begin:end].int() @ raw_w8[expert].t()
                direct_expected[begin:end] = (
                    dot.float() * w8_scales[expert].cpu().unsqueeze(0) * graph_direct_s[begin:end, :1].cpu()
                ).half()
            torch.testing.assert_close(direct_output.cpu(), direct_expected, rtol=0, atol=0)
            if packed_projection is not None:
                torch.testing.assert_close(packed_output.cpu(), direct_expected, rtol=0, atol=0)
            if args.local_activation:
                torch.testing.assert_close(local_q[:256].cpu(), graph_expected_q[:256], rtol=0, atol=0)
                torch.testing.assert_close(
                    local_s[:256].cpu(), graph_expected_s[:256].expand(-1, 8), rtol=1e-7, atol=1e-10
                )
                torch.testing.assert_close(local_output.cpu(), direct_expected, rtol=0, atol=0)
            del graph
            cases.append(
                {
                    "projection": name,
                    "symbol": symbol,
                    "channel_integer_recipe_graph_exact": True,
                    "direct_integer_recipe_graph_exact": True,
                    "packed_weight_integer_recipe_graph_exact": packed_projection is not None,
                }
            )
        del w8, w8_scales, raw_w8
        patterns = ([0, 127, 128, 129, 1, 126], [128, 0, 129, 0, 127, 256], [0] * layer.num_local_experts)
        for counts in patterns:
            ends = (
                torch.tensor(counts + [0] * (layer.num_local_experts - len(counts)), dtype=torch.int64)
                .cumsum(0)
                .to(x.device)
            )
            expected = bank.native_linear(prepared, ends).cpu()
            for mode in modes:
                actual = resources[mode](bank, prepared, ends).cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            cases.append({"projection": name, "counts": counts, "modes": list(modes), "exact_bytes": True})
        # Reuse captured addresses with new activations. Configs/resources are
        # prewarmed, and the group partitions straddle 127/128/129 exactly.
        counts = [0, 127, 128, 129, 1, 126]
        ends = (
            torch.tensor(counts + [0] * (layer.num_local_experts - len(counts)), dtype=torch.int64)
            .cumsum(0)
            .to(x.device)
        )
        for mode in modes:
            resources[mode](bank, prepared, ends)
            graph = torch.npu.NPUGraph()
            torch.npu.synchronize()
            with torch.npu.graph(graph):
                actual = resources[mode](bank, prepared, ends)
            changed = pack_activation_device(torch.randn(ROWS, inputs, generator=generator).half().to(x.device))
            for target, source in zip(prepared, changed):
                target.copy_(source)
            expected = bank.native_linear(prepared, ends).cpu()
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            assert torch.equal(actual.cpu().view(torch.uint8), expected.view(torch.uint8))
            del graph
            cases.append({"projection": name, "mode": mode, "changing_input_graph_exact": True})
    after = server_activity("http://127.0.0.1:8001")
    if after[0] or after[1] != before[1]:
        raise RuntimeError("traffic overlapped validation")
    record = {
        "rank": args.rank,
        "layer": args.layer,
        "build": str(args.build),
        "real_weights": True,
        "server_modified": False,
        "packed_weight_enabled": args.packed_weight,
        "local_activation_enabled": args.local_activation,
        "cases": cases,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
