# SPDX-License-Identifier: Apache-2.0
"""Explicit INT8 ownership trial; validate resident real weights before capture."""

import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

SMALL_DECODE_TOKENS = 3
OWNED_GATHER_MIN_TOKENS = 640


def make_replacements(build):
    """Use immutable compiled assets; keep larger groups on their old schedule."""
    from tools.qwen4exp.channel_int8_server import ServingConfiguration, ServingProjection, forward_grouped
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend.models.qwen4_exp import w4_moe

    build = Path(build)
    provenance = json.loads((build / "m16-provenance.json").read_text())
    for entry in provenance["assets"]:
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("owned INT8 compiled asset changed")
    scope = {"__name__": "qwen_owned_int8_resources"}
    source = build / "native_channel_int8.py"
    exec(compile(source.read_text(), str(source), "exec"), scope)

    class ServingGather(ServingConfiguration, scope["NativeLocalChannelGather"]):
        pass

    kernel, launch = torch.classes.qwen_prefill_v27.Kernel, torch.ops.qwen_prefill_v27.launch
    small = ServingProjection(
        kernel(str(build / "native-hybrid-m16.bin"), "qwen_channel_int8_projection_m16_v1"), launch
    )
    gather = ServingGather(kernel(str(build / "owned-int8-route-gather.bin"), "qwen_local_channel_gather_v1"), launch)

    def forward(self, inputs, weights, ids):
        import torch_npu

        projection, activation = self._channel_int8_resources
        if inputs.shape[0] <= SMALL_DECODE_TOKENS:
            projection = small
        dispatch = w4_moe.build_grouped_expert_dispatch(
            weights,
            ids,
            num_local_experts=self.num_local_experts,
            expert_offset=self.expert_offset,
            weight_dtype=self.compute_dtype,
            count_mode=self.grouped_route_count_mode,
        )
        order = dispatch.token_indices.index_select(0, dispatch.order.to(torch.int32)).to(torch.int32).contiguous()
        ends = dispatch.group_list.contiguous()
        token_codes, token_scales = activation(inputs)
        if inputs.shape[0] >= OWNED_GATHER_MIN_TOKENS:
            routed_codes, routed_scales = gather(token_codes, token_scales, order, ends)
        else:
            routed_codes = token_codes.index_select(0, order).contiguous()
            routed_scales = token_scales.index_select(0, order).contiguous()
        gate = self.projections["gate_up_proj"]
        projected = projection(routed_codes, gate.weight, gate.weight_scale, routed_scales, ends)
        hidden = torch_npu.npu_swiglu(projected, dim=-1)
        hidden_codes, hidden_scales = activation(hidden, ends)
        down = self.projections["down_proj"]
        output = projection(hidden_codes, down.weight, down.weight_scale, hidden_scales, ends)
        return w4_moe.finalize_grouped_routes(output, dispatch, weights, self.compute_dtype, self.grouped_finalize)

    original_capture = NPUModelRunner310.capture_model

    def capture(self, *args, **kwargs):
        # Diagnostics are outside serving execution. Never copy device counts
        # or run timing synchronizations in the forward path above.
        layer = next(module for module in self.model.runnable.modules() if isinstance(module, w4_moe.W4SparseMoE))
        if not hasattr(layer, "_channel_int8_resources"):
            raise ValueError("ownership candidate requires the experimental INT8 target")
        result = qualify_real_layer(layer, forward_grouped, forward)
        (build / f"real-layer-rank-{torch.distributed.get_rank()}.json").write_text(json.dumps(result, indent=2) + "\n")
        return original_capture(self, *args, **kwargs)

    return {
        "vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE._forward_routed": forward,
        "vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE._forward_grouped_chunk": forward,
        "vllm_ascend._310p.model_runner_310p:NPUModelRunner310.capture_model": capture,
    }


@torch.inference_mode()
def qualify_real_layer(layer, original, candidate):
    """Exact real-weight partial/replay gate, then alternating stage timings."""
    from vllm_ascend.models.qwen4_exp.moe import route_topk

    generator = torch.Generator().manual_seed(7401)
    result = {"passed": True, "scope": "one resident real-weight routed MoE partial", "cases": []}
    for tokens in (3, 6, 640):
        inputs = (torch.randn(tokens, layer.gate.shape[1], generator=generator) * 0.1).half().to(layer.gate.device)
        weights, ids = route_topk(
            torch.nn.functional.linear(inputs, layer.gate),
            layer.top_k,
            renormalize=layer.renormalize,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        expected = original(layer, inputs, weights, ids)
        actual = candidate(layer, inputs, weights, ids)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        timing = {"baseline": [], "candidate": []}
        graphs = {}
        if tokens <= SMALL_DECODE_TOKENS * 2:
            for name, operation in (("baseline", original), ("candidate", candidate)):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = operation(layer, inputs, weights, ids)
                graphs[name] = (graph, output)
            for round_id in range(3):
                changed = ids.roll(round_id + 1, dims=1).contiguous()
                ids.copy_(changed)
                for graph, _ in graphs.values():
                    graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(graphs["candidate"][1].cpu(), graphs["baseline"][1].cpu(), rtol=0, atol=0)
        for repeat in range(4):
            for name in ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline"):
                operation = original if name == "baseline" else candidate
                torch.npu.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    if graphs:
                        graphs[name][0].replay()
                    else:
                        operation(layer, inputs, weights, ids)
                torch.npu.synchronize()
                timing[name].append((time.perf_counter() - started) * 1000 / 10)
        medians = {name: statistics.median(samples) for name, samples in timing.items()}
        result["cases"].append({"tokens": tokens, "graph": bool(graphs), "median_ms": medians, "samples_ms": timing})
        if medians["candidate"] > medians["baseline"] * 1.05:
            raise RuntimeError(f"ownership candidate regressed resident partial at {tokens} tokens: {medians}")
        del graphs
    return result
