# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the fixed-shape portions of the W8A16 MTP MoE.

This opt-in experiment keeps only dynamic expert dispatch eager. Routing,
route-slot reduction, the shared expert, and TP reduction can be captured.
The dispatch callback reads the small top-k ID array once, instead of device
nonzero followed by a counts readback. Activations and route weights remain
on device, and the original weight-only expert matmuls are unchanged.

This does not make dynamic W8A16 expert dispatch fully graphable. It must pass
changing-route NPU replay and custom-collective lifetime gates before use.
"""

from collections.abc import Callable, Sequence

import torch

MAX_TOKENS = 12


def plan_local_routes(
    ids: Sequence[int], expert_offset: int, num_local_experts: int
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Return stable expert-ordered slots and their contiguous group ranges."""
    selected = sorted(
        (expert - expert_offset, slot)
        for slot, expert in enumerate(ids)
        if expert_offset <= expert < expert_offset + num_local_experts
    )
    slots = [slot for _, slot in selected]
    groups = []
    start = 0
    while start < len(selected):
        expert = selected[start][0]
        stop = start + 1
        while stop < len(selected) and selected[stop][0] == expert:
            stop += 1
        groups.append((expert, start, stop))
        start = stop
    return slots, groups


def fill_routed_output(self, x, weights, ids, route_outputs, *, swiglu: Callable) -> None:
    """Overwrite all slots, including ranks with no local routes on this step."""
    route_outputs.zero_()
    # One deliberate synchronization boundary. Data-dependent expert dispatch
    # still needs host IDs, but not an intermediate nonzero/counts pipeline.
    slots, groups = plan_local_routes(
        ids.flatten().to(device="cpu").tolist(), self.expert_offset, self.num_local_experts
    )
    if not groups:
        return
    indices = torch.tensor([slots, [slot // self.top_k for slot in slots]], dtype=torch.int64)
    if x.device.type == "npu":
        indices = indices.pin_memory()
    indices = indices.to(device=x.device, non_blocking=True)
    slot_ids, token_ids = indices.unbind(0)
    local_weights = weights.flatten().index_select(0, slot_ids)
    for expert, start, stop in groups:
        selected = x[token_ids[start:stop]]
        gate_up = self._expert_linear(selected, "gate_up_proj", expert)
        expert_output = self._expert_linear(swiglu(gate_up), "down_proj", expert)
        route_outputs.index_copy_(
            0, slot_ids[start:stop], expert_output.to(route_outputs.dtype) * local_weights[start:stop, None]
        )


def _supports_device(device) -> bool:
    return device.type == "npu"


def make_forward(*, linear, route_topk, swiglu, current_capture, weak_ref_tensor, fallback):
    """Build an instance-free resident replacement with the old forward fallback."""

    def forward(self, x):
        if (
            torch.is_grad_enabled()
            or not _supports_device(x.device)
            or x.dtype != torch.float16
            or not self.quantized_experts
            or self.grouped_experts
            or not 0 < x.shape[0] <= MAX_TOKENS
        ):
            return fallback(self, x)
        logits = linear(x, self.gate, self.policy.router_dtype)
        weights, ids = route_topk(
            logits,
            self.top_k,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        route_outputs = torch.empty((ids.numel(), x.shape[-1]), dtype=self.policy.accumulation_dtype, device=x.device)
        capture = current_capture()
        if capture is None or not capture._capturing:
            fill_routed_output(self, x, weights, ids, route_outputs, swiglu=swiglu)
        else:
            # Only weak graph tensor references survive in the eager callback;
            # the graph pool owns their storage until reset.
            weak_x = weak_ref_tensor(x)
            weak_weights = weak_ref_tensor(weights)
            weak_ids = weak_ref_tensor(ids)
            weak_output = weak_ref_tensor(route_outputs)

            def dispatch_routed_experts() -> None:
                fill_routed_output(self, weak_x, weak_weights, weak_ids, weak_output, swiglu=swiglu)

            capture.add_eager(dispatch_routed_experts)
        output = route_outputs.view(x.shape[0], self.top_k, x.shape[-1]).sum(dim=1)
        return self._finish_output(x, output)

    return forward


def replacements():
    # Worker-only imports avoid model registration/device initialization for
    # host route planning and standalone CPU tests.
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    from vllm_ascend.models.qwen4_exp.mtp import _linear, _MTPFP16MoE, route_topk, swiglu_gate_up
    from vllm_ascend.utils import weak_ref_tensor

    return {
        "vllm_ascend.models.qwen4_exp.mtp:_MTPFP16MoE.forward": make_forward(
            linear=_linear,
            route_topk=route_topk,
            swiglu=swiglu_gate_up,
            current_capture=BreakableCUDAGraphCapture.current,
            weak_ref_tensor=weak_ref_tensor,
            fallback=_MTPFP16MoE.forward,
        )
    }
