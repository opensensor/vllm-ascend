# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold resident numerical gate for actual QFN draft W8A16 weight banks.

The controller must drain inference before this collective RPC. Each worker
uses its loaded draft weights through a shallow module view; no checkpoint or
second model is loaded. The view skips TP reduction, so comparisons are exact
local pre-TP outputs. Native collective correctness remains a separate gate.
"""

from copy import copy

import torch

from tools.qwen4exp.benchmark_draft_graph_experts_310 import SplitCapture
from tools.qwen4exp.resident_candidates import draft_graph_experts as candidate

QUALIFIED_HIDDEN = 2560
QUALIFIED_INTERMEDIATE = 640
QUALIFIED_TOP_K = 10
QUALIFIED_EXPERTS = 512
QUALIFIED_LOCAL_EXPERTS = 128


def find_draft_banks(worker):
    drafter = getattr(worker.model_runner, "drafter", None)
    if drafter is None:
        raise RuntimeError("the worker has no loaded draft proposer")
    pending = [("drafter", drafter)]
    seen = set()
    banks = []
    while pending:
        name, root = pending.pop()
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        if type(root).__name__ == "_MTPFP16MoE":
            banks.append((name, root))
            continue
        if isinstance(root, torch.nn.Module):
            pending.extend((f"{name}.{child}", module) for child, module in root.named_children())
        for attribute in ("model", "runnable", "module"):
            child = getattr(root, attribute, None)
            if child is not None:
                pending.append((f"{name}.{attribute}", child))
    if not banks:
        raise RuntimeError("no _MTPFP16MoE found under the loaded drafter")
    return sorted(banks, key=lambda item: item[0])


def local_view(bank):
    """Share immutable loaded weights without mutating the serving module."""
    view = copy(bank)
    view.expert_tp_size = 1
    view._tp_reduce = None
    return view


def validate_bank(bank):
    actual = (bank.hidden_size, bank.intermediate_size, bank.top_k, bank.num_experts, bank.num_local_experts)
    expected = (QUALIFIED_HIDDEN, QUALIFIED_INTERMEDIATE, QUALIFIED_TOP_K, QUALIFIED_EXPERTS, QUALIFIED_LOCAL_EXPERTS)
    if actual != expected or not bank.quantized_experts or bank.grouped_experts:
        raise ValueError(f"requires qualified routed W8A16 geometry {expected}, found {actual}")
    if bank.gate.device.type != "npu" or bank.gate.dtype != torch.float16:
        raise ValueError("requires a resident FP16 Ascend router")
    if not 0 <= bank.expert_offset <= bank.num_experts - bank.num_local_experts:
        raise ValueError("invalid local expert range")
    for projection, shape in (
        ("gate_up_proj", (bank.hidden_size, 2 * bank.intermediate_size)),
        ("down_proj", (bank.intermediate_size, bank.hidden_size)),
    ):
        weights = getattr(bank, projection)
        scales = getattr(bank, projection + "_scale")
        if len(weights) != bank.num_local_experts or len(scales) != bank.num_local_experts:
            raise ValueError("expert bank length does not match rank ownership")
        for weight, scale in zip(weights, scales, strict=True):
            if tuple(weight.shape) != shape or weight.dtype != torch.int8:
                raise ValueError(f"unexpected {projection} weight geometry or dtype")
            if scale.shape != (shape[1],) or scale.dtype != torch.float16 or scale.device != weight.device:
                raise ValueError(f"unexpected {projection} per-channel scale")


def forced_route_ids(offset, local_experts, total_experts, top_k, tokens):
    """Valid top-k sets exercise both local boundaries and empty/full ranks."""
    if not 0 < top_k <= local_experts or top_k > total_experts - local_experts:
        raise ValueError("need at least top-k local and remote experts")
    low = list(range(offset, offset + top_k))
    high = list(range(offset + local_experts - top_k, offset + local_experts))
    remote_left = [(offset - top_k + i) % total_experts for i in range(top_k)]
    remote_right = [(offset + local_experts + i) % total_experts for i in range(top_k)]
    boundary = [offset - 1, offset, offset + local_experts - 1, offset + local_experts]
    boundary.extend(range(offset + 1, offset + 1 + top_k - len(boundary)))
    cases = {
        "local_low": low,
        "local_high": high,
        "remote_left": remote_left,
        "remote_right": remote_right,
        "mixed_boundaries": [expert % total_experts for expert in boundary],
    }
    result = {}
    for name, row in cases.items():
        # Identical experts recur across tokens, but each token has unique IDs.
        result[name] = torch.tensor([row[index % top_k :] + row[: index % top_k] for index in range(tokens)])
    return result


def reference_routed_output(bank, x, weights, ids, *, select_local_routes, swiglu):
    """Original device-selection/count-readback algorithm, excluding router."""
    local_ids, slot_ids, token_ids, local_weights = select_local_routes(
        ids.flatten(), weights.flatten(), bank.expert_offset, bank.num_local_experts, bank.top_k
    )
    route_outputs = torch.zeros((ids.numel(), x.shape[-1]), dtype=bank.policy.accumulation_dtype, device=x.device)
    if local_ids.numel():
        order = torch.argsort(local_ids.to(bank.policy.accumulation_dtype), stable=True)
        local_ids, slot_ids, token_ids, local_weights = (
            local_ids[order],
            slot_ids[order],
            token_ids[order],
            local_weights[order],
        )
        counts = torch.bincount(local_ids, minlength=bank.num_local_experts)
        start = 0
        for expert, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            stop = start + count
            gate_up = bank._expert_linear(x[token_ids[start:stop]], "gate_up_proj", expert)
            output = bank._expert_linear(swiglu(gate_up), "down_proj", expert)
            route_outputs.index_copy_(
                0, slot_ids[start:stop], output.to(route_outputs.dtype) * local_weights[start:stop, None]
            )
            start = stop
    return bank._finish_output(x, route_outputs.view(x.shape[0], bank.top_k, x.shape[-1]).sum(1))


def parameter_versions(bank):
    versions = []
    for name, tensor in bank.named_parameters():
        try:
            version = tensor._version
        except RuntimeError:
            # Inference-created parameters do not expose mutation counters.
            version = None
        versions.append((name, tensor.data_ptr(), version))
    return tuple(versions)


def fixed_routing(weights, ids):
    return lambda *args, **kwargs: (weights, ids)


def unexpected_fallback(*args):
    raise AssertionError("unexpected candidate fallback")


@torch.inference_mode()
def gate_bank(bank, *, tokens, repeats, seed):
    from vllm_ascend.models.qwen4_exp.mtp import _linear, _select_local_mtp_routes, route_topk, swiglu_gate_up
    from vllm_ascend.utils import weak_ref_tensor

    validate_bank(bank)
    versions = parameter_versions(bank)
    view = local_view(bank)
    generator = torch.Generator().manual_seed(seed)
    records = []
    active_capture = [None]
    for count in tokens:
        inputs = torch.zeros((count, bank.hidden_size), dtype=bank.policy.main_dtype, device=bank.gate.device)
        route_ids = torch.zeros((count, bank.top_k), dtype=torch.int64, device=inputs.device)
        route_weights = torch.zeros((count, bank.top_k), dtype=torch.float32, device=inputs.device)
        cases = forced_route_ids(bank.expert_offset, bank.num_local_experts, bank.num_experts, bank.top_k, count)
        for mode in ("forced", "natural"):
            routing = fixed_routing(route_weights, route_ids) if mode == "forced" else route_topk
            forward = candidate.make_forward(
                linear=_linear,
                route_topk=routing,
                swiglu=swiglu_gate_up,
                current_capture=lambda: active_capture[0],
                weak_ref_tensor=weak_ref_tensor,
                fallback=unexpected_fallback,
            )
            route_ids.copy_(cases["local_low"])
            route_weights.fill_(1 / bank.top_k)
            inputs.copy_((torch.randn(inputs.shape, generator=generator) * 0.1).half())
            for _ in range(3):
                forward(view, inputs)
            torch.npu.synchronize()
            capture = SplitCapture()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                active_capture[0] = capture
                capture.begin()
                try:
                    output = forward(view, inputs)
                finally:
                    capture.end()
                    active_capture[0] = None
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            output_address = output.data_ptr()
            for repeat in range(repeats):
                for label, ids in cases.items():
                    current = (torch.randn(inputs.shape, generator=generator) * (0.1 + repeat * 0.03)).half()
                    inputs.copy_(current)
                    route_ids.copy_(ids)
                    weights = torch.rand(route_weights.shape, generator=generator)
                    weights = weights / weights.sum(1, keepdim=True) * bank.routed_scaling_factor
                    route_weights.copy_(weights)
                    if mode == "forced":
                        expected = reference_routed_output(
                            view,
                            inputs,
                            route_weights,
                            route_ids,
                            select_local_routes=_select_local_mtp_routes,
                            swiglu=swiglu_gate_up,
                        )
                    else:
                        expected = view._forward_eager(inputs)
                    capture.replay()
                    actual_cpu, expected_cpu = output.cpu(), expected.cpu()
                    torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0, equal_nan=False)
                    if output.data_ptr() != output_address:
                        raise RuntimeError("replay replaced the graph output allocation")
                    records.append(dict(tokens=count, mode=mode, case=label, repeat=repeat, exact=True))
            torch.npu.synchronize()
            capture.reset()
    if parameter_versions(bank) != versions:
        raise RuntimeError("resident parameter storage or version changed during numerical gate")
    return dict(
        source="already_loaded_draft_bank",
        scope="local_pre_tp_output_real_weights_synthetic_activations",
        expert_offset=bank.expert_offset,
        local_experts=bank.num_local_experts,
        hidden=bank.hidden_size,
        intermediate=bank.intermediate_size,
        top_k=bank.top_k,
        parameter_versions_unchanged=True,
        parameters_with_version_counters=sum(version is not None for _, _, version in versions),
        comparisons=len(records),
        records=records,
    )


class QfnDraftRealWeightMixin:
    def qfn_draft_real_weight_gate(self, tokens=(1, 3, 6), repeats=2, seed=9401):
        """Controller-only cold RPC; inference must be drained on every rank."""
        if not tokens or any(type(count) is not int or not 0 < count <= candidate.MAX_TOKENS for count in tokens):
            raise ValueError("tokens must be a nonempty list of sizes in 1..12")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be positive")
        banks = find_draft_banks(self)
        return dict(
            rank=self.rank,
            passed=True,
            banks=[
                dict(name=name, **gate_bank(bank, tokens=tokens, repeats=repeats, seed=seed)) for name, bank in banks
            ],
        )
