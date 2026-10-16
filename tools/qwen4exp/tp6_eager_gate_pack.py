# SPDX-License-Identifier: Apache-2.0
"""Source-qualified eager gate/packing binding; retain the loaded finalizer."""

import hashlib
import inspect
import textwrap
from types import FunctionType

import torch

QUALIFIED_CHUNK = "a27b331cbf44aeed8d02a3387f0d1c17a4bdea0a57757454b070653dcd7954d5"
MIN_TOKENS = 13
MAX_TOKENS = 256
MAX_CONFIGS = 256


def admitted(module, inputs, weights, ids, resource):
    if resource.failed:
        raise RuntimeError("failed native resource cannot serve")
    if torch.is_grad_enabled() or inputs.device.type != "npu" or torch.npu.is_current_stream_capturing():
        return False
    if (
        inputs.ndim != 2
        or inputs.shape[1] != 2560
        or not MIN_TOKENS <= inputs.shape[0] <= MAX_TOKENS
        or inputs.dtype != torch.float16
        or module.params_dtype != torch.float16
        or module.compute_dtype != torch.float32
        or not module.native_int4
        or module.top_k != 10
        or module.num_local_experts not in (85, 86)
        or inputs.shape[0] * 10 <= module.max_routed_rows
        or module.grouped_activation != "cann_builtin_fp16"
        or module.grouped_finalize != "cann_v2"
        or weights.shape != ids.shape
        or weights.shape != (inputs.shape[0], 10)
        or weights.dtype != torch.float32
        or ids.dtype != torch.int64
        or any(t.device != inputs.device or not t.is_contiguous() for t in (inputs, weights, ids))
    ):
        return False
    key = ((inputs.shape[0] * 10, module.num_local_experts, 1280, 2560, 8, 0, 1), inputs.device)
    return key in resource._configs or len(resource._configs) < MAX_CONFIGS


def make_chunk(original, resource, state):
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_CHUNK:
        raise ValueError("unqualified grouped chunk; preserve current serving composition")
    projection = "            projected = gate_up_bank.native_linear(prepared, group_ends)\n"
    activation = '        if self.grouped_activation == "cann_swiglu_pack":\n'
    finalizer = '        if self.grouped_finalize == "torch":\n'
    if any(source.count(seam) != 1 for seam in (projection, activation, finalizer)):
        raise ValueError("ambiguous gate/pack seam")
    source = source.replace(
        projection,
        "            packed = _resource(gate_up_bank, prepared, group_ends)\n"
        '            output = self.projections["down_proj"].native_linear(packed, group_ends)\n',
    )
    source = source[: source.index(activation)] + source[source.index(finalizer) :]
    factory = (
        "def factory(_resource):\n"
        + textwrap.indent(textwrap.dedent(source), "    ")
        + "    return _forward_grouped_chunk\n"
    )
    namespace = dict(original.__globals__)
    exec(compile(factory, "<tp6-eager-gate-pack>", "exec"), namespace)
    compiled = namespace["factory"](resource)
    implementation = FunctionType(
        compiled.__code__, original.__globals__, compiled.__name__, closure=compiled.__closure__
    )

    def chunk(module, inputs, weights, ids):
        if resource.failed:
            raise RuntimeError("failed native resource cannot serve")
        if state["mode"] == "off" or not admitted(module, inputs, weights, ids, resource):
            return original(module, inputs, weights, ids)
        state["eligible_calls"] += 1
        if state["mode"] == "enabled":
            return implementation(module, inputs, weights, ids)
        if state["mode"] != "shadow":
            raise ValueError("unknown eager gate/pack mode")
        expected = original(module, inputs, weights, ids)
        actual = implementation(module, inputs, weights, ids)
        # Deliberately synchronous diagnostics; never part of enabled serving.
        equal = bool(torch.equal(actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)))
        state["checked_calls"] += 1
        state["checked_modules"].add(id(module))
        if not equal:
            state["mismatches"] += 1
            state["mode"] = "off"
        return expected

    return chunk
