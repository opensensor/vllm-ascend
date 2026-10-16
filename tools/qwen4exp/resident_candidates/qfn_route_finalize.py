# SPDX-License-Identifier: Apache-2.0
"""Exact-source-guarded grouped W4 epilogue candidate; opt-in native resource."""

import hashlib
import inspect
import textwrap
from types import FunctionType

import torch

from tools.qwen4exp import route_finalize
from tools.qwen4exp.native_descriptor_admission import descriptors_available

RESOURCE_NAME = "qwen_route_finalize_v1"
TARGET = "vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE._forward_grouped"
QUALIFIED_SOURCE_SHA256 = "21968c3d2ee79e438fdddffed11e08b4b207cd8644c27621acc02ba882371ceb"
ORIGINAL = (
    '        output = self.projections["down_proj"].grouped_linear(activation, group_ends).to(self.compute_dtype)\n'
    """        output *= dispatch.route_weights.index_select(0, dispatch.order)
        result[start:stop] = (
            output.index_select(0, dispatch.inverse_order).reshape(tokens, self.top_k, hidden).sum(1)
        )"""
)
REPLACEMENT = """        output = self.projections["down_proj"].grouped_linear(activation, group_ends)
        ordered = _finalize(output, dispatch.inverse_order, dispatch.route_weights)
        result[start:stop] = ordered.reshape(tokens, self.top_k, hidden).sum(1)"""


def build_forward(original, resource, *, source=None, eager_only=False, activation=None):
    original = getattr(original, "_qfn_route_finalize_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256 or source.count(ORIGINAL) != 1:
        raise RuntimeError("route finalization requires exact qualified grouped W4 source")
    body = source.replace(ORIGINAL, REPLACEMENT)
    if activation is not None:
        seam = (
            "        gate, up = projected.to(self.compute_dtype).chunk(2, -1)\n"
            "        activation = (F.silu(gate) * up).to(self.params_dtype)"
        )
        if body.count(seam) != 1:
            raise RuntimeError("activation fusion requires the qualified grouped W4 activation seam")
        body = body.replace(seam, "        activation = _activation(projected, self.compute_dtype, self.params_dtype)")
    factory = "def factory(_finalize, _activation):\n" + textwrap.indent(body, "    ") + "    return _forward_grouped\n"
    namespace = dict(original.__globals__)
    exec(compile(factory, "<qfn-qualified-route-finalize>", "exec"), namespace)
    compiled = namespace["factory"](resource, activation)
    implementation = FunctionType(
        compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )

    def forward(self, block_input, weights, ids):
        if eager_only and route_finalize._is_capturing():
            return original(self, block_input, weights, ids)
        if (
            not self.native_int4
            or self.compute_dtype != torch.float32
            or self.params_dtype != torch.float16
            or self.top_k != 10
            or block_input.ndim != 2
            or block_input.shape[1] != 2560
            or not 0 < self.grouped_chunk_tokens <= 2560
        ):
            return original(self, block_input, weights, ids)
        full_chunks, tail = divmod(block_input.shape[0], self.grouped_chunk_tokens)
        rows = []
        if full_chunks:
            rows.append(self.grouped_chunk_tokens * self.top_k)
        if tail:
            rows.append(tail * self.top_k)
        if not descriptors_available(
            resource,
            ((count, block_input.device) for count in rows),
            capacity=route_finalize.MAX_CONFIGS,
            is_capturing=route_finalize._is_capturing,
        ):
            return original(self, block_input, weights, ids)
        return implementation(self, block_input, weights, ids)

    forward._qfn_route_finalize_base = original
    return forward


def replacements(native_resources):
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

    # The resident native loader must have completed its private-tensor gate.
    resources = native_resources[RESOURCE_NAME]
    return {
        TARGET: build_forward(
            W4SparseMoE._forward_grouped, resources["finalize"], eager_only=resources.get("eager_only", False)
        )
    }
