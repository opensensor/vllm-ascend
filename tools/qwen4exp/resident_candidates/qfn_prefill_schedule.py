# SPDX-License-Identifier: Apache-2.0
"""Qualified W4 grouped-prefill binding, prepared from worker-owned resources.

This module does not load a binary, discover workers, or install a patch. The
controller supplies the drained worker to make_replacements during preparation.
The resulting native_linear closure owns its selected already-qualified kernel.
"""

import hashlib
import inspect
import textwrap

import torch

from tools.qwen4exp import native_prefill
from tools.qwen4exp.native_descriptor_admission import descriptors_available

TARGET = "vllm_ascend.models.qwen4_exp.w4_moe:PackedExpertBank.native_linear"
QUALIFIED_SOURCE_SHA256 = "c4ee7f25ff35ff7d3f15ed2c3e1e45a180c4ef8767f629900ddf9da8f6fee706"
STATE_ATTRIBUTE = "_qfn_w4_prefill_probe"
STATE_ATTRIBUTE_V2 = "_qfn_w4_prefill_probe_v2"
STATE_ATTRIBUTE_V3 = "_qfn_w4_prefill_probe_v3"
MIN_GROUPED_ROWS = 128
MAX_QUALIFIED_ROWS = 20480
EXPERTS = 128
PROJECTION_SHAPES = ((1280, 2560), (2560, 640))
REQUIRED_PROJECTIONS = ("gate_up_proj", "down_proj")
REQUIRED_ROWS = (650, 20480)
REQUIRED_SCHEDULES = frozenset(("baseline", "small", "local_count"))


def _on_npu(device):
    return device.type == "npu"


def eligible(bank, prepared, group_ends, native_backend):
    """Metadata-only admission: never inspect route values or synchronize."""
    # Routed decode uses INT32 IDs, so take this cheap exit before inspecting
    # large-bank geometry. Small grouped calls also retain their original op.
    if not isinstance(group_ends, torch.Tensor) or group_ends.dtype != torch.int64:
        return False
    if bank.backend != native_backend or not isinstance(prepared, (tuple, list)) or len(prepared) != 4:
        return False
    low, high, scale, total = prepared
    if not isinstance(low, torch.Tensor) or low.ndim != 2 or not MIN_GROUPED_ROWS < low.shape[0] <= MAX_QUALIFIED_ROWS:
        return False
    tensors = (*prepared, bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum, group_ends)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if bank.weight.ndim != 3 or low.ndim != 2:
        return False
    experts, outputs, packed_inputs = bank.weight.shape
    inputs = packed_inputs * 2
    rows = low.shape[0]
    metadata_shape = (rows, inputs // 128, 8)
    weight_metadata_shape = (experts, outputs, inputs // 128)
    return (
        experts == EXPERTS
        and (outputs, inputs) in PROJECTION_SHAPES
        and MIN_GROUPED_ROWS < rows <= MAX_QUALIFIED_ROWS
        and low.shape == (rows, packed_inputs)
        and high.shape == low.shape
        and low.dtype == high.dtype == bank.weight.dtype == torch.int8
        and scale.shape == total.shape == metadata_shape
        and scale.dtype == total.dtype == torch.float32
        and group_ends.shape == (EXPERTS,)
        and group_ends.dtype == torch.int64
        and all(
            tensor.shape == weight_metadata_shape and tensor.dtype == torch.float16
            for tensor in (bank.weight_scale, bank.weight_offset, bank.weight_sum)
        )
        and _on_npu(low.device)
        and all(tensor.device == low.device and tensor.is_contiguous() for tensor in tensors)
    )


def qualified_resource(worker, mode, *, version=1):
    """Require synchronized real-bank receipts, then borrow one immutable owner."""
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("prefill binding version must be 1, 2 or 3")
    if mode not in ("small", "local_count"):
        raise ValueError("select small or local_count schedule explicitly")
    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("drain model execution before preparing a prefill binding")
    attribute = (STATE_ATTRIBUTE, STATE_ATTRIBUTE_V2, STATE_ATTRIBUTE_V3)[version - 1]
    state = getattr(worker, attribute, None)
    if not isinstance(state, dict) or not state.get("loaded") or state.get("failed"):
        raise RuntimeError("prefill binding requires a loaded, healthy worker-owned probe")
    if state.get("version", 1) != version:
        raise RuntimeError("prefill binding resource version does not match the requested version")
    resources = state.get("resources", {})
    if version == 3 and resources.get("eager_only") is not True:
        raise RuntimeError("v3 prefill binding requires explicitly eager-only loaded resources")
    resource = resources.get(mode)
    if not callable(resource):
        raise RuntimeError("qualified prefill schedule resource is absent")
    covered = set()
    for receipt in state.get("schedule_receipts", ()):
        if receipt.get("passed") is not True or receipt.get("projection") not in REQUIRED_PROJECTIONS:
            continue
        for record in receipt.get("records", ()):
            rows = record.get("rows")
            if rows not in REQUIRED_ROWS or record.get("passed") is not True:
                continue
            cases = record.get("cases", ())
            exact_schedules = {
                case.get("schedule") for case in cases if case.get("exact") is True and case.get("rows") == rows
            }
            if not REQUIRED_SCHEDULES.issubset(exact_schedules):
                continue
            covered.add((receipt["projection"], rows, record.get("local_rows"), record.get("skew")))
    required = {
        (projection, rows, local, skew)
        for projection in REQUIRED_PROJECTIONS
        for rows in REQUIRED_ROWS
        for local, skew in ((0, False), (rows // 4, False), (rows, False), (rows // 4, True))
    }
    if not required.issubset(covered):
        raise RuntimeError("prefill binding requires exact real-bank gates for both projections and all route cases")
    return resource


def build_forward(original, resource, *, source=None, eager_only=False):
    original = getattr(original, "_qfn_prefill_schedule_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("prefill binding requires the exact qualified native_linear source")
    native_backend = original.__globals__["NATIVE_INT4_BACKEND"]

    def forward(self, prepared, group_ends):
        if eligible(self, prepared, group_ends, native_backend):
            rows, device = prepared[0].shape[0], prepared[0].device
            # Queue-draining v3 resources cannot run inside capture, even if
            # this shape already has a cached descriptor. Check before lookup.
            if eager_only and native_prefill._capturing(device):
                return original(self, prepared, group_ends)
            experts, outputs, packed_inputs = self.weight.shape
            key = ((rows, experts, outputs, packed_inputs * 2, 8, 0, 1), device)
            if descriptors_available(
                resource,
                (key,),
                capacity=native_prefill.MAX_CONFIGS,
                is_capturing=lambda: native_prefill._capturing(device),
            ):
                return resource(self, prepared, group_ends)
        return original(self, prepared, group_ends)

    forward._qfn_prefill_schedule_base = original
    return forward


def _bank_type():
    from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank

    return PackedExpertBank


def make_replacements(worker, *, mode="local_count", version=1):
    resource = qualified_resource(worker, mode, version=version)
    if version == 3:
        return {TARGET: build_forward(_bank_type().native_linear, resource, eager_only=True)}
    return {TARGET: build_forward(_bank_type().native_linear, resource)}


def replacements():
    raise RuntimeError("prepare this binding explicitly with make_replacements(drained_worker, mode=...)")
