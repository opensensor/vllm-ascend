# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly admitted QSA candidate: cold dual capture, live replay only.

Default qualification uses C1 prefix plus C1/C2 general graphs. The resident
coverage gate can admit C1-C4 general and prefix variants together with resized
metadata buffers. No request-time capture, reset or native lease mutation.
Extra pool memory/collective events require hardware gating.
"""

from collections import namedtuple
from contextlib import contextmanager
from copy import copy

import torch

from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import (
    build_qualified_forward,
    discover_geometries,
    inspect_admission,
    is_target_wrapper,
    require_empty_graph_parameters,
    validate_target_capture,
)

CAPTURE_SIZES = (3, 6)
EXTENDED_CAPTURE_SIZES = (3, 6, 9, 12)
PREFIX_SIZES = (3,)
DERIVED_METADATA_CACHES = (
    "_gdn_recurrent_step_meta",
    "_gdn_chunk_plan",
    "_qwen4exp_mixed_views",
    "_qwen4exp_query_ranges",
)


class VariantContract(namedtuple("VariantContractBase", "capture binding metadata")):
    __slots__ = ()


def isolate_capture_metadata(metadata_by_layer):
    """Keep staging buffers, but derive step scratch independently in each graph.

    GDN memoizes derived device tensors on metadata. Sharing that memo between
    captures makes the second graph consume scratch written only by the first.
    Keep one shallow copy per shared metadata object and strip only audited
    per-step derived caches; the contract retains these copies through reset.
    """
    if not isinstance(metadata_by_layer, dict):
        raise ValueError("cold QSA capture requires layer metadata")
    copies = {}
    result = {}
    for prefix, metadata in metadata_by_layer.items():
        key = id(metadata)
        if key not in copies:
            isolated = copy(metadata)
            for attribute in DERIVED_METADATA_CACHES:
                if hasattr(isolated, attribute):
                    delattr(isolated, attribute)
            copies[key] = isolated
        result[prefix] = copies[key]
    return result


def metadata_ownership(metadata_by_layer):
    """Cold identity evidence only; never read device values or export pointers."""
    objects = {id(metadata): metadata for metadata in metadata_by_layer.values()}
    storages = set()
    caches = 0

    def visit(value):
        if isinstance(value, torch.Tensor):
            storages.add((str(value.device), value.untyped_storage().data_ptr()))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    for metadata in objects.values():
        cache = getattr(metadata, "_gdn_recurrent_step_meta", None)
        if cache is not None:
            caches += 1
            visit(cache)
    return set(objects), storages, caches


@contextmanager
def context_attributes(context, **values):
    old = {name: (hasattr(context, name), getattr(context, name, None)) for name in values}
    for name, value in values.items():
        setattr(context, name, value)
    try:
        yield
    finally:
        for name, (present, value) in old.items():
            if present:
                setattr(context, name, value)
            else:
                delattr(context, name)


@contextmanager
def capture_pool(wrapper, controller, variant, allocator, make_pool):
    """Restore the actual allocator selection, which can differ from the wrapper."""
    previous_pool = wrapper.graph_pool
    previous_selection = allocator._graph_pool_id
    try:
        if variant == "prefix":
            if controller.prefix_pool is None:
                controller.prefix_pool = make_pool()
            wrapper.graph_pool = controller.prefix_pool
        yield
    finally:
        wrapper.graph_pool = previous_pool
        allocator.set_graph_pool_id(previous_selection)


class ColdVariants:
    def __init__(self, geometries, observer, *, capture_sizes=CAPTURE_SIZES, prefix_sizes=PREFIX_SIZES):
        capture_sizes = tuple(capture_sizes)
        if capture_sizes not in (CAPTURE_SIZES, EXTENDED_CAPTURE_SIZES):
            raise ValueError("cold QSA requires the bounded C1/C2 or C1-C4 capture profile")
        self.geometries = geometries
        self.observer = observer
        prefix_sizes = tuple(prefix_sizes)
        if prefix_sizes not in (PREFIX_SIZES, capture_sizes):
            raise ValueError("prefix coverage must be C1-only or all admitted sizes")
        self.capture_sizes = capture_sizes
        self.prefix_sizes = prefix_sizes
        self.contracts = {}
        self.prefix_pool = None
        self.failed = False

    def call(self, wrapper, context, args, kwargs, *, capture_allowed, capture, replay, eager):
        descriptor = context.batch_descriptor
        size = getattr(descriptor, "num_tokens", None)
        if self.failed:
            raise RuntimeError("failed cold capture requires paused native recovery")
        if size not in self.capture_sizes:
            return eager()
        general_key = (descriptor, "general")
        try:
            admission = inspect_admission(context, self.geometries, args, kwargs)
        except (ValueError, TypeError, AttributeError) as error:
            if general_key in self.contracts:
                raise RuntimeError(
                    "cold QSA metadata changed; coordinated recovery required before submission"
                ) from error
            return eager()
        if general_key not in self.contracts:
            if not capture_allowed:
                return eager()
            if any(key[0] == descriptor for key in wrapper.entries):
                raise RuntimeError("cold QSA variants require empty descriptor entries")
            variants = ("general", "prefix") if size in self.prefix_sizes else ("general",)
            result = None
            try:
                for variant in variants:
                    key = (descriptor, variant)
                    capture_metadata = isolate_capture_metadata(context.attn_metadata)
                    with context_attributes(
                        context,
                        attn_metadata=capture_metadata,
                        capturing=getattr(context, "capturing", False),
                        _qfn_qsa_prefix_capture=self.geometries if variant == "prefix" else None,
                        _qfn_qsa_force_sparse_capture=True,
                    ):
                        entry, result = capture(key, variant)
                    if entry.capture is None:
                        raise RuntimeError("cold variant capture did not produce an owned graph")
                    self.contracts[key] = VariantContract(entry.capture, admission.binding, capture_metadata)
            except BaseException:
                self.failed = True
                raise
            # Startup capture records without executing, exactly as baseline.
            return result
        variant = "prefix" if size in self.prefix_sizes and admission.prefix_eligible else "general"
        key = (descriptor, variant)
        contract = self.contracts.get(key)
        entry = wrapper.entries.get(key)
        if (
            contract is None
            or entry is None
            or entry.capture is not contract.capture
            or contract.binding != admission.binding
        ):
            raise RuntimeError("cold QSA binding changed; coordinated recovery required before submission")
        return replay(entry)

    def snapshot(self):
        ownership = {key: metadata_ownership(contract.metadata) for key, contract in self.contracts.items()}
        separation = []
        for (descriptor, variant), (metadata, tensors, _) in ownership.items():
            if variant == "prefix" and (descriptor, "general") in ownership:
                general_metadata, general_tensors, _ = ownership[(descriptor, "general")]
                separation.append(
                    {
                        "descriptor": str(descriptor),
                        "shared_metadata_objects": len(metadata & general_metadata),
                        "shared_gdn_tensor_storages": len(tensors & general_tensors),
                        "prefix_gdn_tensor_storages": len(tensors),
                        "general_gdn_tensor_storages": len(general_tensors),
                    }
                )
        return {
            "failed": self.failed,
            "capture_sizes": list(self.capture_sizes),
            "prefix_sizes": list(self.prefix_sizes),
            "prefix_pool_retained": self.prefix_pool is not None,
            "variants": [
                {
                    "descriptor": str(key[0]),
                    "variant": key[1],
                    "native_graphs": value.capture.num_graphs,
                    "eager_breaks": value.capture.num_eager_breaks,
                    "metadata_objects": len(ownership[key][0]),
                    "gdn_tensor_storages": len(ownership[key][1]),
                    "gdn_cache_objects": ownership[key][2],
                }
                for key, value in self.contracts.items()
            ],
            "variant_metadata_separation": separation,
        }


def controller_for_observer(wrapper, observer):
    controller = wrapper.__dict__.get("_qfn_qsa_cold_variants")
    if controller is not None and controller.observer is observer:
        return controller
    if wrapper.entries:
        raise RuntimeError("cold variant generation requires retired wrapper entries")
    if controller is not None:
        receipt = controller.observer.snapshot()
        if not receipt["retired"] or receipt["native_leases_acquired"] != receipt["native_leases_released"]:
            raise RuntimeError("previous cold variant generation retains native leases")
    observer._require_live()
    if observer.track_replays or observer.active_capture is not None:
        raise RuntimeError("cold variants require an unobserved, idle collective owner")
    if getattr(wrapper.vllm_config.scheduler_config, "async_scheduling", False):
        raise RuntimeError("cold variants require serialized synchronous workers")
    geometries = discover_geometries(wrapper.runnable, wrapper.vllm_config.model_config.max_model_len)
    if not geometries:
        raise RuntimeError("cold target variant has no QSA geometry")
    compilation = getattr(wrapper.vllm_config, "compilation_config", None)
    sizes = getattr(compilation, "cudagraph_capture_sizes", CAPTURE_SIZES)
    controller = ColdVariants(
        geometries,
        observer,
        capture_sizes=sizes,
        prefix_sizes=getattr(wrapper, "_qfn_qsa_prefix_sizes", PREFIX_SIZES),
    )
    wrapper._qfn_qsa_cold_variants = controller
    return controller


def cold_wrapper_call(self, *args, **kwargs):
    from vllm.compilation import breakable_cudagraph as upstream
    from vllm.compilation import monitor
    from vllm.config import CUDAGraphMode
    from vllm.distributed.device_communicators import pynccl_allocator
    from vllm.forward_context import get_forward_context, is_forward_context_available

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    if not is_forward_context_available():
        return upstream.BreakableCUDAGraphWrapper.__call__(self, *args, **kwargs)
    context = get_forward_context()
    if (
        _EXTRA_CTX.is_draft_model
        or not is_target_wrapper(self)
        or context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        or (self.runtime_mode is not None and self.runtime_mode != context.cudagraph_runtime_mode)
    ):
        return upstream.BreakableCUDAGraphWrapper.__call__(self, *args, **kwargs)
    owner = ACLGraphWrapper._resolve_collective_graph_observer(self)
    controller = controller_for_observer(self, owner)

    def capture(key, variant):
        require_empty_graph_parameters()
        if torch.npu.current_stream() == torch.npu.default_stream():
            raise RuntimeError("cold QSA variants must use capture_model's non-default stream")
        entry = upstream._BreakableEntry(batch_descriptor=context.batch_descriptor)
        self.entries[key] = entry  # All live captures remain discoverable for retirement.
        with capture_pool(self, controller, variant, pynccl_allocator, torch.npu.graph_pool_handle):
            output = self._capture(entry, args, kwargs)
            validate_target_capture(entry.capture)
            return entry, output

    def eager():
        with context_attributes(context, cudagraph_runtime_mode=CUDAGraphMode.NONE, capturing=False):
            return self.runnable(*args, **kwargs)

    return controller.call(
        self,
        context,
        args,
        kwargs,
        capture_allowed=monitor.cudagraph_capturing_enabled,
        capture=capture,
        replay=lambda entry: self._replay(entry, args, kwargs),
        eager=eager,
    )


def cold_inventory(self):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    from tools.qwen4exp.resident_candidates.qsa_admission_audit import inventory_with_audit

    result = inventory_with_audit(self)
    result.pop("qsa_admission_audit", None)
    result["qsa_cold_variants"] = [
        {"model": type(wrapper.runnable).__name__, **wrapper._qfn_qsa_cold_variants.snapshot()}
        for wrapper in tuple(BreakableCUDAGraphWrapper._all_instances)
        if "_qfn_qsa_cold_variants" in wrapper.__dict__
    ]
    return result


def make_replacements(*, qualification_gate=False):
    """Explicit parent-only qualification entry point; not a hardware claim."""
    if qualification_gate is not True:
        raise RuntimeError("cold QSA variants are blocked pending explicit resident qualification")
    validate_qualified_wrapper()
    return {
        "vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward": build_qualified_forward(),
        "vllm_ascend.compilation.breakable_aclgraph:BreakableACLGraphWrapper.__call__": cold_wrapper_call,
        "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_graph_inventory": cold_inventory,
    }


def replacements():
    return make_replacements()


def validate_wrapper_method(effective, upstream):
    original = getattr(effective, "_qfn_qsa_cold_base_call", effective)
    if original is not upstream:
        raise RuntimeError("cold QSA variants require the qualified inherited wrapper __call__")
    return original


def validate_qualified_wrapper():
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper

    cold_wrapper_call._qfn_qsa_cold_base_call = validate_wrapper_method(
        BreakableACLGraphWrapper.__call__,
        BreakableCUDAGraphWrapper.__call__,
    )
