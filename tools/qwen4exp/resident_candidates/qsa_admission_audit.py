# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded diagnostic admission sampling that always delegates model execution.

No selection replacement, graph admission guard, device content read, device
synchronization or collective observer hook. This explicitly diagnostic payload
adds host work; its request rates are not performance claims. Remove it before
measuring the production candidate.
"""

import torch

from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import discover_geometries, inspect_admission

MAX_DESCRIPTORS = 8
MAX_METADATA_GROUPS = 4
MAX_PREFIX_NAMES = 16
MAX_LENGTH_VALUES = 16
MAX_ERROR_LENGTH = 240
MAX_COUNTER = (1 << 31) - 1


def _increment(value):
    return min(MAX_COUNTER, value + 1)


def _tensor_summary(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def _host_summary(value):
    result = _tensor_summary(value)
    if result is not None and value.device.type == "cpu" and value.ndim == 1 and value.numel() <= MAX_LENGTH_VALUES:
        result["values"] = value.tolist()
    return result


def _metadata_summary(context, geometries):
    layers = getattr(context, "attn_metadata", None)
    if not isinstance(layers, dict):
        return {"available": False}
    groups = {}
    for prefix in geometries:
        metadata = layers.get(prefix)
        key = id(metadata)
        if key not in groups:
            if len(groups) >= MAX_METADATA_GROUPS:
                continue
            groups[key] = {
                "prefixes": [],
                "num_actual_tokens": getattr(metadata, "num_actual_tokens", None),
                "num_prefills": getattr(metadata, "num_prefills", None),
                "num_decodes": getattr(metadata, "num_decodes", None),
                "attn_state": getattr(getattr(metadata, "attn_state", None), "name", None),
                "seq_lens_cpu": _host_summary(getattr(metadata, "seq_lens_cpu", None)),
                "query_lens_cpu": _host_summary(getattr(metadata, "query_lens_cpu", None)),
                "buffers": {
                    name: _tensor_summary(getattr(metadata, name, None))
                    for name in ("block_tables", "query_start_loc", "slot_mapping", "seq_lens")
                },
            }
        if len(groups[key]["prefixes"]) < MAX_PREFIX_NAMES:
            groups[key]["prefixes"].append(prefix)
    return {"available": True, "groups": list(groups.values())}


def _input_summary(args, kwargs):
    values = [(f"arg{index}", value) for index, value in enumerate(args)] + list(kwargs.items())
    return {name: _tensor_summary(value) for name, value in values if isinstance(value, torch.Tensor)}


def mismatch_categories(first, current):
    """Classify differences without exposing addresses or reading tensor data."""
    categories = set()
    first_layers, first_inputs = first.binding
    current_layers, current_inputs = current.binding
    if len(first_layers) != len(current_layers):
        categories.add("layer_count")
    for original, actual in zip(first_layers, current_layers):
        if original[:2] != actual[:2]:
            categories.add("layer_geometry")
        shape_names = ("num_actual_tokens", "query_lens", "num_prefills", "num_decodes", "attn_state")
        for name, old, new in zip(shape_names, original[2], actual[2]):
            if old != new:
                categories.add(name)
        for (name, old), (new_name, new) in zip(original[3], actual[3]):
            if name != new_name:
                categories.add("metadata_fields")
            if old[0] != new[0]:
                categories.add(f"{name}_address")
            if old[1:] != new[1:]:
                categories.add(f"{name}_layout")
    if len(first_inputs) != len(current_inputs):
        categories.add("model_input_count")
    for old, new in zip(first_inputs, current_inputs):
        if old[0] != new[0]:
            categories.add("model_input_address")
        if old[1:] != new[1:]:
            categories.add("model_input_layout")
    if first.prefix_eligible != current.prefix_eligible:
        categories.add("prefix_band")
    return tuple(sorted(categories))


class AdmissionAudit:
    """At most eight first contracts per wrapper and fixed-size counters."""

    def __init__(self, geometries, generation):
        self.geometries = geometries
        self.generation = generation
        self.records = {}
        self.overflow = 0
        self.internal_errors = 0

    def observe(self, wrapper, context, args, kwargs):
        descriptor = str(context.batch_descriptor)
        record = self.records.get(descriptor)
        if record is None:
            if len(self.records) >= MAX_DESCRIPTORS:
                self.overflow = _increment(self.overflow)
                return
            record = {
                "calls": 0,
                "capture_calls": 0,
                "replay_calls": 0,
                "successes": 0,
                "errors": 0,
                "mismatches": {},
                "first": None,
                "first_success_phase": None,
                "first_error": None,
                "first_metadata": None,
                "first_inputs": None,
                "first_mismatch": None,
            }
            self.records[descriptor] = record
        record["calls"] = _increment(record["calls"])
        entry = wrapper.entries.get(context.batch_descriptor)
        phase = "capture" if entry is None or entry.capture is None else "replay"
        record[phase + "_calls"] = _increment(record[phase + "_calls"])
        try:
            current = inspect_admission(context, self.geometries, args, kwargs)
            record["successes"] = _increment(record["successes"])
            if record["first"] is None:
                record["first"] = current
                record["first_success_phase"] = phase
                record["first_metadata"] = _metadata_summary(context, self.geometries)
                record["first_inputs"] = _input_summary(args, kwargs)
            else:
                differences = mismatch_categories(record["first"], current)
                for category in differences:
                    record["mismatches"][category] = _increment(record["mismatches"].get(category, 0))
                if differences and record["first_mismatch"] is None:
                    record["first_mismatch"] = {
                        "phase": phase,
                        "categories": list(differences),
                        "metadata": _metadata_summary(context, self.geometries),
                        "inputs": _input_summary(args, kwargs),
                    }
        except Exception as error:
            record["errors"] = _increment(record["errors"])
            if record["first_error"] is None:
                record["first_error"] = {
                    "phase": phase,
                    "type": type(error).__name__,
                    "message": str(error)[:MAX_ERROR_LENGTH],
                    "metadata": _metadata_summary(context, self.geometries),
                    "inputs": _input_summary(args, kwargs),
                }

    def snapshot(self):
        return {
            "scope": "diagnostic_admission_only_not_performance",
            "descriptors": [
                {"descriptor": descriptor, **{key: value for key, value in record.items() if key != "first"}}
                for descriptor, record in self.records.items()
            ],
            "overflow_calls": self.overflow,
            "internal_errors": self.internal_errors,
        }


def audit_call(wrapper, context, args, kwargs, invoke):
    """Observe if possible; a diagnostic error never replaces model execution."""
    audit = None
    try:
        audit = wrapper.__dict__.get("_qfn_qsa_admission_audit")
        if audit is None or audit.generation is not audit_wrapper_call:
            geometries = discover_geometries(wrapper.runnable, wrapper.vllm_config.model_config.max_model_len)
            audit = AdmissionAudit(geometries, audit_wrapper_call)
            wrapper._qfn_qsa_admission_audit = audit
        if audit.geometries:
            audit.observe(wrapper, context, args, kwargs)
    except Exception:
        if audit is not None:
            audit.internal_errors = _increment(audit.internal_errors)
    return invoke()


def audit_wrapper_call(self, *args, **kwargs):
    """Qualified wrapper passthrough with bounded diagnostic sampling."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    invoke = lambda: BreakableCUDAGraphWrapper.__call__(self, *args, **kwargs)
    if not is_forward_context_available():
        return invoke()
    context = get_forward_context()
    if context.cudagraph_runtime_mode == CUDAGraphMode.NONE or (
        self.runtime_mode is not None and self.runtime_mode != context.cudagraph_runtime_mode
    ):
        return invoke()
    return audit_call(self, context, args, kwargs, invoke)


def inventory_with_audit(self):
    """Cold RPC preserving ordinary graph inventory and bounded audit evidence."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    from tools.sidecard.qfn_collective_profile import QfnProfileWorkerExtension
    from vllm_ascend.compilation.acl_graph import (
        get_draft_graph_params,
        get_draft_graph_prefill_params,
        get_graph_params,
    )

    graphs = QfnProfileWorkerExtension.qfn_graph_inventory(self)
    audits = []
    for wrapper in tuple(BreakableCUDAGraphWrapper._all_instances):
        audit = wrapper.__dict__.get("_qfn_qsa_admission_audit")
        if audit is not None:
            audits.append({"model": type(wrapper.runnable).__name__, **audit.snapshot()})
    parameters = {
        "target": summarize_graph_parameters(get_graph_params()),
        "draft": summarize_graph_parameters(get_draft_graph_params()),
        "draft_prefill": summarize_graph_parameters(get_draft_graph_prefill_params()),
    }
    return {"rank": self.rank, "graphs": graphs, "qsa_admission_audit": audits, "graph_parameters": parameters}


def summarize_graph_parameters(parameters):
    """Cold ownership counts only: do not dereference events, handles or tensors."""
    if parameters is None:
        return {"initialized": False, "sizes": {}}
    fields = ("events", "handles", "attn_params", "workspaces")
    sizes = sorted(set().union(*(getattr(parameters, field).keys() for field in fields)))
    return {
        "initialized": True,
        "sizes": {
            str(size): {
                **{field: len(getattr(parameters, field).get(size, ())) for field in fields[:-1]},
                "workspace": parameters.workspaces.get(size) is not None,
            }
            for size in sizes
        },
    }


def replacements():
    return {
        "vllm_ascend.compilation.breakable_aclgraph:BreakableACLGraphWrapper.__call__": audit_wrapper_call,
        "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_graph_inventory": inventory_with_audit,
    }
