# SPDX-License-Identifier: Apache-2.0
"""Vector query casts for the complete loaded target/draft forward pipeline."""

import torch

from ..instance_bindings import InstanceBindings, bind_indexer_forwards
from ..query_cast_binding import wrap_forward
from ..resident_rpc_guard import WORKER_PREFIX, guard_replacements


def loaded_indexers(runner):
    roots = [runner.model]
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is not None:
        roots.append(draft)
    return [
        module
        for root in roots
        for module in root.modules()
        if hasattr(module, "indexer_op") and hasattr(module, "_native_bf16_cast")
    ]


def extend_replacements(changes, native, authoritative_source, *, capture_hook, apply_hook):
    """Explicit hooks must prepare other components without rebinding forward."""
    bindings = InstanceBindings()
    state = {"indexers": 0, "native_calls": 0, "legacy_calls": 0}
    status_hook = changes[WORKER_PREFIX + "resident_status"]

    def factory(original, legacy):
        def convert(value, dtype):
            if value.dtype == torch.float16 and dtype == torch.bfloat16:
                state["native_calls"] += 1
                return native(value.contiguous())
            state["legacy_calls"] += 1
            return legacy(value, dtype)

        return wrap_forward(original, convert, authoritative_source)

    def capture(self):
        try:
            indexers = loaded_indexers(self.model_runner)
            counts = [
                rows * indexer.n_head * indexer.head_dim
                for indexer in indexers
                for rows in range(1, indexer.topk_indices_buffer.shape[0] + 1)
            ]
            native.prepare_counts(counts)
            state["indexers"] = bind_indexer_forwards(bindings, self.model_runner, factory)
            expected = [indexer.forward.__func__ for indexer in indexers]
            result = capture_hook(self)
            if "error" in result:
                bindings.restore()
                return result
            if any(indexer.forward.__func__ is not function for indexer, function in zip(indexers, expected)):
                raise ValueError("parent capture replaced a vector query forward")
            return result
        except Exception as error:
            bindings.restore()
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    def apply(self, generation):
        bindings.restore()
        return apply_hook(self, generation)

    def status(self):
        receipt = status_hook(self)
        receipt["vector_query_cast"] = dict(state, methods=len(bindings.originals), prepared_outside_capture=True)
        return receipt

    result = dict(changes)
    for name, function in (("resident_capture", capture), ("resident_apply", apply), ("resident_status", status)):
        result[WORKER_PREFIX + name] = function
    return guard_replacements(result)
