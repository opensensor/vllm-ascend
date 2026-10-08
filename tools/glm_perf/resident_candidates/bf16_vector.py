# SPDX-License-Identifier: Apache-2.0
"""Bind vector modes on permanent converters shared by compression and scoring."""

from ..bf16_vector import MODES
from ..instance_bindings import InstanceBindings
from ..resident_rpc_guard import WORKER_PREFIX, guard_replacements

MAX_PREFILL_ROWS = 640
POOL_SIZE = 4


def bind_converters(bindings, runner, native):
    roots = [runner.model]
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is not None:
        roots.append(draft)
    modules = [
        module
        for root in roots
        for module in root.modules()
        if hasattr(module, "indexer_op") and hasattr(module, "_native_bf16_cast")
    ]
    converters = {id(module._native_bf16_cast): module._native_bf16_cast for module in modules}
    if not converters:
        raise ValueError("no permanent GLM BF16 converters found")
    counts = {key[1] for converter in converters.values() for key in converter.configs if key[1] > 0}
    for module in modules:
        for width in (module.rope_dim, module.head_dim, POOL_SIZE * module.head_dim, module.n_head * module.head_dim):
            if width > 0:
                counts.update(rows * width for rows in range(1, MAX_PREFILL_ROWS + 1))
    native.prepare_counts(counts)
    try:
        for converter in converters.values():
            original = converter._convert

            def convert(self, source, dtype, mode, original=original):
                if mode in MODES and (not source.numel() or (source.numel(), mode) in native.configs):
                    return native.convert(source, dtype, mode)
                return original(source, dtype, mode)

            bindings.bind(converter, "_convert", convert)
    except Exception:
        bindings.restore()
        raise
    return len(converters)


def extend_replacements(changes, native):
    bindings = InstanceBindings()
    state = {"converters": 0}
    capture_base, apply_base, status_base = (
        changes[WORKER_PREFIX + name] for name in ("resident_capture", "resident_apply", "resident_status")
    )

    def capture(self):
        try:
            state["converters"] = bind_converters(bindings, self.model_runner, native)
            result = capture_base(self)
            if "error" in result:
                bindings.restore()
            return result
        except Exception as error:
            bindings.restore()
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    def apply(self, generation):
        bindings.restore()
        return apply_base(self, generation)

    def status(self):
        receipt = status_base(self)
        receipt["vector_bf16_conversion"] = dict(
            state, calls_by_mode=dict(native.calls), descriptors=len(native.configs), prepared_outside_capture=True
        )
        return receipt

    result = dict(changes)
    for name, function in (("resident_capture", capture), ("resident_apply", apply), ("resident_status", status)):
        result[WORKER_PREFIX + name] = function
    return guard_replacements(result)
