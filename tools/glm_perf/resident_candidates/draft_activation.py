# SPDX-License-Identifier: Apache-2.0
"""Trial a separately qualified activation quantizer on draft MoE instances."""

from ..instance_bindings import InstanceBindings
from ..resident_rpc_guard import WORKER_PREFIX, guard_replacements
from .expert_reconstruction import wrap_fused_moe


def bind_draft(bindings, runner, native, audit, *, wrapper_factory=wrap_fused_moe):
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is None:
        raise ValueError("draft activation candidate requires a loaded MTP model")
    target_methods = {
        id(module._method)
        for module in runner.model.modules()
        if getattr(module, "w2_experts", None) is not None and hasattr(module, "_method")
    }
    seen, layers = set(), []
    try:
        for module in draft.modules():
            bank = getattr(module, "w2_experts", None)
            method = getattr(module, "_method", None)
            if bank is None or method is None or id(method) in seen:
                continue
            if id(method) in target_methods:
                raise ValueError("draft method is shared with the target model")
            original = method._apply_device_grouped.__func__
            bindings.bind(method, "_apply_device_grouped", wrapper_factory(original, native, audit))
            seen.add(id(method))
            layers.append(bank.layer_key)
        if not seen:
            raise ValueError("no permanent draft expert method found")
    except Exception:
        bindings.restore()
        raise
    return layers


def extend_replacements(changes, native):
    if native.activation_bits != 8 or not native.prepared_weight_layout:
        raise ValueError("draft A8 trial requires the qualified permanent native resource")
    bindings = InstanceBindings()
    audit = {"native_dispatches": 0, "fallback_dispatches": 0, "bank_coverage": {}}
    state = {"layers": []}
    capture_base, apply_base, status_base = (
        changes[WORKER_PREFIX + name] for name in ("resident_capture", "resident_apply", "resident_status")
    )

    def capture(self):
        try:
            state["layers"] = bind_draft(bindings, self.model_runner, native, audit)
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
        receipt["draft_activation_trial"] = dict(
            state, target_activation_bits=4, draft_activation_bits=8, methods=len(bindings.originals), audit=dict(audit)
        )
        return receipt

    result = dict(changes)
    for name, function in (("resident_capture", capture), ("resident_apply", apply), ("resident_status", status)):
        result[WORKER_PREFIX + name] = function
    return guard_replacements(result)
