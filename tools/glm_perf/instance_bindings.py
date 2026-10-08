# SPDX-License-Identifier: Apache-2.0
"""Reversible resident methods for already loaded, permanently bound indexers."""

from types import MethodType


class InstanceBindings:
    def __init__(self):
        self.originals = []

    def bind(self, owner, name, function):
        if any(saved_owner is owner and saved_name == name for saved_owner, saved_name, _, _ in self.originals):
            raise ValueError("instance method already bound in this candidate")
        self.originals.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))
        setattr(owner, name, MethodType(function, owner))

    def restore(self):
        for owner, name, owned, original in reversed(self.originals):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.originals.clear()


def bind_indexers(bindings, runner, writer_factory, selector_factory):
    roots = [runner.model]
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is not None:
        roots.append(draft)
    count = 0
    try:
        for root in roots:
            for module in root.modules():
                if not hasattr(module, "_native_bf16_cast") or not hasattr(module, "indexer_op"):
                    continue
                operation = module.indexer_op
                # These per-instance functions hold the permanent converter's
                # private globals. Preserve them instead of rebuilding a class
                # method with another converter or losing cached operands.
                writer = writer_factory(operation._write_pools.__func__)
                selector = selector_factory(operation._select_tokens_fixed.__func__)
                bindings.bind(operation, "_write_pools", writer)
                bindings.bind(operation, "_select_tokens_fixed", selector)
                count += 1
        if not count:
            raise ValueError("no permanently bound indexer found")
    except Exception:
        bindings.restore()
        raise
    return count


def extend_bindings(changes, writer_factory, selector_factory, *, capture_hook=None, apply_hook=None):
    """Install before capture and restore through the next transition's hook."""
    bindings = InstanceBindings()
    state = {"indexers": 0}
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    capture_base, apply_base, status_base = (
        changes[prefix + name] for name in ("resident_capture", "resident_apply", "resident_status")
    )
    # A composed factory can close over the previously installed worker hooks.
    # Give the caller an explicit immutable base so an old capture cannot
    # reinstall its methods after this candidate has bound new ones.
    capture_base = capture_hook if capture_hook is not None else capture_base
    apply_base = apply_hook if apply_hook is not None else apply_base

    def capture(self):
        try:
            state["indexers"] = bind_indexers(bindings, self.model_runner, writer_factory, selector_factory)
            result = capture_base(self)
            if "error" in result:
                bindings.restore()
            return result
        except Exception as error:
            bindings.restore()
            return self._resident_error(error)

    def apply(self, generation):
        # First activation enters the previous hook. The next transition
        # enters this hook before its successor is installed.
        bindings.restore()
        return apply_base(self, generation)

    def status(self):
        result = status_base(self)
        result["indexer_instance_bindings"] = {"indexers": state["indexers"], "methods": len(bindings.originals)}
        return result

    result = dict(changes)
    for name, function in (("resident_capture", capture), ("resident_apply", apply), ("resident_status", status)):
        result[prefix + name] = function
    return result
