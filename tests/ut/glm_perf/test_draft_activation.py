# SPDX-License-Identifier: Apache-2.0
"""Draft role isolation, shared-method rejection and transition restoration."""

from types import SimpleNamespace

import pytest

from tools.glm_perf.instance_bindings import InstanceBindings
from tools.glm_perf.resident_candidates.draft_activation import bind_draft


class Method:
    def _apply_device_grouped(self, *args):
        return "original", args


def owner(method, layer):
    return SimpleNamespace(_method=method, w2_experts=SimpleNamespace(layer_key=layer))


def test_target_keeps_its_method_while_only_draft_instances_change():
    target, draft = Method(), Method()
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [owner(target, "layers.44")]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: [owner(draft, "layers.45")])),
    )
    bindings, native, audit = InstanceBindings(), object(), {}

    def factory(original, supplied, supplied_audit):
        assert supplied is native and supplied_audit is audit
        return lambda self, *args: ("draft", original(self, *args))

    assert bind_draft(bindings, runner, native, audit, wrapper_factory=factory) == ["layers.45"]
    assert target._apply_device_grouped(1) == ("original", (1,))
    assert draft._apply_device_grouped(2) == ("draft", ("original", (2,)))
    bindings.restore()
    assert "_apply_device_grouped" not in draft.__dict__
    assert draft._apply_device_grouped(3) == ("original", (3,))


def test_shared_target_and_draft_method_never_changes():
    method = Method()
    module = owner(method, "layers.45")
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [module]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: [module])),
    )
    bindings = InstanceBindings()
    with pytest.raises(ValueError, match="shared with the target"):
        bind_draft(bindings, runner, None, {})
    assert not bindings.originals and "_apply_device_grouped" not in method.__dict__


def test_missing_draft_rejected_before_any_binding():
    with pytest.raises(ValueError, match="loaded MTP"):
        bind_draft(None, SimpleNamespace(), None, {})
