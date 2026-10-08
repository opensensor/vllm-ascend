# SPDX-License-Identifier: Apache-2.0
"""Per-instance overrides must run during the experiment and restore exactly."""

from types import MethodType, SimpleNamespace

import pytest

from tools.glm_perf.instance_bindings import InstanceBindings, bind_indexers, extend_bindings


class Operation:
    def _write_pools(self):
        return "class writer"

    def _select_tokens_fixed(self):
        return "class selector"


def permanent(self):
    return "permanent converter"


def changed(self):
    return "candidate converter"


def test_permanent_instance_method_bypasses_class_patch_until_bound():
    operation = Operation()
    old = operation._write_pools = MethodType(permanent, operation)
    bindings = InstanceBindings()
    assert operation._write_pools() == "permanent converter"
    bindings.bind(operation, "_write_pools", changed)
    assert operation._write_pools() == "candidate converter"
    bindings.restore()
    assert operation._write_pools is old and operation._write_pools() == "permanent converter"


def test_class_only_method_is_removed_from_instance_on_restore():
    operation = Operation()
    bindings = InstanceBindings()
    bindings.bind(operation, "_write_pools", changed)
    assert "_write_pools" in operation.__dict__
    bindings.restore()
    assert "_write_pools" not in operation.__dict__ and operation._write_pools() == "class writer"


def test_partial_install_failure_restores_target_and_draft():
    operations = [Operation(), Operation()]
    modules = [SimpleNamespace(indexer_op=op, _native_bf16_cast=object()) for op in operations]
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: modules[:1]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: modules[1:])),
    )
    bindings = InstanceBindings()
    calls = []

    def factory(function):
        calls.append(function)
        if len(calls) == 4:
            raise ValueError("changed draft selector")
        return changed

    with pytest.raises(ValueError, match="changed draft selector"):
        bind_indexers(bindings, runner, factory, factory)
    assert not bindings.originals and all(not op.__dict__ for op in operations)


def test_target_and_draft_bind_and_restore_together():
    operations = [Operation(), Operation()]
    modules = [SimpleNamespace(indexer_op=op, _native_bf16_cast=object()) for op in operations]
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: modules[:1]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: modules[1:])),
    )
    bindings = InstanceBindings()
    assert bind_indexers(bindings, runner, lambda old: changed, lambda old: changed) == 2
    assert all(op._select_tokens_fixed() == "candidate converter" for op in operations)
    with pytest.raises(ValueError, match="already bound"):
        bindings.bind(operations[0], "_write_pools", changed)
    bindings.restore()
    assert all(not op.__dict__ for op in operations)


@pytest.mark.parametrize("fail", [False, True])
def test_capture_installs_and_next_apply_restores_before_delegate(fail):
    operation = Operation()
    old = operation._write_pools = MethodType(permanent, operation)
    module = SimpleNamespace(indexer_op=operation, _native_bf16_cast=object())
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda: [module])))
    worker._resident_error = lambda error: {"error": str(error)}
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."

    def capture(self):
        assert operation._write_pools() == "candidate converter"
        return {"error": "capture failed"} if fail else {"captured": True}

    def apply(self, generation):
        assert operation._write_pools is old
        return {"generation": generation}

    changes = {
        prefix + "resident_capture": capture,
        prefix + "resident_apply": apply,
        prefix + "resident_status": lambda self: {},
    }
    candidate = extend_bindings(changes, lambda old: changed, lambda old: changed)
    receipt = candidate[prefix + "resident_capture"](worker)
    if fail:
        assert receipt["error"] and operation._write_pools is old
    else:
        assert candidate[prefix + "resident_status"](worker)["indexer_instance_bindings"]["methods"] == 2
    assert candidate[prefix + "resident_apply"](worker, "next")["generation"] == "next"


def test_explicit_base_bypasses_prior_candidate_hooks():
    operation = Operation()
    module = SimpleNamespace(indexer_op=operation, _native_bf16_cast=object())
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda: [module])))
    worker._resident_error = lambda error: {"error": str(error)}
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."

    def old(self, *args):
        raise AssertionError("old candidate hook ran")

    def capture(self):
        assert operation._write_pools() == "candidate converter"
        return {"captured": True}

    def apply(self, generation):
        assert operation._write_pools() == "class writer"
        return {"generation": generation}

    changes = {
        prefix + "resident_capture": old,
        prefix + "resident_apply": old,
        prefix + "resident_status": lambda self: {},
    }
    candidate = extend_bindings(
        changes, lambda original: changed, lambda original: changed, capture_hook=capture, apply_hook=apply
    )
    assert candidate[prefix + "resident_capture"](worker) == {"captured": True}
    assert candidate[prefix + "resident_apply"](worker, "next") == {"generation": "next"}


def test_query_forward_uses_its_permanent_converter_and_restores_on_failure():
    from tools.glm_perf.instance_bindings import bind_indexer_forwards

    class Indexer:
        def forward(self):
            return "ordinary cast"

    indexers = [Indexer(), Indexer()]
    for indexer in indexers:
        indexer._native_bf16_cast = object()
        indexer.indexer_op = object()
    permanent_forward = indexers[0].forward = MethodType(lambda self: "permanent ordinary cast", indexers[0])
    roots = [SimpleNamespace(modules=lambda indexer=indexer: [indexer]) for indexer in indexers]
    runner = SimpleNamespace(model=roots[0], drafter=SimpleNamespace(model=roots[1]))
    bindings = InstanceBindings()
    observed = []

    def factory(original, converter):
        observed.append(converter)
        return lambda self: converter

    assert bind_indexer_forwards(bindings, runner, factory) == 2
    assert observed == [i._native_bf16_cast for i in indexers]
    assert all(i.forward() is i._native_bf16_cast for i in indexers)
    bindings.restore()
    assert indexers[0].forward is permanent_forward
    assert indexers[1].forward() == "ordinary cast" and "forward" not in indexers[1].__dict__
    observed.clear()

    def fail_second(original, converter):
        if observed:
            raise ValueError("failed draft query cast")
        return factory(original, converter)

    with pytest.raises(ValueError, match="failed draft"):
        bind_indexer_forwards(bindings, runner, fail_second)
    assert not bindings.originals and indexers[0].forward is permanent_forward
    assert "forward" not in indexers[1].__dict__
