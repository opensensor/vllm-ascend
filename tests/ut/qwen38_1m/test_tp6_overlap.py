# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest

from tools.qwen4exp import tp6_overlap


def test_compiled_selection_keeps_original_module_bindings(monkeypatch):
    source = "def select(value):\n    return helper(value)\n"
    namespace = dict(helper=lambda value: value * 2)
    exec(source, namespace)
    original = namespace["select"]
    monkeypatch.setattr(tp6_overlap.inspect, "getsource", lambda function: source)
    monkeypatch.setattr(tp6_overlap, "BASE_QSA_SHA256", hashlib.sha256(source.encode()).hexdigest())
    compiled = tp6_overlap.build_selection(original, source.replace("helper(value)", "helper(value) + 1"))
    assert compiled(3) == 7
    namespace["helper"] = lambda value: value * 3
    assert compiled(3) == 10
    assert compiled.__globals__ is original.__globals__


def test_unreviewed_source_fails_before_compilation(monkeypatch):
    monkeypatch.setattr(tp6_overlap.inspect, "getsource", lambda function: "unknown")
    with pytest.raises(ValueError, match="profiled QSA"):
        tp6_overlap.build_selection(lambda value: value, 'raise AssertionError("must not execute")')


def owner():
    resource = object.__new__(tp6_overlap.OverlapControl)
    resource.worker = SimpleNamespace(rank=5, model_runner=SimpleNamespace(execute_model_state=None))
    resource.early = resource.prefix = resource.verify = resource.failed = False
    resource.early_calls = resource.prefix_calls = resource.checked_calls = 0
    resource.checked_capacities = set()
    return resource


def test_prefix_fast_mode_requires_real_verification_and_drained_state():
    resource = owner()
    with pytest.raises(RuntimeError, match="verify actual"):
        resource.configure(early=False, prefix=True, verify=False)
    resource.configure(early=False, prefix=True, verify=True)
    resource.checked_calls = 10
    assert resource.configure(early=True, prefix=True, verify=False)["early"]
    resource.worker.model_runner.execute_model_state = object()
    with pytest.raises(RuntimeError, match="drain"):
        resource.configure(early=False, prefix=False, verify=False)


def test_failure_rejects_all_later_reconfiguration():
    resource = owner()
    resource.failed = True
    with pytest.raises(RuntimeError, match="healthy"):
        resource.configure(early=False, prefix=False, verify=False)


@pytest.mark.parametrize(
    "options", [dict(early=1, prefix=False, verify=False), dict(early=False, prefix=False, verify=True)]
)
def test_invalid_controls_do_not_mutate_owner(options):
    resource = owner()
    with pytest.raises(ValueError):
        resource.configure(**options)
    assert not resource.early and not resource.prefix
