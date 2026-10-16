# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tools.qwen4exp.single_card_ple import build_forward, candidate_source


@pytest.mark.parametrize("fail", [False, True])
def test_forward_restores_instance_binding_on_success_and_failure(fail):
    method = SimpleNamespace(gather_rows=lambda ids: ids)
    initial = method.gather_rows
    layer = SimpleNamespace(ple=SimpleNamespace(ple_method=method), _ensure_ple_method=lambda device: None)

    class Gather:
        def __init__(self, method, metadata):
            self.original = method.gather_rows

    class Binding:
        def __init__(self, method, gather, verify_calls):
            self.method, self.original = method, gather.original
            self.failed = False
            assert verify_calls == 8

        def install(self):
            self.method.gather_rows = self

        def restore(self):
            self.method.gather_rows = self.original

    def original(self, hidden):
        assert isinstance(self.ple.ple_method.gather_rows, Binding)
        if fail:
            raise ValueError("projection failed")
        return hidden

    forward, owners = build_forward(original, {}, Gather, Binding)
    tensor = SimpleNamespace(device="npu")
    if fail:
        with pytest.raises(ValueError, match="projection failed"):
            forward(layer, tensor)
    else:
        assert forward(layer, tensor) is tensor
        assert forward(layer, tensor) is tensor
    assert method.gather_rows is initial
    assert len(owners) == 1
    next(iter(owners.values())).failed = True
    with pytest.raises(RuntimeError, match="shadow mismatch"):
        forward(layer, tensor)
    assert method.gather_rows is initial


def test_candidate_is_self_contained():
    namespace = {}
    exec(compile(candidate_source({"test": 1}), "<test>", "exec"), namespace)
    assert callable(namespace["replacements"])
    assert namespace["METADATA"] == {"test": 1}
