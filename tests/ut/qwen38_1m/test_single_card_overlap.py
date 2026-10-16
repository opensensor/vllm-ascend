# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.qwen4exp.single_card_overlap import candidate_source


@pytest.mark.parametrize("early,prefix", [(True, False), (False, True), (True, True)])
def test_source_is_self_contained_for_the_frozen_runtime(early, prefix):
    source = candidate_source(early=early, prefix=prefix)
    namespace = {}
    exec(source, namespace)
    assert namespace["EARLY"] is early
    assert namespace["PREFIX"] is prefix
    assert callable(namespace["replacements"])
    assert "if 0 < capacity <= block_topk:" in namespace["QSA_SOURCE"]
    assert "from tools." not in source


@pytest.mark.parametrize("early,prefix,verify", [(False, False, False), (1, True, False), (True, False, True)])
def test_invalid_controls_are_rejected(early, prefix, verify):
    with pytest.raises(ValueError):
        candidate_source(early=early, prefix=prefix, verify=verify)


def test_new_source_revision_requires_qualification(monkeypatch):
    monkeypatch.setattr("tools.qwen4exp.single_card_overlap.QUALIFIED_PREFIX_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source changed"):
        candidate_source(early=False, prefix=True)


def test_positions_only_source_is_self_contained():
    namespace = {}
    exec(candidate_source(early=False, prefix=False, positions=True), namespace)
    assert namespace["POSITIONS"]
    assert callable(namespace["build_positions_forward"])
    assert "pin_memory()" in namespace["POSITION_SOURCE"]
    assert "weak_output.copy_(host_positions, non_blocking=True)" in namespace["POSITION_SOURCE"]
