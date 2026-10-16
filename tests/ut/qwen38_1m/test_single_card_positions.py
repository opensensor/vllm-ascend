# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from tools.qwen4exp import single_card_positions as candidate

SOURCE = "def forward(self, value):\n    return self._logical_query_positions(value)\n"
HOST = "def _host_qsa_logical_positions(value):\n    return dependency(value)\n"
POSITION = "def _logical_query_positions(value):\n    return _host_qsa_logical_positions(value) + 1\n"


def test_nested_helpers_keep_live_globals_and_reapplication_unwraps(monkeypatch):
    namespace = dict(dependency=lambda value: value * 2)
    exec(SOURCE, namespace)
    original = namespace["forward"]
    monkeypatch.setattr(candidate, "FORWARD_SHA256", hashlib.sha256(SOURCE.encode()).hexdigest())
    forward = candidate.build_positions_forward(original, HOST, POSITION, source=SOURCE)
    assert forward(None, 3) == 7
    namespace["dependency"] = lambda value: value * 3
    assert forward(None, 3) == 10
    again = candidate.build_positions_forward(forward, HOST, POSITION, source=SOURCE)
    assert again._single_card_position_base is original
    assert again.__globals__ is original.__globals__


def test_unqualified_forward_is_rejected_before_helpers_execute():
    with pytest.raises(ValueError, match="exact frozen"):
        candidate.build_positions_forward(lambda: None, 'raise AssertionError("must not execute")', "", source=SOURCE)
