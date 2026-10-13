# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from tools.sidecard.expert_prefill_gate import validate


@pytest.mark.parametrize("fail", [False, True])
def test_real_bank_gate_checks_active_peer_zero_routes_and_records_failure(tmp_path, fail):
    class Actor:
        rank = 0
        banks = {18: None, 19: None}

        def __call__(self, layer, x, ids, weights):
            owned = ids < 144
            return x.float() * (weights * owned).sum(-1, keepdim=True)

    eager = Actor()

    def captured(*values):
        return eager(*values) + int(fail)

    path = tmp_path / "gate.json"
    if fail:
        with pytest.raises(AssertionError, match="real-bank"):
            validate(eager, captured, torch, output=path, device="cpu")
    else:
        report = validate(eager, captured, torch, output=path, device="cpu")
        assert report["passed"] and len(report["cases"]) == 6
        assert [v["routes"] for v in report["cases"][:3]] == ["local", "mixed_peer", "zero_weights"]
    report = json.loads(path.read_text())
    assert report["passed"] is not fail
    assert all(row["bitwise_equal"] is not fail for row in report["cases"])
