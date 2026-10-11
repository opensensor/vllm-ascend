# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import numpy as np

from tools.sidecard import batch_projection_gate as gate


def test_batch_gate_uses_streaming_bank_interface_and_restores_route_order(monkeypatch, tmp_path):
    plan = SimpleNamespace(authenticate=lambda: None, by_layer={3: tuple(range(72))}, manifest={"activation_bits": 4})
    monkeypatch.setattr(gate, "NativeResidentPlan", lambda *args: plan)
    monkeypatch.setattr(gate, "digest_file", lambda path: "a" * 64)
    monkeypatch.setattr(gate, "LAYERS", (3,))
    monkeypatch.setattr(gate, "BATCH_TOKENS", (2,))
    monkeypatch.setattr(gate, "ROUTES_PER_TOKEN", (2,))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    closed = []

    class Oracle:
        def __init__(self, banks, specification, routes):
            assert banks is plan
            self.library = specification["library"]

        def compute_bytes(self, layer, body, experts, bits):
            inputs = np.frombuffer(body, dtype="<f2").reshape(len(experts), 4096)
            return np.add(inputs, np.asarray(experts, dtype="<f2")[:, None], dtype=np.float16).tobytes()

        def close(self):
            closed.append(self.library)

    monkeypatch.setattr(gate, "CudaOracle", Oracle)
    report = gate.run(tmp_path, "a" * 64, tmp_path / "old", tmp_path / "new", tmp_path / "report.json")
    assert report["passed"]
    assert report["cases"][0]["bitwise_equal"]
    assert report["cases"][0]["routes"] == 4
    assert closed == [str(tmp_path / "new"), str(tmp_path / "old")]
