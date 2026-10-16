# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported gates before process-group or device initialization."""

from pathlib import Path

import pytest

from tools.glm_perf.probe_router_owner import run


@pytest.mark.parametrize("backend,tp,graphs", [("bad", 4, False), ("hccl", 6, False), ("gloo", 4, True)])
def test_router_probe_rejects_unsupported_backend_topology_and_capture(backend, tp, graphs):
    with pytest.raises(ValueError, match="TP2/TP4"):
        run(backend, tp, Path("/unused"), graphs)
