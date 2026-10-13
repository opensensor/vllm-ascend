# SPDX-License-Identifier: Apache-2.0
"""Exclusive real-weight TP2 block benchmark and socket/DMA vocabulary gate."""

import json
from pathlib import Path

import pytest


def test_complete_layer_socket_dma_equivalence(request):
    paths = request.config.getoption("--sidecard-stage-specs")
    if paths is None:
        pytest.skip("requires explicit fresh specifications and exclusive ownership of the assigned accelerators")
    # Device imports belong to the process that owns this explicitly requested run.
    from tools.sidecard.stage_benchmark import run

    socket_path, dma_path = map(Path, paths)
    socket_spec, dma_spec = [json.loads(p.read_bytes()) for p in (socket_path, dma_path)]
    assert socket_spec.get("stage_transport", "socket") == "socket"
    assert dma_spec["stage_transport"] == "direct-dma"
    assert socket_spec["checkpoint_sha256"] == dma_spec["checkpoint_sha256"]
    assert run(socket_path)["passed"]
    reference = Path(socket_spec["output"]) / "stage-benchmark.json"
    assert run(dma_path, reference)["passed"]
    for rank in (0, 1):
        receipt = json.loads((Path(dma_spec["output"]) / f"stage-dma-rank{rank}.json").read_bytes())
        assert receipt["state"] == "closed" and receipt["guard_poison_exact"]
        assert receipt["payload_host_staging"] is False and receipt["calls"] > 0
