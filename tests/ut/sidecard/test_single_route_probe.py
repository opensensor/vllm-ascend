# SPDX-License-Identifier: Apache-2.0
"""Reject unsafe device scope and unauthenticated model references offline."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from tools.sidecard.single_route_benchmark import run as benchmark
from tools.sidecard.single_route_probe import run as probe


@pytest.mark.parametrize("rank,visible", [(0, "0,1"), (1, "2"), (2, "2,3"), (True, "2,3")])
def test_probe_rejects_other_card_before_device_import(tmp_path, monkeypatch, rank, visible):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="physical chips"):
        probe(tmp_path / "absent.json", tmp_path / "output", rank)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("error", ["signature", "reference", "snapshots", "cuda", "npu"])
def test_benchmark_rejects_unqualified_modes_before_allocating(tmp_path, monkeypatch, error):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object()))
    config, reference = tmp_path / "runtime.json", tmp_path / "reference.json"
    config.write_text(
        json.dumps(
            dict(
                prompt_prefix_cache=error == "snapshots",
                cuda_indexer_hadamard=error != "cuda",
                direct_dma=dict(npu_expert_graphs=True, npu_single_route_graphs=error != "npu"),
            )
        )
    )
    reference.write_text(json.dumps(dict(passed=error != "reference")))
    signature = hashlib.sha256(reference.read_bytes()).hexdigest() if error != "signature" else "0" * 64
    with pytest.raises(ValueError):
        benchmark(config, tmp_path / "private.json", reference, signature)
