# SPDX-License-Identifier: Apache-2.0
"""Benchmark switches restore owners and full-vector checks reject failures."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.indexer_benchmark import OUTPUT_TOKENS, decode, rotation_mode, run


@pytest.mark.parametrize("fail", [False, True])
def test_rotation_mode_restores_each_owner_after_success_or_failure(fail):
    indexers = [SimpleNamespace(fused_cuda_rotation=value) for value in (True, False)]
    engine = SimpleNamespace(
        layers={
            0: SimpleNamespace(linear=True),
            1: SimpleNamespace(linear=False, indexer=indexers[0]),
            2: SimpleNamespace(linear=False, indexer=indexers[1]),
        }
    )

    def run():
        with rotation_mode(engine, False):
            assert all(not x.fused_cuda_rotation for x in indexers)
            if fail:
                raise RuntimeError("failed probe")

    if fail:
        with pytest.raises(RuntimeError, match="failed probe"):
            run()
    else:
        run()
    assert [x.fused_cuda_rotation for x in indexers] == [True, False]


def test_decode_checks_every_vocabulary_and_uses_exact_teacher_tokens():
    calls = []
    forced = list(range(OUTPUT_TOKENS))
    logits = torch.tensor([1.0, -2.0, 3.0])

    def forward(tokens):
        calls.append(tokens)
        return logits

    engine = SimpleNamespace(forward=forward)
    hashes, outputs, spans = decode(engine, logits, forced=forced)
    assert len(hashes) == len(outputs) == OUTPUT_TOKENS
    assert len(spans) == OUTPUT_TOKENS - 1
    assert outputs == forced
    assert calls == [[token] for token in forced[:-1]]
    assert len(set(hashes)) == 1
    with pytest.raises(AssertionError, match="non-finite"):
        decode(engine, torch.tensor([1.0, float("nan"), 3.0]))


@pytest.mark.parametrize("error", ["signature", "failed_reference", "snapshots", "disabled"])
def test_benchmark_rejects_unqualified_reference_and_mode_before_allocation(tmp_path, monkeypatch, error):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object()))
    config, reference = tmp_path / "runtime.json", tmp_path / "reference.json"
    config.write_text(
        json.dumps(dict(prompt_prefix_cache=error == "snapshots", cuda_indexer_hadamard=error != "disabled"))
    )
    reference.write_text(json.dumps(dict(passed=error != "failed_reference")))
    signature = hashlib.sha256(reference.read_bytes()).hexdigest() if error != "signature" else "0" * 64
    with pytest.raises(ValueError):
        run(config, tmp_path / "missing-private-request.json", reference, signature)
