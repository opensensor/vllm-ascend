# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.prefill_gate import validate


@pytest.mark.parametrize("fail", [False, True])
def test_prefill_parity_restores_choices_and_records_numerical_failure(tmp_path, fail):
    layers = {0: SimpleNamespace(vectorized_prefill_conv=True)}
    engine = SimpleNamespace(layers=layers, prefill_chunk_size=32, max_context=4096, fingerprints=None)
    seen = []

    def reset(**kwargs):
        assert kwargs == {"context_limit": 2112}
        engine.position = 0

    def forward(tokens):
        seen.append(tuple(tokens))
        changed = fail and layers[0].vectorized_prefill_conv
        engine.fingerprints.append(dict(position=engine.position, logits_sha256="changed" if changed else "same"))
        engine.position += len(tokens)
        return torch.tensor([0.0, 1.0 + int(changed)])

    engine.reset, engine.prefill, engine.forward = reset, forward, forward
    path = tmp_path / "gate.json"
    if fail:
        with pytest.raises(AssertionError, match="full-vocabulary"):
            validate(engine, [1, 2, 3], output=path)
    else:
        report = validate(engine, [1, 2, 3], output=path)
        assert report["passed"] and report["all_prefill_boundaries_compared"]
    assert path.exists()
    assert engine.prefill_chunk_size == 32 and layers[0].vectorized_prefill_conv
    assert engine.fingerprints is None
    assert seen[:64] == seen[64:]


@pytest.mark.parametrize("chunk", [64, 128])
def test_larger_chunks_compare_final_prefill_and_every_decode_logit(tmp_path, chunk):
    engine = SimpleNamespace(
        layers={0: SimpleNamespace(vectorized_prefill_conv=True)},
        prefill_chunk_size=chunk,
        max_context=4096,
        fingerprints=None,
    )

    def reset(**kwargs):
        engine.position = 0

    def forward(tokens):
        engine.position += len(tokens)
        engine.fingerprints.append(dict(position=engine.position, logits_sha256=str(engine.position)))
        return torch.tensor([0.0, 1.0])

    def prefill(tokens):
        for offset in range(0, len(tokens), engine.prefill_chunk_size):
            logits = forward(tokens[offset : offset + engine.prefill_chunk_size])
        return logits

    engine.reset, engine.prefill, engine.forward = reset, prefill, forward
    report = validate(engine, [1, 2], output=tmp_path / "gate.json")
    assert report["passed"] and not report["all_prefill_boundaries_compared"]
    assert len(report["traces"][0]) == 64 + 63
    assert len(report["traces"][1]) == 2048 // chunk + 63
    assert engine.prefill_chunk_size == chunk and engine.fingerprints is None
