# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
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


@pytest.mark.parametrize("change", [None, "logit", "token", "auth"])
def test_authenticated_reference_detects_shared_worker_arithmetic_changes(tmp_path, change):
    engine = SimpleNamespace(
        layers={0: SimpleNamespace(vectorized_prefill_conv=True)},
        prefill_chunk_size=32,
        max_context=4096,
        fingerprints=None,
    )

    def forward(tokens):
        engine.fingerprints.append(dict(logits_sha256="same"))
        return torch.tensor([0.0, 1.0])

    engine.reset = lambda **kwargs: None
    engine.prefill = engine.forward = forward
    path = tmp_path / "reference.json"
    reference = validate(engine, [1, 2], output=path)
    if change == "logit":
        reference["traces"][1][0]["logits_sha256"] = "changed"
    if change == "token":
        reference["teacher_forced_tokens"][0] = 0
    path.write_text(json.dumps(reference))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if change == "auth":
        digest = "wrong"
    arguments = dict(reference_path=path, reference_sha256=digest)
    if change is None:
        assert validate(engine, [1, 2], **arguments)["reference_bitwise_equal"]
    else:
        error = ValueError if change == "auth" else AssertionError
        with pytest.raises(error):
            validate(engine, [1, 2], **arguments)
    assert engine.fingerprints is None and engine.layers[0].vectorized_prefill_conv


@pytest.mark.parametrize("fail", [False, True])
def test_cold_window_gate_disables_batching_for_baseline_and_restores_it(tmp_path, fail):
    engine = SimpleNamespace(
        layers={0: SimpleNamespace(vectorized_prefill_conv=True)},
        prefill_chunk_size=32,
        prefill_window_size=64,
        max_context=4096,
        fingerprints=None,
    )
    seen = []

    def forward(tokens):
        changed = fail and engine.prefill_window_size == 64
        engine.fingerprints.append(dict(logits_sha256="changed" if changed else "same"))
        return torch.tensor([0.0, 1.0 + int(changed)])

    def prefill(tokens):
        seen.append(engine.prefill_window_size)
        return forward(tokens)

    engine.reset = lambda **kwargs: None
    engine.prefill, engine.forward = prefill, forward
    path = tmp_path / "gate.json"
    if fail:
        with pytest.raises(AssertionError, match="full-vocabulary"):
            validate(engine, [1, 2], output=path)
    else:
        assert validate(engine, [1, 2], output=path)["candidate_window_rows"] == 64
    assert seen == [32, 64]
    assert engine.prefill_window_size == 64
    assert engine.fingerprints is None and engine.layers[0].vectorized_prefill_conv
