# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.dsa_graph_gate import validate


@pytest.mark.parametrize("fail", [False, True])
def test_parity_compares_same_teacher_forced_inputs_and_restores_graph_owners(fail):
    stages = tuple(SimpleNamespace(dsa_graphs={index: object()}) for index in (3, 19, 35))
    saved = tuple(stage.dsa_graphs for stage in stages)
    seen = []
    engine = SimpleNamespace(
        prefix_graphs=stages[0],
        middle_graphs=stages[1],
        suffix_graphs=stages[2],
        max_context=4096,
        config={"index_topk": 2048},
        fingerprints=None,
    )

    def reset(**kwargs):
        assert kwargs == {"context_limit": 2056}
        engine.position = 0

    def forward(tokens):
        seen.append(tuple(tokens))
        captured = bool(stages[0].dsa_graphs)
        engine.fingerprints.append(dict(position=engine.position, hash="changed" if captured and fail else "same"))
        engine.position += len(tokens)
        return torch.tensor([0.0, 1.0])

    engine.reset, engine.prefill, engine.forward = reset, forward, forward
    if fail:
        with pytest.raises(AssertionError, match="full-vocabulary"):
            validate(engine, [1, 2, 3])
    else:
        report = validate(engine, [1, 2, 3])
        assert report["passed"] and report["sparse_indexer_exercised"]
        assert report["captured_layers"] == [3, 19, 35]
    assert seen[:8] == seen[8:]
    assert engine.fingerprints is None
    assert all(stage.dsa_graphs is graphs for stage, graphs in zip(stages, saved, strict=True))
