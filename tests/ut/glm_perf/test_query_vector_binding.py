# SPDX-License-Identifier: Apache-2.0
"""The complete loaded target/draft query path uses reversible vector casts."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_candidates.query_vector import extend_replacements
from tools.glm_perf.resident_rpc_guard import WORKER_PREFIX

SOURCE = "class Indexer:\n    def forward(self, q):\n        q = q.to(torch.bfloat16)\n        return q\n"


class Indexer:
    n_head = 2
    head_dim = 4
    topk_indices_buffer = torch.empty(640, 4)
    indexer_op = object()

    def _native_bf16_cast(self, value, dtype):
        return value.to(dtype)

    def forward(self, q):
        return q


class Vector:
    def __init__(self):
        self.prepared = set()
        self.calls = []

    def prepare_counts(self, counts):
        self.prepared.update(counts)

    def __call__(self, value):
        assert value.is_contiguous() and value.numel() in self.prepared
        self.calls.append(value.numel())
        return value.bfloat16()


@pytest.fixture
def binding():
    target, draft, native = Indexer(), Indexer(), Vector()
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [target]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: [draft])),
    )
    session = SimpleNamespace(graphs_dirty=False)
    worker = SimpleNamespace(
        model_runner=runner,
        _resident_error=lambda error: {"error": str(error)},
        _resident_session=lambda: session,
    )
    changes = {WORKER_PREFIX + "resident_status": lambda self: {"rank": 0}}
    return target, draft, native, worker, changes, session


def test_main_draft_decode_and_large_prefill_use_vector_and_restore_permanent_converter(binding):
    target, draft, native, worker, changes, _ = binding
    original = target.forward.__func__
    changes = extend_replacements(
        changes,
        native,
        SOURCE,
        capture_hook=lambda self: {"captured": True},
        apply_hook=lambda self, generation: generation,
    )
    assert changes[WORKER_PREFIX + "resident_capture"](worker) == {"captured": True}
    for module in (target, draft):
        for rows in (2, 8, 640):
            value = torch.arange(rows * 8, dtype=torch.float16).reshape(8, rows).t()
            assert torch.equal(module.forward(value), value.bfloat16())
        value = torch.randn(2, 8, dtype=torch.float32)
        assert torch.equal(module.forward(value), value.bfloat16())
    receipt = changes[WORKER_PREFIX + "resident_status"](worker)["vector_query_cast"]
    assert receipt["indexers"] == 2 and receipt["methods"] == 2 and receipt["native_calls"] == 6
    assert receipt["legacy_calls"] == 2
    assert changes[WORKER_PREFIX + "resident_apply"](worker, 19) == 19
    assert target.forward.__func__ is original and draft.forward.__func__ is original
    assert "_native_bf16_cast" not in target.__dict__


@pytest.mark.parametrize("failure", ["exception", "receipt", "rebind"])
def test_capture_failure_restores_every_forward_and_reports_error(binding, failure):
    target, draft, native, worker, changes, session = binding
    original = target.forward.__func__

    def capture(self):
        if failure == "exception":
            raise RuntimeError("capture failed")
        if failure == "receipt":
            return {"error": "parent failed"}
        target.forward = original.__get__(target)
        return {"captured": True}

    changes = extend_replacements(
        changes, native, SOURCE, capture_hook=capture, apply_hook=lambda self, generation: generation
    )
    assert "error" in changes[WORKER_PREFIX + "resident_capture"](worker)
    assert target.forward.__func__ is original and draft.forward.__func__ is original
    assert not {"forward"} & target.__dict__.keys()
    if failure != "receipt":
        assert session.graphs_dirty
