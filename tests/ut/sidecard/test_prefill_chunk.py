# SPDX-License-Identifier: Apache-2.0
import struct
from types import SimpleNamespace

import pytest

from tools.sidecard.dma_protocol import CAPACITY, COMMAND, OUTPUT_ROW_BYTES, validate_command
from tools.sidecard.dma_worker import validate_routes
from tools.sidecard.expert_actor import ExpertCoordinator
from tools.sidecard.layer_weights import MAX_PREFILL_CHUNK, prefill_chunk_size


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_prefill_capacity_uses_existing_slots_and_checks_complete_metadata(rows):
    assert prefill_chunk_size({"expert_transport": "direct-dma", "prefill_chunk_size": rows}) == rows
    assert rows * OUTPUT_ROW_BYTES <= CAPACITY
    assert validate_command(COMMAND.pack(0, 18, rows), 0, max_chunk=rows) == (0, 18, rows)
    metadata = struct.pack(f"<{rows * 8}i{rows * 8}f", *([2] * (rows * 8)), *([0.5] * (rows * 8)))
    validate_routes(metadata, rows, frozenset(), max_chunk=rows)
    with pytest.raises(ValueError, match="row capacity"):
        validate_routes(metadata, rows + 1, frozenset(), max_chunk=rows)
    with pytest.raises(ValueError, match="identity"):
        validate_command(COMMAND.pack(1, 18, rows + 1), 1, max_chunk=rows)


@pytest.mark.parametrize("value", [0, 16, 33, 129, 256, True, 64.0, "128", None])
def test_prefill_rejects_invalid_configuration_before_allocation(value):
    with pytest.raises(ValueError, match="chunk size"):
        prefill_chunk_size({"expert_transport": "direct-dma", "prefill_chunk_size": value})


def test_socket_default_stays_32_rows_and_larger_batches_require_dma():
    assert prefill_chunk_size({}) == 32
    assert MAX_PREFILL_CHUNK * OUTPUT_ROW_BYTES == CAPACITY
    with pytest.raises(ValueError, match="direct DMA"):
        prefill_chunk_size({"prefill_chunk_size": 64})


@pytest.mark.parametrize("chunk", [32, 64, 128])
def test_coordinator_prefill_keeps_tail_and_checks_cancellation_between_chunks(chunk):
    calls = []
    engine = object.__new__(ExpertCoordinator)
    engine.prefill_chunk_size = chunk
    engine.forward = lambda tokens: calls.append(list(tokens)) or len(tokens)
    tokens = list(range(chunk * 2 + 7))
    assert engine.prefill(tokens) == 7
    assert [len(part) for part in calls] == [chunk, chunk, 7]
    assert sum(calls, []) == tokens
    cancelled = SimpleNamespace(is_set=lambda: len(calls) == 4)
    with pytest.raises(InterruptedError, match="cancelled"):
        engine.prefill(tokens, cancelled=cancelled)
    assert len(calls) == 4
