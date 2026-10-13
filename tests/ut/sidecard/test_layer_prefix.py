# SPDX-License-Identifier: Apache-2.0
"""Recurrent rollback, append-only KV, geometry and failed-prefix isolation."""

import threading
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.expert_actor import ExpertCoordinator
from tools.sidecard.layer_prefix import PromptPrefix
from tools.sidecard.layer_server import Generation, prefix_boundary


class Engine:
    def __init__(self):
        self.config, self.spec = {"index_topk": 8}, {}
        self.prefill_chunk_size = 4
        self.position, self.processed, self.fail_at = 0, [], None
        self.layers = {
            0: SimpleNamespace(linear=True, state=torch.zeros(1), conv=torch.zeros(1), length=0),
            1: SimpleNamespace(
                linear=False,
                length=0,
                cache=torch.zeros(64),
                indexer=SimpleNamespace(
                    pool_count=0,
                    tail_length=0,
                    tail_keys=torch.zeros(1),
                    tail_gates=torch.zeros(1),
                    pools=torch.zeros(16),
                ),
            ),
        }

    def reset(self, context_limit=None):
        self.position = 0
        for layer in self.layers.values():
            layer.length = 0
        self.layers[0].state.zero_()
        self.layers[0].conv.zero_()
        indexer = self.layers[1].indexer
        indexer.pool_count = indexer.tail_length = 0

    def prefill(self, tokens, **kwargs):
        if kwargs.get("cancelled"):
            raise InterruptedError("cancelled")
        for token in tokens:
            if self.position == self.fail_at:
                raise RuntimeError("failed forward")
            self.processed.append(token)
            linear, dsa = self.layers.values()
            linear.state.mul_(2).add_(token)
            linear.conv.fill_(token)
            dsa.cache[self.position] = linear.state
            self.position += 1
            dsa.length = self.position
            indexer = dsa.indexer
            indexer.pool_count, indexer.tail_length = divmod(self.position, 4)
            if indexer.tail_length == 0:
                indexer.pools[indexer.pool_count - 1] = linear.state
            indexer.tail_keys.fill_(token)
            indexer.tail_gates.fill_(self.position)
        return linear.state.clone() + dsa.cache[: self.position].sum() + indexer.tail_gates


def test_reuse_restores_recurrent_and_indexer_state_after_decode():
    engine = Engine()
    cache = PromptPrefix(engine)
    prompt = [1, 2, 3, 4, 5]
    cache.prefill(prompt, 12)
    engine.prefill([9, 8, 7])  # Mutate all attention state during decode.
    engine.processed = []
    extended = prompt + [6, 7]
    actual, cached = cache.prefill(extended, 16)
    reference, _ = PromptPrefix(Engine()).prefill(extended, 16)
    assert torch.equal(actual, reference)
    assert cached == 4 and engine.processed == [5, 6, 7]
    assert engine.position == 7


def test_identical_aligned_prompt_reuses_saved_logits():
    engine = Engine()
    cache = PromptPrefix(engine)
    logits, _ = cache.prefill([1, 2, 3, 4], 12)
    engine.prefill([99])
    engine.processed = []
    actual, cached = cache.prefill([1, 2, 3, 4], 12)
    assert torch.equal(actual, logits) and cached == 4 and not engine.processed


@pytest.mark.parametrize("tokens,limit", [([1, 2, 9, 4, 5], 12), ([1, 2, 3], 12), ([1, 2, 3, 4], 8)])
def test_changed_prefix_shorter_prompt_and_sparse_mode_force_cold(tokens, limit):
    engine = Engine()
    cache = PromptPrefix(engine)
    cache.prefill([1, 2, 3, 4, 5], 12)
    engine.processed = []
    _, cached = cache.prefill(tokens, limit)
    assert cached == 0 and engine.processed == tokens


def test_snapshot_keeps_graph_tensor_addresses():
    engine = Engine()
    cache = PromptPrefix(engine)
    addresses = [engine.layers[0].state.data_ptr(), engine.layers[0].conv.data_ptr()]
    cache.prefill([1, 2, 3, 4], 12)
    engine.prefill([5])
    cache.prefill([1, 2, 3, 4, 6], 12)
    assert addresses == [engine.layers[0].state.data_ptr(), engine.layers[0].conv.data_ptr()]


@pytest.mark.parametrize("cancelled", [True, False])
def test_failed_or_cancelled_prefill_invalidates_entry(cancelled):
    engine = Engine()
    cache = PromptPrefix(engine)
    cache.prefill([1, 2, 3, 4], 12)
    engine.fail_at = None if cancelled else 4
    with pytest.raises(InterruptedError if cancelled else RuntimeError):
        cache.prefill([1, 2, 3, 4, 5], 12, cancelled=cancelled)
    assert cache.entry is None


def test_short_prompt_does_not_retain_a_partial_chunk():
    engine = Engine()
    cache = PromptPrefix(engine)
    _, cached = cache.prefill([1, 2], 4)
    assert cached == 0 and cache.entry is None


@pytest.mark.parametrize("queued", [False, True])
def test_cancelled_request_preserves_previous_live_prefix(queued):
    engine = Engine()
    engine.architecture = ExpertCoordinator.architecture
    engine.spec["prompt_prefix_cache"] = True
    engine.config["eos_token_id"] = [0]
    generation = Generation(engine, None)
    generation.prefix.prefill([1, 2, 3, 4], 12)
    previous = generation.prefix.entry
    cancelled, started = threading.Event(), threading.Event()
    results = []

    def run():
        started.set()
        results.extend(generation.generate([1, 2, 3, 4, 5], 1, cancelled=cancelled))

    if queued:
        generation.lock.acquire()
    else:
        cancelled.set()
    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(1)
    cancelled.set()
    if queued:
        generation.lock.release()
    worker.join(1)
    assert not worker.is_alive() and not results
    assert generation.prefix.entry is previous and engine.position == 4


def test_interleaved_request_restores_independent_kv_and_pools():
    engine = Engine()
    cache = PromptPrefix(engine)
    cache.prefill([1, 2, 3, 4], 12)
    cache.prefill([9, 9, 9, 9], 12)
    actual, cached = cache.prefill([1, 2, 3, 4, 5], 12)
    reference_engine = Engine()
    reference, _ = PromptPrefix(reference_engine).prefill([1, 2, 3, 4, 5], 12)
    assert cached == 4 and torch.equal(actual, reference)
    assert torch.equal(engine.layers[1].indexer.pools[:1], reference_engine.layers[1].indexer.pools[:1])


def test_bounded_cache_evicts_oldest_equally_sized_prefix():
    cache = PromptPrefix(Engine())
    for value in (1, 2, 3):
        cache.prefill([value] * 4, 12)
    assert len(cache.entries) == 2
    _, cached = cache.prefill([1] * 4, 12)
    assert cached == 0


def test_context_boundary_reuses_fixed_prompt_for_a_changed_new_message():
    cache = PromptPrefix(Engine())
    cache.prefill([1, 2, 3, 4, 5, 6, 7], 12, boundary=4)
    actual, cached = cache.prefill([1, 2, 3, 4, 9, 8, 7], 12, boundary=4)
    reference, _ = PromptPrefix(Engine()).prefill([1, 2, 3, 4, 9, 8, 7], 12)
    assert cached == 4 and torch.equal(actual, reference)


@pytest.mark.parametrize("boundary", [-1, 3, 8, True])
def test_invalid_boundaries_are_rejected(boundary):
    with pytest.raises(ValueError, match="boundary"):
        PromptPrefix(Engine()).prefill([1, 2, 3, 4], 12, boundary=boundary)


@pytest.mark.parametrize("prefix,expected", [([1, 2, 3, 4, 5], 4), ([1, 2, 9], 0)])
def test_chat_boundary_checks_exact_template_prefix(prefix, expected):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "system", "content": "Rules"}]
            assert kwargs["add_generation_prompt"] is False
            return {"input_ids": prefix}

    body = {"messages": [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Task"}]}
    assert prefix_boundary(body, [1, 2, 3, 4, 5, 6, 7], Tokenizer(), 4) == expected


def test_snapshot_byte_budget_evicts_oversized_entries(monkeypatch):
    monkeypatch.setattr("tools.sidecard.layer_prefix.MAX_PREFIX_BYTES", 1)
    cache = PromptPrefix(Engine())
    cache.prefill([1, 2, 3, 4], 12)
    assert not cache.entries
    _, cached = cache.prefill([1, 2, 3, 4], 12)
    assert cached == 0


def test_single_message_uses_the_regular_chunk_boundary():
    assert prefix_boundary({"messages": [{"role": "user", "content": "Hello"}]}, [1, 2], None, 4) is None


def test_short_background_prompt_preserves_expensive_contexts():
    cache = PromptPrefix(Engine())
    cache.prefill([1] * 8, 16)
    cache.prefill([2] * 8, 16)
    cache.prefill([3] * 4, 16)
    assert sorted(len(entry[0]) for entry in cache.entries) == [8, 8]
    _, cached = cache.prefill([1] * 8 + [4], 16)
    assert cached == 8
