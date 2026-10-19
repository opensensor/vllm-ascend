# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from tools.dflash_research.qfn_dspark_vocab_shard import canonical_input, shard_bounds, shard_weight
from vllm_ascend._310p.qwen4exp_mtp import greedy_tokens_from_local_logits


def test_qfn_vocabulary_partition_has_no_gaps_or_overlap():
    bounds = [shard_bounds(248320, rank, 6) for rank in range(6)]
    assert bounds[0][0] == 0 and bounds[-1][1] == 248320
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert len({width for _, _, width in bounds}) == 1
    assert all(width % 64 == 0 for _, _, width in bounds)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("draft_k", [2, 7])
def test_partitioned_markov_proposals_match_full_vocabulary(batch, draft_k):
    generator = torch.Generator().manual_seed(1941)
    vocabulary, hidden, markov_width, world = 997, 16, 8, 6
    weight = torch.randn(vocabulary, hidden, generator=generator)
    markov = torch.randn(vocabulary, markov_width, generator=generator)
    embeddings = torch.randn(vocabulary, markov_width, generator=generator)
    states = torch.randn(batch, draft_k, hidden, generator=generator)
    previous = torch.arange(batch)
    base = F.linear(states, weight)
    for step in range(draft_k):
        bias = F.linear(embeddings[previous], markov)
        expected = (base[:, step] + bias).argmax(-1)
        locals_, pairs = [], []
        for rank in range(world):
            start, end, width = shard_bounds(vocabulary, rank, world)
            logits = F.linear(states[:, step], shard_weight(weight, rank, world))
            logits += F.linear(embeddings[previous], shard_weight(markov, rank, world))
            padding = width - (end - start)
            if padding:
                logits[:, -padding:] = -torch.inf
            values, ids = logits.max(-1)
            pairs.append(torch.stack([values, (ids + start).float()], -1))
            locals_.append((logits, start, padding))
        gathered = torch.cat(pairs, -1)
        for logits, start, padding in locals_:
            actual = greedy_tokens_from_local_logits(logits, start, padding, lambda _, dim, gathered=gathered: gathered)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        previous = expected


def test_padding_cannot_win_even_when_all_valid_scores_are_negative():
    vocabulary, world = 701, 6
    bounds = [shard_bounds(vocabulary, rank, world) for rank in range(world)]
    pairs, shards = [], []
    for start, end, width in bounds:
        padding = width - (end - start)
        logits = torch.full((1, width), -2.0)
        if padding:
            logits[:, -padding:] = 100.0
        masked = logits.clone()
        if padding:
            masked[:, -padding:] = -torch.inf
        values, ids = masked.max(-1)
        pairs.append(torch.stack([values, (ids + start).float()], -1))
        shards.append((logits, start, padding))
    for logits, start, padding in shards:
        actual = greedy_tokens_from_local_logits(logits, start, padding, lambda _, dim: torch.cat(pairs, -1))
        assert actual.item() == 0  # first vocabulary ID wins ties across ranks


@pytest.mark.parametrize("args", [(0, 0, 6), (128, 6, 6), (128, -1, 6), (16, 5, 6)])
def test_invalid_or_empty_partition_is_rejected(args):
    with pytest.raises(ValueError):
        shard_bounds(*args)


def test_canonical_input_owns_storage_and_preserves_replica_input():
    original = torch.arange(12.0).reshape(3, 4)
    before = original.clone()
    source = original + 0.125

    class Group:
        def broadcast(self, value, src):
            assert src == 0
            assert value.data_ptr() != original.data_ptr()
            value.copy_(source)
            return value

    actual = canonical_input(original, Group())
    torch.testing.assert_close(actual, source, rtol=0, atol=0)
    torch.testing.assert_close(original, before, rtol=0, atol=0)
