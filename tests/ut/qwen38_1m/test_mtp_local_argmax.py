# SPDX-License-Identifier: Apache-2.0
"""Greedy distributed drafts must match the complete vocabulary argmax."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/qwen4exp_mtp.py"
spec = importlib.util.spec_from_file_location("qwen_argmax_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 6, 18])
def test_six_shards_match_full_argmax_with_ties_padding_and_large_ids(dtype, rows):
    generator = torch.Generator().manual_seed(17)
    shards = [torch.randn(rows, 11, generator=generator).to(dtype) for _ in range(6)]
    # Tie within and across ranks: the lowest global ID must win.
    for logits in shards:
        logits[0].fill_(-3)
    shards[1][0, 2:4] = 4
    shards[4][0, 1] = 4
    padding = [0, 0, 0, 0, 0, 3]
    shards[-1][:, -3:] = 60000
    starts = [100001 + 11 * rank for rank in range(6)]
    pairs = []
    for logits, start, pad in zip(shards, starts, padding):
        valid = logits[:, : logits.shape[1] - pad]
        values, ids = valid.max(dim=-1)
        pairs.append(torch.stack((values.float(), (ids + start).float()), dim=-1))
    gathered = torch.cat(pairs, dim=-1)
    full = torch.cat([logits[:, : 11 - pad] for logits, pad in zip(shards, padding)], dim=-1)
    expected = full.argmax(dim=-1) + starts[0]
    for rank in range(6):
        # Strided input also exercises the local reduction contract.
        storage = torch.empty(rows, 22, dtype=dtype)
        storage[:, ::2] = shards[rank]
        logits = storage[:, ::2]

        def gather(actual, dim, rank=rank):
            assert dim == -1
            torch.testing.assert_close(actual, pairs[rank], rtol=0, atol=0)
            assert actual.numel() == rows * 2
            return gathered

        actual = module.greedy_tokens_from_local_logits(logits, starts[rank], padding[rank], gather)
        assert actual.dtype == torch.int64
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("start,padding", [(-1, 0), (0, -1), (0, 5), (1 << 24, 0)])
def test_invalid_shard_contract_rejected_before_collective(start, padding):
    with pytest.raises(ValueError):
        module.greedy_tokens_from_local_logits(torch.zeros(1, 4), start, padding, lambda **kwargs: pytest.fail())


def make_proposer(**changes):
    logits = torch.tensor([[0.5, 2.0, 60000.0]])
    head = SimpleNamespace(
        num_added_embeddings=0,
        shard_indices=SimpleNamespace(org_vocab_start_index=70000, num_org_vocab_padding=1),
    )

    class Processor:
        org_vocab_size = 100000

        def __call__(self, actual_head, hidden, *, skip_gather):
            assert actual_head is head
            assert skip_gather
            return logits.clone()

    model = SimpleNamespace(lm_head=head, logits_processor=Processor(), draft_id_to_target_id=None)
    proposer = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen4ExpMTP"])),
        _enable_probabilistic_draft_probs=False,
        use_heterogeneous_vocab=False,
        model=SimpleNamespace(unwrap=lambda: model),
    )
    for key, value in changes.items():
        setattr(proposer, key, value)
    return proposer, model


def test_eligible_wrapped_model_uses_local_projection():
    proposer, _ = make_proposer()
    result = module.try_qwen4exp_draft_argmax(
        proposer, torch.zeros(1, 2), lmhead_tp_enabled=False, all_gather=lambda pair, dim: pair
    )
    assert result.tolist() == [70001]


@pytest.mark.parametrize(
    "reason", ["method", "architecture", "probabilistic", "heterogeneous", "lmhead_tp", "added", "remap", "large"]
)
def test_ineligible_models_keep_existing_logits_path(reason):
    proposer, model = make_proposer()
    if reason == "method":
        proposer.method = "other"
    elif reason == "architecture":
        proposer.draft_model_config.hf_config.architectures = ["OtherMTP"]
    elif reason == "probabilistic":
        proposer._enable_probabilistic_draft_probs = True
    elif reason == "heterogeneous":
        proposer.use_heterogeneous_vocab = True
    elif reason == "added":
        model.lm_head.num_added_embeddings = 3
    elif reason == "remap":
        model.draft_id_to_target_id = torch.tensor([1, 0])
    elif reason == "large":
        model.logits_processor.org_vocab_size = (1 << 24) + 1
    result = module.try_qwen4exp_draft_argmax(
        proposer,
        torch.zeros(1, 2),
        lmhead_tp_enabled=reason == "lmhead_tp",
        all_gather=lambda *args, **kwargs: pytest.fail("fallback must not enter the compact collective"),
    )
    assert result is None


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_reduction_rejects_dtypes_that_cannot_preserve_score_order(dtype):
    with pytest.raises(ValueError):
        module.greedy_tokens_from_local_logits(torch.zeros(1, 4, dtype=dtype), 0, 0, lambda **kwargs: pytest.fail())


def test_entirely_padded_rank_cannot_win_over_a_valid_vocabulary():
    valid = torch.tensor([[1.0, 0.0], [-float("inf"), -float("inf")]])
    padded = torch.full((2, 2), 60000.0)
    expected_pairs = torch.tensor([[1.0, 0.0, -float("inf"), 2.0], [-float("inf"), 0.0, -float("inf"), 2.0]])
    actual = module.greedy_tokens_from_local_logits(padded, 2, 2, lambda pairs, dim: expected_pairs)
    assert actual.tolist() == [0, 0]
    assert torch.isneginf(padded).all()
    assert valid.argmax(dim=-1).tolist() == [0, 0]
