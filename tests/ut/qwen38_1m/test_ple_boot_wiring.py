# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot wiring for the lazy-shard PLE transport + vocab-parallel load (#1/#2).

Covers two pieces of model-boot plumbing:

1. ``_PLEInjection`` builds the lazy per-shard mmap transport (transport c) when
   a ``checkpoint_dir`` is supplied, and keeps the synthetic stub otherwise; the
   real-config forward gathers rows lazily and stays finite.
2. ``load_weights`` routes the vocab-parallel ``embed_tokens`` / ``lm_head``
   through their ``weight_loader`` (dim-0 TP shard) instead of the strict
   full-shape copy, which would otherwise silently skip them on TP>1.

Run with ``--noconftest``:

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_ple_boot_wiring.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.qwen38_1m.test_model_load_and_moe import (
    _tiny_moe_config,
    _vllm_config,
)
from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.model import (
    AscendQwen4ExpForCausalLM,
    _PLEInjection,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    AscendPLELazyShardEmbeddingMethod,
    AscendPLEPinnedHostEmbeddingMethod,
)

_CKPT = Path(
    "/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/models/ascend/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i"
)


def _real_text_config() -> SimpleNamespace:
    cfg_path = _CKPT / "config.json"
    if not cfg_path.exists():
        pytest.skip(f"checkpoint config not mounted at {cfg_path}")
    return SimpleNamespace(**json.loads(cfg_path.read_text())["text_config"])


# --------------------------------------------------------------------------- #
# 1. lazy transport wiring in _PLEInjection
# --------------------------------------------------------------------------- #
def test_ple_injection_uses_stub_without_checkpoint_dir():
    cfg = _tiny_moe_config(num_layers=1, num_experts=8, top_k=3)
    inj = _PLEInjection(config=cfg, layer_idx=0, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY)
    inj._ensure_ple_method(torch.device("cpu"))
    assert isinstance(inj.ple.ple_method, AscendPLEPinnedHostEmbeddingMethod)
    assert inj.ple.ple_method.num_embeddings == inj._STUB_TABLE_ROWS


def test_ple_injection_builds_lazy_transport_with_checkpoint_dir():
    cfg = _real_text_config()
    inj = _PLEInjection(
        config=cfg,
        layer_idx=1,
        dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY,
        checkpoint_dir=str(_CKPT),
    )
    inj._ensure_ple_method(torch.device("cpu"))
    method = inj.ple.ple_method
    assert isinstance(method, AscendPLELazyShardEmbeddingMethod)
    assert method.num_embeddings == inj.ngram.padded_vocab_size
    assert method.embedding_dim == inj.per_head_dim
    assert method.physical_bytes == 0

    # The forward path must use FULL padded-vocab ids (no stub reduction) and
    # gather real rows lazily; output stays finite.
    seq_len = 8
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % int(cfg.vocab_size)
    ids = inj._real_ngram_ids(input_ids, reduce=False)
    assert ids.max() >= inj._STUB_TABLE_ROWS  # proves no modulo reduction

    hidden = torch.zeros(seq_len, inj.ple.hc_hidden_size, dtype=torch.float16)
    out = inj(hidden, input_ids)
    assert out.shape == (seq_len, inj.ple.hc_hidden_size)
    assert torch.isfinite(out.to(torch.float32)).all()


# --------------------------------------------------------------------------- #
# 2. vocab-parallel embed / lm_head sharding via weight_loader
# --------------------------------------------------------------------------- #
_VMOD = "vllm.model_executor.layers.vocab_parallel_embedding"
_LMOD = "vllm.model_executor.layers.logits_processor"


def test_vocab_and_lm_head_shard_under_tp4():
    cfg = _tiny_moe_config(num_layers=1, num_experts=8, top_k=3, shared_inter=0)
    vocab, hidden, tp = int(cfg.vocab_size), int(cfg.hidden_size), 4

    for rank in range(tp):
        vc = _vllm_config(cfg)
        vc.parallel_config = type(vc.parallel_config)(tensor_parallel_size=tp)
        with (
            patch(f"{_VMOD}.get_tensor_model_parallel_world_size", return_value=tp),
            patch(f"{_VMOD}.get_tensor_model_parallel_rank", return_value=rank),
            patch(f"{_VMOD}.tensor_model_parallel_all_reduce", side_effect=lambda x: x),
            patch(f"{_LMOD}.tensor_model_parallel_gather", side_effect=lambda x: x),
            patch(f"{_LMOD}.tensor_model_parallel_all_gather", side_effect=lambda x, dim=-1: x),
            patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=rank),
        ):
            model = AscendQwen4ExpForCausalLM(vllm_config=vc)

        full_embed = torch.arange(vocab * hidden, dtype=torch.float32).view(vocab, hidden).to(torch.float16)
        full_lm = (torch.arange(vocab * hidden, dtype=torch.float32).view(vocab, hidden) + 1).to(torch.float16)
        loaded = model.load_weights(
            iter(
                [
                    ("model.language_model.embed_tokens.weight", full_embed),
                    ("lm_head.weight", full_lm),
                ]
            )
        )
        assert "model.embed_tokens.weight" in loaded
        assert "lm_head.weight" in loaded

        emb = model.model.embed_tokens
        start, end = emb.shard_indices.org_vocab_start_index, emb.shard_indices.org_vocab_end_index
        assert torch.equal(emb.weight[: end - start], full_embed[start:end])
        if emb.weight.shape[0] > end - start:
            assert emb.weight[end - start :].eq(0).all()

        lm = model.lm_head
        start, end = lm.shard_indices.org_vocab_start_index, lm.shard_indices.org_vocab_end_index
        assert torch.equal(lm.weight[: end - start], full_lm[start:end])
        if lm.weight.shape[0] > end - start:
            assert lm.weight[end - start :].eq(0).all()
