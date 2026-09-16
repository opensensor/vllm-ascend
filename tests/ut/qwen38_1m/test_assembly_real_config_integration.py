# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration smoke: the assembly wires the REAL n-gram hasher on the real config.

The tiny-config assembly tests exercise the stub id path; this test loads the real
Qwen3.8-Flash-Next checkpoint config and asserts `_PLEInjection` builds the real
SplitMix64 `AscendQwen4ExpNGramEmbedding` and forwards through it (bounded to the
host synthetic table — the real 128-shard gather is the load_weights/device path).
Skips cleanly when the checkpoint mount is absent (CI-safe).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.model import _PLEInjection

_CKPT = Path(
    "/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/models/ascend/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i"
)


def _real_text_config() -> SimpleNamespace:
    cfg_path = _CKPT / "config.json"
    if not cfg_path.exists():
        pytest.skip(f"checkpoint config not mounted at {cfg_path}")
    return SimpleNamespace(**json.loads(cfg_path.read_text())["text_config"])


def test_assembly_uses_real_ngram_hasher_on_real_config():
    cfg = _real_text_config()
    inj = _PLEInjection(config=cfg, layer_idx=1, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY)

    # Real config carries the full n-gram vocab layout -> the real hasher is built
    # (not the tiny-config stub fallback).
    assert inj.ngram is not None
    assert inj.ngram_size == int(cfg.ngram_size)

    seq_len = 12
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % int(cfg.vocab_size)

    # The forward path resolves ids through the real SplitMix64 hasher.
    ids_via_forward = inj._real_ngram_ids(input_ids)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int64)
    ngram_context = torch.full((1, inj.ngram_size - 1), inj.eos_token_id, dtype=torch.int64)
    expected = inj.ngram.compute_ngram_ids(input_ids, query_start_loc, ngram_context).remainder(inj._STUB_TABLE_ROWS)
    assert torch.equal(ids_via_forward, expected)
    assert ids_via_forward.shape == (seq_len, inj.num_ngram_heads)

    # End-to-end injection forward runs and is finite (gather+project+gate+conv).
    # The PLE injection operates on the hyperconnection-expanded stream.
    hidden = torch.zeros(seq_len, inj.ple.hc_hidden_size, dtype=torch.float16)
    out = inj(hidden, input_ids)
    assert out.shape[0] == seq_len
    assert torch.isfinite(out.to(torch.float32)).all()


def test_real_ngram_ids_are_deterministic():
    cfg = _real_text_config()
    inj = _PLEInjection(config=cfg, layer_idx=1, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY)
    input_ids = torch.arange(5, 5 + 20, dtype=torch.int64) % int(cfg.vocab_size)
    a = inj._real_ngram_ids(input_ids)
    b = inj._real_ngram_ids(input_ids)
    assert torch.equal(a, b)
