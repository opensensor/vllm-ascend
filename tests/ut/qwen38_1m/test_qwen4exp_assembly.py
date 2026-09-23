# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Ascend Qwen4Exp full-model assembly (plan T1.5).

Everything runs host-side with NO NPU and NO Triton kernel: a tiny random
Qwen4Exp config builds the 48-layer graph (36 GDN linear-attention layers, 12
QSA sparse-attention layers, PLE injection at ``ple_layer_ids``, MoE with a
shared expert, gated-residual / hyperconnection wiring), loads dummy weights,
forwards on CPU, and samples a token. The KV-cache spec materialization is
checked against the T1.4 expectations, determinism is asserted across two
fixed-seed greedy forwards, and the 310P flag path is verified to import no
NPU / GDN-kernel module.

The vLLM ``VocabParallelEmbedding`` / ``ParallelLMHead`` / ``LogitsProcessor``
call tensor-parallel collectives even at TP=1; on this single-process host there
is no initialized TP group, so those collectives are patched to identity (the
same single-rank shimming the sibling registration test uses for the vocab
layers).

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_qwen4exp_assembly.py
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_PKG_DIR = Path(__file__).parents[3] / "vllm_ascend" / "models" / "qwen4_exp"

# --- T1.4 expected QSA side-cache geometry (see test_kv_cache_specs.py) -----
_EXPECTED_RING_PAGE_BYTES = 1_024
_EXPECTED_COMPRESSED_PAGE_BYTES = 8_192


# ---------------------------------------------------------------------------
# Tiny random config + fakes
# ---------------------------------------------------------------------------
def _tiny_text_config(
    *,
    num_layers: int = 48,
    qsa: bool = True,
    moe: bool = True,
    tie: bool = False,
    ple_layer_ids: tuple[int, ...] = (4,),
) -> SimpleNamespace:
    """A tiny random Qwen4Exp text config (duck-typed, host-safe).

    ``num_layers`` layers alternate so every 4th layer is ``full_attention``
    (12 of 48) and the rest are ``linear_attention`` (36 of 48), mirroring the
    real 36-GDN / 12-QSA split.
    """
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]
    cfg = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie,
        eos_token_id=0,
        layer_types=layer_types,
        # hyperconnection / gated residual
        hc_count=2,
        hc_lowrank=8,
        # PLE / n-gram
        ple_layer_ids=list(ple_layer_ids),
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        # GDN (linear attention)
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        # MoE
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    if qsa:
        cfg.indexer_n_heads = 2
        cfg.indexer_kv_heads = 1
        cfg.indexer_head_dim = 16
        cfg.indexer_budget = 4
        cfg.indexer_compress_ratio = 4
    if moe:
        cfg.num_experts = 4
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 32
        cfg.shared_expert_intermediate_size = 32
    return cfg


def _vllm_config(cfg: SimpleNamespace) -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_text_config=cfg,
        hf_config=SimpleNamespace(text_config=cfg, vision_config=None),
        dtype=torch.float16,
        multimodal_config=None,
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        cache_config=SimpleNamespace(mamba_cache_mode="align", mamba_ssm_cache_dtype="float32"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=None,
    )


@contextlib.contextmanager
def _single_rank_tp():
    """Shim TP collectives to identity so the model runs on one CPU process."""
    vmod = "vllm.model_executor.layers.vocab_parallel_embedding"
    lmod = "vllm.model_executor.layers.logits_processor"
    with (
        patch(f"{vmod}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{vmod}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{vmod}.tensor_model_parallel_all_reduce", side_effect=lambda x: x),
        patch(f"{lmod}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{lmod}.tensor_model_parallel_gather", side_effect=lambda x: x),
        patch(f"{lmod}.tensor_model_parallel_all_gather", side_effect=lambda x, dim=-1: x),
    ):
        yield


def _build(cfg: SimpleNamespace):
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    with _single_rank_tp():
        return AscendQwen4ExpForCausalLM(vllm_config=_vllm_config(cfg))


def _load_dummy_weights(model, *, seed: int = 0) -> tuple[int, int]:
    """Generate + load deterministic dummy weights via the model's loader."""
    generator = torch.Generator().manual_seed(seed)
    weights = []
    for name, param in model.named_parameters():
        weights.append((name, (torch.randn(param.shape, generator=generator) * 0.02).to(param.dtype)))
    loaded = model.load_weights(weights)
    return len(loaded), len(weights)


def _greedy_step(model, input_ids, positions):
    with torch.no_grad(), _single_rank_tp():
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)
    return logits


# ---------------------------------------------------------------------------
# Boot: construct + load dummy weights + forward + sample
# ---------------------------------------------------------------------------
def test_full_model_boots_forwards_and_samples():
    cfg = _tiny_text_config(num_layers=48)
    model = _build(cfg)
    model.eval()
    assert len(model.model.layers) == 48

    loaded, total = _load_dummy_weights(model, seed=0)
    assert loaded == total > 0  # every dummy weight lands by name+shape

    seq_len = 8
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % cfg.vocab_size
    positions = torch.arange(seq_len, dtype=torch.int64)

    logits = _greedy_step(model, input_ids, positions)
    assert logits is not None
    assert logits.shape == (seq_len, cfg.vocab_size)
    assert torch.isfinite(logits.float()).all()

    token = int(logits[-1].argmax().item())
    assert 0 <= token < cfg.vocab_size


def test_model_forwards_serving_ngram_context_to_ple():
    """Decode/chunk history prepared by model state must reach PLE hashing."""
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    class _BackboneRecorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.call = None

        def forward(self, *args, **kwargs):
            self.call = (args, kwargs)
            return torch.ones(1, 2)

    model = AscendQwen4ExpForCausalLM.__new__(AscendQwen4ExpForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = _BackboneRecorder()

    input_ids = torch.tensor([17], dtype=torch.int64)
    positions = torch.tensor([9], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    ngram_context = torch.tensor([[15, 16]], dtype=torch.int32)
    output = model(
        input_ids,
        positions,
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
    )

    assert torch.equal(output, torch.ones(1, 2))
    assert model.model.call is not None
    _args, kwargs = model.model.call
    assert kwargs["query_start_loc"] is query_start_loc
    assert kwargs["ngram_context"] is ngram_context


def test_ple_decode_hash_uses_previous_tokens():
    from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
    from vllm_ascend.models.qwen4_exp.model import _PLEInjection

    cfg = _tiny_text_config(num_layers=1, ple_layer_ids=(1,))
    cfg.ngram_vocab_size_base = 257
    cfg.seed = 1234
    ple = _PLEInjection(config=cfg, layer_idx=0, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY)
    assert ple.ngram is not None
    token = torch.tensor([23], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)

    ids_with_history = ple._real_ngram_ids(
        token,
        query_start_loc,
        torch.tensor([[21, 22]], dtype=torch.int32),
    )
    ids_at_sequence_start = ple._real_ngram_ids(
        token,
        query_start_loc,
        torch.tensor([[cfg.eos_token_id, cfg.eos_token_id]], dtype=torch.int32),
    )

    assert not torch.equal(ids_with_history, ids_at_sequence_start)


def test_qsa_dense_decode_only_when_selection_is_exact():
    from vllm_ascend.models.qwen4_exp.model import _QSAAttention

    metadata = SimpleNamespace(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        seq_lens_cpu=torch.tensor([2, 2048]),
    )
    assert _QSAAttention._dense_decode_is_exact(metadata, 2048)
    metadata.seq_lens_cpu = torch.tensor([2, 2049])
    assert not _QSAAttention._dense_decode_is_exact(metadata, 2048)
    metadata.attn_state.name = "PrefillOnly"
    metadata.seq_lens_cpu = torch.tensor([2, 2048])
    assert not _QSAAttention._dense_decode_is_exact(metadata, 2048)


def test_composition_places_gdn_qsa_ple_and_moe():
    from vllm_ascend.models.qwen4_exp.model import (
        _EagerSparseMoE,
        _GDNAttention,
        _PLEInjection,
        _QSAAttention,
    )

    cfg = _tiny_text_config(num_layers=48, ple_layer_ids=(4,))
    model = _build(cfg)
    layers = model.model.layers

    gdn = sum(isinstance(layer.attention, _GDNAttention) for layer in layers)
    qsa = sum(isinstance(layer.attention, _QSAAttention) for layer in layers)
    assert (gdn, qsa) == (36, 12)

    # PLE sits on the layer whose absolute id (layer_idx + 1) is listed.
    ple_layers = [layer.layer_idx for layer in layers if isinstance(layer.ple, _PLEInjection)]
    assert ple_layers == [3]  # ple_layer_ids == [4] -> layer_idx 3

    # MoE (routed + shared expert) is built for the sparse layers.
    assert any(isinstance(layer.mlp, _EagerSparseMoE) for layer in layers)
    a_moe = next(layer.mlp for layer in layers if isinstance(layer.mlp, _EagerSparseMoE))
    assert a_moe.has_shared_expert is True
    assert a_moe.num_experts == 4 and a_moe.top_k == 2


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_greedy_forward_is_deterministic():
    cfg = _tiny_text_config(num_layers=48)
    seq_len = 8
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % cfg.vocab_size
    positions = torch.arange(seq_len, dtype=torch.int64)

    # Same model forwarded twice -> bitwise-identical logits.
    model = _build(cfg)
    model.eval()
    _load_dummy_weights(model, seed=7)
    logits_a = _greedy_step(model, input_ids, positions)
    logits_b = _greedy_step(model, input_ids, positions)
    assert torch.equal(logits_a, logits_b)

    # A fresh model with the same fixed-seed dummy weights -> same greedy token.
    model2 = _build(cfg)
    model2.eval()
    _load_dummy_weights(model2, seed=7)
    logits_c = _greedy_step(model2, input_ids, positions)
    assert int(logits_a[-1].argmax()) == int(logits_c[-1].argmax())
    assert torch.equal(logits_a, logits_c)


# ---------------------------------------------------------------------------
# KV-cache spec materialization (T1.4)
# ---------------------------------------------------------------------------
def test_kv_group_report_matches_t1_4():
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    from vllm_ascend.models.qwen4_exp.kv_cache import AscendQSARawRingSpec

    cfg = _tiny_text_config(num_layers=48)
    model = _build(cfg)

    report = model.kv_group_report()
    # 36 GDN mamba layers, 12 QSA raw-ring layers, 12 QSA compressed layers.
    assert report == {"MambaSpec": 36, "AscendQSARawRingSpec": 12, "MLAAttentionSpec": 12}

    groups = model.get_kv_cache_groups()
    # Layers that share a spec merge into one group per cache class -> 3 groups.
    assert len(groups) == 3
    by_type = {type(g.kv_cache_spec).__name__: g for g in groups}
    assert set(by_type) == {"MambaSpec", "AscendQSARawRingSpec", "MLAAttentionSpec"}
    assert len(by_type["MambaSpec"].layer_names) == 36
    assert len(by_type["AscendQSARawRingSpec"].layer_names) == 12
    assert len(by_type["MLAAttentionSpec"].layer_names) == 12

    ring = by_type["AscendQSARawRingSpec"].kv_cache_spec
    compressed = by_type["MLAAttentionSpec"].kv_cache_spec
    assert isinstance(ring, AscendQSARawRingSpec)
    assert isinstance(compressed, MLAAttentionSpec)
    assert isinstance(by_type["MambaSpec"].kv_cache_spec, MambaSpec)
    # Byte-math ties back to the T1.4 QSA geometry (1M reference page sizes).
    assert ring.page_size_bytes == _EXPECTED_RING_PAGE_BYTES
    assert compressed.page_size_bytes == _EXPECTED_COMPRESSED_PAGE_BYTES


def test_dense_full_attention_and_dense_mlp_variant_boots():
    from vllm_ascend.models.qwen4_exp.model import _EagerDenseAttention, _EagerMLP

    # No indexer config -> dense full attention; no experts -> dense MLP.
    cfg = _tiny_text_config(num_layers=4, qsa=False, moe=False, ple_layer_ids=(2,))
    model = _build(cfg)
    model.eval()
    _load_dummy_weights(model, seed=3)

    assert any(isinstance(layer.attention, _EagerDenseAttention) for layer in model.model.layers)
    assert all(isinstance(layer.mlp, _EagerMLP) for layer in model.model.layers)

    seq_len = 6
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % cfg.vocab_size
    positions = torch.arange(seq_len, dtype=torch.int64)
    logits = _greedy_step(model, input_ids, positions)
    assert logits.shape == (seq_len, cfg.vocab_size)

    # Dense full-attention layers materialize FullAttentionSpec groups
    # (num_layers=4 -> 1 full_attention at idx 3, 3 linear_attention).
    report = model.kv_group_report()
    assert report.get("FullAttentionSpec", 0) == 1
    assert report.get("MambaSpec", 0) == 3


def test_embedding_tie_shares_lm_head_weight():
    cfg = _tiny_text_config(num_layers=4, tie=True)
    model = _build(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# 310P flag path: no NPU / GDN-kernel import
# ---------------------------------------------------------------------------
def test_no_npu_or_kernel_import_on_310p_flag_path(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_310P", "1")

    cfg = _tiny_text_config(num_layers=48)
    # Snapshot before this test builds/forwards so the assertion is scoped to what
    # THIS test imports, not modules a sibling test left in the shared interpreter
    # (e.g. a torch_npu host stub). The package-source grep test is the absolute
    # no-triton gate; this one guards that the 310P eager path pulls no runtime.
    pre_modules = set(sys.modules)
    model = _build(cfg)
    _load_dummy_weights(model, seed=0)
    seq_len = 8
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % cfg.vocab_size
    positions = torch.arange(seq_len, dtype=torch.int64)
    _greedy_step(model, input_ids, positions)

    newly_imported = set(sys.modules) - pre_modules
    # Building + forwarding on the host 310P path pulls no NPU runtime.
    assert not any(name == "torch_npu" or name.startswith("torch_npu.") for name in newly_imported)
    # The GDN eager backend never pulls the fla (chunk/recurrent) kernel modules.
    assert not any(name.startswith("vllm_ascend._310p.ops.fla") for name in newly_imported)


def test_package_source_has_no_triton_import():
    hits = {}
    for path in _PKG_DIR.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "import triton" in code or "from triton" in code or "triton" in code and "import" in code:
                hits.setdefault(path.name, []).append((lineno, line))
    assert not hits, f"triton import reachable in package: {hits}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "--noconftest"]))
