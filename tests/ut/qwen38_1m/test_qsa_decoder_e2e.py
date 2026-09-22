# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU end-to-end parity for the Ascend 310P Qwen4Exp QSA decoder layer (plan T6.4).

This is the assembly gate that ties the T6.1 indexer, the T6.2 sparse attention
and the T6.3 chunked-prefill seams into ONE decoder-layer code path and proves it
reproduces the T0.6 *composite* reference (indexer + sparse attention composed,
with the surrounding project -> select -> attend -> output-gate -> out-proj
wiring) at short context, for fixed seeds.

Three levels of coverage:

* **Composition entry point** -- :func:`run_qsa_decoder_attention` in
  ``qsa.py`` is the single implementation every one of the 12 QSA layers runs.
  It is driven directly here (float64 policy) against a composite reference built
  from the T0.6 ``qsa_indexer_reference`` + ``qsa_attention_reference`` modules,
  using the *same* projection weights on both sides so only the wiring/order and
  the arithmetic are under test.
* **Wired module** -- the assembly's ``_QSAAttention`` (which delegates to the
  same composition entry point) matches the composite reference too, and every
  QSA layer in a full tiny model is the identical ``_QSAAttention`` class.
* **Determinism** -- greedy logits of a full tiny model (pinned float16-main /
  float32-accum policy) are bitwise-identical across two forwards and across a
  fresh model loaded from the same fixed seed, on a synthetic sequence.

Context length: the plan names 8K. The T0.6 sparse-attention reference is an
O(T * Hq * S) float64 double loop in Python, which is intractable at 8,192 tokens
on CPU. The sparse-selection + budget-pruning code path is *independent of the
absolute sequence length* -- once the context exceeds the indexer token budget
the selection prunes, exercising exactly the same branches an 8K context would.
So parity is asserted on sequences that are (a) below the budget (dense
selection) and (b) several times the budget (pruned/sparse selection), with the
budget shrunk to a tiny value so pruning is genuine; :func:`_selection_is_sparse`
asserts the sparse case really prunes. This keeps the reference cheap while
covering the same code as an 8K run.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import here):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_qsa_decoder_e2e.py
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.qwen38_1m.reference.qsa_attention_reference import (
    project_qk_norm_rope,
    sparse_gqa_attention,
)
from tests.ut.qwen38_1m.reference.qsa_indexer_reference import qsa_select_tokens
from tests.ut.qwen38_1m.reference.tolerances import QSA_ATTN_ATOL, QSA_ATTN_RTOL
from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.kv_cache import AscendQSAFullAttentionSpec
from vllm_ascend.models.qwen4_exp.model import _QSAAttention
from vllm_ascend.models.qwen4_exp.qsa import (
    QSADecoderProjections,
    run_qsa_decoder_attention,
)

_EPS = 1e-6
_ROPE_THETA = 10_000.0
_PARTIAL_ROTARY_FACTOR = 0.25

# A float64 storage + accumulation policy: the composition runs at float64, so it
# agrees with the float64 composite reference at rounding level and the tight
# T0.6 QSA-attention tolerance (rtol 1e-8 / atol 1e-9) applies -- a real wiring
# or ordering regression (dropped gate, swapped norm/RoPE, wrong scale) shifts
# outputs by O(1) and fails, while two correct float64 formulations agree.
_POLICY_F64 = replace(
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    main_dtype=torch.float64,
    accumulation_dtype=torch.float64,
    qsa_main_dtype=torch.float64,
    qsa_indexer_dtype=torch.float64,
    attention_dtype=torch.float64,
    attention_accumulation_dtype=torch.float64,
    kv_cache_dtype=torch.float64,
)


# ---------------------------------------------------------------------------
# Tiny QSA config + helpers
# ---------------------------------------------------------------------------
def _qsa_config(
    *,
    hidden_size: int = 32,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    index_n_heads: int = 2,
    index_head_dim: int = 16,
    indexer_budget: int = 8,
    compress_ratio: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        rope_theta=_ROPE_THETA,
        rms_norm_eps=_EPS,
        indexer_n_heads=index_n_heads,
        indexer_kv_heads=1,
        indexer_head_dim=index_head_dim,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=compress_ratio,
    )


def _rand(shape, seed, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


def _init_module(module: _QSAAttention, seed: int) -> None:
    """Fill every parameter of a QSA module with small deterministic noise."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_((torch.randn(param.shape, generator=gen, dtype=torch.float64) * 0.1).to(param.dtype))


@pytest.mark.parametrize("state_name", ["PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"])
def test_qsa_dense_prefill_fast_path_covers_complete_selection(state_name):
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        attn_state=SimpleNamespace(name=state_name),
        seq_lens_cpu=torch.tensor([896, 2048], dtype=torch.int32),
        seq_lens_list=None,
    )

    assert _QSAAttention._dense_prefill_is_exact(metadata, token_budget=2048)


def test_qsa_dense_prefill_fast_path_preserves_sparse_and_decode_dispatch():
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        attn_state=SimpleNamespace(name="ChunkedPrefill"),
        seq_lens_cpu=torch.tensor([2049], dtype=torch.int32),
        seq_lens_list=None,
    )
    assert not _QSAAttention._dense_prefill_is_exact(metadata, token_budget=2048)

    metadata.seq_lens_cpu = torch.tensor([32], dtype=torch.int32)
    metadata.num_decodes = 1
    assert not _QSAAttention._dense_prefill_is_exact(metadata, token_budget=2048)


def test_qsa_dense_prefill_fast_path_uses_host_list_fallback():
    metadata = SimpleNamespace(
        num_prefills=2,
        num_decodes=0,
        attn_state=SimpleNamespace(name="ChunkedPrefill"),
        seq_lens_cpu=None,
        seq_lens_list=[256, 512],
    )

    assert _QSAAttention._dense_prefill_is_exact(metadata, token_budget=2048)


def test_qsa_layer_registers_custom_cache_spec_owner():
    """The registered layer must own both the main and index-cache spec."""
    static_forward_context = {}
    fake_vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=static_forward_context)
    )
    backend = object()
    with (
        patch(
            "vllm_ascend.models.qwen4_exp.model.get_current_vllm_config_or_none",
            return_value=fake_vllm_config,
        ),
        patch("vllm_ascend.models.qwen4_exp.model._resolve_attn_backend", return_value=backend),
    ):
        module = _QSAAttention(config=_qsa_config(), layer_idx=1, dtype_policy=_POLICY_F64, prefix="qsa.1")

    assert static_forward_context == {"qsa.1": module}
    assert module.get_attn_backend() is backend
    assert isinstance(module.get_kv_cache_spec(fake_vllm_config), AscendQSAFullAttentionSpec)


def _linear_f64(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.double() @ weight.double().t()


def _composite_reference(module: _QSAAttention, cfg, block_input, positions):
    """Compose the T0.6 references using ``module``'s own projection weights.

    project (q/k/v/gate/index) -> T0.6 indexer selection -> T0.6 Q/K
    GemmaRMSNorm + partial RoPE -> T0.6 sparse GQA attention with output gate ->
    out-projection. Everything in float64.
    """
    seq_len = block_input.shape[0]
    q = _linear_f64(block_input, module.q_proj).view(seq_len, cfg.num_attention_heads, cfg.head_dim)
    k = _linear_f64(block_input, module.k_proj).view(seq_len, cfg.num_key_value_heads, cfg.head_dim)
    v = _linear_f64(block_input, module.v_proj).view(seq_len, cfg.num_key_value_heads, cfg.head_dim)
    gate = _linear_f64(block_input, module.gate_proj).view(seq_len, cfg.num_attention_heads, cfg.head_dim)
    index_q = _linear_f64(block_input, module.iq_proj).view(seq_len, cfg.indexer_n_heads, cfg.indexer_head_dim)
    index_k = _linear_f64(block_input, module.ik_proj)  # [T, D]

    packed, valid_counts = qsa_select_tokens(
        index_q,
        index_k,
        positions,
        compress_ratio=cfg.indexer_compress_ratio,
        token_topk=cfg.indexer_budget,
    )
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    q_n, k_n = project_qk_norm_rope(
        q,
        k,
        positions,
        module.attn.q_norm_weight,
        module.attn.k_norm_weight,
        cfg.rms_norm_eps,
        rotary_dim,
        cfg.rope_theta,
    )
    attn_out = sparse_gqa_attention(q_n, k_n, v, gate, packed, valid_counts, cfg.num_key_value_heads)
    ref = _linear_f64(attn_out.reshape(seq_len, cfg.num_attention_heads * cfg.head_dim), module.o_proj)
    return ref, packed, valid_counts


def _selection_is_sparse(packed, valid_counts, context_len) -> bool:
    """True if at least one query row selected fewer than the causal prefix."""
    for t in range(valid_counts.shape[0]):
        causal_prefix = t + 1  # positions == arange(context_len)
        if int(valid_counts[t].item()) < min(causal_prefix, context_len):
            return True
    return False


# ---------------------------------------------------------------------------
# Composition entry point (qsa.py) vs T0.6 composite reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("seq_len", "expect_sparse"),
    [
        (6, False),  # below the token budget -> dense causal selection
        (40, True),  # several x the budget -> pruned/sparse selection (8K-shape)
        (96, True),
    ],
)
def test_run_qsa_decoder_attention_matches_composite_reference(seq_len, expect_sparse):
    cfg = _qsa_config(indexer_budget=8, compress_ratio=4)
    module = _QSAAttention(config=cfg, layer_idx=1, dtype_policy=_POLICY_F64).double()
    _init_module(module, seed=11)

    block_input = _rand((seq_len, cfg.hidden_size), seed=101) * 0.2
    positions = torch.arange(seq_len, dtype=torch.int64)

    projections = QSADecoderProjections(
        q_proj=module.q_proj,
        k_proj=module.k_proj,
        v_proj=module.v_proj,
        gate_proj=module.gate_proj,
        index_q_proj=module.iq_proj,
        index_k_proj=module.ik_proj,
        out_proj=module.o_proj,
    )
    with torch.no_grad():
        actual = run_qsa_decoder_attention(
            block_input,
            positions,
            projections=projections,
            indexer=module.indexer,
            attention=module.attn,
            num_query_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            index_n_heads=cfg.indexer_n_heads,
            index_head_dim=cfg.indexer_head_dim,
            store_dtype=module.params_dtype,
            compute_dtype=module.compute_dtype,
        )

    ref, packed, valid_counts = _composite_reference(module, cfg, block_input, positions)

    assert _selection_is_sparse(packed, valid_counts, seq_len) == expect_sparse
    torch.testing.assert_close(actual.double(), ref, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_run_qsa_decoder_attention_accepts_multimodal_positions():
    cfg = _qsa_config(indexer_budget=8, compress_ratio=4)
    module = _QSAAttention(config=cfg, layer_idx=1, dtype_policy=_POLICY_F64).double()
    _init_module(module, seed=17)

    seq_len = 6
    block_input = _rand((seq_len, cfg.hidden_size), seed=107) * 0.2
    text_positions = torch.arange(seq_len, dtype=torch.int64)
    multimodal_positions = text_positions.repeat(3, 1)
    projections = QSADecoderProjections(
        q_proj=module.q_proj,
        k_proj=module.k_proj,
        v_proj=module.v_proj,
        gate_proj=module.gate_proj,
        index_q_proj=module.iq_proj,
        index_k_proj=module.ik_proj,
        out_proj=module.o_proj,
    )
    common = {
        "projections": projections,
        "indexer": module.indexer,
        "attention": module.attn,
        "num_query_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "index_n_heads": cfg.indexer_n_heads,
        "index_head_dim": cfg.indexer_head_dim,
        "store_dtype": module.params_dtype,
        "compute_dtype": module.compute_dtype,
    }

    with torch.no_grad():
        text_output = run_qsa_decoder_attention(block_input, text_positions, **common)
        multimodal_output = run_qsa_decoder_attention(block_input, multimodal_positions, **common)

    torch.testing.assert_close(multimodal_output, text_output)


def test_wired_qsa_module_matches_composite_reference():
    """The assembly's ``_QSAAttention.forward`` (which delegates to the shared
    composition entry point) matches the composite reference bit-for-bit at f64."""
    cfg = _qsa_config(indexer_budget=8, compress_ratio=4)
    module = _QSAAttention(config=cfg, layer_idx=2, dtype_policy=_POLICY_F64).double()
    _init_module(module, seed=23)

    seq_len = 48
    block_input = _rand((seq_len, cfg.hidden_size), seed=202) * 0.2
    positions = torch.arange(seq_len, dtype=torch.int64)

    with torch.no_grad():
        actual = module(block_input, positions)
    ref, packed, valid_counts = _composite_reference(module, cfg, block_input, positions)

    assert _selection_is_sparse(packed, valid_counts, seq_len)
    torch.testing.assert_close(actual.double(), ref, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


# ---------------------------------------------------------------------------
# Full tiny model: identical QSA code path + determinism
# ---------------------------------------------------------------------------
def _tiny_model_config(*, num_layers: int = 8) -> SimpleNamespace:
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]
    return SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        eos_token_id=0,
        layer_types=layer_types,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[4],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=4,
        indexer_compress_ratio=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )


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


def _load_dummy_weights(model, *, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    weights = [
        (name, (torch.randn(param.shape, generator=gen) * 0.02).to(param.dtype))
        for name, param in model.named_parameters()
    ]
    model.load_weights(weights)


def _greedy_logits(model, input_ids, positions):
    with torch.no_grad(), _single_rank_tp():
        return model.compute_logits(model(input_ids, positions))


def test_all_qsa_layers_share_the_full_path():
    cfg = _tiny_model_config(num_layers=8)
    model = _build(cfg)
    qsa_layers = [layer.attention for layer in model.model.layers if isinstance(layer.attention, _QSAAttention)]
    # 8 layers, every 4th (idx 3, 7) is full_attention -> QSA -> 2 layers, all
    # the identical class, i.e. one shared composition entry point.
    assert len(qsa_layers) == 2
    assert all(type(a) is _QSAAttention for a in qsa_layers)


def test_qsa_decoder_greedy_is_deterministic():
    cfg = _tiny_model_config(num_layers=8)
    seq_len = 24  # > indexer_budget (4) -> QSA selection is genuinely sparse
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64) % cfg.vocab_size
    positions = torch.arange(seq_len, dtype=torch.int64)

    model = _build(cfg)
    model.eval()
    _load_dummy_weights(model, seed=7)
    logits_a = _greedy_logits(model, input_ids, positions)
    logits_b = _greedy_logits(model, input_ids, positions)
    assert torch.equal(logits_a, logits_b)  # same model, two runs -> bitwise

    model2 = _build(cfg)
    model2.eval()
    _load_dummy_weights(model2, seed=7)
    logits_c = _greedy_logits(model2, input_ids, positions)
    assert torch.equal(logits_a, logits_c)  # fresh model, same seed -> bitwise
    assert int(logits_a[-1].argmax()) == int(logits_c[-1].argmax())
    assert torch.isfinite(logits_a.float()).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "--noconftest"]))
