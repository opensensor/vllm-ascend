# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend GLM-5.3-Flash W2 DSA sparse-attention path (plan G5).

DSA = ``deepseek_sparse_attention``, the 11 ``full_attn_layers`` of GLM-5.3-Flash
(``glm5_next``). Structurally it is a *Lightning-Indexer top-k token selection*
feeding a *full (MLA-latent) attention core* -- the same family as DeepSeek
V4.1's sparse attention. GLM's MLA is the **NoPE** variant (``qk_rope_head_dim=0``)
with a **kpool** indexer (``index_kpool``), but the indexer *scoring + top-k
selection* math is exactly the deepseek_v41 one, so the 310P GLM DSA module
REUSES ``vllm_ascend.models.deepseek_v41.indexer`` for selection and only wraps a
thin GLM adapter (kpool compression) + GLM's own q/kv/o projections + attention
core.

Everything is host-side, Triton-free, NPU-free (CPU-testable). Run with
``--noconftest`` (the shared tests/ut/conftest.py fails to import on this host):

    python3 -m pytest -q --noconftest \
        tests/ut/glm_w2/test_glm5next_w2_dsa.py
"""

from pathlib import Path

import torch

_DSA_SRC = Path(__file__).parents[3] / "vllm_ascend" / "models" / "glm5next_w2" / "dsa.py"

# Small, faithful test geometry (NoPE MLA + kpool indexer).
_DIMS = dict(
    hidden_size=32,
    num_attention_heads=2,
    q_lora_rank=16,
    kv_lora_rank=12,
    qk_nope_head_dim=8,
    v_head_dim=8,
    index_n_heads=2,
    index_head_dim=8,
    index_topk=8,  # token budget -> block_topk = index_topk // index_kpool
    index_kpool=2,  # kpool==2 so it matches a deepseek_v41 compress_ratio=2 module
    rms_norm_eps=1e-5,
)


def _build_dsa(dtype=None):
    from vllm_ascend.models.glm5next_w2.dsa import AscendGlm5NextW2DSA

    mod = AscendGlm5NextW2DSA(dtype=dtype, **_DIMS)
    mod.reset_parameters(seed=1234)
    return mod


def _build_indexer(dtype=None):
    from vllm_ascend.models.glm5next_w2.dsa import Glm5NextW2DsaIndexer

    idx = Glm5NextW2DsaIndexer(
        hidden_size=_DIMS["hidden_size"],
        q_lora_rank=_DIMS["q_lora_rank"],
        index_n_heads=_DIMS["index_n_heads"],
        index_head_dim=_DIMS["index_head_dim"],
        index_topk=_DIMS["index_topk"],
        index_kpool=_DIMS["index_kpool"],
        dtype=dtype,
    )
    idx.reset_parameters(seed=99)
    return idx


# ---------------------------------------------------------------------------
# Shape / dtype contracts
# ---------------------------------------------------------------------------


def test_dsa_forward_shape_and_dtype_fp16():
    from vllm_ascend.models.glm5next_w2.dtype_policy import (
        ASCEND_GLM5NEXT_W2_DTYPE_POLICY as p,
    )

    mod = _build_dsa()  # fp16 main (from policy)
    T = 12
    hidden = torch.randn(T, _DIMS["hidden_size"], dtype=torch.float16)
    out = mod(hidden)
    assert out.shape == (T, _DIMS["hidden_size"])
    # DSA IO rides the policy's fp16 dsa dtype.
    assert out.dtype == p.cast_site("dsa")
    assert out.dtype is torch.float16
    assert torch.isfinite(out).all()


def test_dsa_reads_dtypes_from_policy():
    # The module must not spell dtype literals: its compute/accum dtypes come
    # from the authoritative G3 policy sites.
    from vllm_ascend.models.glm5next_w2.dtype_policy import (
        ASCEND_GLM5NEXT_W2_DTYPE_POLICY as p,
    )

    mod = _build_dsa()
    assert mod.param_dtype is p.cast_site("dsa")
    assert mod.accum_dtype is p.cast_site("dsa_accumulation")
    assert mod.indexer.dtype is p.cast_site("indexer")


# ---------------------------------------------------------------------------
# Causality: output at t is independent of tokens > t
# ---------------------------------------------------------------------------


def test_dsa_causality():
    # float64 parity mode -> exact equality (no fp tolerance games).
    mod = _build_dsa(dtype=torch.float64)
    T = 16
    torch.manual_seed(7)
    hidden = torch.randn(T, _DIMS["hidden_size"], dtype=torch.float64)
    out_ref = mod(hidden)

    # Perturb a strictly-later token and re-run; earlier outputs must not move.
    t_cut = 9
    hidden2 = hidden.clone()
    hidden2[t_cut + 1 :] += torch.randn_like(hidden2[t_cut + 1 :]) * 3.0
    out_pert = mod(hidden2)

    assert torch.allclose(out_ref[: t_cut + 1], out_pert[: t_cut + 1], atol=0.0, rtol=0.0), (
        "DSA output at position <= t changed when a future token was perturbed"
    )


# ---------------------------------------------------------------------------
# Top-k selection: exactly block_topk indices per query (when enough visible)
# ---------------------------------------------------------------------------


def test_indexer_selects_exactly_block_topk_and_is_causal():
    idx = _build_indexer(dtype=torch.float64)
    T = 16
    torch.manual_seed(3)
    query = torch.randn(T, _DIMS["index_n_heads"], _DIMS["index_head_dim"], dtype=torch.float64)
    weights = torch.randn(T, _DIMS["index_n_heads"], dtype=torch.float64)
    raw_keys = torch.randn(T, _DIMS["index_head_dim"], dtype=torch.float64)
    positions = torch.arange(T)

    block_topk = idx.block_topk  # index_topk // index_kpool = 8 // 2 = 4
    assert block_topk == 4

    blocks_per_token = idx.select_blocks(query, weights, raw_keys, positions)
    num_blocks = T // _DIMS["index_kpool"]
    for t in range(T):
        vis = min((t + 1) // _DIMS["index_kpool"], num_blocks)
        chosen = blocks_per_token[t]
        # Never select a non-causal (not-yet-formed) block.
        assert all(0 <= b < vis for b in chosen), f"token {t} selected block outside causal range: {chosen}"
        # Exactly block_topk when enough visible blocks exist, else all visible.
        assert len(chosen) == min(vis, block_topk), f"token {t}: got {len(chosen)} blocks, vis={vis}"
        assert len(set(chosen)) == len(chosen), "duplicate blocks selected"


# ---------------------------------------------------------------------------
# Attention is restricted to the selected (+ local) tokens
# ---------------------------------------------------------------------------


def test_dsa_attention_restricted_to_selected_tokens():
    mod = _build_dsa(dtype=torch.float64)
    T = 16
    torch.manual_seed(11)
    hidden = torch.randn(T, _DIMS["hidden_size"], dtype=torch.float64)

    out, sel = mod(hidden, return_selection=True)
    mask = sel.token_mask  # [T, S] bool, True = attended
    assert mask.shape == (T, T)
    assert mask.dtype == torch.bool

    # (a) restriction never crosses the causal boundary.
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.all(mask <= causal), "attention mask allows a non-causal (future) token"

    # (b) somewhere the mask is strictly sparser than dense causal -- i.e. the
    # top-k selection actually drops causal tokens (sparsity is real).
    assert (mask < causal).any(), "selection never dropped any causal token (not sparse)"

    # (c) the module's output equals attention computed under exactly that mask,
    # and DIFFERS from the dense-causal attention -> attention is restricted to
    # the selected tokens, not all causal tokens.
    out_sparse = mod.attend_with_mask(hidden, mask)
    out_dense = mod.attend_with_mask(hidden, causal)
    assert torch.allclose(out, out_sparse, atol=0.0, rtol=0.0), "forward() did not use its own selection mask"
    assert not torch.allclose(out_sparse, out_dense), "sparse output identical to dense -> restriction is a no-op"


# ---------------------------------------------------------------------------
# Reuse equivalence: GLM selection == deepseek_v41 indexer selection
# ---------------------------------------------------------------------------


def test_selection_matches_deepseek_v41_indexer():
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41.indexer import AscendDeepseekV41Indexer

    idx = _build_indexer(dtype=torch.float64)
    ratio = 2  # both GLM (kpool) and deepseek_v41 support ratio 2
    T = 16
    torch.manual_seed(5)
    query = torch.randn(T, _DIMS["index_n_heads"], _DIMS["index_head_dim"], dtype=torch.float64)
    weights = torch.randn(T, _DIMS["index_n_heads"], dtype=torch.float64)
    raw_keys = torch.randn(T, _DIMS["index_head_dim"], dtype=torch.float64)
    positions = torch.arange(T)

    # GLM reused-selection path.
    glm_blocks = idx.select_blocks(query, weights, raw_keys, positions, compress_ratio=ratio)

    # DeepSeek V4.1 indexer module driven with the SAME projected inputs.
    ds_config = SimpleNamespace(
        index_n_heads=_DIMS["index_n_heads"],
        index_head_dim=_DIMS["index_head_dim"],
        index_topk=_DIMS["index_topk"],
        q_lora_rank=None,
        hidden_size=None,
    )
    ds_idx = AscendDeepseekV41Indexer(ds_config, compress_ratio=ratio, build_projections=False)
    ds_result = ds_idx.forward(
        None, None, raw_keys, positions, precomputed_query=query, precomputed_weights=weights
    )
    assert ds_result is not None
    ds_blocks = ds_result.blocks_per_token

    assert glm_blocks == ds_blocks, "GLM DSA selection diverged from the reused deepseek_v41 indexer"


# ---------------------------------------------------------------------------
# Triton-free grep gate
# ---------------------------------------------------------------------------


def test_dsa_source_has_no_triton_import():
    hits = []
    for lineno, line in enumerate(_DSA_SRC.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]  # comments are documentation, not code
        if "import triton" in code or "from triton" in code or ("triton" in code and "import" in code):
            hits.append((lineno, line))
    assert not hits, f"triton import reachable in dsa.py: {hits}"


# ---------------------------------------------------------------------------
# Lazy package export (PEP 562), consistent with G3 -- no G3 regression
# ---------------------------------------------------------------------------


def test_package_lazily_exports_dsa_classes():
    import vllm_ascend.models.glm5next_w2 as pkg

    exported = set(dir(pkg))
    assert "AscendGlm5NextW2DSA" in exported
    assert "Glm5NextW2DsaIndexer" in exported
    # Lazy attribute access resolves to the real classes.
    from vllm_ascend.models.glm5next_w2.dsa import (
        AscendGlm5NextW2DSA,
        Glm5NextW2DsaIndexer,
    )

    assert pkg.AscendGlm5NextW2DSA is AscendGlm5NextW2DSA
    assert pkg.Glm5NextW2DsaIndexer is Glm5NextW2DsaIndexer
    # G3 dtype policy export still present (no regression).
    assert hasattr(pkg, "ASCEND_GLM5NEXT_W2_DTYPE_POLICY")


# ---------------------------------------------------------------------------
# GLM kpool parity: softmax(gate+ape)-weighted pooled key (missing-gate fix)
# ---------------------------------------------------------------------------


def _ref_kpool_compress(raw_keys, gate_score, ape, ratio, dtype=torch.float64):
    """Direct reference for the shipped ``kpool_compress_and_write_cache`` math."""
    k = raw_keys.to(dtype)
    g = gate_score.to(dtype)
    a = ape.to(dtype)
    S, D = k.shape
    nb = S // ratio
    trimmed_k = k[: nb * ratio].view(nb, ratio, D)
    trimmed_g = g[: nb * ratio].view(nb, ratio, D)
    scores = trimmed_g + a[None, :, :]
    mx = scores.max(dim=1, keepdim=True).values
    probs = torch.exp(scores - mx)
    denom = probs.sum(dim=1)
    return (trimmed_k * probs).sum(dim=1) / denom


def test_kpool_softmax_compress_matches_reference():
    from vllm_ascend.models.glm5next_w2.dsa import _kpool_softmax_compress

    T, D, ratio = 12, _DIMS["index_head_dim"], _DIMS["index_kpool"]
    torch.manual_seed(17)
    raw_keys = torch.randn(T, D, dtype=torch.float64)
    gate_score = torch.randn(T, D, dtype=torch.float64)
    ape = torch.randn(ratio, D, dtype=torch.float64)

    got = _kpool_softmax_compress(raw_keys, gate_score, ape, ratio, accum_dtype=torch.float64)
    ref = _ref_kpool_compress(raw_keys, gate_score, ape, ratio)
    assert got.shape == (T // ratio, D)
    assert torch.allclose(got, ref, atol=0.0, rtol=0.0), "kpool compression diverged from reference"


def test_glm_parity_indexer_uses_gate_ape_weighted_pool():
    from vllm_ascend.models.glm5next_w2.dsa import Glm5NextW2DsaIndexer

    T = 16
    torch.manual_seed(23)
    hidden = torch.randn(T, _DIMS["hidden_size"], dtype=torch.float64)
    qr = torch.randn(T, _DIMS["q_lora_rank"], dtype=torch.float64)
    positions = torch.arange(T)

    idx = Glm5NextW2DsaIndexer(
        hidden_size=_DIMS["hidden_size"],
        q_lora_rank=_DIMS["q_lora_rank"],
        index_n_heads=_DIMS["index_n_heads"],
        index_head_dim=_DIMS["index_head_dim"],
        index_topk=_DIMS["index_topk"],
        index_kpool=_DIMS["index_kpool"],
        dtype=torch.float64,
        use_compress_ape=True,
        use_compress_gate=True,
    )
    idx.reset_parameters(seed=99)

    # A non-trivial gate/ape must move the pooled key away from the mean pool.
    raw_keys, _ = idx.project_k_and_weights(hidden)
    gate = idx.project_gate(hidden)
    assert gate is not None and gate.shape == (T, _DIMS["index_head_dim"])

    pooled_gate = idx._compress_glm(raw_keys, gate, _DIMS["index_kpool"])
    pooled_mean = idx._compress(raw_keys, _DIMS["index_kpool"])
    assert not torch.allclose(pooled_gate, pooled_mean), "gate+ape pooling must differ from mean pool"

    ref = _ref_kpool_compress(raw_keys, gate, idx.compress_ape, _DIMS["index_kpool"])
    assert torch.allclose(pooled_gate, ref, atol=0.0, rtol=0.0)

    # The full indexer forward must route through the gate/ape-weighted pool.
    result = idx(hidden, qr, positions)
    assert result.token_mask.shape == (T, T)

