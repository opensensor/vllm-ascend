# Plan: GLM-5.3-Flash (glm5_next) 2-bit Experts on Four Ascend 310P Chips

**Generated**: 2026-09-16
**Precedent**: the DeepSeek V4.1 W2 plan (`deepseek-v41-w2-310p-plan.md`) — reuse the whole W2 pipeline (format/kernel/loader/converter/assembly pattern), host-first discipline, pathspec commits, `--noconftest`, per-op tolerances.

## Overview

Deliver a 310P W2 execution path for **GLM-5.3-Flash** (`Glm5NextForConditionalGeneration` / `glm5_next`: 45 layers, hidden 4096, **288 routed experts top-8**, moe_inter 2048, 1 shared expert, MTP-1), text-only, TP4/EP4 eager, to a real-weight 8K boot. **Structurally lighter than DeepSeek** — no Engram, no sparse indexer; only routed experts are quantized. The routed experts are packed **W2** (unpacked to INT8 per active expert, reusing the E1.2/E1.3 kernel), everything else FP16; `Glm5Next` is already registered in vllm-ascend so the model adaptation is light.

## Source (verified 2026-09-16)

- FP8 base: `zai-org/GLM-5.3-Flash` at `/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8` (62 shards, 306 GB; chosen over `nvidia/GLM-5.3-Flash-NVFP4` because FP8→W2 avoids a double-4-bit-quant and needs no new NVFP4 dequant). arch `Glm5NextForConditionalGeneration`, 45 layers, hidden 4096, 288 experts top-8, `weight_scale_inv` F32 [128,128] block scales, MTP-1, vocab 154880, multimodal (vision excluded, text-only).
- Routed experts `mlp.experts.{E}.{gate,up,down}_proj.weight` = **F8_E4M3 + F32 [128,128] block** → **W2** (37,152 tensors). 186 non-routed F8 tensors (dense L0-2, shared_experts, some MLA proj) → FP16 via dequant. Everything else BF16/F32 → FP16. `model.visual.*` (347) EXCLUDE.
- NVFP4 partial (`GLM-5.3-Flash-NVFP4`, 28 GB) parked for a future **RTX PRO 6000** (native Blackwell) deployment — not the Ascend source.

## Reuse (all under vllm-ascend, from the DeepSeek W2 work)

W2 format/pack (`tools/deepseek_w2/w2_format.py`), unpack kernel + 310P method (`models/deepseek_v41/w2_unpack.py`, `_310p/quantization/methods/w2_dynamic.py` — `(W2A8_DYNAMIC, moe)`), streamed sharded loader + placement (`_310p/sharded_state_loader_310p.py`), per-rank accounting (`observability/deepseek_w2_mem_accounting.py`), converter machinery (`tools/deepseek_w2/convert_full.py`), and the **shipped `Glm5Next` model** in vllm-ascend.

## Tasks

### G1: W2 manifest + FP8→W2 converter — COMPLETED (2026-09-16, commit 75f3b273b)
`tools/glm_w2/{build_manifest,convert_full}.py` + `tests/ut/glm_w2/` (35 UTs). New `dequant_fp8_e4m3_f32block` (F8_E4M3 × plain F32 [128,128] block, validated bit-exact — `0xE8`→-64.0 × 1.918e-4 = -0.012277). GLM routing: routed experts→W2, 186 non-routed F8→FP16-dequant, rest BF16/F32→FP16, vision excluded. Manifest `artifacts/glm-5.3-flash-w2/manifest.json` (76,108 tensors, 62 shards). Estimated output **~97.3 GB** (W2 ~79.1 + FP16 ~18.2). Reuses the DeepSeek convert_full (sharding/resume/fadvise/sha256). **Flag**: router `e_score_correction_bias` + `hc_*` gates + `A_log`/`dt_bias` (genuine F32) are cast to FP16 — revisit if accuracy needs F32.

### G2: Run full conversion + verify — COMPLETED (2026-09-16)
Full FP8→W2 done → `/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-W2-310p`: **19 shards, 97.4 GB, peak RSS 26.7 GB (bounded), 347 vision excluded, ~37 min**. Verified: W2=37152 (==manifest routed experts), FP16=1271; in-shard dtypes codes=U8/scale=F32/plain=F16; all 19 shard sha256 recorded, shards 1/10/19 re-hashed OK. Fits 4×310P with ample headroom (W2 experts ~20 GB/chip, no host Engram table).

### G3–G7: Glm5Next 310P W2 adaptation — ADAPT existing package (recon 2026-09-16)
**vllm-ascend already ships a FULL `vllm_ascend/models/glm5next/` package** — `model.py`, `kv_cache.py`, `cache_config.py`, **`kda.py`** (KDA linear attention — glm5_next is a HYBRID: KDA linear-attn + full-attn layers, like Qwen4Exp GDN/QSA), `mtp.py`, `multimodal.py`, `processor.py`, `ops/`, + `patch/platform/patch_glm5next_config.py`; registered `Glm5NextForCausalLM`/`ForConditionalGeneration`/`Glm5NextMTPModel`. So G3–G7 is an ADAPT (like DeepSeek's deepseek_v4→deepseek_v41), with MORE reuse (attention/KDA/kv/mtp exist). **Next: a Glm5Next delta review** (Triton/910 vs 310P-ready? which MoE method?) then the adaptation wave: package/dtype-policy override, MoE→E1.3 W2 method (swap FusedMoE + any Triton op), W2 weight-mapping/streamed load, assembly + dummy-weight CPU boot. No Engram, no sparse indexer.

## G3–G7 breakdown (from the glm5next recon)

glm5_next is a HYBRID: `layer_types` = 34 **KDA** linear-attention layers (`kda_layers`) + 11 **DSA** deepseek-sparse-attention layers (`full_attn_layers` [3,7,...,43]); `first_k_dense_replace=3` (first 3 dense, rest MoE 288/top-8, `routed_scaling_factor=2.5`); KDA cfg 64 heads/head_dim 128/short_conv 4. Shipped `glm5next/model.py` uses `FusedMoEFactory`+`QuantizationConfig`+Triton KDA (910 path) — NO `_310p` glm5next. Reuse: **DSA ≈ deepseek_v41 indexer (E3.2)**, **MoE→E1.3 `AscendW2DynamicFusedMoEMethod310`**, hyperconnection (`mhc`) ≈ Qwen; the one NEW component is **KDA** (gated-delta linear attn, ~like Qwen GDN — reuse `_310p/ops/fla/*` or eager).

- **G3** package + dtype policy + registration (adapt shipped glm5next → 310P W2 variant; text-only alias). dep: none.
- **G-ref** eager KDA reference (new) + reuse DeepSeek indexer/W2 refs for DSA/MoE. dep: none.
- **G4** KDA linear attn on 310P (Triton-free — adapt kda.py, reuse `_310p/ops/fla` GDN kernels or eager) + parity. dep: G3,G-ref.
- **G5** DSA sparse attention on 310P — REUSE the deepseek_v41 `indexer.py`/`mla`-style path. dep: G3.
- **G6** MoE→E1.3 W2 method (swap FusedMoEFactory, top-8/288, scaling 2.5) + hyperconnection + W2 weight-mapping. dep: G3,E1.3.
- **G7** assembly + dummy-weight CPU boot + MTP-1. dep: G4,G5,G6.

### G3: package + dtype policy + registration — COMPLETED (2026-09-16, commit 948332f3b)

**Status**: DONE, TDD RED→GREEN, 11/11 UTs pass (`python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm5next_w2_package.py`). DeepSeek V4.1 E2.1 package test still 11/11 (additive, no regression). ruff clean.

**Work log** (ADAPT mirroring `deepseek_v41` E2.1):
- New additive package `vllm_ascend/models/glm5next_w2/` — does NOT touch the shipped `glm5next/`.
- `dtype_policy.py`: `Glm5NextW2DtypePolicy` (frozen) + `ASCEND_GLM5NEXT_W2_DTYPE_POLICY` singleton + `REQUIRED_CAST_SITES`. W2 experts `expert_weight`=uint8 / `expert_activation`=int8 / `expert_accumulation`=fp32; fp16 for `kda`,`dsa`,`dense`,`shared_expert`,`lm_head`,`main` (+ `mla`/`indexer` companions); fp32 for `accumulation`,`router`,`logits`. **No Engram sites** (GLM has none — dropped vs DeepSeek). `from_vllm_config(None)` pins fp16 main; honors a `torch.dtype` KV override only.
- `model.py`: PEP-562-lazy `AscendGlm5NextW2ForCausalLM` (subclasses shipped `glm5next.model.Glm5NextForCausalLM`, base resolved LAZILY via `_shipped_causal_lm_base()` so the import path stays Triton-free) + `AscendGlm5NextW2ForConditionalGeneration` (alias, `_reject_multimodal` first gate rejecting `vision_config`/`multimodal_config` — GLM `model.visual.*` excluded). Staged no-op W2 hooks `_swap_kda_to_eager` (G4) / `_override_dsa_indexer` (G5) / `_swap_moe_to_w2` (G6). Registration-only MTP-1 stub `Glm5NextW2MTP` (fails fast on construct).
- `__init__.py`: eager dtype-policy export; PEP-562-lazy forward of the reused DeepSeek E1.2/E1.3 W2 host-math (`w2_active_moe_forward` etc. from `deepseek_v41`) — expresses "GLM reuses the DeepSeek W2 kernel" without pulling anything heavy at import.
- Additive registration in `vllm_ascend/models/__init__.py`: `Glm5NextW2ForCausalLM` / `Glm5NextW2ForConditionalGeneration` / `Glm5NextW2MTPModel` → the new classes. Shipped `Glm5Next*` rows untouched (test asserts intact).

**Config-contract constants** (in `model.py`, all VERIFIED against `/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8/config.json` `text_config`): `GLM5NEXT_NUM_HIDDEN_LAYERS=45`, `N_ROUTED_EXPERTS=288`, `NUM_EXPERTS_PER_TOK=8`, `FIRST_K_DENSE_REPLACE=3`, `FULL_ATTN_LAYERS=(3,7,11,15,19,23,27,31,35,39,43)` (11 DSA; other 34 KDA), `KDA_NUM_HEADS=64`, `KDA_HEAD_DIM=128`, `ROUTED_SCALING_FACTOR=2.5`, `NUM_NEXTN_PREDICT_LAYERS=1`. **No deltas** from the task brief — every stated constant matched the source config exactly. Also captured: `hidden_size=4096`, `n_shared_experts=1`, `moe_intermediate_size=2048`, `vocab_size=154880`, `short_conv_kernel_size=4` (`KDA_SHORT_CONV_KERNEL_SIZE`), `KDA_LAYERS` (derived 34-tuple). NB source config `model_type` is `glm5_next` (top) / `glm5_next_text` (text_config).

**Gotchas**:
- Shipped `glm5next.model` pulls Triton via `glm5next.kda` → `vllm_ascend.ops.triton.kda.kda` (the G4 swap target) AND `FusedMoEFactory` (G6). Import-hygiene test greps the W2 source for triton AND asserts a fresh-interpreter import of the W2 package+model pulls neither `vllm_ascend.models.glm5next.model` nor `vllm_ascend.ops.triton.kda.kda` (analogue of DeepSeek's `mul_add` gate; KDA op path substituted).
- New files must be `git add`-ed explicitly before a pathspec commit (`git commit -- <files>` won't stage untracked). Used explicit `git add <files>` (never `-A`/`.`).
- MTP registration points at the `model:Glm5NextW2MTP` stub (self-contained in G3, like DeepSeek's original E2.1 `model:DeepSeekV41MTP` stub); G7 repoints to the real drafter.

**Files**: `vllm_ascend/models/glm5next_w2/{__init__,dtype_policy,model}.py`, `vllm_ascend/models/__init__.py` (additive block), `tests/ut/glm_w2/test_glm5next_w2_package.py`.

## Quality note (same correction as DeepSeek)
Naive RTN W2 is **provisional** (the W2 sensitivity finding applies to GLM experts too); the algorithm-agnostic infra is reused; a calibrated-2-bit path + an end-to-end perplexity gate (on the cards) settle quality. FP8→W2 (vs DeepSeek's FP4→W2) starts from a higher-precision source, so GLM's W2 quality should be no worse.

## Device wave
Same as DeepSeek: D1 probe/freeze → D1.5 component parity → D2 real-weight 8K → D3 quant/perplexity gate → long context. Needs the 4×310P hardware.

### G-ref: Eager KDA linear-attention reference — COMPLETED (2026-09-16, commit 0dddfcbdd)
Pure-PyTorch (CPU, NO NPU, NO Triton) parity oracle for GLM-5.3-Flash KDA (Kimi Delta Attention), mirroring the shipped 910/Triton path `vllm_ascend/models/glm5next/kda.py` + `vllm_ascend/ops/triton/kda/*`. New `tools/glm_w2/kda_reference.py` + `tests/ut/glm_w2/test_kda_reference.py` (22 UTs, RED→GREEN proven by removing the module for the RED run).

**Math implemented (fp32, matches shipped semantics step-for-step):**
1. `short_conv1d_causal` — depthwise causal conv1d (`F.conv1d`, `groups=dim`, `padding=width-1` truncated to seqlen) + SiLU, per `ops/causal_conv1d.py::causal_conv1d_ref` (short_conv_kernel_size=4).
2. `kda_safe_gate` — bounded log-decay `g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))` (`ops/triton/kda/gate.py::apply_kda_gate`, safe_gate=True, lower_bound=-5.0). **Cross-checked bit-identical (0.0 max abs diff) against the shipped `apply_kda_gate`.**
3. `kda_recurrent_reference` — the required sequential oracle, mirroring `fused_recurrent_kda.py` `IS_KDA=True` inner loop per head with state `S[V,K]=[128,128]`: L2-norm(q,k) eps 1e-6 → `q*=scale (K**-0.5)` → `S*=exp(g)[None,:]` (decay along K) → `u=(v - S@k)*beta` (delta rule, per-head scalar beta=sigmoid(beta_raw)) → `S+=outer(u,k)` → `o=S@q`.
4. `gated_rmsnorm` — sigmoid-gated RMSNorm (eps 1e-5) per `FusedRMSNormGated(activation="sigmoid").forward_native` (`x_normed*weight*sigmoid(g)`), the `o_norm` before `o_proj`.
Plus `kda_chunked_reference` (state-carry chunked driver; consistency oracle — must equal the recurrence) and `kda_layer_reference` (end-to-end conv→gate→recurrence→o_norm, stops before o_proj).

**Assumptions/ambiguities for G4 to reconcile:** (a) L2-norm eps=1e-6, o_norm eps=1e-5, scale=K**-0.5 — all defaults read from the shipped kda wrapper/kernels. (b) beta is a per-head **scalar** for GLM (`IS_BETA_HEADWISE=False`, since `beta.ndim==3 != v.ndim==4`); a head-wise-beta checkpoint would need beta shape [T,H,V]. (c) The chunked *prefill* Triton path (`chunk_kda_with_fused_gate`) does log-space cumulative sums and may reorder fp32 adds vs. this strict left-to-right recurrence — small numeric drift there is expected; the **sequential recurrent form is the authoritative oracle** (the decode path `fused_recurrent_kda` is the exact structural match). (d) Bounded gate saturates to exactly 0.0 / lower_bound in fp32 at the extremes → decay∈[e^lb,1] with inclusive bounds (documented in the gate-bounds test). No NPU hardware needed to run the oracle; `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_kda_reference.py` → 22 passed.

### G5: DSA sparse attention on 310P — COMPLETED (2026-09-16)
Triton-free, NPU-optional (CPU-testable) 310P path for GLM-5.3-Flash's 11 DSA (`deepseek_sparse_attention`) `full_attn_layers`. TDD RED→GREEN, 8/8 UTs (`python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm5next_w2_dsa.py`); G3 package test still 11/11 (no regression); ruff clean; `dsa.py` has ZERO `triton` references.

**Files**: `vllm_ascend/models/glm5next_w2/dsa.py` (new), `vllm_ascend/models/glm5next_w2/__init__.py` (lazy `_DSA_EXPORTS` block only), `tests/ut/glm_w2/test_glm5next_w2_dsa.py` (new).

**Structural finding (drives G7 assembly): GLM DSA = MLA-latent NoPE + kpool Lightning-Indexer, NOT plain GQA and NOT DeepSeek's rope-tail MLA.** Verified against `/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8/config.json` `text_config`: `qk_rope_head_dim=0` (NoPE — the MLA core has NO decoupled RoPE tail, unlike deepseek_v41.mla which always rotates one), `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=256`, `v_head_dim=256`, `num_attention_heads=64`; indexer `index_topk=2048`, `index_n_heads=32`, `index_head_dim=128`, `index_kpool=4`, `indexer_rope_interleave=True`, `rms_norm_eps=1e-5`. Shipped `glm5next.attention.Glm5NextMLAAttention` uses `fused_qkv_a_proj`(q_lora|kv_lora)+`q_b_proj`+`kv_b_proj`+`o_proj` via the vLLM MLA wrapper, and its `Indexer` uses `wq_b`(q_lora→n_head·head_dim) + fused `wk_weights_proj`(hidden→[head_dim|n_head]) + `k_norm`(LayerNorm) + softmax/head weight scaling. The shipped `SparseAttnIndexerKpool.forward_oot` raises `NotImplementedError` on Ascend (CUDA-only DeepGEMM/radix-topk) — so G5 IS the first working 310P DSA path.

**REUSED from deepseek_v41 (imported verbatim, not forked)** — `vllm_ascend.models.deepseek_v41.indexer`: `mean_pool_compress` (kpool CSA2 pooling), `lightning_indexer_scores` (`score[t,m]=Σ_h w[t,h]·relu(scale·(q_{t,h}·k_pool_m))`), `select_topk_blocks`/`block_topk_for_ratio` (deterministic causal descending-score top-k, ascending block-id tie-break). A test asserts the GLM selection is **bit-identical** to a `AscendDeepseekV41Indexer(compress_ratio=2, build_projections=False)` module driven with the same precomputed query/weights/keys.

**GLM adapter (the thin new code)**: (a) kpool ratio — deepseek's indexer *module* gates ratios {0,1,2} but its *functions* work for any ratio≥1, so we drive them with `index_kpool` (=4); (b) GLM's own `wq_b`/`wk_weights_proj`/`k_norm` projections + weight scaling (`Glm5NextW2DsaIndexer`); (c) the **MLA-NoPE attention core** (`AscendGlm5NextW2DSA`): q down/up + kv down + `kv_b_proj`→per-head `[qk_nope|v]`, NO rope, `o_proj`, softmax restricted to selected tokens; (d) a **local current-pool window** `[floor(t/kpool)·kpool, t]` always allowed alongside the selected fully-formed visible blocks — guarantees non-empty self-inclusive attention for the first `kpool` tokens and keeps every attended key causal (a visible fully-formed block ⇒ all its tokens ≤ t), so DSA output at t is provably independent of tokens > t.

**Dtypes** from the G3 policy: fp16 IO (`dsa`/`indexer` sites), fp32 accumulation (`dsa_accumulation`); `dtype=torch.float64` override runs the whole module in fp64 for exact parity tests. The selection reduction is always float64 (matching the deepseek_v41 host indexer) so reuse-equivalence is exact. `AscendGlm5NextW2DSA.from_config(text_config)` is the G7 entry point; `attend_with_mask(hidden, mask)` exposes the bare core for G7 wiring / parity.

**Gotchas / for G7**: GLM's real kpool indexer additionally folds an absolute-position embedding (`index_kpool_compress_ape`) + a gate (`index_kpool_compress_gate`) into the pooled keys before scoring — this host module runs the plain mean-pool reuse (`use_compress_ape=False` default, exact deepseek selection) and exposes `compress_ape` as an opt-in adapter so a later device-parity task can wire the GLM APE/gate without touching call sites. The `_override_dsa_indexer` hook in `model.py` (staged no-op at G3) is where G7 attaches `AscendGlm5NextW2DSA` to the 11 DSA layers. **Note**: the G5 code commit's message carries `Signed-off-by` but the `Co-Authored-By: Claude Opus 4.8` trailer was omitted from that commit; not re-written because the strict G5 rules forbid `--amend`/history rewrites in the shared parallel tree (a parallel G4 agent was active on HEAD-adjacent commits).
