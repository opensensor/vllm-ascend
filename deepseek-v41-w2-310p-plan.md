# Plan: DeepSeek V4.1 (552B) 2-bit Experts on Four Ascend 310P Chips

**Generated**: 2026-09-16
**PRD**: `docs/source/developer_guide/Design_Documents/deepseek_v41_w2_310p_prd.md`
**Precedent**: the Qwen3.8-Flash-Next 1M 310P plan (`qwen38-flash-next-1m-310p-plan.md`) — reuse its host-first / device-last discipline, pathspec commits, `--noconftest` test runs, and per-op tolerance-before-comparison rule.
**Constraint**: 4×310P target NOT yet available. All code + host parity tests execute first; device work is a final serialized wave (D1–D5). The W2 checkpoint is producible locally (chunked); a rental is needed only for calibration/quality.

## Overview

Deliver a 310P execution path for DeepSeek V4.1 (`DeepseekV41ForCausalLM`: 40 layers, hidden 5120, MLA, sparse-attention indexer, 384-expert top-6 MoE, two Engram host layers, MTP-3), text-only, TP4/EP4 eager, to a validated real-weight **8K** boot. The routed experts are **W2 (2-bit) storage** unpacked to INT8 per active expert (top-6) and run through the existing `npu_quant_grouped_matmul_dequant`; the two Engram tables live in host RAM at ~W4. The heavy lifting reused from the Qwen work is the streamed loader, TP4/EP4 accounting, host-table ownership + prefetch, W-quant MoE mapping/math, sparse-attention indexer scaffolding, KV specs, gates and observability. The new critical path is (1) a packed W2 format + streaming converter, (2) a W2→INT8 active-expert unpack + 310P W2 MoE method, (3) the DeepSeek assembly (MLA, indexer/CSA2, Engram×2, DSpark, top-6/384, MTP).

## Key codebase facts (verified 2026-09-16)

- Source FP8 checkpoint: `/run/media/matteius/20TB-drive/models/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (48 shards, 476 GB). `config.json`: `DeepseekV41ForCausalLM`, `deepseek_v41`, 40 layers, hidden 5120, vocab 129280, `n_routed_experts 384`, `num_experts_per_tok 6`, `moe_intermediate_size 2304`, `n_shared_experts 1`, `q_lora_rank 1280`, `qk_rope_head_dim 64`, `num_nextn_predict_layers 3`, `quantization_config {fp8, expert_dtype fp4, weight_block_size [32,32], scale_fmt ue8m0}`, nested `text_config`+`vision_config`. Index has `layers.N.engram.{embed,q_weight,k_weight,wkv}…` (≈2 Engram layers) and `…indexer…` tensors.
- 310P quant registry is **W8-only**: `vllm_ascend/_310p/quantization/methods/__init__.py` (w8a8_dynamic/static/s/sc). Device MoE = `torch_npu.npu_quant_grouped_matmul_dequant` (`_310p/quantization/methods/w8a8_dynamic.py`); host math mirror in `vllm_ascend/models/qwen4_exp/moe.py`.
- A W4A16 fused-MoE exists but is NOT 310P-registered and its 310P op support is unverified: `vllm_ascend/quantization/methods/wna16/w4a16.py` (`AscendW4A16FusedMoEMethod`, `npu_convert_weight_to_int4pack` + `npu_grouped_matmul` antiquant). No W2 anywhere.
- Reuse surface (all under `vllm_ascend/`): `_310p/sharded_state_loader_310p.py`, `observability/qwen38_mem_accounting.py`, `models/qwen4_exp/{ngram_embedding,ple_prefetch,weight_mapping,moe,indexer_qsa,qsa,kv_cache,dtype_policy}.py`, `tools/qwen38_1m/{build_manifest,hw_probe,env_freeze}.py`, `observability/qwen38_runlog.py`.

**SCOPE UPDATE (2026-09-16, recon)**: the DeepSeek architecture is NOT green-field.
- vllm-ascend already ships `vllm_ascend/models/deepseek_v4/` — full model + `compressor.py` (CSA2), `indexer.py` (sparse attn), `dspark.py`, `mtp.py`, `vision.py`, `vl_model.py` — but it is the **910/A2 Triton path** (`ops/triton/mul_add.muls_add_triton`, general `FusedMoEFactory`/`QuantizationConfig`), NOT the 310P Triton-free W8 grouped-matmul path. Also present: `ops/mla.py`, `attention/{mla_v1,sparse_flash_mla}.py`, `attention/context_parallel/mla_cp.py`, `models/deepseek_mtp.py`.
- The vLLM fork has `vllm/models/deepseek_v41/` — `common/engram.py`, `nvidia/{model,model_state,engram}.py`, `attention.py`, `sparse_mla.py`, `compressor.py`, `quant_config.py` — the authoritative V4.1 port source (like `qwen4_exp/nvidia/*`).
- **Consequence**: E2.1/E3.1/E3.2/E4.1/E4.2 are **ADAPT existing code into a Triton-free 310P variant** (mirror the Qwen `_310p` pattern), not port-from-scratch. The genuinely-new critical path is the **W2 arithmetic** (E0.1, E0.4-W2, E1.1, E1.2, E1.3) + the 310P Triton-free wiring. Re-examine deepseek_v4 / fork deepseek_v41 before writing any MLA/indexer/engram/assembly code.

## QUALITY CORRECTION (2026-09-16, from the W2 sensitivity harness — commit decb018d5)

The host proxy (`tools/deepseek_w2/w2_quality.py`) shows **naive round-to-nearest (RTN) 2-bit is quality-risky**: per-expert SwiGLU output vs the FP4 source has median rel-MSE ≈0.98 (cosine ≈0.58); W3 ≈0.26, W4 ≈0.046; ~22% of experts poor even at W4. Engram W4 is fine (0.019). The packed-W2 **format is correct** (error == exact scale/2 bound) — the loss is intrinsic to 2-bit RTN, not a bug. **Caveat**: isolated per-expert *weight* error only — excludes INT8 activations, the top-6/384 routing average (MoE is quant-robust), residual scale, and task accuracy, so end-to-end may be softer.

**Corrections (do NOT discard infra — format/unpack/method/loader/assembly are quant-algorithm-agnostic):**
1. Treat the current RTN-W2 artifact as **provisional** (a bring-up vehicle, not a quality claim).
2. Add **E-CAL: calibrated 2-bit** (GPTQ/AWQ-style — minimize expert *output* error with calibration data; emits the SAME packed-W2 codes our pipeline consumes). Calibration runs on a GPU (RTX 6000 PRO or rental) or the cards once up. Depends on E0.4/E1.1.
3. Add an **end-to-end perplexity gate** (once E4.1 assembly runs on the cards, D3/G2): measure the REAL W2 delta before trusting it — the routing average may rescue much of the per-expert loss. Same gate applies to GLM.
4. **W4 fallback** for the most sensitive experts / capacity-permitting models is documented (mixed-bit), though W4 does not fit DeepSeek 552B on 4×310P.

**Documentation policy**: op availability (int4pack, antiquant, MLA fused, sub-INT8 unpack) MUST be verified against the pinned CANN container, not assumed.

## Prerequisites

- vllm-ascend repo write access (this repo); vLLM fork at `/run/media/matteius/20TB-drive/vllm` (DeepSeek V4.1 upstream is preview-grade — pin the rev).
- Source FP8 checkpoint (above); optional BF16 source for higher-quality requant (decision E1.1).
- Pinned container per R1; `ruff`, `pytest`; pathspec commits (`git commit -- <files>`), Conventional Commits, `git commit -s`.
- 4×310P target — device wave only.

## Dependency Graph

```
LEGEND  [asc]=vllm-ascend  [x]=external/local-convert  [hw]=device wave  (R#)=reuse Qwen module

Wave 0 (infra, parallel):
  E0.1 W2 manifest+provenance   E0.2 env freeze(R)   E0.3 hw probe(R)
  E0.4 eager refs (W2 QDQ/MLA/indexer/Engram)   E0.5 mem accounting extend(R)
  E2.1 pkg + registry + precision policy (independent)
Wave 1 (W2 arithmetic — critical path):
  E1.1 packed W2 format + streaming converter ← E0.1
  E1.2 W2→INT8 unpack + host QDQ parity ← E0.4,E1.1
  E1.3 310P W2 fused-MoE method (registry) ← E1.2
Wave 2:
  E2.2 model state + MLA latent-KV specs ← E2.1
  E2.3 Engram host lookup (~W4) (R:T4.1/T4.4) ← E2.1
  E3.4 weight mapping + streamed W2 load ← E1.1,E2.1,E0.5
Wave 3 (components):
  E3.1 MLA attention + parity ← E2.1,E0.4
  E3.2 indexer / CSA2 sparse attn + parity ← E2.1,E0.4
  E3.3 W2 MoE forward wiring ← E1.3,E2.1
Wave 4:
  E4.1 full-model assembly + dummy boot ← E2.2,E2.3,E3.1,E3.2,E3.3,E0.4
  E4.2 MTP-3 registration (stub)     EOBS observability(R) ← E0.5

DEVICE WAVE (hardware, serial):
  D1 probe+freeze ← E0.2,E0.3,hw
  D1.5 on-device component parity (W2 unpack, MLA, indexer, Engram) ← D1,E1.3,E3.1,E3.2,E2.3
  D2 G1 real-weight 8K ← E4.1,E3.4,EOBS,D1,D1.5
  D3 G2 quant correctness 8K ← D2
  D4 headroom + long-context strategy decision ← D3
  D5+ long context ← D4

Follow-on: GLM 5.3 753B (bounded expert cache/offload), MTP decode, vision, 1M.
```

## Tasks

### E0.1 [x/asc]: DeepSeek V4.1 W2 manifest + provenance
- **depends_on**: []
- **location**: extend `tools/qwen38_1m/build_manifest.py` (or new `tools/deepseek_w2/build_manifest.py`); `artifacts/deepseek-v41-w2/manifest.json`
- **description**: Parse the FP8 source config/index + safetensors headers → manifest: layer/expert/Engram/indexer/MLA tensor families + counts, observed dtypes (FP8 weights, FP4 experts, scales), Engram table shapes, EOS, MTP layers, `weight_block_size`/`scale_fmt`. Reserve the frozen G2 threshold (pending runtime). Record source rev + intended per-family target precision (W2 experts, ~W4 Engram, FP16 rest).
- **validation**: manifest validates; expert/Engram/indexer families enumerated; dtype map matches headers.
- **status**: Completed (2026-09-16, commit 93e85b9a2) — `tools/deepseek_w2/build_manifest.py` → `artifacts/deepseek-v41-w2/manifest.json` (96085 tensors, 48 shards, 21 families, header-authoritative dtypes). **Real source scheme**: routed experts `layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight` = **I8 (FP4 packed)** + `.scale` = **F8_E8M0 (ue8m0)**, `weight_block_size [32,32]` (47,232 weights + 47,232 scales incl. MTP; main block 46080); **Engram** (layers 1,14) `embed.weight` F8_E4M3 `[~384M,256]` + `embed.scale` F8_E8M0 `[~384M,8]`, `wkv` F8_E4M3, `q/k` BF16; **MLA** `attn.{wq_a,wq_b,q_norm,wkv,kv_norm,wo_a,wo_b,attn_sink(F32)}` (687) + `mla_compressor` (11, BF16); **indexer** `attn.indexer.{wq_b,wk,k_norm,weights_proj}` (32); shared_expert F8_E4M3; embed/lm_head/norms BF16; vision recorded but text-only. No generation_config → EOS=1/BOS=0/PAD=2 from config. Target precision recorded per family (W2 experts / ~W4 Engram / FP16 rest); g2 threshold pending. 8 UTs; ruff clean.
- **files**: `tools/deepseek_w2/{build_manifest.py,__init__.py}` (new), `tests/ut/deepseek_w2/test_build_manifest.py` (new), `artifacts/deepseek-v41-w2/manifest.json` (new)

### E0.2 [asc]: Environment freeze (reuse)
- **depends_on**: []
- **location**: reuse `tools/qwen38_1m/env_freeze.py`
- **description**: Add DeepSeek source rev + W2 converter rev to the frozen record. UT reuses the T0.2 pattern.
- **status**: Reuse-complete (2026-09-16) — `tools/qwen38_1m/env_freeze.py` (T0.2) is generic; add the DeepSeek source rev + W2 converter rev as recorded fields when E1.1 lands. No new code required now.

### E0.3 [asc]: Hardware probe (reuse as-is)
- **depends_on**: []
- **location**: `tools/qwen38_1m/hw_probe.py` (unchanged)
- **description**: Runs on target at D1; no code change expected (within/cross-card classification already fits the 2×300I-Duo topology).
- **status**: Reuse-complete (2026-09-16) — `tools/qwen38_1m/hw_probe.py` (T0.3) reused as-is; no code change. Runs on target at D1.

### E0.4 [asc]: Eager reference harness (W2 QDQ, MLA, indexer, Engram)
- **depends_on**: []
- **location**: `tests/ut/deepseek_w2/reference/`
- **description**: Pure-PyTorch FP64/FP32 references: W2 pack/unpack + INT8 grouped-QDQ MoE (per-block W2 scale, per-token act quant); DeepSeek MLA (q_lora/kv_lora down/up, decoupled rope, latent KV); sparse-attention indexer scoring/selection (CSA2 compression); Engram hash + gather + projection. Tolerances declared before any final comparison. Reuse the Qwen `w8a8_reference`/`qsa_*_reference` where shapes allow.
- **validation**: self-consistency UTs (W2 round-trip within bound; MLA vs dense-latent; indexer vs brute force).
- **status**: Completed (2026-09-16, commits 8ff7e26df W2+Engram, 2791496c6 MLA+indexer) — all 4 pure-torch references under `tests/ut/deepseek_w2/reference/`, 106 self-consistency UTs, ruff clean. **W2 reference contract**: signed int2 codes `{-2,-1,0,1}` packed 4/byte + per-`[32,32]`-block FP scale (round-trip ≤ scale/2), `unpack_w2_to_int8` + grouped==per-token MoE (reuses the qwen38_1m W8A8 per-token quantizer); skew + unrouted-skip covered. **Engram**: rolling-XOR n-gram hash (exact int64, prime-bucket + per-layer multipliers) with pad/dead-token blocking, int8×ue8m0 gather, signed-sqrt sigmoid gated projection. **MLA**: absorbed-latent vs dense oracle (q/kv lora, decoupled RoPE on 64-dim tail). **Indexer/CSA2**: mean-pool compression + Lightning-Indexer scoring + top-k, vs brute force at boundaries. Tolerances declared in `tolerances.py` before all asserts. NOTE for E1.2/E1.3: this file is the W2 pack/scale contract — reconcile the E1.1 converter format against it.
- **files**: `tests/ut/deepseek_w2/reference/{__init__,tolerances,w2_moe_reference,engram_reference,mla_reference,indexer_reference}.py` + 4 `test_*_selfconsistency.py` (new)

### E0.5 [asc]: Per-rank memory accounting extension
- **depends_on**: []
- **location**: extend `vllm_ascend/observability/qwen38_mem_accounting.py` (add `W2_EXPERT`, `ENGRAM_HOST` (~W4), `UNPACK_CACHE` components) or a thin DeepSeek wrapper
- **description**: Track W2 packed experts (HBM), the INT8 unpack cache (HBM), and the ~W4 Engram host table (counted once, host-resident, excluded from device totals). Same 5% imbalance guard.
- **validation**: UT: component sums vs a synthetic W2/Engram trace; Engram not ×rank.
- **status**: Completed (2026-09-16, commit 90cf438fe) — `observability/deepseek_w2_mem_accounting.py`: thin layer importing (not editing) the Qwen accountant. `DeepSeekW2MemComponent` adds `W2_EXPERT`/`UNPACK_CACHE` (device, under the 5% imbalance guard) and `ENGRAM_HOST` (host); `DEEPSEEK_HOST_COMPONENTS` unions with the Qwen `PLE_HOST_TABLE` so `DeepSeekW2RankMemoryReport` excludes Engram from device totals and `host_table_bytes()` counts it once (raises on cross-rank divergence). 20 UTs green; ruff clean.
- **files**: `vllm_ascend/observability/deepseek_w2_mem_accounting.py` (new), `tests/ut/deepseek_w2/test_mem_accounting.py` (new)

### E1.1 [x]: Packed W2 format + streaming converter
- **depends_on**: [E0.1]
- **location**: `tools/deepseek_w2/w2_convert.py`, format spec in the PRD/doc; output `artifacts/deepseek-v41-w2/` shards + manifest
- **description**: Define the packed W2 layout (2-bit codes + per-block scale, block shape aligned to source `weight_block_size [32,32]` AND the INT8 grouped-matmul `w13`/`w2` per-output-channel layout). Stream the FP8/FP4 source in bounded chunks (never materialize a full expert bank); quantize routed experts → W2, Engram → ~W4, leave MLA/indexer/dense/LM-head FP16. Record provenance (source rev, per-tensor precision, packing params). Decision logged: requant from FP8/FP4 source vs a BF16 source (quality).
- **validation**: converter round-trips a few real experts within the E0.4 W2 tolerance; bounded RSS during conversion; manifest updated with per-shard sha256.
- **status**: Completed (2026-09-16, commit 3f1b2d841) — `tools/deepseek_w2/w2_format.py` (packed layout) + `w2_convert.py` (streaming FP4→W2). **Source dequant bit-exact** vs the fork `reference_mxfp4.dq_mxfp4_torch` on real layer-6 experts (MXFP4 E2M1, low-nibble-first, ue8m0=2^(byte-127) per 32). **Packed-W2 contract (E1.2/E1.3 must match, bit-identical to E0.4 ref)**: codes int2 `{-2,-1,0,1}` packed 4/byte little-endian → `uint8[out, in/4]`; scale fp32 per-`[32,32]`-block → `[out/32, in/32]`; dequant = `code * broadcast(block_scale)`. W4 (Engram): int4 `{-8..7}`, 2/byte. Round-trip bound `|w-recon| ≤ 0.5·blockscale·(1+1e-6)+1e-9` (asserted; 2-bit is coarse — sampled half-step ~0.021 vs source absmax 0.125). Bounded RSS: peak tile 2.36–5.24 MB vs 47.2 MB full weight; never holds a bank. `dequant_fp8_e4m3` available for the FP8→FP16/W4 source path. 11 UTs; ruff clean; sample `.pt` binaries gitignored (only manifest committed).
- **files**: `tools/deepseek_w2/{w2_format.py,w2_convert.py}` (new), `tests/ut/deepseek_w2/test_w2_convert.py` (new), `artifacts/deepseek-v41-w2/sample/{conversion_manifest.json,.gitignore}` (new)

### E1.2 [asc]: W2→INT8 active-expert unpack + host QDQ parity
- **depends_on**: [E0.4, E1.1]
- **location**: `vllm_ascend/models/deepseek_v41/w2_unpack.py`, `tests/ut/deepseek_w2/test_w2_moe_parity.py`
- **description**: Unpack packed W2 → INT8 (per-block scale applied) for a set of active experts into a bounded cache, then run the existing INT8 grouped-QDQ math (reuse `qwen4_exp/moe.py` grouped path). Compare 384-expert top-6 + shared forward vs the E0.4 W2 reference at declared tolerances; cover skew (all tokens→1 expert), router renorm, per-block scale application order. No `.item()` in hot paths.
- **validation**: parity UT green; skew + renorm + scale-order guards have teeth (wrong order diverges).
- **status**: Completed (2026-09-16, commit eeb61bf34) — new package `vllm_ascend/models/deepseek_v41/` + `w2_unpack.py`. `unpack_active_experts(experts, topk_ids)` widens packed W2 (`uint8[out,in/4]`) → signed int8 codes + broadcast `[out/32,in/32]`→`[out,in]` block scale (dequant operand `codes*scale` applied element-wise pre-GEMM, since the per-32-column block scale can't fold into a per-output-channel factor); gate/up fused to `w13`, down→`w2`; bounded to the UNIQUE active experts (`torch.unique`, one boundary sync — cache size == #active, 6/12; 1 under skew; never the 384 bank). `w2_active_moe_forward` mirrors `qwen4_exp/moe.py` grouped INT8 QDQ (per-token act quant, one GEMM/expert, SwiGLU, softmax→top6→renorm, higher-precision shared expert; no hot `.item()`). **Parity == E0.4 `w2_moe_reference`, max abs err 0.0** (tol 1e-9). Guards bite: skew, active-only size, renorm-off diverges, per-block-scale-order diverges if folded post-matmul. 15 UTs + 106 E0.4 = **121 green**; ruff clean. **E1.3 wraps**: `unpack_active_experts()` + `w2_active_moe_forward(cache=...)`.
- **files**: `vllm_ascend/models/deepseek_v41/{__init__,w2_unpack}.py` (new), `tests/ut/deepseek_w2/test_w2_moe_parity.py` (new)

### E1.3 [asc]: 310P W2 fused-MoE method + registry
- **depends_on**: [E1.2]
- **location**: `vllm_ascend/_310p/quantization/methods/w2_dynamic.py`, register in `_310p/quantization/methods/__init__.py` + `registry.py`
- **description**: `AscendW2DynamicFusedMoEMethod310`: holds packed W2 params + per-block scales; `apply` unpacks the active experts (E1.2) and calls `npu_quant_grouped_matmul_dequant`. Mirror `AscendW8A8DynamicFusedMoEMethod310`'s param/weight-loading surface so the loader (E3.4) and mapping reuse hold. Unpack op verified against pinned CANN (fused if available, else elementwise on ≤6 experts).
- **validation**: CPU UT: method builds params from a synthetic W2 index; `apply` (host-math stub) equals the E1.2 path; registry resolves the method.
- **status**: Completed (2026-09-16, commit 0bf0147be) — `AscendW2DynamicFusedMoEMethod310` (`_310p/quantization/methods/w2_dynamic.py`): packed-W2 params (`w13_codes`/`w2_codes` uint8[E,out,in//4] + block scales fp32[E,out//32,in//32]) + shared-expert params, mirroring the W8A8 method surface (E3.4 reuse). `apply` dispatches on `_device_kernel_available()`: device path (guarded) unpacks ≤top_k active experts (E1.2) then `npu_quant_grouped_matmul_dequant`+`npu_swiglu`; host path re-expresses via E1.2 `w2_active_moe_forward`. Registered additively as `(W2A8_DYNAMIC, moe)` via the `@register_scheme` decorator + package import (registry.py untouched); the 5 W8 schemes still resolve. Parity == E0.4/E1.2 reference @ 1e-9 (with/without shared expert). **D1.5 device note (in-module)**: pinned CANN has NO fused W2→INT8 unpack op → elementwise unpack over active experts only; per-`[32,32]` block scale applied into the weight pre-matmul (not foldable to per-output-channel). 15 UTs; full deepseek_w2 suite **175 passed**; ruff + py_compile clean. (Commit lacks the Co-Authored-By trailer — no-`--amend` rule; left as-is.)
- **files**: `vllm_ascend/_310p/quantization/methods/w2_dynamic.py` (new), `_310p/quantization/methods/__init__.py` (+import), `tests/ut/deepseek_w2/test_w2_method.py` (new)

### E1.4 [x]: Full-model resumable W2/W4/FP16 converter (produces the 552B artifact)
- **depends_on**: [E1.1, E0.1]
- **location**: `tools/deepseek_w2/convert_full.py`, test `tests/ut/deepseek_w2/test_convert_full.py`
- **description**: Walk the source index; family→precision routing (W2 routed experts / W4 engram embed+wkv / FP16 shared+MLA+indexer+MTP+dense via FP8→FP16 dequant or BF16 cast / EXCLUDE vision); streaming + bounded RSS; ~5 GB safetensors shards + `model.safetensors.index.json` + `conversion_manifest.json` (per-shard sha256+bytes, per-tensor provenance) + `progress.json` + config/tokenizer copy. Resumable: deterministic pre-planned shard bins, atomic temp+rename, sha256-skip on re-run, Ctrl-C safe; huge ~384M-row engram embeds row-partitioned (10 parts) with 32-row block padding.
- **validation**: 24 CPU UTs (routing, dtype/shape, index+manifest+sha256, partition/pad, W2/W4 half-step recon, FP16 fidelity, resume/skip + partial-recovery, bounded RSS) on a synthetic real-dtype checkpoint.
- **status**: Completed (2026-09-16, commit 793c42d6d) — 24 UTs green; ruff clean; safetensors 0.7.0. Real-index dry-run: **47232 W2 / 22 W4 / 1122 FP16 / 138 excluded, ~50 shards, ~258 GB**.  **VERIFIED (step 3, 2026-09-16)**: full run complete — 50/50 shards, 258.2 GB, peak RSS 24.6 GB (bounded), ~79 min. Counts: W2 47232 (==manifest), FP16 1122, W4 4 source/22 parts, excluded 3 families/138 tensors; in-shard dtypes codes=U8, scale=F32, plain=F16; all 50 shard sha256 recorded, shards 1/26/50 re-hashed OK. **Engram W4 ≈99 GB host** (~2× the ~50 GiB estimate but fits 256 GB host; W3≈74/W2≈49 GB fallback via `bits`). Needed an OOM fix: posix_fadvise DONTNEED per shard (commit b79684234) on the 122 GB dev box. **Full 476 GB→W2 run launched by orchestrator** to `/run/media/matteius/20TB-drive/models/DeepSeek-V4.1-W2-310p` (background; verify counts/precision/RSS/sizes/sha256 on completion — plan step 3). **Note**: real Engram at W4 (~2×384M×256 params) is larger than the ~50 GiB first estimated — verify host-RAM budget in step 3.
- **files**: `tools/deepseek_w2/convert_full.py` (new), `tests/ut/deepseek_w2/test_convert_full.py` (new)

## E2–E4 REWRITE (2026-09-16, from the adaptation-seams review — supersedes the green-field E2.x/E3.x/E4.x blocks below)

Detail + config-delta table + per-seam guidance: `docs/source/developer_guide/Design_Documents/deepseek_v41_w2_310p_adaptation_seams.md` (commit a334ef3ed). Key facts: shipped `vllm_ascend/models/deepseek_v4/` is **config-driven and `torch_npu`-native** (only ONE Triton op: `muls_add_triton` at `deepseek_v4/model.py:425,432` = `x*scale+y` → replace with eager `addcmul`); the W2 kernel is **already built** (E1.2 `w2_unpack.py`, E1.3 `w2_dynamic.py` registered `(W2A8_DYNAMIC, moe)`). Three real V4→V4.1 deltas: **Engram** at `engram_layer_ids=[1,14]` (the only NEW sub-block; shipped V4 has hash-MoE, no Engram), **`compress_ratios` {0,1,2}** (shipped indexer gates on `==4`), and **W2 experts** (done). V4.1 config: 40 layers, hidden 5120, 384 experts top-6, moe_inter 2304, q_lora 1280, MTP-3, dspark [37,38,39].

| Task | Verdict | Concrete seam |
| --- | --- | --- |
| **E2.1** | ADAPT/CREATE | thin `deepseek_v41/` package importing shipped `deepseek_v4`; add `dtype_policy.py`; register `DeepseekV41ForCausalLM` (+ CondGen alias rejecting MM). dep: none |
| **E2.2** | ADAPT | reuse asc KV specs; recompute per-layer plan for `compress_ratios {0,1,2}`. dep: E2.1 |
| **E2.3** | CREATE (only new sub-block) | Engram host table = Qwen `ngram_embedding.py`+`ple_prefetch.py` host method (one shared ~W4 384M-row copy, never ×rank) + port the fork's **torch/numpy** n-gram hashing (NOT the Triton kernels); inject at layers [1,14]. dep: E2.1 |
| **E3.1** | ADAPT | `mla.py`: reuse asc `DeepseekV4Attention` linears/RoPE + `ops/mla.py`/`attention/mla_v1.py`, eager on 310P. dep: E2.1,E0.4 |
| **E3.2** | ADAPT | `indexer.py`: reuse asc `deepseek_v4/indexer.py`; ratio {1,2} gate + torch fallback for the `npu_*` selection ops (verify 310P availability). dep: E2.1,E0.4 |
| **E3.3** | ADAPT | `moe.py`: drop hash-MoE; run router (softmax→top6→renorm, `routed_scaling_factor=1.5`) → hand topk to **E1.3 `AscendW2DynamicFusedMoEMethod310`** (not `FusedMoEFactory`); replace `muls_add_triton` with eager. dep: E1.3,E2.1 |
| **E3.4** | ADAPT | `weight_mapping.py`: map W2 codes+block scales to the E1.3 layout via the `_310p` sharded loader. dep: E1.1,E2.1,E0.5 |
| **E4.1** | ADAPT | `model.py`: subclass `DeepseekV4Model`, compose 40-layer stack + Engram hook at [1,14]; no FusedMoEFactory / Triton / 910-DSA. dep: E2.2,E2.3,E3.1,E3.2,E3.3,E0.4 |
| **E4.2** | ADAPT | register MTP-3 / DSpark reusing asc `dspark.py`/`mtp.py` (inherits E3.3 fixes). dep: E2.1 |

The original green-field E2.x/E3.x/E4.x blocks below are retained for history but SUPERSEDED by the table above; implement the adapt-based versions.

### E2.1 [asc]: Ascend DeepseekV41 package + registration + precision policy
- **depends_on**: []
- **location**: `vllm_ascend/models/deepseek_v41/{__init__,model,mla,indexer,engram,moe,mtp}.py`, register in `vllm_ascend/models/__init__.py`
- **description**: `AscendDeepseekV41ForCausalLM` (+ ConditionalGeneration alias rejecting multimodal at first gate; MTP registered, wired later). Authoritative precision policy: W2 experts / INT8 act, ~W4 Engram, FP16 MLA/indexer/dense/LM-head, FP32 accumulation — a single object all modules read (mirror Qwen `dtype_policy`). Expose `get_model_state_cls`, MLA/latent-KV hooks, per-component `load_weights` hooks. No Triton/CUDA on 310P.
- **validation**: CPU UT: import under faked NPU platform; tiny random deepseek_v41 config constructs on meta; all arch names register; grep-gate no triton; precision-policy table read by every module (no local literals).
- **status**: Not Completed

### E2.2 [asc]: Model state + MLA latent-KV specs
- **depends_on**: [E2.1]
- **location**: `vllm_ascend/_310p/worker/v2/` state, `vllm_ascend/models/deepseek_v41/kv.py`; reuse `kv_cache.py` plumbing
- **description**: MLA latent KV-cache spec (compressed kv_lora latent + decoupled rope key), block math, hybrid packaging with the indexer's compressed history; per-chip byte table (BF16/C8 variants for later long context). PP=1.
- **validation**: CPU UT: specs from a tiny config yield a valid KVCacheConfig; 8K + long-context block math exact.
- **status**: Not Completed

### E2.3 [asc]: Engram host lookup (~W4) — reuse T4.1/T4.4
- **depends_on**: [E2.1]
- **location**: `vllm_ascend/models/deepseek_v41/engram.py` (subclass/reuse `AscendPLEEmbeddingMethod` + `ple_prefetch`)
- **description**: One shared ~W4 host Engram table across 4 workers; quantized row gather (dequant on gather), batched dedup + async prefetch; fail-fast host accounting; never ×rank copies. Reuse the pinned-UVA / shared-mmap dual transport.
- **validation**: UT with /dev/shm ~W4 table: 4 procs share one copy; row reads correct (dequant); host bytes exact; no full-table pin by default.
- **status**: Not Completed

### E3.1 [asc]: MLA attention on 310P + parity
- **depends_on**: [E2.1, E0.4]
- **location**: `vllm_ascend/models/deepseek_v41/mla.py` (+ ops); reuse `vllm_ascend/attention/` MLA precedent where present (`attention/context_parallel/mla_cp.py`)
- **description**: DeepSeek MLA: q down/up (q_lora_rank 1280), kv down/up (kv_lora_rank), decoupled rope (qk_rope_head_dim 64), latent KV write/read; eager torch on 310P (no Triton). Check existing `vllm_ascend` MLA ops for 310P reuse.
- **validation**: CPU parity UT vs E0.4 MLA reference at boundary lengths; latent-KV shape/dtype asserts from config.
- **status**: Not Completed

### E3.2 [asc]: Sparse-attention indexer / CSA2 + parity
- **depends_on**: [E2.1, E0.4]
- **location**: `vllm_ascend/models/deepseek_v41/indexer.py` (+ ops); adapt Qwen `indexer_qsa.py`/`qsa.py`
- **description**: Port the DeepSeek sparse-attention indexer (scoring, selection, compression/CSA2) to torch/NPU ops, deterministic top-k, cache slot mappings. Reuse the Qwen QSA indexer structure where the math aligns.
- **validation**: CPU parity UT vs E0.4 indexer reference at boundary lengths; deterministic selection.
- **status**: Not Completed

### E3.3 [asc]: W2 MoE forward wiring
- **depends_on**: [E1.3, E2.1]
- **location**: `vllm_ascend/models/deepseek_v41/moe.py`, assembly
- **description**: Wire the assembly's MoE block to the E1.3 W2 method (top-6/384 router renorm + shared expert). Host-math path mirrors E1.2 for CPU validation.
- **validation**: CPU parity UT vs E0.4/E1.2 W2 reference; router renorm correct.
- **status**: Not Completed

### E3.4 [asc]: Weight mapping + streamed W2 load
- **depends_on**: [E1.1, E2.1, E0.5]
- **location**: `vllm_ascend/models/deepseek_v41/weight_mapping.py`, reuse `_310p/sharded_state_loader_310p.py`
- **description**: Map W2 expert tensors + per-block scales → the E1.3 method layout; FP16 MLA/indexer/dense/LM-head by name; Engram → host (E2.3). Stream by TP4 (EP4 ready), no full-bank materialization; integrate E0.5 accounting; simulate per-chip bytes from the manifest. Reject missing/extra/duplicate/wrong-shape/dtype.
- **validation**: CPU sim UT: per-rank bytes ≤ HBM target; no double instantiation; bounded RSS; all rejection classes fire.
- **status**: Not Completed

### E4.1 [asc]: Full-model assembly + dummy-weight boot
- **depends_on**: [E2.2, E2.3, E3.1, E3.2, E3.3, E0.4]
- **location**: `vllm_ascend/models/deepseek_v41/model.py`
- **description**: Assemble the 40-layer model (MLA/indexer attention, top-6/384 MoE + shared, two Engram layers at their positions, DSpark, norms, LM-head) using real wired components + E0.4 eager references as stubs where needed. Dummy-weight CPU/meta boot exercises control flow + KV-spec materialization; deterministic greedy smoke.
- **validation**: CPU: tiny random model loads dummy weights, forwards, samples; KV-group report matches E2.2; two fixed-seed forwards identical; no CUDA/Triton on 310P path.
- **status**: Not Completed

### E4.2 [asc]: MTP-3 registration (stub)
- **depends_on**: [E2.1]
- **description**: Register the `num_nextn_predict_layers=3` MTP class; not wired into decode (follow-on).
- **status**: Not Completed

### EOBS [asc]: Observability (reuse run-log)
- **depends_on**: [E0.5]
- **location**: reuse `vllm_ascend/observability/qwen38_runlog.py`
- **description**: Run artifacts + first-fatal-rank + within/cross-card collective trace; add W2/Engram/unpack-cache memory sections and W2 unpack metrics.
- **validation**: UT: injected one-rank failure names the rank; W2/Engram sections present.
- **status**: Not Completed

## Device wave (hardware only, serial)

- **D1** probe + env freeze on target; record ≈46 GiB actual free/chip, HCCL topology.
- **D1.5** on-device component parity (G0): W2 unpack+MoE, MLA, indexer, Engram lookup vs E0.4 refs on NPU.
- **D2** G1 — real-weight startup, TP4, eager, 8K: non-empty completion; per-rank weight report (≤5% imbalance); ≥8 GiB free/chip after load.
- **D3** G2 — quantized correctness at 8K: deterministic across restarts; quality delta vs the frozen FP8-source threshold. **W2-from-FP8 quality is the gate risk** — mixed-precision or BF16-requant fallback are recorded decisions.
- **D4** headroom + long-context strategy decision (MLA latent-KV layout, Engram device hot-cache?).
- **D5+** long context.

## Testing Strategy

- Host-only before D1: `pytest -sv tests/ut/deepseek_w2/... --noconftest` (the shared conftest breaks in this env; run isolated). No NPU before the device wave.
- Parity tolerances committed in-source before final comparisons.
- Import-hygiene: no triton/cuda reachable from the deepseek_v41 package on 310P.
- Determinism: bitwise-rerun for W2 unpack, indexer top-k, MoE grouping.

## Risks & Mitigations

- **W2-from-FP8 quality** — biggest risk; mixed precision (W2 experts / ~W4 Engram / higher-bit sensitive groups), BF16-requant fallback, rental calibration before any quality claim.
- **No native W2 GEMM** — unpack active experts (top-6) to INT8, reuse the validated grouped matmul; unpack is the only new device op and runs on ≤6 experts.
- **310P op gaps (int4pack/antiquant/MLA fused)** — verify pinned CANN first; eager/elementwise fallbacks.
- **HBM headroom** — measured ≥8 GiB/chip at G1; expert-cache/offload fallback (mandatory for GLM 5.3, optional here).
- **DeepSeek V4.1 preview-grade upstream** — real port; pin rev; validate assembly vs eager reference before device.

## Open decisions (resolve in task logs)

1. W2 packing layout vs source `weight_block_size [32,32]` and the INT8 grouped-matmul expectation (E1.1).
2. Unpack target INT8 (confirmed) vs INT4 (if 310P antiquant available) (E1.3/D1.5).
3. Engram precision (W4 vs W3) + host-only vs device hot cache (E2.3/D4).
4. Frozen G2 threshold + whether the FP8 reference runs on 310P or only a rental GPU (E0.1/D3).
5. Requant from FP8/FP4 vs BF16 source (E1.1).
