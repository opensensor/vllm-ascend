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

## Quality note (same correction as DeepSeek)
Naive RTN W2 is **provisional** (the W2 sensitivity finding applies to GLM experts too); the algorithm-agnostic infra is reused; a calibrated-2-bit path + an end-to-end perplexity gate (on the cards) settle quality. FP8→W2 (vs DeepSeek's FP4→W2) starts from a higher-precision source, so GLM's W2 quality should be no worse.

## Device wave
Same as DeepSeek: D1 probe/freeze → D1.5 component parity → D2 real-weight 8K → D3 quant/perplexity gate → long context. Needs the 4×310P hardware.
