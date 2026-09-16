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
- **status**: Not Completed

### E0.2 [asc]: Environment freeze (reuse)
- **depends_on**: []
- **location**: reuse `tools/qwen38_1m/env_freeze.py`
- **description**: Add DeepSeek source rev + W2 converter rev to the frozen record. UT reuses the T0.2 pattern.
- **status**: Not Completed

### E0.3 [asc]: Hardware probe (reuse as-is)
- **depends_on**: []
- **location**: `tools/qwen38_1m/hw_probe.py` (unchanged)
- **description**: Runs on target at D1; no code change expected (within/cross-card classification already fits the 2×300I-Duo topology).
- **status**: Not Completed

### E0.4 [asc]: Eager reference harness (W2 QDQ, MLA, indexer, Engram)
- **depends_on**: []
- **location**: `tests/ut/deepseek_w2/reference/`
- **description**: Pure-PyTorch FP64/FP32 references: W2 pack/unpack + INT8 grouped-QDQ MoE (per-block W2 scale, per-token act quant); DeepSeek MLA (q_lora/kv_lora down/up, decoupled rope, latent KV); sparse-attention indexer scoring/selection (CSA2 compression); Engram hash + gather + projection. Tolerances declared before any final comparison. Reuse the Qwen `w8a8_reference`/`qsa_*_reference` where shapes allow.
- **validation**: self-consistency UTs (W2 round-trip within bound; MLA vs dense-latent; indexer vs brute force).
- **status**: Not Completed

### E0.5 [asc]: Per-rank memory accounting extension
- **depends_on**: []
- **location**: extend `vllm_ascend/observability/qwen38_mem_accounting.py` (add `W2_EXPERT`, `ENGRAM_HOST` (~W4), `UNPACK_CACHE` components) or a thin DeepSeek wrapper
- **description**: Track W2 packed experts (HBM), the INT8 unpack cache (HBM), and the ~W4 Engram host table (counted once, host-resident, excluded from device totals). Same 5% imbalance guard.
- **validation**: UT: component sums vs a synthetic W2/Engram trace; Engram not ×rank.
- **status**: Not Completed

### E1.1 [x]: Packed W2 format + streaming converter
- **depends_on**: [E0.1]
- **location**: `tools/deepseek_w2/w2_convert.py`, format spec in the PRD/doc; output `artifacts/deepseek-v41-w2/` shards + manifest
- **description**: Define the packed W2 layout (2-bit codes + per-block scale, block shape aligned to source `weight_block_size [32,32]` AND the INT8 grouped-matmul `w13`/`w2` per-output-channel layout). Stream the FP8/FP4 source in bounded chunks (never materialize a full expert bank); quantize routed experts → W2, Engram → ~W4, leave MLA/indexer/dense/LM-head FP16. Record provenance (source rev, per-tensor precision, packing params). Decision logged: requant from FP8/FP4 source vs a BF16 source (quality).
- **validation**: converter round-trips a few real experts within the E0.4 W2 tolerance; bounded RSS during conversion; manifest updated with per-shard sha256.
- **status**: Not Completed

### E1.2 [asc]: W2→INT8 active-expert unpack + host QDQ parity
- **depends_on**: [E0.4, E1.1]
- **location**: `vllm_ascend/models/deepseek_v41/w2_unpack.py`, `tests/ut/deepseek_w2/test_w2_moe_parity.py`
- **description**: Unpack packed W2 → INT8 (per-block scale applied) for a set of active experts into a bounded cache, then run the existing INT8 grouped-QDQ math (reuse `qwen4_exp/moe.py` grouped path). Compare 384-expert top-6 + shared forward vs the E0.4 W2 reference at declared tolerances; cover skew (all tokens→1 expert), router renorm, per-block scale application order. No `.item()` in hot paths.
- **validation**: parity UT green; skew + renorm + scale-order guards have teeth (wrong order diverges).
- **status**: Not Completed

### E1.3 [asc]: 310P W2 fused-MoE method + registry
- **depends_on**: [E1.2]
- **location**: `vllm_ascend/_310p/quantization/methods/w2_dynamic.py`, register in `_310p/quantization/methods/__init__.py` + `registry.py`
- **description**: `AscendW2DynamicFusedMoEMethod310`: holds packed W2 params + per-block scales; `apply` unpacks the active experts (E1.2) and calls `npu_quant_grouped_matmul_dequant`. Mirror `AscendW8A8DynamicFusedMoEMethod310`'s param/weight-loading surface so the loader (E3.4) and mapping reuse hold. Unpack op verified against pinned CANN (fused if available, else elementwise on ≤6 experts).
- **validation**: CPU UT: method builds params from a synthetic W2 index; `apply` (host-math stub) equals the E1.2 path; registry resolves the method.
- **status**: Not Completed

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
