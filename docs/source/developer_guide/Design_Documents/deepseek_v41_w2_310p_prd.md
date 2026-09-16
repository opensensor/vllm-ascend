# PRD: DeepSeek V4.1 (552B) 2-bit Experts on Four Ascend 310P Chips

**Generated**: 2026-09-16
**Companion plan**: `deepseek-v41-w2-310p-plan.md`
**Precedent**: the Qwen3.8-Flash-Next 1M 310P work
(`qwen38_flash_next_1m_310p_prd.md`) — this port reuses its streamed loader,
TP4/EP4 placement/accounting, host-table ownership, batched gather/prefetch,
eager sparse-MoE scaffolding, KV-spec plumbing, and gates/observability. The
genuinely new work is the **W2 (2-bit) expert arithmetic** and the DeepSeek
assembly (MLA, indexer/sparse attention, two Engram layers, DSpark, top-6/384
routing, MTP-3).

| | |
| --- | --- |
| Target host | Two Atlas 300I Duo cards = four independent 48 GB Ascend 310P chips; 256 GB DDR host |
| Aggregate HBM | 192 GB raw; **~172.8 GB usable** under the 90% planning rule (~160.9 GiB) |
| Source checkpoint | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` — FP8 weights, **FP4 experts** (`weight_block_size [32,32]`, `scale_fmt ue8m0`) |
| Initial target | Text-only, W2 experts + dense weights, ~W4 host Engram, TP4/EP4, **eager 8K** |

---

## 1. Problem statement

DeepSeek V4.1 (`DeepseekV41ForCausalLM`, ~552 B params) cannot fit on four 310P
chips at any precision above 2-bit for the routed experts: at 4-bit the ~450 B
non-Engram params alone are ~209 GiB, over even the raw 192 GB. At 2-bit the
routed/dense payload is ~105 GiB — comfortably under the 160.9 GiB usable HBM —
leaving room for KV cache, workspaces and communication buffers. The two large
Engram tables (~203 GB in the source) stay in host RAM at reduced precision. The
Ascend 310P has **no native sub-INT8 matmul**, so W2 is a *storage* format that is
unpacked to INT8 for the active experts and run through the existing INT8 grouped
matmul. This PRD defines a text-only, 8K-first bring-up of that path.

## 2. Goals

1. A packed **W2 weight format** + streaming converter that quantizes the FP8/FP4
   source in bounded chunks locally (no full-model materialization, no rental
   required to produce a weight-only checkpoint).
2. A 310P **W2→INT8 active-expert unpack** feeding the existing
   `npu_quant_grouped_matmul_dequant` grouped matmul (no new device GEMM).
3. A correct **DeepSeek V4.1 assembly** on 310P: MLA attention, the sparse-
   attention indexer, two Engram host-lookup layers, top-6/384 MoE, DSpark, MTP-3
   (MTP excluded from the first text gate).
4. **Host-resident Engram** at ~W4 with quantized row lookup, one shared logical
   copy across the four workers.
5. Fit a real-weight boot with measured per-chip headroom; **gate at 8K** before
   any long-context attempt.

## 3. Non-goals for the first release

- Long context / 1M (a later gate, after the W2 kernel + Engram cache are measured).
- Vision / multimodal (rejected at the first gate; `DeepseekV41ForConditionalGeneration` config carries a `vision_config`).
- MTP speculative decoding (registered, wired later).
- GLM 5.3 753B (the second experiment — it exceeds the safe HBM budget at W2 and needs a bounded expert cache/offload; tracked separately).
- W2 quality *guarantees*: requantizing from an FP8/FP4 source is lossy; calibration/quality validation needs a GPU (rental), and is a gate, not a goal, of the first weight-only bring-up.

## 4. Fixed inputs and assumptions (from the real config)

### 4.1 Model (`DeepSeek-V4.1-Flash-UNCENSORED-FP8/config.json`, `text_config`)
- Architecture `DeepseekV41ForCausalLM`, `model_type deepseek_v41`.
- 40 decoder layers; hidden 5120; vocab 129,280.
- MoE: **384 routed experts, top-6**, `moe_intermediate_size 2304`, 1 shared expert.
- Attention: **MLA** — `q_lora_rank 1280`, `qk_rope_head_dim 64` (+ `kv_lora_rank`, `v_head_dim` per config); plus a sparse-attention **indexer** (`…indexer…` tensors) — DeepSeek Sparse Attention / CSA2 compression.
- **Engram**: two Engram layers (e.g. `layers.14.engram.{embed,q_weight,k_weight,wkv}…`) — a host-resident n-gram-style table analogous to Qwen4Exp PLE.
- MTP: `num_nextn_predict_layers 3`.
- Source quantization: `fp8` weights, **`expert_dtype fp4`**, `weight_block_size [32,32]`, `scale_fmt ue8m0`.

### 4.2 Hardware — same as the Qwen PRD §5.2
Four independent 48 GB 310P domains, 256 GB host. Actual free NPU bytes, HCCL
topology (within- vs cross-card), PCIe, NUMA and bandwidth are recorded on target
(`tools/qwen38_1m/hw_probe.py`, reused).

### 4.3 Quantization target (this project)
- Routed experts → **W2** (packed), unpacked to INT8 per active expert.
- Engram tables → **~W4** in host RAM (quality-sensitive embeddings; matches the operator's ~47–55 GiB host figure).
- MLA / indexer / dense / LM-head / norms → **FP16** (as in the Qwen policy).
- Activations INT8 (per-token dynamic) for the expert GEMM; FP16 elsewhere.

## 5. Capacity model (planning — replace with allocator measurements)

Operator analysis, to be confirmed against the actual W2 export and D1 probe:

| Component | Aggregate | Placement | Notes |
| --- | ---: | --- | --- |
| W2 routed experts (packed) | ~85–110 GiB | HBM (TP4/EP4) | ~450 B non-Engram params @ 2-bit ≈ 105 GiB |
| Engram tables (~W4) | ~47–55 GiB | **Host RAM** | one shared logical copy; ~203 GB source → ~4-bit |
| MLA/indexer/dense/LM-head (FP16) | to measure | HBM | |
| INT8 active-expert unpack cache | small, bounded | HBM | top-6/384 active per token |
| KV (MLA latent) + workspaces + HCCL | to measure | HBM | |

Usable HBM budget **160.9 GiB** (172.8 GB). W2 experts (~105 GiB) + FP16
non-expert + caches must leave a measured **≥ 8 GiB free per chip** after load and
a successful worst-case prefill (mirrors the Qwen §6 floor). Host must reserve
≥ 48 GiB for OS/transfer/pinned buffers; ~W4 Engram (~50 GiB) leaves ample host
headroom in 256 GB.

## 6. The W2 kernel contract (the critical new work)

**There is no native W2 GEMM on Ascend.** The 310P quant registry
(`_310p/quantization/methods/__init__.py`) is W8-only; the MoE device path is
`torch_npu.npu_quant_grouped_matmul_dequant` (INT8). A general-tree
`AscendW4A16FusedMoEMethod` (`quantization/methods/wna16/w4a16.py`,
`npu_convert_weight_to_int4pack` + `npu_grouped_matmul` antiquant) exists but is
**not** 310P-registered and its 310P op availability is **unverified** — and W4
does not fit DeepSeek in HBM regardless.

Therefore W2 is a **storage** format, not a GEMM:

1. **Packed W2 format**: 2-bit codes + per-block scales (and per-group zero/offset
   where needed), block shape chosen to match the source `weight_block_size` and
   the INT8 grouped-matmul layout (`w13`/`w2`, per-output-channel scale/offset).
2. **Active-expert unpack**: for the top-6 experts selected per token, unpack W2 →
   INT8 (or FP16) on device into a bounded cache, then run the **existing**
   `npu_quant_grouped_matmul_dequant`. Only active experts are unpacked; the W2
   bank stays packed in HBM. This is the "unpack active experts into an INT8
   cache" design — memory-bound and cache-friendly given top-6/384 sparsity.
3. **Host-side reference**: the unpack+QDQ math is expressed in pure PyTorch and
   parity-checked against a W2 QDQ reference (mirrors `moe.py` / T3.3), so
   correctness is validated with no hardware.

The unpack step is the only genuinely new device op; if it cannot be a fused
`torch_npu` op on 310P it is a small elementwise unpack, which is acceptable
because it runs on ≤6 experts per token, not the full 384-bank.

## 7. Product requirements (mirroring the Qwen PRD)

- **R1 Reproducible environment** — reuse `env_freeze` (`tools/qwen38_1m/`).
- **R2 Architecture registration** — register `DeepseekV41ForCausalLM` (+ ConditionalGeneration alias rejecting multimodal) OOT via `ModelRegistry.register_model`.
- **R3 Parallel topology** — TP4 default, EP4 ready; per-rank accounting reused (`qwen38_mem_accounting`).
- **R4 W2 execution** — packed W2 experts, INT8 active-expert unpack + existing grouped matmul; FP16 for MLA/indexer/dense/LM-head; a single authoritative dtype/precision policy (as in Qwen T1.2).
- **R5 Engram host lookup** — one shared ~W4 host table; quantized row gather; batched dedup + prefetch (reuse the T4.1/T4.4 host method + prefetcher).
- **R6 MLA + indexer** — port DeepSeek MLA and the sparse-attention indexer to 310P torch/NPU ops (no Triton), parity vs an eager reference.
- **R7 One-shot correctness** — deterministic greedy at 8K; quality delta vs the FP8 source within a frozen threshold (needs a runtime; the FP8 reference itself may need a GPU).
- **R8 Scheduling/serving** — no truncation; report accepted tokens; concurrency policy per gate.
- **R9 Observability** — reuse the run-log (`qwen38_runlog`) incl. first-fatal-rank + within/cross-card collective tracing.
- **R10 W2 checkpoint provenance** — the converter records source rev, per-tensor precision, packing params, and a manifest (extend `build_manifest.py`).

## 8. Validation and release gates

- **G0** component parity (CPU): W2 unpack+QDQ MoE, MLA, indexer, Engram lookup vs eager references at pre-declared tolerances.
- **G1** real-weight startup, TP4, eager, 8K — non-empty completion; per-rank weight report; ≤5% imbalance.
- **G2** quantized correctness at 8K — deterministic across restarts; quality delta vs the frozen FP8-source threshold.
- **G3+** long context — only after the W2 kernel + Engram cache behavior are measured.

Accuracy criteria frozen **before** inspecting Ascend output (as in Qwen §8.1).
**W2-from-FP8 is the dominant quality risk**; if G2 fails, options are ~W3/W4 for
the most sensitive expert groups (mixed precision) or requantizing from a BF16
source — both recorded decisions, not silent changes.

## 9. Reuse inventory (what carries over from the Qwen 310P work)

| Capability | Reused module | Adaptation |
| --- | --- | --- |
| Streamed sharded load, no full-bank | `_310p/sharded_state_loader_310p.py` (T3.2) | W2 tensor families |
| TP4/EP4 per-rank accounting | `observability/qwen38_mem_accounting.py` (T0.5) | add W2/Engram components |
| Host-table ownership + dedup/prefetch | `models/qwen4_exp/ngram_embedding.py` (T4.1) + `ple_prefetch.py` (T4.4) | Engram, ~W4 rows |
| W-quant fused-MoE mapping + grouped math | `weight_mapping.py` (T3.1), `moe.py` (T3.3) | W2 pack + INT8 unpack |
| Sparse-attention indexer + kernel | `indexer_qsa.py`/`qsa.py` (T6.x) | DeepSeek indexer / CSA2 |
| KV-cache specs + block math | `kv_cache.py` (T1.4) | MLA latent KV |
| Checkpoint manifest | `tools/qwen38_1m/build_manifest.py` (T0.1) | DeepSeek/engram/W2 |
| Hardware probe / env freeze / run-log | `tools/qwen38_1m/{hw_probe,env_freeze}.py`, `qwen38_runlog.py` | as-is |
| Gates / determinism / import-hygiene discipline | plan + tests | as-is |

## 10. New work (the port)

1. Packed W2 format + streaming converter (local, chunked; also handles the giant Engram → ~W4).
2. W2→INT8 active-expert unpack + a 310P W2 fused-MoE method registered in the quant registry.
3. DeepSeek assembly: MLA, indexer/CSA2 compression, two Engram host layers, DSpark, top-6/384 routing, MTP-3.
4. Host quantized-Engram row lookup (~W4 dequant on gather).
5. Gates at 8K then long context.

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| W2-from-FP8 quality loss | mixed precision (W2 experts, ~W4 Engram, higher-bit sensitive groups); requant-from-BF16 fallback; calibrate on a rental before claiming quality |
| No native W2 GEMM | unpack only active experts (top-6) to INT8, reuse the validated grouped matmul |
| 310P lacks an assumed op (int4pack / antiquant / MLA fused) | verify against the pinned CANN first; fall back to elementwise unpack / eager MLA |
| Engram host stalls | shared allocation, NUMA placement, batched dedup + prefetch (reuse T4.4), ~W4 to shrink the table |
| DeepSeek V4.1 upstream is preview-grade | this is a real port; pin the source rev; validate the assembly against an eager reference before device |
| HBM headroom after W2 | measured ≥8 GiB/chip floor at G1; expert-cache/offload fallback if tight (as GLM will need) |

## 12. Open decisions

1. W2 block/packing layout vs the source `weight_block_size [32,32]` and the INT8 grouped-matmul expectation.
2. Unpack target: INT8 (reuse W8 grouped matmul, confirmed) vs INT4 (if 310P antiquant is available) — measured.
3. Engram precision (W4 vs W3) and whether it is host-only or has a device hot cache.
4. Frozen G2 quality threshold and whether the FP8-source reference runs on 310P or only on a rental GPU.
5. MLA latent-KV cache layout and long-context strategy (later gate).
