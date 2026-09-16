# Runtime Requirements: Qwen3.8-Flash-Next (Qwen4Exp) 1M Context on Four Ascend 310P Chips

**Generated**: 2026-09-15
**Companion to**: `qwen38_flash_next_1m_310p_prd.md` (PRD), `qwen38-flash-next-1m-310p-plan.md` (plan)
**Status of inputs**: quantization and native Ascend checkpoint export are **complete**; the
figures below are derived from the exported checkpoint, not from allocator readings. Per-chip
*allocator* measurements are still pending the device wave (plan D1).

This document records the runtime memory requirements now that the checkpoint exists, states the
placement the loader must implement, and defines the staged bring-up. It replaces the pre-export
planning estimates in PRD §6 with measured checkpoint sizes (PRD §6 explicitly requires this
substitution). It does **not** re-decide the 1M cache architecture — that remains plan D4's call.

---

## 1. Why the checkpoint is large

Only the ordinary linear (routed-expert) weights were quantized to W8A8. The PLE n-gram
parameter table remains FP16. The FP16 PLE table alone is 95.43 GiB, which is why the checkpoint
is dominated by embedding parameters rather than compute weights.

## 2. Measured compressed-checkpoint breakdown

| Component | Size (GB, 1e9 B) | Size (GiB) | Placement |
| --- | ---: | ---: | --- |
| Routed experts (W8A8) | 125.83 | 117.19 | NPU (sharded ×4) |
| PLE tensors (FP16) | 102.47 | 95.43 | **Host RAM** |
| Other model tensors (FP16) | 10.43 | 9.71 | NPU (sharded ×4) |
| Shared experts | 0.48 | 0.45 | NPU (sharded ×4) |
| Quantization scales | 0.19 | 0.18 | NPU (sharded ×4) |
| **Total** | **239.39** | **222.95** | |

The native Ascend export adds ~0.57 GB of packaging overhead, producing the reported
**239.96 GB** on disk.

## 3. It does not fit entirely in NPU memory

Two Atlas 300I Duo cards expose four 48 GB 310P memory domains: **192 GB raw**, roughly
**184 GB usable**. The 239.96 GB checkpoint therefore cannot be NPU-resident. A naive loader that
places every tensor on the NPUs will OOM. The PLE table must live in host RAM and be looked up by
row.

## 4. Intended placement

| Region | Bytes | Notes |
| --- | ---: | --- |
| PLE table (host RAM) | 102.47 GB / 95.43 GiB | One shared logical copy; **never** ×rank copies |
| Non-PLE weights (NPU, all four) | 136.93 GB / 127.53 GiB | Routed + shared experts, other tensors, scales |
| Non-PLE weights per NPU | 34.23 GB / **31.88 GiB** | TP4/EP4 balanced sharding |
| Remaining per NPU | ~11–14 GiB | Runtime, workspaces, context cache (see §6) |

The 256 GB host holds the 95.43 GiB PLE table with ~48 GiB reserved for the OS, server
processes, pinned transfer buffers and filesystem activity, leaving ~112 GiB for host KV and any
optional prefix state (PRD §6).

## 5. Delta versus PRD §6 planning estimates

| Line | PRD §6 (pre-export estimate) | Measured (this doc) | Delta |
| --- | ---: | ---: | ---: |
| Accelerator-resident non-PLE model, aggregate | 123.25 GiB | 127.53 GiB | +4.28 GiB |
| Accelerator-resident non-PLE model, per chip | 30.81 GiB | **31.88 GiB** | **+1.07 GiB/chip** |
| Host FP16 PLE | 95.37 GiB | 95.43 GiB | +0.06 GiB |

The device weight footprint is ~1 GiB/chip larger than planned. That reduces per-chip headroom
by the same amount and tightens the 1M cache decision (plan D4). All arithmetic downstream of
PRD §6 (plan T3.2, T8.3) must use the measured 31.88 GiB/chip figure, not 30.81.

**Refinement (2026-09-16, plan T3.2 from the checkpoint manifest)**: simulating placement
directly from the exported checkpoint's weight index gives **32.01 GiB/chip** non-PLE (128.06 GiB
aggregate / 4), ~0.13 GiB/chip above the 31.88 estimate. The difference is the W8A8 quant
scales, which aggregate to ~0.70 GiB (F32 `[out,1]` per projection × 73,728) rather than the
~0.19 GB first attributed. PLE stays 95.43 GiB host-resident (counted once, not ×4). This
manifest-derived 32.01 GiB/chip is the authoritative model figure; it shifts the T8.3 per-chip
projection by ~0.13 GiB (A ≈ 35.83, B ≈ 38.83 GiB/chip) and changes no pass/fail conclusion.

## 6. Per-chip headroom and the 1M cache problem

Headroom after loading the non-PLE weights depends on the *actual* free bytes per chip, which is a
device-wave measurement (plan D1, PRD open decision #1):

| Assumed free per chip | Headroom after 31.88 GiB model |
| --- | ---: |
| ~46 GB decimal (42.84 GiB) | ~10.96 GiB |
| ~48 GB nominal (44.70 GiB) | ~12.82 GiB |
| a true 46 GiB | ~14.12 GiB |

So roughly **11–14 GiB/chip** is available for runtime, workspaces and the context cache — enough
for initial shorter-context inference **only if** the PLE stays in host memory and the loader
streams shards without building a second full in-memory copy.

At 1M tokens an ordinary conventional TP4 BF16 QSA K/V cache costs ~12 GiB/chip (PRD §6), which
would consume essentially all of that headroom and OOM once workspaces and activations are added.
The 1M target therefore requires one of the planned cache layouts:

- **Candidate A** — C8 (8-bit) QSA main K/V: ~6 GiB/chip.
- **Candidate B** — QSA-aware DCP4 (sequence-sharded BF16 main K/V, replicated indexer): ~6 GiB/chip logical.

The choice between them is measured on hardware at 128K and decided at plan D4; this document
pre-decides nothing.

## 7. Current runtime-support gap

The checkpoint is done. What blocks serving is runtime support in vLLM Ascend:

1. **310P PLE host lookup / cache path** — host-resident FP16 PLE table with row-gather transport
   (pinned-UVA or shared-mmap + registered windows), batched dedup and async prefetch. Plan
   T4.1/T4.3/T4.4; PRD R5.
2. **Four-chip sharding** — stream-load the non-PLE tensors across the four 310P chips (TP4
   default) with no rank materializing the full expert bank or a second in-memory copy. Plan
   T3.2; PRD R2/R3.
3. **1M cache path** — C8 QSA cache or QSA-aware context sharding, without which the BF16 QSA cache
   exceeds per-chip memory at 1M. Plan T8.1/T8.2/D4.

## 8. Staged bring-up (initial practical target)

1. Stream-load the non-PLE tensors across all four chips (31.88 GiB/chip).
2. Keep the 95.43 GiB PLE table resident in host RAM; fetch only requested rows.
3. Validate at **8K** context (plan D2/D3, gates G1/G2).
4. Increase to **128K** and **262K**, measuring actual per-chip headroom at each step
   (plan D5/D6, gates G3/G4).
5. Add compressed (C8) or distributed (DCP4) QSA state **before** attempting **1M**
   (plan T8.x/D4 → D7/D8).

## 9. Future host-RAM reduction (deferred)

Quantizing the PLE table to INT8 would save ~51 GB of host RAM and disk (95.43 GiB → ~47.7 GiB),
but it requires a compatible sparse-lookup kernel and a separate quality-validation pass. It is
out of scope for the first correctness milestone (PRD §3, non-goal) and is tracked as a later
optimization, not a requirement for reaching 1M.

## 10. Open items feeding the device wave

- Actual free bytes per chip (§6) — plan D1 / PRD open decision #1.
- PLE row-gather transport winner (pinned-UVA vs mmap+registered window) — plan D1-MB.
- 1M cache candidate A vs B — plan D4.
