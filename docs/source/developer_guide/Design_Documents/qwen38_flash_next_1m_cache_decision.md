# Qwen3.8-Flash-Next 1M QSA Cache — Allocation Projection & Candidate Decision Record

**Generated**: 2026-09-16
**Plan task**: T8.3
**Companion to**: `qwen38_flash_next_1m_310p_prd.md` (PRD §6/§8/§13),
`qwen38_flash_next_1m_runtime_requirements.md` (measured checkpoint),
prototypes T8.1 (`qsa_c8.py`, Candidate A) and T8.2 (`models/qsa_dcp/`, Candidate B).

> **This document pre-decides nothing.** It supplies the per-chip allocation
> projection and the acceptance *formulas*. **D4 (on hardware) is the sole
> decision authority** for Candidate A vs B, using measured allocator numbers.

---

## 1. Inputs

| Input | Value | Source |
| --- | ---: | --- |
| Accelerator-resident non-PLE model, per chip (TP4) | **31.88 GiB** | measured export (runtime-requirements doc) |
| QSA main K/V @1M, BF16, sequence-sharded | 6.00 GiB/chip | T8.1 math / PRD §6 |
| QSA main K/V @1M, C8 (int8), sequence-sharded | 3.00 GiB/chip | T8.1 (exactly ½ BF16) |
| QSA compressed indexer history, BF16, replicated | 0.75 GiB/chip | T1.4 (64 MiB×12 layers) / PRD §6 |
| QSA raw ring (12 layers) | ~12 KiB/chip | T1.4 (negligible) |
| GDN state @concurrency 1 (36 layers, ssm fp32 + conv fp16) | ~0.072 GiB/chip | T5.1/T5.2 shapes |

**Note on the model input**: the plan lists T8.3 as depending on T3.2's
manifest-derived per-rank load simulation. T3.2 is checkpoint-blocked (no
checkpoint on host). This record instead uses the **measured exported** model
footprint (31.88 GiB/chip), which supersedes a manifest estimate. T3.2's
distinct guarantees — bounded peak load RSS and no-double-instantiation during
streamed load — remain a **separate checkpoint-blocked verification**, not a
substitute for these bytes.

## 2. Per-chip persistent footprint at 1M

| | Candidate A (C8 main + BF16 indexer) | Candidate B (DCP4 BF16 main + BF16 indexer) |
| --- | ---: | ---: |
| Model (measured) | 31.88 | 31.88 |
| QSA main K/V | 3.00 | 6.00 |
| QSA indexer (compressed, replicated) | 0.75 | 0.75 |
| QSA raw ring + GDN state | 0.07 | 0.07 |
| **Model + persistent cache / chip** | **35.70 GiB** | **38.70 GiB** |

## 3. Headroom projection vs the 8 GiB floor

PRD §6 requires **≥ 8 GiB measured free per chip after model and persistent
cache**, followed by a successful worst-case prefill. Actual free bytes per chip
are a **D1 measurement** (PRD open decision #1); three planning scenarios:

| Free per chip (D1 measures the truth) | Cand A headroom | Cand B headroom |
| --- | ---: | ---: |
| ~46 GB decimal (42.84 GiB) | 7.14 GiB — **below floor** | 4.14 GiB — **below floor** |
| ~48 GB nominal (44.70 GiB) | 9.00 GiB — clears | 6.00 GiB — **below floor** |
| true 46 GiB | 10.30 GiB — clears | 7.30 GiB — **below floor** |

## 4. Projection finding (informative, NOT a decision)

Under the **measured** model footprint (≈ +1.07 GiB/chip vs the PRD pre-export
estimate of 30.81):

- **Candidate B (BF16 DCP4) is projected below the 8 GiB floor at 1M in every
  free-per-chip scenario.** Its viability depends on either a larger-than-planned
  free-per-chip measurement at D1, reduced workspace/fragmentation, or dropping to
  a C8 indexer.
- **Candidate A (C8) clears the floor only if D1 measures ≥ ~44 GiB free/chip**
  and is below it in the pessimistic 46-GB-decimal case.
- This tightening is consistent with the PRD §13 **fallback-C** contingency
  (host main KV + device hot buffers) being a live risk if both candidates miss
  headroom on real silicon.

D4 must confirm or refute this with allocator measurements; the projection here
only tells D4 where the margins are thin.

## 5. Acceptance formulas for D4 (encoding PRD §13 stop rules)

Let `free_meas` = allocator free bytes/chip after weight load; `cache_meas(X)` =
measured persistent 1M cache/chip for candidate X; `WCP` = worst-case-prefill run.

**Candidate A (C8) accepted iff ALL hold:**
1. `free_meas − cache_meas(A) ≥ 8 GiB` (or ≥ measured graph-capture requirement if larger), AND
2. WCP@1M completes without OOM/sustained swap, AND
3. **Accuracy stop**: C8-vs-BF16 quality delta within the frozen G2 threshold (T0.1).
   *If accuracy fails, A is rejected regardless of memory* (PRD §13 A-stop).

**Candidate B (DCP4 BF16) accepted iff ALL hold:**
1. `free_meas − cache_meas(B) ≥ 8 GiB` (or ≥ measured graph-capture requirement), AND
2. WCP@1M completes without OOM/sustained swap, AND
3. **Communication stop**: after one DCP optimization pass, per-step selected-row
   exchange cost (within/cross-card, from TOBS collective trace) keeps decode
   ≥ 10 tok/s and TTFT within ceiling (PRD §8.2/§13 B-stop).

**Selection rule**: among candidates that pass, D4 picks the one with the larger
measured post-allocation headroom, tie-broken by measured decode tok/s. If
neither passes, escalate to fallback-C design (PRD §13) before the 262K gate.

## 6. What D4 must measure (hands to T8.4 the winner)

1. `free_meas` per chip after real-weight load (feeds every formula above).
2. `cache_meas(A)` and `cache_meas(B)` from the allocator (not projection) at 128K, extrapolated/confirmed toward 1M.
3. C8 accuracy delta vs the frozen G2 threshold on the T0.4 probes.
4. DCP per-step exchanged-row bytes and collective time, classified within/cross-card (TOBS R3 trace).
5. Worst-case-prefill headroom (graph capture supersedes the 8 GiB floor if larger).

## 7. Open dependencies

- **D1**: actual free bytes/chip (§3 scenarios collapse to one number).
- **T0.1**: frozen G2 accuracy threshold (gates Candidate A's accuracy stop) — checkpoint-blocked.
- **T3.2**: streamed-load peak-RSS / no-double-instantiation verification — checkpoint-blocked; independent of the byte projection here.
