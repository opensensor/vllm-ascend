# GLM-5.3-Flash on Ascend 310P — Progress Summary

_Last updated: 2026-09-21_

## Goal
Run GLM-5.3-Flash (~304B params; 45-layer hybrid of KDA linear-attention +
DeepSeek-Sparse-Attention, 288-expert MoE) **coherently** on 4× Ascend 310P at TP4.
GLM is bf16-designed; the 310P is effectively fp16-only for NZ weights — the whole
effort is fighting range/precision and quantization loss on constrained HBM.

## Where it stands
The model **loads, fits (~34 GB/chip), and generates** end-to-end at TP4 — but the
output is **not yet coherent**. On the 67-token parity prompt the next token is
`' the'` (golden reference: `' forty'`). Every catastrophic failure has been fixed;
what remains is an accuracy/drift problem, not a "doesn't run" problem.

## Fixed along the way
- **Fit + expert parallelism** — 72 experts/rank + all-reduce (was replicating 288
  → OOM); eager fp32 MoE device path.
- **Custom AscendC Cube kernel** `npu_w2_blocked_dequant_matmul_310` — ~17× faster
  decode than eager; handles the per-[32,32] block scale the stock op can't.
- **Bug #1 dead FFN** — dense-MLP/shared-expert were fp8-mishandled → zero output;
  fixed by marking them fp16.
- **Bug #3 KDA eps** — an earlier "under-contribution fix" was backwards, inflating
  KDA ~15×; reverted to `o_norm_eps`. L0–L2 KDA now cos ~1.0.
- **Residual explosion** — the MSE-quant era's "RMS 525 at L44" catastrophe. Traced
  to a clipping-driven feedback resonance; **fixed by no-clip (minmax) W4
  quantization**. L44 residual now RMS ~10, bounded at every depth. Bit-depth was
  never the issue — clipping was.

## The enabling tool
A **GPU golden reference** (NVFP4, NVIDIA-validated cosine 0.9999, coherent
`' forty'`) captured on a rented **Shadeform RTX PRO 6000** — the local box could
never run the golden math correctly. Per-layer cosine-diffing the Ascend capture
against this golden is what localizes each bug.

## Current blocker — sharply localized
Attention diverges at layer 7 and cascades:

| Layer | type | attn cos vs golden | expert cos |
|---|---|---|---|
| L3 | DSA | **0.998** ✓ | 0.918 |
| L7 | DSA | **0.624** ✗ | 0.302 |
| L44 | DSA | 0.425 | 0.116 |

**Decisive finding (2026-09-21)** — via a new `forward_pre_hook` on `self_attn`
(the attention *input*, a tap never recorded before; prior captures hooked a
nonexistent `input_layernorm`): L3 and L7 attention inputs are **near-identical and
both healthy** (RMS 0.0161 vs 0.0163; conditioning max/rms 6.3 vs 5.3; bounded
everywhere). Yet L3 is perfect and L7 breaks. So:
- The break is **not** a residual blow-up or ill-conditioning (that theme is closed).
- The DSA has **~20× gain** (0.016 in → 0.3 out), so it amplifies a small
  *directional* input error into a large output error.
- **L7's expert is W4** (verified: W4early = W4 on L3–26, W2 on L27–45, from
  down_proj_codes widths [4096,1024]=W4 vs [4096,512]=W2), weights ~0.9935 — its
  0.302 output is a **symptom** of the drifted attention input, not an
  expert-bit problem.

## Ruled out (code-level)
- **DSA softmax scale**: eager `qk_head_dim**-0.5 = 256**-0.5` matches the reference
  `Glm5NextMLAAttention.scaling` exactly; GLM is NoPE (`qk_rope_head_dim=0`) so the
  YARN `mscale**2` branch does not run.
- **Indexer sparsity**: at 67 tokens `block_topk = index_topk//kpool = 512` ≫ ~17
  blocks → all blocks selected → L3 and L7 run *identical* dense MLA-NoPE. Selection
  is not the differentiator.

## The one open question
Is L7's divergence **(a)** high-gain amplification of upstream **directional
residual drift** — from the MoE contributions (the dominant residual terms), whose
per-layer cosine is only ~0.86–0.92 even with a clean input and W4 weights — or
**(b)** a DSA compute issue biting only at L7? L3 DSA being perfect argues for (a);
the lone datum for (b) is L7's attn_out running 43% hot (0.299 vs golden 0.209).

**Settling it needs the golden *attention-input* direction at L3/L7**, which requires
re-running the golden with the same pre-hook — and the golden math only works on
**Shadeform**, not the local box. Alternatively, an **on-box DSA input-swap test**
(L3-weights·L7-input vs L7-weights·L3-input) isolates weights-vs-input without a
golden, but needs the Ascend box.

## Working hypothesis for the fix
The per-layer MoE fidelity floor (~0.92 vs the NVFP4 golden — likely the W4
block-scale being coarser than NVFP4's fp8-e4m3 per-16 scales, plus router top-k
overlap of 0.97 flipping near-tie experts) injects a small directional error into
the dominant residual term every layer → accumulates → the high-gain DSA layers
amplify it into incoherence. The golden is itself 4-bit and coherent, so **4-bit is
sufficient if per-layer fidelity matches NVFP4**.

## Artifacts (in `artifacts/`)
- `glm-w2-residual-explosion-diagnosis.md` — full running diagnosis (UPDATE 4 =
  the 2026-09-21 attn-input analysis).
- `golden_ascend_activations_attnin.npz` + `capture_ascend_attnin.py` +
  `compare_attnin.py` — the new attention-input capture and tooling.
- `golden_gpu_activations.npz` — the Shadeform NVFP4 golden.
- `golden_ascend_activations_w4early.npz` — current best Ascend capture (no-clip W4).

## UPDATE (2026-09-21, offline golden-diff analysis) — two new findings

Re-analysed the existing captures offline (no box needed). Two material results:

### 1. The "next step" (NVFP4-direct) is code-complete but **memory-infeasible as written**
The NVFP4-direct converter (`tools/glm_w2/convert_nvfp4.py`, commit `1cb1fd588`)
exists and is fidelity-1.0 by construction (verbatim E2M1 codes + folded scale),
but has **never been run** (no `GLM-5.3-Flash-NVFP4-310p` output exists) and its
scale layout does not fit the 310P:

| format | expert codes | expert scales | experts/rank | + non-expert (5.6 GB) |
|---|---|---|---|---|
| W4 (int4, per-[32,32] fp32) | 152.2 GB | 1.2 GB | 38.4 GB | ~44 GB (marginal/over) |
| NVFP4-direct (per-[1,16] fp32 scale) | 152.2 GB | **76.1 GB** | **57.1 GB** | ~63 GB (OOM) |
| NVFP4-direct (per-[1,16] fp8 scale) | 152.2 GB | 19.0 GB | 42.9 GB | ~48 GB (OOM) |

The 310P chips are ~43 GB (npu-smi 44278/43693 MB). Full-NVFP4 does not fit even
with fp8 scales; the converter currently emits **fp32** scales (57 GB/rank).

### 2. ~~near-tie routing is a second amplifier~~ — REVISED by the on-box routing-swap (UPDATE 2)
The earlier "top-k overlap collapses 0.97→0.50" was measured with `argsort(logits)`
which **ignores the `noaux_tc` correction bias** that the real router applies.
A definitive on-box routing-swap (feed the golden's raw gate logits through the
model's own bias-aware `route`, keep W4 weights + ascend input fixed) shows
**routing is a minor factor**, not the blocker:

| layer | MoE cos (ascend routing) | MoE cos (golden routing) | bias-aware topk overlap |
|---|---|---|---|
| L3 | 0.9179 | 0.9181 (+0.0002) | 0.922 |
| L7 | 0.3015 | 0.3324 (+0.031) | 0.636 |

So the L3/L7 expert error is **weight-quantization + input drift (fp16-vs-bf16)**,
not routing. Quant-fidelity ladder (source weights): int4[32x32]=0.991,
e2m1[32x32]=0.9935, e2m1[16x16]=0.9941, e2m1[1x16]=1.0.

## UPDATE 2 (2026-09-21, on-box + conversion)
- **Routing-swap diagnostic** (`diag_routing_swap.py`, v2) → routing is NOT the
  coherence blocker (see table above). Fix must target weight fidelity + input drift.
- **NVFP4-direct converter fixed + run** — added `--nvfp4-layers` /
  `--default-expert-bits` to `tools/glm_w2/convert_nvfp4.py`, fixed the
  `FP16_NVFP4` decode-width bug, and converted `GLM-5.3-Flash-NVFP4mix-310p`
  (NVFP4-direct fidelity-1.0 on L3–L18, W2 no-clip on L19–L44; 157 GB, fits
  ~40 GB/rank). Pending deploy + validation.

## UPDATE 3 (2026-09-21, NVFP4-direct validated — negative result)
NVFP4-direct (byte-identical to the golden's own 4-bit weights) **does not fix
coherence**. Deployed `GLM-5.3-Flash-NVFP4mix-310p` (NVFP4-direct on L3–L18,
W2 no-clip on L19–L44, fp16 scale grid → 34.0 GB/rank, fits) and captured the
parity prompt:

| layer | metric | W4early | NVFP4-direct | golden |
|---|---|---|---|---|
| L0 | layer_out cos | — | 0.985 | 1.0 |
| L3 | expert_out cos | 0.918 | **0.925** (+0.007) | 1.0 |
| L3 | attn_out cos | 0.998 | 0.999 | 1.0 |
| L7 | expert_out cos | 0.302 | 0.314 (+0.012) | 1.0 |
| L44 | layer_out cos | 0.385 | **-0.008** (rms 80k, blow-up) | 1.0 |

next token = `' blo'` (14202), not `' forty'`.

**Conclusion: the coherence blocker is the fp16-vs-bf16 input drift, not weight
quantization.** L0's dense-MLP output is already 0.985 vs the bf16 golden, and
even with expert weights at fidelity 1.0, L3's expert output is only 0.925 — the
residual stream (seeded in L0–L2 by fp16-vs-bf16) dominates. The L3 router logits
stay 0.9999 (the gate projection is blind to the drift), yet the experts see it.
Secondary finding: W2-minmax (0.83 fidelity) on L19–L44 is too coarse and blows up
L44 (the W4early's W4 on L19–L26 kept it bounded). Weight fidelity matters for
deep-layer *stability*, but not for the L3 *accuracy* floor.

## Next steps
1. **The remaining lever is bf16 range/precision in the non-quantised layers**
   (embeddings, dense MLP L0–L2, attention, residual). L0 is already 0.985 — chase
   why the first dense layer drifts (fp16 vs bf16 rounding) before any quant.
2. For deep-layer stability, keep L19–L26 at W4 (0.991), not W2 (0.83), regardless
   of the NVFP4-direct result.
3. If bf16 emulation is infeasible on the fp16-only 310P, this model cannot reach
   golden coherence; the honest deliverable is the diagnostic + the quant tooling.
