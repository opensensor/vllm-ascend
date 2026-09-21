# GLM-5.3-Flash-W2 310P — deep-layer residual explosion (post-MSE) diagnosis

Status: 2026-09-20. Blocks coherence after the KDA + MoE-explosion + MSE-quant fixes.

## Symptom
With the MSE-recalibrated W2 model (`GLM-5.3-Flash-W2-310p-mse`), the mid-layer
experts improved as predicted (L3 0.81→0.89, L4 0.57→0.74, L7 0.26→0.52 cosine
vs the GPU golden), but the **residual/hidden stream balloons with depth** and the
final token is still wrong (`'helper'`; golden `' forty'`).

Residual RMS by layer (Ascend MSE eager capture, `golden_ascend_activations_deep.npz`;
for non-last layers the captured `layer_out` == the MoE output `x`, which equals
`expert_out` exactly):

```
L0 .008  L3 .69  L4 .05  L7 .06     <- contracting/bounded through L7
L10 3.06                            <- ONSET: ~51x jump between L7 and L10
L13 6.9  L16 8.1  L19 8.2  L25 10.4  L31 17  L37 26  L42 53   <- coherent accumulation
L44 525                            <- final hc_contract materialization
```
Golden (GPU NVFP4) stays ~1.2 at every layer. minmax-W2 (pre-MSE) also stayed
bounded (L44 layer_out RMS ~0.82). So MSE **triggers** this; it is not the sole cause.

## Mechanism (established)
- mHC recursion (`vllm/model_executor/kernels/mhc/torch.py`): the residual is a
  4-stream fp32 accumulator; `mhc_post_torch` does
  `residual_new = einsum(comb, residual) + post_mix * x`. `comb` is ~doubly
  stochastic (sinkhorn) → the mix is norm-preserving. So each layer ADDS
  `post_term = post_mix * x` (post_mix ∈ (0, hc_post_mult_value=2.0), `x` = sublayer output).
- The per-sublayer input RMSNorm (`vllm_ascend/patch/worker/patch_mhc_norm.py`
  `_mhc_rms_norm`, fused into `hc_pre`/`hc_fused_post_pre`) normalizes `layer_input`
  to unit-RMS×gamma REGARDLESS of the residual magnitude, so sublayer inputs stay
  bounded (~0.3–0.6 RMS; norm gammas are small/flat, checked: post-attn_ln RMS
  0.03–0.62 across depth — NOT the driver). Sublayer outputs are therefore ~1–2 RMS.
- So `post_term` is bounded per layer, but it **accumulates coherently** into the
  residual (~+1–3 RMS/layer → ~50 by L42). The GOLDEN's per-layer post_terms must
  largely CANCEL (residual stays ~1.2). Both use the same eager `mhc_post_torch`
  math, so the divergence is in the INPUTS to the recursion (`x` direction, `comb`,
  `post_mix`) between the 310P eager path and the golden's tilelang fused kernel —
  OR the tilelang kernel does an extra bound the torch reference omits.
- Why MSE triggers it: MSE experts are higher-fidelity (cos 0.92 vs 0.83), so their
  outputs align with the "true" accumulating direction → coherent addition. minmax's
  noisier outputs partially cancel → stayed bounded (masking the latent bug).

## Ruled out
- L44 expert WEIGHTS are correct (cos 0.92 vs full FP8, bounded |w|max ~0.05).
- Norm gammas do not grow with depth (checked).
- Not fp16 saturation (RMS 525 << fp16 max 65504; genuine accumulation).
- `patch_mhc_norm` IS effective (early layers L0–L7 stay bounded).

## Next on-device experiments (need the box; do NOT blind-patch the shared mHC)
1. **Capture mHC intermediates** at L5/L10/L20/L30/L40: the fp32 residual
   accumulator RMS, `layer_input` pre- and post-norm RMS, `post_mix` mean, and the
   per-layer `post_term` direction. Confirms whether post_terms add coherently and
   whether `comb` is actually norm-preserving on-device.
2. **Compare 310P `mhc_post_torch` vs the golden tilelang mHC** on identical inputs
   (read the opensensor tilelang mHC kernel at `/srv/ai/src/vllm-opensensor`
   `vllm/model_executor/kernels/mhc/` — it was unreachable during this session) —
   look for a normalization/contraction the torch reference drops, or a
   comb transpose/normalization-axis mismatch (torch column-normalizes `dim=-2`).
3. **A/B the routed_scaling_factor** (2.5): it multiplies every MoE `post_term`;
   temporarily set 1.0 to see if the accumulation slope drops proportionally
   (confirms the MoE post_term is the growth term).
4. Once the divergence is a known line, fix it and re-capture; expect L44 residual
   → ~1 and next token → coherent.

## Notes
- MSE-W2 is correct and kept (commit 7c822d035); it exposed this pre-existing latent
  bug rather than causing it. This is the "Bug #2 residual-range" theme, now the #1
  blocker, better characterized.
- Artifacts: `golden_ascend_activations_deep.npz` (dense-tap Ascend MSE),
  `golden_ascend_activations_72mse.npz`, `golden_gpu_activations.npz` (golden),
  `compare_activations.py`.

## UPDATE 2 (2026-09-20, input/output probe — narrowed to a runtime MoE effect)

Probe (`golden_ascend_activations_probe.npz`) captured actual module INPUTS
(forward_pre_hook) vs outputs at dense depths. Decisive results:

| L | mlp_IN rms | mlp_IN max/rms | mlp_OUT rms | gain (OUT/IN) |
|---|---|---|---|---|
| 4 | 0.25 | 5.2 | 0.05 | 0.21 |
| 7 | 0.26 | 5.1 | 0.06 | 0.23 |
| 10 | 0.27 | 4.8 | **3.06** | **11.3** |
| 25 | 0.32 | 1.3 | 10.35 | 32 |
| 42 | 0.62 | 1.1 | 53.1 | 85 |

**Ruled out (all measured):**
- mHC input norm WORKS — `mlp_IN` is bounded ~0.25–0.62 RMS at every layer, and
  at deep layers it is WELL-CONDITIONED (max/rms ~1.1, no outlier channels).
- Expert weights are correct and LOW-gain, constant across depth: local
  reconstruction (dequant MSE weights, random RMS-0.3 input) gives per-routed-expert
  gain ~0.16 and shared-expert gain ~0.13; spectral norms only ~3–7. A single MoE
  layer's correct output for a 0.6 input is ~0.1–0.3, NOT 53.
- Not input outliers, not weight magnitude, not the mHC.

**The unresolved core:** the RUNTIME MoE output at L10+ is ~500× larger than the
correct expert math, from a bounded well-conditioned input, with correct low-gain
weights. L3/L4/L7 MoE layers are correct (gain ~0.2); the blowup switches on at
~L10 and compounds. AND it is MSE-specific: the SAME eager `_apply_device` path
gives bounded output for the minmax-W2 weights but exploding output for MSE-W2,
even though both weight sets reconstruct to identical local gain (~0.16). So it is
neither a pure weight bug nor a pure code bug — it is a runtime interaction that
local reconstruction does not reproduce.

**Leading hypotheses (need on-device to disambiguate):**
1. Runtime combine/router: topk_weights not summing to 1 on-device, or too many
   experts summed — but router_logits matched golden (cos 0.99) at L3–L7.
2. A precision/accumulation effect in the eager device GEMM sensitive to MSE's
   distribution (smaller scales → larger codes: MSE scale ~0.30·absmax vs minmax
   ~0.67, so MSE codes use more of {-2..1}) that only bites past some depth.
3. A per-layer structural difference at L8–L10 (attention type / MoE variant) that
   changes the runtime path.

**Exact next on-device experiments (when box is available):**
1. Inside `Glm5NextW2MoE.forward` / `_apply_device`, capture SEPARATELY: the actual
   expert input tensor, the routed-only output (pre-`*2.5`), the shared-only output,
   and `topk_weights.sum(-1)`, at L7 (good) vs L10 (bad). Pinpoints the component.
2. A/B `routed_scaling_factor` 2.5→1.0 and re-diff (confirms the routed term).
3. If it is MSE-distribution precision: re-check the eager dequant/GEMM dtype on
   310P for MSE codes vs minmax.

**Reframed recommendation:** since golden (nvfp4, 0.99) is stable+coherent and the
310P3 chips are ~43 GB each (full W4 experts = 38.3 GB/chip FITS), **W4 experts is
the most promising path** to reach the golden's stable regime AND likely sidesteps
this MSE-specific runtime blowup — pending the loader/kernel W4 support scoped in
`glm-w2-quant-quality-plan.md` and on-device validation.

## UPDATE 3 (2026-09-20, runtime MoE instrumentation — ROOT CAUSE: feedback resonance)

Wrapped `AscendW2DynamicFusedMoEMethod310.apply` to log per-MoE-layer (rank-0)
input/output RMS + topk_weights (`moe_probe_log.json`). Findings:
- Per-layer routed gain (out/in) is BOUNDED but grows with depth: ~0.3 (L3) →
  ~9 (L42), tracking the experts' spectral norm (~6–9). NOT the 500× I feared —
  the earlier 85× was `mlp_out` after the `×2.5` combine + EP all-reduce (this
  probe sees only rank-0's expert shard: `tw_sum`~0.2 partials, and out=0 on
  layers where rank-0 held none of the selected experts — an EP artifact, NOT
  dead layers).
- **ROOT CAUSE = positive-feedback resonance.** The mHC residual accumulates →
  the (RMS-normed, bounded) layer input direction increasingly aligns with each
  MoE's dominant singular vector → per-layer gain climbs to ~spectral (6–9) →
  `routed_scaling_factor=2.5` pushes the residual-accumulation loop gain >1 at
  depth → the residual grows geometrically (L8–L10 onset, RMS 525 by L44).
- **Why MSE-specific / not a code bug:** golden (bf16) and minmax-W2 don't
  resonate (golden accurate; minmax noisier + no clipping). MSE's MSE-optimal
  scale CLIPS block outliers, introducing a *correlated* bias that stays aligned
  across layers and sustains the resonance. Dequant is bit-identical to the
  reference (checked), weights/gammas/input all correct — this is a
  fidelity-driven numerical instability, not a bug in the eager path.

## Path to a WORKING model
1. **W4 experts (recommended).** Golden (nvfp4, ~4-bit) is stable+coherent; W4 on
   310P should replicate that regime (fidelity 0.99, minimal clipping → no
   correlated bias → no resonance). Now known to FIT: 310P3 chips are ~43 GB each
   (full W4 experts = 38.3 GB/chip). Needs converter W4 emit + loader W4 read
   (`[out, in//2]`, 2 codes/byte) + eager `dequantize_w4` in `_apply_device`
   (kernel later for speed). Biggest lift; highest confidence of coherence.
2. **Cheap de-risk first (optional):** re-convert with a MILDER MSE scale
   (`frac≈0.45` vs the 0.30 optimum → less clipping, fidelity ~0.88, between
   minmax 0.83 and MSE 0.92). If the residual stays bounded, it confirms
   clipping-bias drives the resonance; but W2 fidelity may still be too low for
   full coherence (minmax 0.83 is stable yet incoherent), so this is a diagnostic
   step, not necessarily the final fix.

Both need the box + a re-convert (~75 min) + a capture. W4 is the likely endgame.

## UPDATE 4 (2026-09-21, attn-INPUT tap on W4early — the residual is HEALTHY; drift is directional; high-gain DSA amplifies it)

Ran the missing tap: a `forward_pre_hook` on `self_attn` (whose `.forward` is
rebound to the eager KDA/DSA core) captures the **attention input**. Prior
captures hooked a nonexistent `input_layernorm`, so the attention input had
NEVER been recorded on either side. Capture: `golden_ascend_activations_attnin.npz`
(W4early no-clip, TP4 eager, 67-token parity prompt), taps at all 11 DSA layers
+ KDA neighbours. Next token still `' the'` (279), unchanged.

**Decisive result — the input into L7 is NOT corrupted in magnitude/conditioning:**

| L | type | attn_in rms | attn_in cond (max/rms) | attn_out rms | g:attn cos | g:expert cos |
|---|---|---|---|---|---|---|
| 3 | DSA | 0.0161 | 6.32 | 0.328 (gold 0.331) | **0.998** | 0.918 |
| 4 | kda | 0.136 | 5.56 | 0.014 | 0.955 | 0.863 |
| 7 | DSA | 0.0163 | 5.25 | 0.299 (gold 0.209) | **0.624** | 0.302 |
| 11| DSA | 0.0200 | 5.61 | 0.404 | — | — |
| 43| DSA | 0.0299 | 9.34 | 0.444 | — | — |
| 44| kda | 0.214 | 7.23 | 0.159 | 0.408 | 0.127 |

- **L3 and L7 attention inputs are essentially identical**: rms 0.0161 vs 0.0163,
  conditioning 6.3 vs 5.3, both bounded and well-conditioned. Yet L3→cos 0.998,
  L7→cos 0.624. So the L7 divergence is **NOT** a residual magnitude blow-up or
  ill-conditioning (that theme, from the MSE era, is fully closed by no-clip W4:
  attn_in conditioning stays 5–9 at every depth; residual only grows gently,
  L44 layer_out rms 9.95 vs the MSE 525).
- **The DSA has ~20x gain** (0.016 input → 0.3–0.5 output). A high-gain map
  amplifies a small *directional* input error into a large output error — so a
  modest residual direction drift (invisible in rms/cond) at L7's input suffices
  to produce cos 0.624.
- **L7 expert is W4** (W4early = W4 on L3–L26, W2 on L27–45; verified from code
  widths: down_proj_codes [4096,1024]=W4 vs [4096,512]=W2). Its weights are
  ~0.9935 cos, yet expert_out is 0.302 — because the expert INPUT is the drifted
  0.624 attention output. **So L7's expert error is a SYMPTOM of the attention
  divergence, not an expert-bit problem.** Root = the attention.

**Ruled out this pass (code-level, both cheap and clean):**
- DSA softmax scale: eager `qk_head_dim**-0.5 = 256**-0.5` == reference
  `Glm5NextMLAAttention.scaling`; GLM is NoPE (`qk_rope_head_dim=0`) so the YARN
  `mscale**2` branch does NOT run (reference takes the `rotary_emb=None` else).
- Indexer sparsity: `block_topk = index_topk//kpool = 512` ≫ ~17 blocks for a
  67-token prompt → ALL blocks selected → L3 and L7 run identical DENSE causal
  MLA-NoPE. Selection is not the differentiator.

**Remaining single question:** is L7's attention divergence (a) high-gain
amplification of an **upstream directional residual drift** (accumulated from the
MoE contributions — the dominant residual terms — whose per-layer cos is only
~0.86–0.92 even with clean input + W4 weights), or (b) a **DSA compute/weight
issue** that only bites at L7? L3 DSA (same code) is perfect, which argues for (a).
The 43% magnitude inflation of L7 attn_out (0.299 vs gold 0.209) is the one datum
that also admits (b). **This cannot be settled without the golden attention-INPUT
direction at L3/L7** — which the local (free, non-borrowed) RTX PRO 6000 can
capture by re-running the GPU golden with the same `self_attn` pre-hook, then
diffing against the Ascend attn_in we now have. Alternatively an on-box DSA
input-swap test (L3-weights on L7-input and vice-versa) isolates weights-vs-input
without the golden.

**Most-probable root (working hypothesis):** the per-layer MoE output fidelity
floor (~0.92 cos vs the NVFP4 golden, even with clean input + W4 weights — likely
the W4 block-scale scheme being coarser than NVFP4's fp8-e4m3 per-16 block scales,
plus the router topk_ids 0.97 overlap flipping near-tie experts) injects a small
directional error into the dominant residual term every layer; it accumulates and
the high-gain DSA layers amplify it into incoherence. Path to test/fix: (1) golden
attn_in diff to confirm residual drift; (2) raise W4 expert fidelity toward NVFP4
(match its scale structure) — the golden is itself 4-bit and coherent, so 4-bit is
sufficient IF the per-layer fidelity matches.
