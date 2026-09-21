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
