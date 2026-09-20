# GLM-5.3-Flash-W2 on 310P — expert quantization quality plan

Status: 2026-09-20. Author: bring-up session (golden-diff analysis).

## Problem

After the two catastrophic bugs were fixed (KDA `o_norm` eps revert; MoE Cube
packed-codes contract), the model is coherent-ish but the next token is a
plausible-but-wrong `' the'` vs the golden `' forty'`. The residual error is
**inherent 2-bit expert quantization loss**, localized by cosine-diffing the
Ascend W2 capture against the NVIDIA-validated NVFP4 GPU golden
(`artifacts/golden_gpu_activations.npz`) on the 67-token parity prompt:

| MoE layer | `expert_out` cos vs golden |
|---|---|
| L3 (first MoE, clean input) | 0.81 |
| L4 | 0.57 |
| L7 | 0.26 |
| L44 | 0.29 |

Router logits stay ~0.999 and topk-overlap is 0.97 at L3, so routing is fine —
the error is the **W2 expert GEMM output**, compounding over 42 MoE layers.

Root cause is quantization fidelity, confirmed against the fp8 source
(`GLM-5.3-Flash-FP8`) at the weight level:

| scheme | weight cosine vs fp8 |
|---|---|
| W2 current (`max(pos/1.5, neg/2.5)`) | **0.833** |
| stored codes == re-quantize(ref) | 1.0000 → converter faithful, no bug |
| W2 MSE-optimal scale (~0.30·absmax) | **0.921** |
| **W4** (`{-8..7}`) | **0.991** |

## Fix ladder (cheapest first)

### Step 1 — MSE-recalibrated W2 (DONE in code, free, no memory cost)
`compute_block_scales(method="mse")` + `convert_full.py` default `method="mse"`
(commit `7c822d035`). Lifts weight cosine 0.833 → ~0.921 with the **same code
format, dequant path, and kernels** — only the stored scale value changes.
- **Action:** re-convert the checkpoint offline
  (`GLM_W2_SCALE_METHOD=mse python -m tools.glm_w2.convert_full ...`, ~hours CPU),
  re-capture on the box (`glm_launch.sh capture_ascend_w2_72.py`,
  `VLLM_ASCEND_W2_DISABLE_CUBE=1`), re-run `compare_activations.py`.
- **Expected:** L3 expert_out ~0.81 → ~0.90; much gentler accumulation. May be
  sufficient for coherence on its own. **Do this before spending any memory.**

### Step 2 — mixed-precision W4 on the highest-impact layers (if Step 1 is not enough)

Full W4 is **38.3 GB/chip** of experts (TP4) vs W2's **19.3 GB/chip** — over
budget (W2 already fits at ~24–25.5 GB/chip total incl. fp16 non-experts + KV +
activations). So keep W2 as the base and promote only the most impactful MoE
layers to W4.

- **Per-layer cost:** W2→W4 adds **+0.453 GB/chip** per MoE layer (TP4).
  Number of W4 layers that fit `N_W4 = floor(headroom_per_chip / 0.453)`.
  → Confirm real per-chip HBM headroom on the box (`npu-smi`), then pick `N_W4`.
  Freeing budget if needed: lower `max_model_len`/KV, or shrink `max_num_seqs`.
- **Which layers:** two candidate policies, pick via a quick ablation —
  1. **Deepest-K** MoE layers (their expert error routes most directly to logits;
     empirically L7/L44 are worst). Start with the last ~8–12 MoE layers.
  2. **Sensitivity-ranked:** use the golden capture — per MoE layer, substitute
     the golden `expert_out` for the Ascend one and measure final-logit / top-1
     recovery; W4 the layers with the largest recovery. More principled; needs a
     one-off ablation script over the saved `.npz` pair.
  Likely also promote **L3** (first MoE; feeds every downstream layer).
- **Format:** W4 primitives already exist (`w2_format.quantize_weight_w4`,
  `pack_w4_codes` `{-8..7}` 2/byte, `dequantize_w4`) and MSE scaling applies to
  W4 too (`method="mse"`).

#### Code changes required for mixed W2/W4
1. **Converter** (`tools/glm_w2/convert_full.py`): accept a per-layer bit-width
   map (e.g. `--w4-layers 34,35,...,44`); emit W4 codes (`[out, in//2]` uint8) +
   scales for those layers, W2 elsewhere; record bit-width per layer in the
   output config/manifest.
2. **Loader** (`vllm_ascend/models/glm5next_w2/{model,weight_mapping,moe}.py`,
   `_PackedW2Expert`): read per-layer bit-width; W4 experts have
   `codes [out, in//2]` (vs W2 `in//4`) and the same `[32,32]` fp32 scale grid.
   `weight_mapping.expected_expert_shape` must branch on bits.
3. **Compute** (`vllm_ascend/_310p/quantization/methods/w2_dynamic.py`
   `_apply_device`): dispatch per layer. Simplest first cut — **W4 layers run the
   eager fp32 path** (`dequantize_w4`; no new kernel), W2 layers keep the Cube
   fast path. Optional later: a W4 Cube kernel variant of
   `npu_w2_blocked_dequant_matmul_310` (unpack 4-bit on-chip) for speed.

### Step 3 — validate
Re-capture on the box, `compare_activations.py` vs the golden; success = deep-layer
`expert_out` cos → ~0.95+, next token → `' forty'` (or coherent continuation).
Keep the broken/fixed `.npz` baselines in `artifacts/` for regression.

## Notes / gotchas
- Re-conversion and re-capture need the Ascend box (`matteius-threadripper`,
  `/srv/ai/venvs/fork028`, `VLLM_ASCEND_310P_ENABLE_MLA=1`). Eager MoE capture is
  slow (~3.5 min/forward) but fine for a one-shot golden.
- The Cube v3 kernel (packed codes + on-chip unpack) on origin/main is still
  UNBUILT/UNVALIDATED — it's a *speed* fix for W2, orthogonal to this quality work;
  eager is the proven-correct path for validation.
- 2-bit is fundamentally capped (~0.92 even MSE-optimal); if Step 1+2 still fall
  short, the remaining lever is W4 on more layers (memory permitting) or a smarter
  quantizer (GPTQ/AWQ-style error feedback) — a larger effort.
