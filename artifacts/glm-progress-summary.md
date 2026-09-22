# GLM-5.3-Flash on Ascend 310P — Progress Summary

_Last updated: 2026-09-22_

## Goal

Run GLM-5.3-Flash (~304B params; 45-layer hybrid of KDA linear-attention +
DeepSeek-Sparse-Attention, 288-expert MoE) **coherently** on 4× Ascend 310P at TP4.
GLM is bf16-designed; the 310P is effectively fp16-only for NZ weights — the whole
effort is fighting range/precision and quantization loss on constrained HBM.

## Where it stands

The model **loads and generates end-to-end at TP4**, and two independent
catastrophic errors are fixed: the TP shared-expert reduction and packed-W2
prefill corruption. The best fitting checkpoint uses 36.6536 GB/chip and no
longer emits numerically explosive gibberish. Exact golden parity is not yet
reached: on the 67-token parity prompt it selects `' '` while the golden token
`' forty'` is rank 2, only 0.5234 log-prob behind. What remains is quantization
accuracy under the card's memory ceiling, not an execution-correctness failure.

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
(the attention _input_, a tap never recorded before; prior captures hooked a
nonexistent `input_layernorm`): L3 and L7 attention inputs are **near-identical and
both healthy** (RMS 0.0161 vs 0.0163; conditioning max/rms 6.3 vs 5.3; bounded
everywhere). Yet L3 is perfect and L7 breaks. So:

- The break is **not** a residual blow-up or ill-conditioning (that theme is closed).
- The DSA has **~20× gain** (0.016 in → 0.3 out), so it amplifies a small
  _directional_ input error into a large output error.
- **L7's expert is W4** (verified: W4early = W4 on L3–26, W2 on L27–45, from
  down_proj_codes widths [4096,1024]=W4 vs [4096,512]=W2), weights ~0.9935 — its
  0.302 output is a **symptom** of the drifted attention input, not an
  expert-bit problem.

## Ruled out (code-level)

- **DSA softmax scale**: eager `qk_head_dim**-0.5 = 256**-0.5` matches the reference
  `Glm5NextMLAAttention.scaling` exactly; GLM is NoPE (`qk_rope_head_dim=0`) so the
  YARN `mscale**2` branch does not run.
- **Indexer sparsity**: at 67 tokens `block_topk = index_topk//kpool = 512` ≫ ~17
  blocks → all blocks selected → L3 and L7 run _identical_ dense MLA-NoPE. Selection
  is not the differentiator.

## The one open question

Is L7's divergence **(a)** high-gain amplification of upstream **directional
residual drift** — from the MoE contributions (the dominant residual terms), whose
per-layer cosine is only ~0.86–0.92 even with a clean input and W4 weights — or
**(b)** a DSA compute issue biting only at L7? L3 DSA being perfect argues for (a);
the lone datum for (b) is L7's attn_out running 43% hot (0.299 vs golden 0.209).

**Settling it needs the golden _attention-input_ direction at L3/L7**, which requires
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
deep-layer _stability_, but not for the L3 _accuracy_ floor.

## Next steps

1. **The remaining lever is bf16 range/precision in the non-quantised layers**
   (embeddings, dense MLP L0–L2, attention, residual). L0 is already 0.985 — chase
   why the first dense layer drifts (fp16 vs bf16 rounding) before any quant.
2. For deep-layer stability, keep L19–L26 at W4 (0.991), not W2 (0.83), regardless
   of the NVFP4-direct result.
3. If bf16 emulation is infeasible on the fp16-only 310P, this model cannot reach
   golden coherence; the honest deliverable is the diagnostic + the quant tooling.

## UPDATE 4 (2026-09-21, expanded GPU mHC golden)

Re-ran the coherent Shadeform NVFP4 reference with direct instrumentation of the
mHC custom-op inputs and outputs at L0/L3/L7. The run reproduced next token
`35223` (`' forty'`) and saved 77 finite tensors in
`golden_gpu_mhc_activations.npz` plus metadata in
`golden_gpu_mhc_meta.json`. The capture includes residual, post/combination
mixes, normalized layer input, attention output, MLP output, and layer output.

The existing Ascend scalar diagnostic already excludes an L0 mHC magnitude bug:

| boundary | GPU RMS | Ascend RMS | Ascend/GPU |
|---|---:|---:|---:|
| L0 pre layer input | 0.087670 | 0.087670 | 0.999994 |
| L0 fused residual | 0.007827 | 0.007828 | 1.000075 |
| L0 fused layer input | 0.046793 | 0.046787 | 0.999882 |
| L3 fused layer input | 0.381908 | 0.380776 | 0.997036 |
| L7 fused layer input | 0.266913 | 0.263441 | 0.986992 |

The matching Ascend tensor capture subsequently completed against
`GLM-5.3-Flash-NVFP4mix-310p` (53 finite tensors, next token `'心安'`). Its exact
cosine comparison settles the mHC question:

| boundary | cosine vs GPU | Ascend/GPU RMS |
|---|---:|---:|
| L0 attention output | 0.999975 | 1.00002 |
| L0 fused-mHC layer input | 0.999967 | 0.99980 |
| L0 dense MLP output | **0.985325** | 1.00081 |
| L3 attention output | 0.998906 | 0.99435 |
| L3 fused-mHC layer input | 0.997193 | 0.99711 |
| L3 MoE output | **0.943309** | 0.82933 |
| L7 attention output | **0.627978** | 1.42046 |
| L7 fused-mHC layer input | 0.843086 | 0.98690 |

The mHC implementation is directionally correct while its inputs are healthy;
at L7 it actually improves the already-divergent attention direction from 0.628
to 0.843. The earliest material error is therefore the L0 dense MLP (0.985),
followed by accumulated MoE drift (L3 output 0.943). L7's high-gain DSA then
amplifies that upstream drift. Further mHC rounding changes are not justified by
the evidence; the next capture/fix target is the internal dense-MLP projection
boundary at L0-L2 (gate/up activation versus down projection).

## UPDATE 5 (2026-09-22, root cause found and fixed)

The expanded L0-L7 capture and code audit found the actual TP4 correctness bug.
The shipped GLM shared-expert MLP is tensor-parallel and is deliberately built
with `reduce_results=False`: the fused MoE path is responsible for reducing the
routed and shared contributions together. The 310P W2 delegate instead did:

```text
all_reduce(local_routed) * 2.5 + local_shared
```

That left a different, partial shared-expert contribution on every TP rank. It
also explains the layer-specific magnitude collapse exactly: at L6 the golden
shared branch has RMS 0.368 and dominates the full MoE RMS 0.380, while the
broken Ascend output had RMS 0.087 (almost routed-only).

`Glm5NextW2MoE.forward` now preserves local routed and shared results, combines
them, and issues one TP all-reduce over the combined tensor:

```text
all_reduce(local_routed * 2.5 + local_shared)
```

Real-weight TP4 validation on `GLM-5.3-Flash-NVFP4mix-310p` confirms the fix:

| boundary | before cosine | after cosine | before RMS ratio | after RMS ratio |
|---|---:|---:|---:|---:|
| L3 MoE output | 0.9433 | **0.9981** | 0.829 | **1.007** |
| L4 MoE output | 0.8867 | **0.9870** | 0.802 | **1.000** |
| L5 MoE output | 0.8308 | **0.9822** | 0.755 | **0.994** |
| L6 MoE output | 0.7915 | **0.9817** | 0.230 | **1.001** |
| L7 attention output | 0.6279 | **0.9931** | 1.420 | **1.003** |
| L7 MoE output | 0.5546 | **0.9797** | 0.412 | **1.009** |

This disproves the prior FP16-vs-BF16-blocker conclusion: the small dense-layer
differences are tolerated, while the missing TP reduction was the compounding
error. The NVFP4-direct/W2-late checkpoint still selects `' blo'` because its
known W2 layers 19-44 are unstable; final coherence validation is being run on
the stable no-clip checkpoint (W4 layers 3-26, W2 layers 27-44).

## UPDATE 6 (2026-09-22, packed-W2 Cube corruption isolated and contained)

With the TP collective fixed, a full-depth TP4 capture found a discontinuous
failure at layer 27, the first W2 layer in the W4-through-26 checkpoint. Layer
23 remained bounded, and layer 27 entered the MoE with healthy activations, but
the packed-W2 Cube down projection returned corrupt values:

| boundary | cosine vs GPU | GPU RMS | bad Cube RMS | bad/golden |
|---|---:|---:|---:|---:|
| L23 MoE output (W4) | 0.8478 | 0.08450 | 0.08365 | 0.990x |
| L27 MoE input | 0.9447 | 0.33461 | 0.33550 | 1.003x |
| L27 MoE output (W2) | -0.0070 | 0.18173 | 39074.1 | 215016x |

A direct packed-code hardware test isolated the kernel boundary for the model's
down-projection shape `[M, 2048] x [4096, 2048]^T`: `M=48` agrees with eager
FP32 math (`max_abs=0.001953`), while `M=49` is the first corrupt case
(`max_abs=18256`). Single-token decode is correct. The W2 runtime now uses the
Cube kernel only for expert groups of at most 48 tokens and falls back to exact
eager dequant/matmul for larger prefill groups. It does not add an environment
variable or disable the optimized decode path.

Real-weight TP4 validation of that mixed dispatch removed the explosion and
matched the forced-eager result:

| boundary | cosine vs GPU | Ascend/GPU RMS |
|---|---:|---:|
| L27 MoE output | 0.6268 | 0.9570 |
| L31 MoE output | 0.4767 | 1.2276 |
| L35 MoE output | 0.5480 | 1.1225 |
| L39 MoE output | 0.4468 | 1.1683 |
| L43 MoE output | 0.6748 | 0.9631 |
| L44 final layer output | 0.7658 | 0.9171 |

The next token is now bounded (`' '`) instead of gibberish caused by a numeric
explosion. The remaining direction loss comes from coarse W2 weights. A
W4-through-34 checkpoint was converted at 155.3 GB, but its 37.50 GB/chip weight
footprint does not survive the 96-token runtime profile. W4-through-32 is the
largest fitting candidate validated end-to-end; it uses 36.6536 GB/chip and
leaves W2 only on layers 33-44.

## UPDATE 7 (2026-09-22, direct server and golden-prompt validation)

Direct `vllm serve` validation used the real W4-through-32 checkpoint, TP4,
FP16, eager execution, `max_model_len=96`, and 64 overridden KV blocks. The
larger cache matters: 16 blocks admit the 13-token smoke prompt but leave the
67-token parity request waiting forever for scheduler capacity; 64 blocks log
768 aggregate cache tokens and admit the parity request immediately.

The OpenAI-compatible `/v1/completions` request returned HTTP 200 after about
151 seconds. During the forward, `npu-smi` sampled all four cards doing useful
work (27-36% initially and later as high as 82/28/50/99% AICore), confirming that
the eager correctness run is slow rather than idle. The deterministic top five
were:

| token | log-prob |
|---|---:|
| `' '` | -1.7226 |
| `' forty'` | **-2.2460** |
| `' twelve'` | -2.8241 |
| `' seven'` | -2.9804 |
| `' the'` | -3.0741 |

This is a substantial quality recovery from arbitrary/gibberish output: the
GPU-golden token is now rank 2 with a 0.5234 log-prob gap, but it is not an exact
correctness pass. A W4-through-33 overlay (154.98 GB total, 37.0755 GB/chip)
was also attempted; startup failed during the 96-token profile because HCCL
could not allocate a 420,478,976-byte all-reduce buffer. W4-through-32 is
therefore the measured memory boundary for this configuration.

## UPDATE 8 (2026-09-22, local-only full-W4 fit plan)

Hardware testing is intentionally deferred while the Ascend instance is used
elsewhere. Local checkpoint and runtime analysis produced a viable way to keep
all 42 main routed-expert layers (3-44) at W4 without dropping a layer:

1. The existing Ascend prefetch offloader can evict registered decoder-layer
   parameters while leaving the packed expert banks resident. The packed banks
   are plain `_PackedW2Expert` tensor holders, not `nn.Parameter`s, whereas the
   attention, dense/shared MLP, router, norm, and mHC tensors are registered.
2. Each of the 11 eager DSA seams allocated a second copy of its shipped
   MLA/indexer projections. At TP4 this is 82.255 MiB/layer, or 0.8836 GiB/rank.
   The eager parameters now share the shipped parameter storage during model
   construction, before checkpoint loading and memory profiling. The first
   forward repeats the binding as a mixed-dtype fallback.
3. Prefetching every decoder layer with one reusable buffer is expected to evict
   about 4.04 GiB/rank and retain about 0.23 GiB/rank of shape-unique static
   buffers, for roughly 3.8 GiB/rank net savings. It preserves the exact loaded
   weights; it does not add another quantization step.

The exact expert-size delta is 0.421875 GiB/rank for each W2 -> W4 layer. Thus:

| configuration | estimated weight footprint/rank |
|---|---:|
| measured W4-through-32 | 36.6536 GiB |
| full W4 before memory fixes | 41.7161 GiB |
| full W4 after early DSA storage sharing | 40.8325 GiB |
| full W4 after DSA sharing + all-layer prefetch | about 37.0 GiB |

The local artifact is:

```text
/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-W4all-overlay-310p
```

It reuses the validated layer-3-34 checkpoint and adds noclip W4 overlays for
layers 35-44. Local integrity validation found all 42 main expert layers, 1,728
expert tensors/layer, correct W4 code/scale geometry, 35.7144 GiB/rank of main
routed weights, and no broken symlinks. Layer 45 is the intentionally unwired
MTP head and remains W2; the loader already excludes it.

The next direct-serve run should add the existing vLLM Ascend prefetch flags to
the prior known-good TP4/eager/96-token/64-block command:

```text
--offload-backend prefetch \
--offload-group-size 1 \
--offload-num-in-group 1 \
--offload-prefetch-step 1
```

Omit `--offload-params` so every registered decoder parameter is eligible. This
will be a correctness-first run: W4 still uses eager FP32 dequant/matmul because
the current packed Cube kernel hardcodes four 2-bit codes per byte. The bounded
performance follow-up is a two-codes-per-byte W4 Cube variant plus packed-width
dispatch. None of this update has yet been validated on Ascend hardware.

## UPDATE 9 (2026-09-22, full-W4 load and offloader correction)

The full-W4 artifact was synced to the Threadripper and passed its remote index
integrity check: all 75,575 mapped tensors resolve across 45 shard files, with
no missing files or broken links. A TP4/eager/96-token/64-block serve attempt
loaded all 45 shards and reported 37.6656 GiB/rank before offloader
finalization. The prefetch offloader selected all 45 decoder layers, reported
4.4410 GB offloaded with a 0.2527 GB static pool, then failed at the first lazy
HCCL all-reduce initialization because the released parameter allocations were
still held by PyTorch's NPU caching allocator.

Code inspection also found that the Ascend specialization called upstream
`post_init()` before converting its static buffers to FRACTAL_NZ. Upstream had
already bound parameters and started the first prefetch into the original ND
buffers, so replacing entries in the pool afterward produced unused NZ buffers
while inference retained stale ND buffer references.

The local implementation now:

1. creates the static pool and converts required entries to NZ before assigning
   any buffer to a parameter or starting any prefetch;
2. returns released cached NPU blocks to the device before the profile run, so
   the external HCCL allocator can use the net offload savings; and
3. subtracts `total_offloaded_bytes - static_buffer_pool_bytes` from the model
   runner's resident-weight accounting before memory profiling.

The correction passes the offloader and GLM/W2 regression set locally (65
tests). Per operator instruction, no further Ascend hardware run has been made;
the corrected startup and golden prompt remain hardware-unvalidated.

## UPDATE 10 (2026-09-22, sparse packed-expert offload)

The all-layer registered-parameter offload made full W4 fit, but transferred
about 4.2 GB/rank for every token. The 310P GLM path can now instead select
packed routed-expert banks with the existing prefetch layer pattern by adding
`packed_experts` to `--offload-params`. Selected banks remain pinned on the
host; after routing, only nonzero locally-owned experts are staged to NPU in
their compact W2/W4 representation. The weights are never widened on host or
while crossing PCIe.

A conservative full-W4 starting point is one expert layer in every eight:

```text
--offload-backend prefetch \
--offload-group-size 8 \
--offload-num-in-group 1 \
--offload-prefetch-step 1 \
--offload-params packed_experts
```

Five selected MoE layers should free about 4.25 GiB/rank. At single-token
decode, top-8 routing transfers at most the selected local experts rather than
all 72 local experts, reducing the expected steady-state transfer from roughly
4 GB/rank/token to about 0.1--0.2 GB/rank/token. Exact throughput and the
minimum safe pattern still require hardware measurement.

## UPDATE 11 (2026-09-22, stateful 310P KDA execution)

The W2 adapter's KDA override was still a CPU-parity implementation: every
forward started from a zero conv/recurrent state and ran a Python loop over the
token dimension. That explains both a fundamental decode-correctness break and
much of the observed approximately 0.1 token/s behavior across 34 KDA layers.

The 310P path now uses operators already shipped in this tree:

1. `npu_causal_conv1d_310` reads and updates the paged convolution state;
2. `chunk_kda_fwd` handles variable-length prefill with FP32 carry
   accumulation; and
3. `npu_recurrent_gated_delta_rule_310` handles decode/spec in place against
   the paged recurrence pool.

The recurrent operator is driven through its per-key-channel `gk` input, not
the scalar GDN `g` input, and receives the exact GLM bounded gate
`lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))`. Thus this is not a
GDN approximation. The persistent recurrence pool is FP16, as required by the
310P kernel, while chunked prefill converts active states to FP32 and casts the
final carry back. This halves recurrence-cache storage compared with the prior
FP32 declaration. Weight-only gate constants and transposed FP16 conv weights
are cached after their first use, and the common non-spec path returns the
operator output directly without an extra zero-and-copy buffer.

Host validation covers the safe-gate formula, variable-length spec slot
flattening, four-entry KDA cache dtype contract, operator-lane structure, and
the existing W2 regressions (80 tests passing). Throughput and real-weight
golden output still require the next authorized 310P run.
