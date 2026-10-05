# Seventh experiment: combine indexer key/head and gate projections

## What changes

The indexer reads `hidden_f32` twice, through two FP32 matrix multiplies:

```python
kw = torch.mm(hidden_f32, key_head_weight.t())
gate_score = torch.mm(hidden_f32, gate_weight.t())
```

The candidate prepares `cat((key_head_weight, gate_weight), dim=0)` before graph
capture, then computes one FP32 matrix multiply and takes two output views.
RoPE, key normalization, head scaling, compression, cache updates and selected
pool indices retain their existing code. The views have a different row stride;
downstream layout conversion costs must be included in the serving measurement.

This removes one projection launch and one logical read of `hidden_f32` per
indexer invocation. Arithmetic and weight elements read are unchanged. For
640x4096 FP32 input, the avoided logical input read is 10 MiB per invocation;
cache reuse can make actual external-memory savings smaller. No timing result
or speedup is claimed.

## Resident preparation and memory

Candidate: `tools/glm_perf/resident_candidates/indexer_projection.py`.

It replaces `Indexer.forward` and wraps `NPUModelRunner310.capture_model` in the
existing paused resident transaction. Before the original capture method runs,
it traverses target and draft models, validates all indexer weight shapes, and
prepares a closure-owned bank. It does not change shared source or the harness.

Important constraints:

- All existing FP32 weight copies must already be materialized; no lazy weight
  initialization is allowed from candidate forward, including MTP's first call.
- Original model weight objects/storage stay intact. Combined copies are extra
  memory. Required bytes are calculated before allocation and capped at **64
  MiB per worker**. For head dimension 128, 32 index heads and hidden size 4096,
  each indexer needs 4.5 MiB. Actual model count, including draft, determines the
  total. This cap is not proof that the live server has sufficient free memory.
- Recapture reuses the same combined allocations. Changed weight identities
  require baseline restoration and a fresh candidate transaction. In-place
  weight mutation is outside this inference experiment's contract.
- The bank uses weak indexer keys and owns no model references. Rollback removes
  both replacements; the existing harness clears graphs before recapture.
- A failed preparation/capture leaves the server paused under the existing
  harness contract. No reduced-context or partial-layer fallback is automatic.

No native library build or full model reload is needed for the proposed
resident experiment. It has **not** been applied to workers.

## Gates before promotion

1. Check available memory against the bank's exact byte estimate and graph
   workspace at the existing configured context. Do not lower context to fit.
2. Compare old/new FP32 projections using actual target and draft weights for
   rows 1/2/8/640 and the proposed larger prefill shapes. A wider GEMM can choose
   different accumulation/tiling; start with exact equality. Investigate any
   mismatch before a serving trial, rather than silently loosening tolerances.
3. Verify exact downstream selected pool indices, including near ties and long
   context, and both captured graph sizes. Projection parity is not sufficient
   evidence for different-stride downstream consumers by itself.
4. Measure c1/c4 and cold prefill in both orders, with all other experiments off.
   Retain real quality and MTP repetition gates for this arithmetic change.

CPU checks cover target/draft preparation ordering, byte admission before
allocation, missing/replaced operands, weak lifetime, repeated capture reuse,
one-GEMM forward, and the actual indexer forward's remaining operations.
Small integer-valued fixtures make those layout checks exact; they do not
qualify real-weight FP32 GEMM numerics on 310P.

Validation: **21 projection-candidate CPU tests pass**. The three queue test
files now total **63 passing tests**, including loading the actual candidate
source through `PatchSession`, preparing it again while active, and restoring
both original methods without changing weight storage pointers.

## Repeated-candidate preparation fix

The source-rewriting KDA and MoE candidates previously looked up the currently
installed method. Re-preparing the same active candidate could then attempt to
rewrite an already rewritten function. Candidate functions now retain an
explicit reference to the original method, so repeated preparation starts from
the same baseline. The indexer capture wrapper follows the same rule to avoid
nested preparation and redundant weight banks. CPU regressions cover these
cases; installed runtime files were not changed.

## Routing audit follow-up

The grouped path builds token IDs as repeated `arange(num_tokens)` and gathers
them by the route permutation. The sorted token IDs are equivalently
`dispatch.order // top_k`, because original routes are token-major. Eliminating
both the token-ID expansion and gather needs a GLM-specific descriptor path;
replacing only the gather would leave the expansion allocated. This is recorded
for follow-up, not claimed as another ready optimization. The shared Qwen
dispatcher has not been modified.
