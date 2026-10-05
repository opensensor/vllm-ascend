# Additional opportunity: reorder expert outputs before widening

## Sixth queued candidate

`AscendW2DynamicFusedMoEMethod310._apply_device_grouped` has a fallback combine:

```python
routed = routed.to(torch.float32)
routed *= dispatch.route_weights.index_select(0, dispatch.order)
output = routed.index_select(0, dispatch.inverse_order).reshape(num_tokens, top_k, hidden).sum(1)
```

The staged `moe_half_unpermute.py` candidate instead does:

```python
restored = routed.index_select(0, dispatch.inverse_order).to(torch.float32)
restored.mul_(dispatch.route_weights)
output = restored.reshape(num_tokens, top_k, hidden).sum(1)
```

The inverse permutation already restores original token/route order, so weights
can be applied in their original order. Multiplication still uses FP32 and the
sum visits the same top-k slots in the same order. This is not FP16 accumulation.

### Expected traffic change, not a measured speedup

The large gather reads and writes FP16 instead of FP32. For 640 tokens, top-8
routing and hidden size 4096, its read-plus-write traffic falls from 160 MiB to
80 MiB per layer per rank. Across 42 routed layers that is approximately
**3.28 GiB less logical gather traffic per 640-token chunk per rank**, conditional
on every layer using this fallback. Actual external-memory transactions depend
on cache and kernel implementation; this is not a bandwidth-counter result.

It also removes one route-weight gather. Decode gets smaller traffic savings
but still removes that operation in every participating layer. The FP32 cast
and weighting remain. A measured faster FP16 gather on 310P is not assumed.

Native FP32 route-combine and CANN unpermute branches are untouched. If either
is active, this candidate will not affect that layer. Check branch coverage
before timing it. The fallback still requires initialized peer-owned rows;
zero-weight masking does not make uninitialized or NaN rows safe.

### Staged and checked

- Reversible resident candidate:
  `tools/glm_perf/resident_candidates/moe_half_unpermute.py`.
- Its source guard matches the exact three fallback statements before replacing
  them. The CPU test checks that restoring this block reconstructs the current
  method's entire AST, so routing, projection, native branches and shared-expert
  addition are unchanged.
- **17 new CPU tests pass**, including bitwise finite output equality at
  1/2/8/640/1280 tokens, strided FP16 inputs, peer routes, exceptional values,
  unchanged inputs, and a check that only the FP16 activation gather executes.
- It is appended to `python -m tools.glm_perf.optimization_queue`.
- No new native operator or weight allocation is needed. NPU parity, graph
  recapture and serving measurements are pending; no hardware was used.

Hardware gate: compare exact combine outputs on actual FP16 grouped results,
including peer-only and mixed routes, then test graph rows 2/8. Use the existing
resident harness to compare c1/c4 and cold prefill without a model reload. Keep
all other candidates off. Only combine with prior winners after its own gate.

## Another item to profile: indexer projection fusion

`Glm5NextIndexer.forward` in `vllm_ascend/models/glm5next/attention.py` computes
two independent FP32 matrix multiplications from the same `hidden_f32`:
one for key/head weights and another for compressor gate scores. Concatenating
the projection weights outside graph capture could replace these with one
projection followed by views. It would remove one launch and one logical read
of the hidden-state matrix; total weight elements and arithmetic are unchanged.

This is an investigation item, **not a seventh ready candidate**. A wider GEMM
may choose different accumulation/tiling and change near-tie pool rankings.
It also needs a resident resource with correctly versioned combined weights,
including MTP layers, prepared outside capture. Retaining existing copies would
add memory at the large configured context. Before implementing, measure the
two current GEMMs and establish exact output/selected-index parity for the
combined shape. Do not infer a serving improvement merely from fewer launches.

The inverse routing permutation also uses a second argsort, but the existing
source explicitly records slower scatter/index-copy experiments on 310P. That
is not being requeued without a different implementation or new evidence.
