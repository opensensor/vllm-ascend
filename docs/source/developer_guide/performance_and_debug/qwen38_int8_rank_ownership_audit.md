# Qwen INT8 rank ownership audit

The six-chip INT8 server passes the retained short text, math, code, image,
and thinking gates. It admits 1,068,514 KV tokens for a configured
1,048,576-token request limit. That is approximately one full independent
window in the shared cache, not six full windows.

## Measured regression

The same cold 10,322-token prompt had zero cached tokens in both runs.
Cold TTFT improved from 22.636 seconds on paired W4 to 20.227 seconds on INT8.
The 512-token coding decode decreased from 32.200 to 26.233 tokens/s.
The hardware kernel microbenchmark's gain therefore does not establish an
end-to-end decode gain. These measurements are initial comparisons, not a
repeated matched-load statistical qualification.

The INT8 resident weights also occupy 22.77–23.24 GiB per rank, compared with
approximately 14.6 GiB for the W4 profile. Wider expert codes can increase
weight traffic even when the dot-product arithmetic becomes cheaper. Separate
weight-load, padding, dispatch, and reduction timings are needed to establish
how much each contributes to the measured decode regression.

## Confirmed ownership and dispatch findings

| Area | Current behavior | Consequence / next change |
| --- | --- | --- |
| Small decode | The INT8 extension replaces both `_forward_routed` and `_forward_grouped_chunk` with the grouped function. | W4's fixed-route decode and fused down reduction are lost. Restore an INT8-specific small-token path before tuning large prefill tiles. |
| Route metadata | Grouped dispatch sorts expert keys, counts routes, computes cumulative ends, and sorts an inverse permutation on every rank and layer. | Extra launches exist on the decode critical path. Fixed-route native kernels can consume local expert IDs directly. |
| Gate inputs | Token activations are quantized once. The updated profile gathers only owned codes/scales at 640 tokens or above; smaller calls retain builtin gathers. | The native gather removes peer operand copies for large prefills. Tiny eager gathers were slower, so they are excluded. |
| SwiGLU | Builtin SwiGLU executes over the complete routed matrix on every rank. | Peer-owned zero rows still undergo activation math. Fuse FP16-boundary SwiGLU and INT8 quantization over the local prefix. |
| Projection output clearing | Whole-K projection explicitly zeros rows from `localEnd` through total routed capacity, for both gate/up and down. | Peer output traffic remains large. Remove clearing only after every downstream reader excludes inactive rows. |
| Finalization | The separate CANN finalizer receives the global route permutation and full-capacity output. It casts FP32 route weights to FP16 and widens the FP16 combined output back to FP32. | A local reducer could remove casts and unused reads, but reproducing the current FP16 boundaries and restoring FP32 weighting are distinct numerical qualifications. |
| QSA indexer | Index query/key projections, index-key cache update, and sparse group selection remain replicated; attention query/KV heads are sharded. | Qualify a designated-owner selection plus compact broadcast. First prove selection/cache parity across ranks, including ties, images, prefix reuse, and graph replay. |
| Shared expert | The checkpoint selects `tp_sharded_uneven`. | Its 640 intermediate channels are already partitioned. Do not replace this with six full shared-expert computations. |
| Router | Every rank calculates global router logits and top-k IDs. | Global top-k is required. Local-only top-k changes the model. Sharded logits plus an exact global merge needs communication measurements before adoption. |

The grouped projection already skips peer expert dots using local cumulative
ends. The confirmed redundancy is in preparation, activation, output clearing,
and finalization around those dots. Quantizing the common token input on every
rank is also replicated work, but those operands are needed by multiple ranks;
centralizing it may add more communication cost than it saves.

## Illustrative traffic accounting

For a 640-token step, ten routes, 2,560 hidden channels, and 1,280 gate/up
channels, each full routed output contains 6,400 rows. The down FP16 matrix is
31.25 MiB per rank and the gate/up matrix is 15.625 MiB per rank.
With evenly distributed routes, five sixths of those rows are peer-owned.
The current explicit peer clears alone therefore write approximately
234.375 MiB across six ranks per layer, or 10.986 GiB across 48 layers,
per step. This excludes subsequent reads, SwiGLU traffic, token gathers,
and useful local output writes.

This is executed-byte accounting from the current dimensions and loops, not
an NPU hardware-counter measurement. Route imbalance changes per-rank load;
each route still has exactly one owning rank. Removing unused writes requires
an ownership proof, not simply deleting initialization.

## Qualification order

1. Instrument stage timings and route ownership without host tensor inspection
   in the serving path. Separate kernel time, launch count, HCCL, MTP acceptance,
   and thermal holds from per-request wall time.
2. Implement fixed-route INT8 decode with ordered local reduction. Test changing
   IDs, repeated experts, peer-only ranks, and graph replay with poisoned unused
   buffers. First reproduce the active INT8 finalizer's FP16 weight/output
   boundaries. Qualify FP32 weighting separately; widening an already rounded
   FP16 output does not restore FP32 accumulation accuracy.
3. Add owned-prefix INT8 gather and fused local SwiGLU/quantization. Prove the
   FP16 rounding and integer quantization recipe on real weights before binding.
4. Stop clearing peer output only when all consumers skip it or map it to a
   valid zero sentinel. Validate zero-local-route ranks on successive replays.
5. Shadow QSA selection parity on all six ranks. An owner/broadcast design must
   synchronize selection buffers before attention consumes them, inside and
   outside captured graphs. Measure the added collective, rather than assuming
   sixfold latency improvement from removing sixfold repeated work.

The live server remains on INT8 for user testing. The owned gather and a
small-row projection tile are now hardware-qualified and deployed as described
below. The remaining ownership redesigns are implementation targets.

## Prepared owned-route gather candidate

`NativeLocalChannelGather` and `qwen_local_channel_gather_v1` copy INT8 codes
and eight-lane FP32 scales only up to the final device expert boundary.
They allocate fixed-capacity outputs for graph compatibility, leave inactive
rows untouched, and never read route counts back to the host. The projection
must consume the same boundaries and skip inactive operands.

The existing real-weight partial benchmark accepts `--owned-route-gather`
with `--direct-activation --local-activation`. It requires exact output parity
with the prior direct INT8 path before recording alternating timing samples.
All six resident workers passed 22 byte-exact gather cases each, including
poisoned inactive IDs/output capacity, zero-owned-route ranks, and changing
counts during graph replay. At 640 tokens, captured gather execution took
approximately 0.053–0.054 ms versus 0.080–0.092 ms for the two full gathers.
Tiny eager gathers were slower, so the deployed path uses the owned gather
only for 640 tokens or above. The serving branch does not read device counts
back to the CPU.

The M16 whole-K projection keeps integer accumulation and scaling unchanged.
Synthetic FP16 output and graph replay checks passed on every chip for both
projection geometries. Larger groups regressed, so only calls with at most
three tokens select M16; other calls keep M32.

Alternating timings and exact comparisons on the first resident real-weight
MoE layer passed on all ranks. Its three-token partial improved by roughly
1–3%; its 640-token partial improved by roughly 0.4–1.4%. Six-token calls
retain the old path and showed ordinary measurement variation. These are
single-layer partial timings, excluding the shared expert and HCCL.

The complete update passed marker, math, code, fresh/cached image, long cold
prefill, and thinking gates. The retained follow-up run measured 18.865 seconds
cold TTFT for the same 10,322-token prompt and 29.551 tokens/s for a 512-token
coding response. That compares with 20.227 seconds and 26.233 tokens/s in the
initial INT8 run. Coding/thinking text differed between those runs; these
observations are not a matched-output or statistical attribution of the
entire improvement to these two kernel changes.

The running candidate is `qwen_channel_owned_int8_tp6`. Worker PIDs and resident
weight storage survived the cutover. The graph recapture transaction reset
reusable prefix caches; the earlier isolated kernel probes preserved them.
The restart profile retains this candidate. Hardware and cutover receipts are
in `artifacts/qwen38-owned-int8-gather-20261011/`.

## QSA owner selection and preprocessing overlap

The GLM frontend observation transfers as a dependency pattern, not as an
already available Qwen overlap point. The active Qwen `_QSAAttention.forward`
currently projects local Q/K/V/gate, normalizes and rotates Q/K, writes local
KV, updates the index cache, and then scores/selects sparse groups. Moving the
owner indexer earlier is necessary before peer Q/K work can hide its latency.
The indexer depends on index Q/K, logical positions, and the index cache;
it does not depend on the local attention Q/K projections or the attention KV
write. Sparse attention depends on both completed paths.

`tools/qwen4exp/qsa_owner_overlap.py` provides an **offline, unbound** schedule
prototype. It admits only uncaptured pure prefill batches of at least 128
tokens by default. Every decode, mixed batch, captured call, and single-rank
call retains replicated selection and the original local-then-indexer order.
The threshold is an experimental admission rule, not a measured crossover.

For admitted batches, a device event makes the completed index-cache update
and indexer operands visible to a resource-owned stream. Only the designated
rank scores/selects. All ranks enqueue one INT32 packet broadcast on that
stream while local Q/K preparation runs on the main stream. A final device
event joins the selection immediately before the caller can consume it.
The packet contains group IDs, group counts, tail starts, and tail counts;
each attention operand is a contiguous view of the same allocation. No score
matrix, FP16 KV tensor, or host route-count read is introduced by the schedule.

Keep the existing index-key projections and index-cache updates on all ranks
for this first experiment. Owner-only cache updates would leave peer caches
stale when execution returns to replicated decode or a mixed batch. Eliminating
those updates requires a separate cache ownership and synchronization design.
Precompute shared RoPE memoization before scheduling concurrent callbacks;
neither callback may mutate that cache during overlap.

Offline tests cover owner-only selection, six-rank packet contents, contiguous
views, padding, allocator stream ownership, event ordering, fallback admission,
and failure before submission for incompatible devices. They do not validate
actual HCCL completion or simultaneous Cube execution. Before model binding,
qualify selection parity on real weights, near-cutoff ties, MRoPE images,
prefix reuse, decode/prefill transitions, and repeated request shapes. Verify
the collective uses the intended group-relative owner and never silently skips
a disabled communicator. Once a collective is submitted, a rank must not
fall back independently. Measure synchronous owner broadcast against overlap
and against replicated selection, including per-rank imbalance and HCCL time.
The live server has not been changed for this prototype.
