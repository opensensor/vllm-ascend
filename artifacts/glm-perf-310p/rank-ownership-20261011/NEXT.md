# Next GLM runtime experiments

This is an offline queue, not a list of measured speedups. Preserve the two-card
W2 and three-card W3 exports. The first three experiments change scheduling or
ownership without requiring another weight quantization. The integer pipeline
and rank-owned indexer already built remain unqualified on NPU hardware.

## First qualify the work already built

Establish the native integer pipeline alone before combining it with indexer
ownership. Compare direct division, integer reconstruction, and source-row
caching independently. Then compare synchronous indexer ownership against
prefill overlap. Each run must record loaded source/bundle hashes, actual operator
dispatch, peak memory, cold time to first token, and steady decode latency.
Changed-input replay and real-model parity are required; CPU transport checks do
not establish graph correctness or HCCL overlap.

Keep the qualified checkpoint and launch settings fixed for the ownership
comparison. Compare the new integer checkpoint against its own unfused reference
before attributing an end-to-end difference to kernel fusion. Use cold prompts
with different lengths and cached continuations; report prompt processing and
decode separately. Do not promote a candidate from reduced operation counts or
a single short request.

## 1. Stream operands instead of fencing the entire core

The [integer projection](../../../tools/glm_perf/glm_channel_projection.cpp)
uses `PIPE_ALL` after operand movement, L1-to-L0 loading, and each `Mmad`.
These fences serialize work that could potentially overlap. Queue dependency
events and selective A/B ping-pong buffers, keeping integer accumulation and the
existing group-scale epilogue unchanged.

The current 64-physical-row build reserves 191,520 bytes of UB for fused A8
gate/up and 224,288 for A4. Duplicating all scratch is not an admissible starting
point. One additional A operand slot is 8192 bytes in UB; duplicating the existing
A/B slots separately adds 24,576 bytes in L1, 8192 in L0A, and 16,384 in L0B.
These are slot sizes, not a complete new memory budget or hardware admission.
Reuse accumulator/readback storage only after its consumers finish.

Offline gate: prove producer/consumer and buffer-reuse ordering across K tiles,
group boundaries, row tails, experts, and output tiles; compile the full fused
variants with a resource ledger. Device gate: exact integer products and matching
outputs, followed by absolute Cube/vector/scalar/memory time and matched
end-to-end prefill/decode latency. Fewer barriers alone are not evidence of gain.

## 2. Compute the router gate and top-k on one local TP owner

The [shipped MoE](../../../vllm_ascend/models/glm5next/model.py) computes the gate
unconditionally, and the [W2 delegate](../../../vllm_ascend/models/glm5next_w2/moe.py)
then computes top-k on every TP rank. Move both to one owner within each stage's
TP group and distribute the original IDs and weights. Skipping only top-k would
leave the replicated gate GEMM intact.

At 640 tokens and top-k eight, INT64 IDs plus FP32 weights occupy 61,440 bytes
(60 KiB). A byte AllGather receive slab would occupy 120 KiB per rank for TP2 or
240 KiB for TP4, excluding packing/send buffers and transport overhead. Reuse
stage scratch across layers. All ranks retain their local routed experts, local
shared-expert partials, and the existing combined reduction.

Offline gate: unchanged score precision, correction bias, normalization, tie
behavior, route order, local-expert masking, and collective ordering. Find a
composition seam that bypasses the gate on peers and accepts preselected routes.
Device gate: compare prefill first; 42 added result transfers per model forward
can hurt decode. Test overlap with each peer's independent shared-expert work.
The owner's gate remains on the critical path, so reduced aggregate work is not
itself a latency improvement.

## 3. Remove prefill combine scalar address walks and route traffic

The [fused prefill down kernel](../../../tools/glm_perf/glm_channel_down_prefill.cpp)
still writes FP16 routed results to GM, reads them back, and obtains inverse route
addresses with scalar `GetValue` calls for every 128-channel output tile.
The smaller retained workspace did not eliminate that traffic.

First queue vector loading/gather of a token block's inverse route addresses and
weights, reused across output tiles where core ownership permits. Then evaluate
a token-owned down/combine schedule that keeps partial output tiles on chip.
Preserve original top-k addition order, repeated experts, invalid routes, and
zero peer contributions; unordered FP32 atomics are not an equivalent substitute.

At 640 tokens, eight routes, and hidden size 4096, the all-routes-local bound is
40 MiB of FP16 writes plus 40 MiB of reads per rank per MoE layer. Actual bytes
scale with valid local routes; TP partitions do not all write this full bound.
Token ownership trades expert weight reuse for gathered input rows, so estimate
both weight reloads and route traffic before selecting a schedule.

Offline gate: deterministic combine parity and complete UB/L1/GM lifetime and
traffic budgets, including uneven expert occupancy. Device gate: absolute scalar
and memory time, total down/combine duration, and cold prefill at growing cached
context. Decode already has an on-chip combine and needs a separate comparison.

## 4. Split indexer heads when owner compute remains dominant

The fixed-owner indexer eliminates duplicate work but leaves one rank doing all
32 heads. A later alternative can split head projections and scoring across TP,
then reduce exact FP32 scores before the owner selects token indices.
Preserve per-head ReLU before summing heads. Merging each rank's local top-k is
not equivalent to selecting from globally summed scores.

Score exchange costs depend on query rows and pooled context length, unlike the
fixed-width index transfer. Begin with a tiled score/reduction memory model and
account for FP32 reduction order and ties. Queue implementation only if the
qualified owner profile shows enough critical-path compute to repay the extra
communication. This is an alternative to full owner scoring, not an automatic
additional optimization.

## Defer lower-priority work

- Reclaim unused peer indexer weights/cache only after ownership is qualified.
  This needs loader and cache-planner changes; it saves capacity, not necessarily
  time, and must retain every rank's local MLA KV.
- Input quantization ownership needs a cost comparison: its output transfer is
  much larger than router IDs/weights and can cost more than repeating the vector
  work.
- Inspect existing vocab-parallel `get_top_tokens` before implementing another
  greedy sampler. The local upstream already has a compact argmax collective.
  General sampling ownership must preserve RNG, logprobs, grammar, and output
  protocols; it is behind cold-prefill work in this queue.
- Broader quantization groups change numerical behavior. Keep that study
  separate from the scheduling/ownership experiments above.

The resource estimates above derive from
[the current memory ledger](../../../tools/glm_perf/channel_scale_memory.py).
They exclude compiler overhead, graph pools, KV/recurrent state, HCCL resources,
and other simultaneously live buffers unless explicitly named. No NPU, live
server, or checkpoint was changed while preparing this queue.
