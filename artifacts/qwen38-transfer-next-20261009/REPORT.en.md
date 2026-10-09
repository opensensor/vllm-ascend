# Qwen transfer, checkpoint and communication candidates

[Chinese report](REPORT.zh.md) · [Deferred runbook](RUNBOOK.en.md).

All five follow-up areas now have offline implementations or explicit resident
candidates. No server was started, stopped, reconfigured or exercised. Native
code was compiled on the serving host without opening an NPU device, loading
the bridge or launching kernels. Throughput, power and thermal gains are unknown.

## Changes

| Area | Delivered behavior | Default and limits |
| --- | --- | --- |
| Transfer telemetry | Prefix copy direction, logical bytes, archive swaps, CoW, host spill/restore and host call duration; explicit drain reasons; optional runner metadata/sample and MoE TP payload records | Prefix cumulative counters are active; timestamp detail and runner/MoE hooks are opt-in |
| Checkpoint phases | Invalidation/CoW across groups shares one required worker drain; admissions across groups share another when primary slots must be reused | Opt-in; retirement was already batched; barriers between phases remain |
| GDN state IO | Native gather + cold-state mask + transpose, and inverse scatter; chunk H/O accepts and returns its native FP32 state layout directly | Opt-in, FP32, 128 × 128 heads, at most four sequences and 48 value heads |
| Native W4 reuse | Cache scale, offset and sum metadata in UB once per expert/output tile, reusing it across M=32 row tiles | Opt-in template specialization; existing packed-weight L1 reuse and default schedule remain |
| TP communication | Eager prefill can reduce one local routed + TP-sharded shared chunk on a communication stream while computing the next chunk | Opt-in, 1,024-token chunks, at most two in flight; decode graphs remain on the original path |

Telemetry never calls `item()`, copies values to the host or queries device
events. It accounts for submitted logical payloads and existing host waits,
not physical DDR traffic or energy. Copy host time measures submission/blocking,
not asynchronous kernel duration. Host timestamps are bounded to 256 recent
records per enabled ledger. `resident_status` exposes rank/PID, prefix, runner
and module ledgers. The saved-receipt comparator requires all four ranks,
unchanged PIDs/configuration and nondecreasing counters.

Coverage is explicit: checkpoint copies, the compact table, PLE history,
Qwen MTP sampled-token delivery, explicit prefix drains and W4 TP calls.
Internal operator DMA, general runner metadata, attention collectives, graph
replay task counts and allocator traffic still require the NPU profiler.
Graph Python counters describe capture/dispatch, not every replayed task.

Checkpoint batching does not change retention, archive capacity, spill policy,
state ownership or captured storage addresses. No completion token is reused
across base runner updates or an admission phase. Empty updates and admission
hits add no drain. Failed initial synchronization preserves checkpoint metadata;
a later copy failure is not an atomic rollback transaction.

GDN retains the existing cache layout and state dtype. Cold rows become literal
zero without reading stale cache bytes; warm rows transpose without arithmetic.
The scalar 8 × 8 native transpose uses 512 bytes of UB per core and needs a
performance gate: fewer intermediate copies does not prove it runs faster.
Upstream generic GDN behavior is untouched; the hook targets this fork's
`_GDNAttention._native_delta_rule`, including its non-speculative mixed view.
Decode and speculative state updates retain their in-place recurrent path.
Device slot IDs are trusted scheduler metadata; validate in-range unique
scatter ownership before qualifying the candidate.

W4 uses the same activation quantization, INT4 products, FP32 correction order,
FP16 boundary and inactive-output zeroing. The resource rejects decode/routed
IDs, older metadata lanes, other expert counts and unsupported geometry. Its
largest N=160 metadata cache adds 19,200 UB bytes per core; the explicit M=32
schedule allocates 141,632 static UB bytes. This is not a memory-capacity
measurement. Packed weights already reside in L1 across row tiles: they are
not redundantly reloaded per row tile by the original grouped schedule.

The TP pipeline trades one large reduction for several smaller reductions,
which can increase HCCL overhead. A 2,560-token MoE call issues three collectives
instead of one. All ranks must agree on chunks and collective order. Shared
weights remain TP-sharded and are added before reduction; replicated policies
are rejected. Chunking can alter GEMM numerics. No cross-layer, attention or
captured-decode communication overlap is claimed.

## Corrected earlier candidate wiring

The prior fused-WY resident candidate patched an upstream GDN alias that this
custom model does not call. It now supplies the callback through the actual
Qwen serving method. Regression tests verify that callback execution, state IO
and restoration occur on the model path. The old kernel component gate did not
establish this integration. Previous artifact reports describe their historical
snapshot; use this corrected runtime for later serving gates. The previous WY
manifest now fingerprints the model file too.

## Validation

- Focused CPU suite: **332 passed, one skipped** for unavailable `msmodelslim`.
  See [CPU receipt](offline-tests.log).
- Native state gather/scatter bodies compiled and executed against bounded CPU
  DMA/vector stubs at 1/12/48 value heads, including four sequences, shuffled
  slots and cold stale NaNs. FP32 values match the reference exactly. Stub
  events are no-ops and do not validate NPU ordering.
- Checkpoint CoW chains, one drain per phase, failure preservation, slot reuse,
  runner staging order, no-op updates and existing prefix behavior pass.
- TP tests verify combined shared/routed math and bounded dependency scheduling;
  these do not emulate HCCL or establish hardware overlap.
- W4 CPU checks verify cached metadata matches strided banks, shape/dtype
  rejection and tiling arguments. Cube arithmetic/event correctness remains
  a device gate.
- CANN 9.1.0, dav-2002 host compilation: **two device binaries with three entry
  points and one versioned bridge**. Source fingerprints match the staged code.
  See [build receipt](host-build.log) and [provenance](host-build-provenance.json).
- An extended run had four existing lifecycle failures from missing
  `torch_npu.float4_e2m1fn_x2` in the local CPU environment. The unchanged parent
  `3802a6e54` reproduces the same four failures, with 16 passes. See
  [extended receipt](extended-tests.log.gz) and
  [parent receipt](baseline-gdn-lifecycle.log.gz).
- Scoped hooks and repository-wide formatting results are recorded in
  [validation metadata](validation.json). No full-suite or NPU pass is claimed.

[Logical bounds](logical-bounds.json) are source-level payload estimates, not
measured bus traffic. For four sequences with twelve TP-local value heads,
state payload is 3 MiB; eliminating intermediate state IO changes the estimated
read/write payload from 30 MiB to 12 MiB per layer, excluding H/O, WY, output
and convolution. Real cache/DDR behavior can differ.

## Deferred qualification

Use a complete runtime snapshot. Qualify each candidate independently with
native component parity, FP32 final states, cold/warm prefixes, C1/C2/C3,
mixed prefill/decode, fresh/cached images, cancellation and prefix CoW under
TP4/EP4. Retain the 94°C hold / all-core 85°C resume policy and independent
96°C cutoff; do not switch candidates or recapture during a thermal hold.
Measure all ranks' memcpy directions/bytes, event intervals and HCCL tasks
against a synchronized thermal timeline. A reduced transfer count alone does
not establish lower heat. Combine candidates only after individual gates.

Dummy, real-weight, image, capacity, ACL graph replay, EP, MTP acceptance,
performance and sustained thermal gates were **not run** under the explicit
offline instruction. Existing image capability is retained in the code, but
these candidates have no new image validation or capacity claim. FlashComm1
remains outside this profile. Server startup remains deferred.
