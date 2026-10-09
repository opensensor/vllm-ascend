# Qwen: five offline optimization candidates

## US English summary

Implemented candidates for fused paged QSA, decode-aware prefill pacing,
fused GDN WY preparation, active local MoE route preparation, and MTP0/1/2
comparison. Defaults remain unchanged. The server was not restarted,
reconfigured or contacted for inference. No NPU workload was submitted.
Host CANN compilation succeeded for three native kernels and the versioned
bridge; this does not qualify device execution, event ordering, quality,
performance or temperature. All hardware and real-weight gates remain pending.

Work began October 8 and finished October 9, 2026. The source base includes
the previous memory fixes (`048676877`) and unrelated committed GLM work.
Uncommitted work in the shared checkout is preserved.

## Changes and limits

| Area | Implementation | Remaining gate or limitation |
| --- | --- | --- |
| QSA | Explicit `ascend_qsa_prefill.backend=paged_native` selects the existing on-chip gather/attention kernel for large pure prefills. Unused parallel gather streams are disabled. | Native versus batched attention can round differently. Benchmark mixed/ragged, long-context and image requests on all TP ranks. No new QSA kernel was invented or performance gain measured. |
| Prefill | Explicit `QwenPrefillPacedScheduler` inherits bounded Mamba retention. It schedules existing decoders first, bounds total mixed-step tokens, and adapts a 128-aligned prefill budget from observed host step latency. It restores the original config and FCFS order after each call. | A 500-ms target is a pacing estimate, not a hard deadline. Pure prefill keeps the original budget. Pending encoder prefills retain original limits so indivisible image spans cannot starve. The separate 94/85 controller is still required. |
| WY | Native FP32 forward substitution retains U/W on chip, fusing lower decay, triangular solve, products and final FP16 writes. Q/K layout, grouped Gram and cumulative gates remain torch operations. A callback seam leaves default math intact. | Summation order differs from blocked inversion. CPU checks cover both 4/12 and 16/48 heads plus multi-chunk FP32 recurrent-state propagation. Ascend H/O, long recurrence, full-model quality and speed remain untested. |
| MoE | One kernel gathers four packed token operands for the device-counted local prefix. A second reuses the existing SwiGLU quantizer unchanged, executing only local rows. Scoped resident bindings restore instance state on failure. | Buffers retain fixed full capacity. Grouped projection still zeroes inactive outputs; projection/finalization are not fused. The route candidate requires native INT4 plus `cann_swiglu_pack`, and must be compared with the same activation mode before comparing against the serving FP16 SwiGLU mode. |
| MTP | Deferred MTP0/1/2 profiles, sustained concurrent collection and an offline matched comparator. Recommendations require three repeats, quality/image receipts, complete thermal coverage and at least 600 seconds per concurrency arm. | Each depth requires cache replanning and graph capture. No automatic live switch or depth recommendation has been made. Client stream timing is approximate; wall throughput includes cooldown holds. |

The paced scheduler reuses upstream Mamba splitting and encoder admission;
it does not copy or replace upstream scheduler internals. Encoder-aware bypass
is conservative and can retain decoder stalls during image-prefill steps.
Existing decoder priority within each class is stable. Later steps retain the
original FCFS ordering and upstream request additions/removals.

The WY kernel uses 83,456 logical UB bytes per core. Its remaining grouped Gram
is read by each value-head task; the candidate trades external intermediate
writes and launch overhead for vector work and on-chip traffic. It may be slower
than the blocked reference. No FP32 recurrent state is narrowed.

For 25,600 routes at hidden width 2,560, four full packed-operand gathers write
93.75 MiB of logical payload. At 25% local routes the new gather writes
23.4375 MiB. This is source-level payload accounting, not measured DDR traffic,
allocator peak, or a thermal prediction. See `logical-bounds.json`.

## Validation

CPU receipts are in `offline-tests.log`. They execute the real native vector
bodies against a bounds-aware DMA/vector stub, compare WY with the blocked
reference, exercise recurrent-state propagation and padding, and prove inactive
route indices are never read. Events are no-ops in the stub and cannot establish
device correctness. Scheduler tests use a synchronous boundary stub; they
cannot establish complete engine behavior or image accuracy.

The focused suite has **360 passed, one optional `msmodelslim` skip**.
Scoped formatting and lint receipts are saved alongside it. The required
full-repository `format.sh ci` check retains unrelated baseline failures;
its compressed receipt is preserved without including formatter changes
to unrelated files.

Host-only CANN 9.1 compilation ran on `matteius@192.168.53.187` in a separate
experiment directory. Final native source fingerprints match
`host-build-provenance.json`. The resource namespace is `qwen_prefill_v2`.
The binaries and bridge were compiled, not loaded, registered on a worker,
or executed. See `host-build.log` and `host-build-provenance.json`.

No dummy serve, real-weight generation, image gate, performance benchmark,
capacity expansion or sustained thermal gate was run. Existing image support
is preserved by design, with a new image gate required for each candidate.
ACLGraph, EP and MTP combinations are pending; FlashComm1 remains outside these
candidate changes. No new environment variable or model-runner behavior was added.

## Deferred validation

Use `RUNBOOK.en.md` and `RUNBOOK.zh.md` after NPU access is authorized again.
`queued-profiles.json` contains partial overrides only and has `autostart=false`.
Gate each change independently against the same real checkpoint, cache settings,
activation mode, request bodies, output caps and cold/warm cache state before
combining them. Maintain the 94-C hold, 85-C resume and existing 96-C hard stop.

Temperature alone cannot identify redundant transfers. Compare sustained wall
throughput, decoder gaps during prefills, profiler copy/barrier critical paths,
peak allocation and observed thermal traces. Energy reporting remains absent
unless a real external measurement is supplied.
