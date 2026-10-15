# QSA four/eight-block staging investigation

The immutable `qfn-qsa-gather-staging-compile-20261014-v1/build-r2` bundle
contains separate four- and eight-block entries. Both use the frozen pinned
gather implementation; only the staging constant differs between them.
The 310P ACLRTC wrapper uses `KERNEL_TYPE_AICORE`. The first CPU build rejected
`KERNEL_TYPE_AIV_ONLY` before any device submission. Existing profile records
also identify the installed gather as AI_CORE, so source-level labels alone
do not establish a runtime core mismatch.

## Hardware correctness

All four resident workers passed 432 full physical-byte comparisons each:
query counts 1/3/64, one/two KV heads, table widths 2048/4096, both K/V
orientations, nine changing count/tail configurations, and both native widths.
Input bytes and reused output storage were preserved. Each worker also passed
eight unchanged-attention comparisons, including 95 queries split into 65/30
requests and serial/parallel K/V streams. Descriptor creation events preserve
cross-stream readiness. No serving worker or model weight storage changed.

Receipts: `qsa-resident-v1/gather-validate-{1,3,64}-r2-v1.json`.
All devices remained healthy at 61–66°C after the largest gate.

## Initial timing and unresolved regression

Three-repeat private timing at 64 queries, one KV head, and full tables gave
the following ranges of per-rank median queued-device intervals:

| Gather case | Installed | Native four blocks | Native eight blocks |
| --- | ---: | ---: | ---: |
| Dense K | 0.623–0.638 ms | 0.606–0.618 ms | 0.571–0.596 ms |
| Dense V | 0.430–0.434 ms | 0.476–0.485 ms | 0.487–0.493 ms |
| Sparse K | 0.724–0.739 ms | 0.763–0.784 ms | 0.657–0.685 ms |
| Sparse V | 0.547–0.558 ms | 0.591–0.612 ms | 0.524–0.538 ms |

These event intervals include device idle time between the starting event and
host submission. They are not isolated kernel durations. The native four-block
control's V regression against the installed four-block operator is a reason
to investigate implementation and launch overhead before blaming larger staging.
The matched private profile separates actual kernel intervals from submission
gaps. At this initial stage, end-to-end prefill had not yet been measured.

The subsequently qualified model binding uses eight-block K only while retaining installed V.
This does not close the V investigation. Full exactness receipts remain available
for both orientations. Native owners and frozen builds are retained throughout.

Timing receipt: `qsa-resident-v1/gather-timing-64-r2-v1.json`.
Build provenance: [gather-staging-build-r2.json](gather-staging-build-r2.json).

## Isolated trace and wrapper correction

The private trace resolves the initial apparent V regression. Dense V native8
kernel execution was 369 microseconds versus 372 installed, but its pre-kernel
gap was about 56 microseconds longer. Sparse V execution improved from 495 to
419 microseconds, with an additional 71-microsecond pre-kernel gap. The native
kernel was not the cause of the queued-event regression.

CPU span attribution places most extra time before native dispatch: admission,
metadata preparation, and descriptor handling. The model binding also repeated
admission inside the public wrapper. The corrected binding admits once, calls
an internal admitted launch, and caches descriptor block counts. It preserves
stream events, producer-thread queue draining, same-device native argument
checks, and all native owners. A paused wrapper upgrade keeps the previous
Python wrapper and transfers existing descriptor/event references.

The corrected wrapper repeated all 432 physical-byte checks and eight complete
attention checks on every rank. The `timing_admitted` diagnostic distinguishes
an intentionally hoisted-admission ceiling from the actual model binding;
ceiling timings must not be reported as serving performance. Dense V still
needs further host-path improvement before enabling its new kernel in serving.

Evidence:

- [Detailed trace attribution](qsa-resident-v1/private-gather-trace-v1/rank0/analysis.md)
- `qsa-resident-v1/gather-fast-validate-{1,3,64}-v1.json`
- `qsa-resident-v1/gather-fast-timing-64-v1.json`

The initial K-only model candidate passed all eight smoke cases. Its first
uncached 8K request took 19.608 seconds, 417.79 prompt tokens/s. A subsequent HC
control took 20.025 seconds, 409.09 prompt tokens/s. These single-request values
are preliminary; the final combined candidate is measured separately.

The [final combined comparison](qsa-prefill-decode-results.md) measured 414.81
prompt tok/s median versus a fresh 415.00 control. The initial apparent prefill
advantage was not reproduced; the exact, effectively neutral candidate is retained.
