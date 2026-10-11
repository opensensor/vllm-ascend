# Initial sidecard hardware gates

October 11, 2026, `matteius-threadripper`. The RTX PRO 6000 Blackwell and four
Ascend 310P3 chips were used after hardware authorization. The existing
GLM TP4 server, parent PID 276072 and four workers, remained resident. Probes
ran in `/srv/ai/storage/artifacts/sidecard-glm-hardware-20261011-v1/source`.

## Host-staged transfer integrity

[hardware-copy-probe.json](hardware-copy-probe.json) records 24 cases:
four chips, 1/4/640 rows, and one/eight returned rows per input. All use width
4096, FP16 input and FP32 output. Each case has two warmups and five measured
iterations. Every round trip matched expected bytes, including signed zero.
All worker processes exited normally. The earlier bounded trial is recorded
in [hardware-small-copy-probe.json](hardware-small-copy-probe.json).

| Input rows | Returned rows per input | Input / output bytes | Median validated round trip, range across chips |
| --- | --- | --- | --- |
| 1 | 1 | 8 KiB / 16 KiB | 0.411–0.479 ms |
| 1 | 8 | 8 KiB / 128 KiB | 0.673–0.751 ms |
| 4 | 1 | 32 KiB / 64 KiB | 0.542–0.604 ms |
| 4 | 8 | 32 KiB / 512 KiB | 1.446–1.878 ms |
| 640 | 1 | 5 MiB / 10 MiB | 39.083–48.201 ms |
| 640 | 8 | 5 MiB / 80 MiB | 512.798–531.844 ms |

These are synchronous diagnostic round trips, including host allocations,
socket serialization, checksums, NPU cast/repetition and extra CUDA copies for
verification. Each device operation is fenced. Host buffers are pageable.
Chips run sequentially in separate Ascend processes; a persistent CUDA
coordinator uses the installed runtime directly. No CUDA PyTorch package was
installed or server environment changed. The large return path needs more
work before being used for prefill; these results are not a PCIe bandwidth
measurement, concurrent-owner benchmark or inference speed forecast.

## Real native expert arithmetic

[hardware-native-expert-probe.json](hardware-native-expert-probe.json) records
expert 72 on chip 0 in layers 3/W4, 8/W3 and 33/W2. Native shard SHA256 values,
manifest/config/index identities and frozen kernel provenance are recorded.
The probe used the existing two-card native checkpoint, permanent Cube layout,
FP16 stored scales and A4 activation quantization. It loaded one expert at a
time without changing checkpoint files or model-runner code.

Each layer ran one-row decode and 17-row prefill cases, then changed the input
by a factor of 0.75. All 12 cases matched the CPU quantization/arithmetic
reference exactly: maximum absolute error and implementation NRMSE were zero.
Three synchronized repeat calls per case were bitwise equal. Median call times
were 0.90–1.71 ms; they include Python dispatch and device synchronization and
are single-expert measurements, not complete MoE or joined serving latency.

Top-k=1 and unit route weights produce one raw expert output per input row.
This suggests an adapter seam using one gathered row per route, but complete
owner-bank loading, arbitrary local expert maps, actual GLM activation inputs,
CUDA expert parity, weighted combination and full-model quality remain pending.
MTP and graph capture were not tested by these probes.

The first arithmetic launch failed because its added Python path replaced
CANN's sourced path, leaving `tbe` unavailable. The corrected launch preserved
that path and passed. The failed and successful logs are retained with the
other probe logs for reproducibility:
[initial launch](native-expert-probe.log),
[successful launch](native-expert-probe-v2.log), and
[copy worker logs](full-probe-logs/).

## Validation and device state

The CPU suite passed 149 tests, including framing/shape/checksum rejection,
CUDA copy completion ordering, allocation bounds, private buffers, shutdown,
native header admission and rejection of incompatible layouts/dtypes/paths.
Ruff check and format passed. Hardware gates supplement these tests; no full
repository NPU suite or joined GLM serving test was run.

[hardware-inventory-during-copy-probe.json](hardware-inventory-during-copy-probe.json)
and [hardware-inventory-after-probes.json](hardware-inventory-after-probes.json)
record the current four-chip mapping. All chips were healthy afterward; the
same GLM workers remained resident and CUDA had no remaining probe allocation.
No service, driver or original checkpoint was modified. Full sidecard memory
admission and expert backends remain unqualified.

[hardware-source-receipt.json](hardware-source-receipt.json) pins all 15 staged
Python files and verifies their hashes match the local implementation. It
also records the retained server process and empty CUDA compute-process list
after probe cleanup.

## Mixed-device expert packets and CUDA reduction

[hardware-split-expert.json](hardware-split-expert.json) records the next gate:
eight real GLM experts per tested layer execute across the RTX and all four
Ascend chips using the actual `SCARD001` expert request/reply protocol.
Layers 3/W4, 8/W3 and 33/W2 each run 1/4/17-token inputs and a changed-input
pass. Source checkpoint and CPU golden identities are pinned; all selected
native source shards are checked before uploading any bank.

| Owner | Selected actual global expert IDs |
| --- | --- |
| CUDA 0 | 0, 5 |
| Ascend chip 0 | 1, 6 |
| Ascend chip 1 | 2 |
| Ascend chip 2 | 3 |
| Ascend chip 3 | 4, 7 |

This fixture differs from the full 72/54 seed. Each worker loads its complete
fixture bank once and retains it across requests. CUDA uses an unpacked-INT8
correctness kernel with the native block32 activation-quantization contract;
the Ascend workers use the qualified permanent Cube/FP16-scale kernels. CUDA
and CANN execute in separate Python environments and processes. No CUDA Torch
installation, driver changes or modification of the running GLM server was
needed.

All 18 mixed-device cases matched the CPU reference bitwise for every owner's
raw rows and the final weighted output. The gate includes arbitrary local
expert maps, reordered route slots, retained zero-weight routes, shared output,
routed scale 2.5, changed activations, complete reply coverage and repeat calls.
The final reducer runs on CUDA after validating all replies. It uses original
route-slot order and explicit unfused FP32 operations. Six additional GPU
cases passed cancellation-order and separate-multiply/add checks exactly.

[hardware-split-expert-cpu-combine.json](hardware-split-expert-cpu-combine.json)
is the earlier control with CPU scalar reduction. Backend service totals
showed that expert device work was much shorter than the full scalar-control
round trip, which motivated moving final reduction to CUDA.

| Tokens | CPU scalar reduction control, round-trip range | CUDA reduction, round-trip range |
| --- | --- | --- |
| 1 | 36.66–39.51 ms | 19.58–21.91 ms |
| 4 | 126.22–132.65 ms | 59.62–70.56 ms |
| 17 | 503.16–519.47 ms | 224.94–245.62 ms |

These are individual correctness-gate timings across three layers and two
input variants, including tuple serialization, finite-value checks, copies,
expert execution, reply validation and reduction. They are not serving
benchmarks or a GLM speedup claim. Timed calls follow raw-output checks and
are warm; first-use runtime costs are not included in this table. Host packet handling and validation remain
substantial. Backend totals include host conversion and device execution,
but exclude wire handling and coordinator reduction; see the
[worker logs and configurations](run-v3/).

Across the three layer runs, CUDA backend service totaled 64.65–66.91 ms per
18 calls. Ascend first-use calls reached 237.81–534.50 ms; these cold costs
are visible in the service totals and remain a startup/warmup integration
concern. They must not be hidden when measuring cold TTFT for a joined model.

The bounded backends refuse full-model use: at most three layers, sixteen
experts per owner per layer and 256 routes. The CUDA reducer supports at most
32 tokens/eight slots. Captured GLM activation/router parity, all 288 experts,
full 42-layer resident-bank admission, CUDA attention/state, optimized packed
CUDA kernels, MTP, graphs and joined model quality remain pending.

The current CPU suite passed 164 tests. It additionally covers external
interpreters, startup failures, poisoned streams, complete pool shutdown,
noncontiguous global/local maps, oversized-bank refusal and FP32 packing.
All device workers exited normally and the CUDA process list was empty after
the coordinator exited. The existing server parent PID 276072 remained.
Ruff check/format, CUDA source formatting and markdown checks passed.

[hardware-split-source-receipt.json](hardware-split-source-receipt.json) pins
the current staged Python/CUDA sources, compiled CUDA library, compiler/build
command and per-owner service totals. Source hashes match the local code.
[split-golden-receipt.json](split-golden-receipt.json) identifies the reference
fixtures retained on the hardware host. The complete staged experiment is at
`/srv/ai/storage/artifacts/sidecard-glm-split-20261011-v3`; the authenticated
golden payloads remain in the v1 experiment directory. The
[post-split inventory](hardware-inventory-after-split.json) records healthy
chips and the retained Ascend server workers. No original checkpoint changed.

## Later joined service

The [service evidence](service/README.md) records the subsequent running
`glm-sidecard` deployment. CUDA holds 72 experts in every main routed layer;
both Ascend cards retain the backbone and remaining native expert partitions.
Seven operational cases passed, including chat, streaming, concurrency and
243-token chunked prefill. All 42 routed layers executed additional CUDA
routes during actual API validation. The 126 real-input expert comparisons
had maximum absolute error 1.1921e-7 and maximum NRMSE 6.8954e-8.

Whole-model validation remains false because greedy repeated completions
varied; the cause is not isolated. The new service is an experimental eager
profile, not a speedup or quality qualification. The earlier server PID
276072 was replaced after complete CUDA preparation; the current API parent
is PID 864265. All original probe reports above describe the earlier device
state. The source checkpoint and driver stack remained unchanged.
