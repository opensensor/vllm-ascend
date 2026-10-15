# Local NPU collective qualification

The new library supports ordered local NPU groups of arbitrary size in its
schedule and owner API. Binomial reduction/broadcast handles non-power-of-two
sizes; recursive doubling handles power-of-two groups. Hardware qualification
covers two, three and four ranks on two Ascend 310P Duos. Larger groups remain
untested. The Torch adapter now supports optional graph execution with native ownership
leases and no Python replay tracking; the original eager comparison is retained below.

## Raw SDK comparison

Medians below are completed batches measured with shared-host CLOCK_MONOTONIC,
including host dispatch and rank arrival spread. Each comparison used seven
alternating pairs of 30 calls, FP16/FP32 SUM, matched geometry and a raw HCCL
reference. These are not pure wire latency or model token rates. Full samples
and configurations accompany every result; no confidence interval was computed.

| Ranks | Elements | Dtype | Algorithm | Custom µs | HCCL µs |
| --- | --- | --- | --- | ---: | ---: |
| 2 | 2560 | float16 | recursive-doubling | 31.39 | 66.18 |
| 3 | 2560 | float32 | binomial | 109.91 | 206.39 |
| 3 | 2560 | float16 | binomial | 107.67 | 190.73 |
| 4 | 2560 | float16 | recursive-doubling | 80.30 | 142.09 |
| 4 | 2560 | float32 | recursive-doubling | 82.86 | 146.87 |
| 4 | 7680 | float16 | recursive-doubling | 80.34 | 143.44 |
| 4 | 7680 | float32 | recursive-doubling | 79.64 | 124.88 |
| 4 | 15360 | float16 | recursive-doubling | 87.35 | 117.58 |
| 4 | 15360 | float32 | recursive-doubling | 84.03 | 165.11 |
| 4 | 81920 | float16 | recursive-doubling | 96.67 | 174.17 |
| 4 | 81920 | float32 | recursive-doubling | 122.38 | 307.07 |

The initial staged binomial two-rank implementation regressed at 2,560 elements
(about 84–86 µs versus HCCL 62–63 µs). The retained candidate removes standalone
staging copies and uses recursive doubling for power-of-two groups. Both the
rejected and retained candidate receipts are included.

## Queued Torch gate

All 14 registered geometries passed on all four ranks: 2,560 / 5,120 / 7,680 /
10,240 / 15,360 / 23,040 / 30,720 elements in FP16 and FP32. Each had complete
CPU/HCCL byte comparisons, 30 distinct live input/output tensors submitted
without an intervening fence, and explicit all-rank shutdown. No fallback calls
occurred. All raw reports are in [the Torch receipt](torch-result.json).

These timings include Torch output allocation/queueing and a clone plus HCCL
for the baseline, matching out-of-place semantics. QFN’s existing HCCL communicator
reduces in place, so this microbenchmark includes an extra baseline clone that
its real-model path does not require. They are synthetic completed
batches, not QFN token throughput.

| Elements | Dtype | Custom µs | Torch HCCL µs |
| --- | --- | ---: | ---: |
| 2560 | float16 | 77.48 | 180.69 |
| 2560 | float32 | 83.98 | 182.47 |
| 7680 | float16 | 89.07 | 192.12 |
| 7680 | float32 | 88.03 | 193.66 |
| 30720 | float16 | 86.01 | 199.14 |
| 30720 | float32 | 92.81 | 192.62 |

## Real QFN TP4 / EP4 comparison

Both runs completed with the same qualified 169.19-GiB W4 checkpoint, coherent
OPP packages, 262,144-token configured context, MTP2, four worker CPU-affinity
sets and eager execution. Only custom SUM configuration differs in the saved
engine options. The private runtime changes are restricted to the communicator
and recognized additional-config routing; model Python sources are unchanged.

| Concurrency | HCCL median tok/s | Custom median tok/s | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 18.03 | 19.03 | 1.056× |
| 4 | 53.45 aggregate | 58.52 aggregate | 1.095× |

Each median has only two timed trials: C1 generated 256 tokens per request;
C4 generated 128 tokens per request. A four-prompt, 32-token warmup preceded
both arms. Rates include prompt processing and scheduling. All prompts were
short, and reported cached-token counts were zero in every request; the
harness's “prefix-warmed” scope string denotes the warmup attempt, not a cache
hit. This did not validate 262K input execution or long-context performance.

Per-trial receipts prove positive custom call deltas on every rank and retain
fallback geometry counters. Unregistered prefill shapes and non-SUM collectives
continue to use HCCL. All four custom owners reached `closed` before engine
shutdown, and both worker trees exited gracefully. Final device health was
captured separately.

The saved outputs were coherent on these explanatory prompts, but token
sequences differed even between repeated HCCL baseline trials. This is a
preliminary throughput comparison, not bit-for-bit model parity or a quality
benchmark. No confidence interval was computed; MTP acceptance and generated
trajectories can also affect timing. The earlier graph-enabled QFN measurements
are a different configuration; this eager candidate is not a graph replacement.

See [matched comparison](qfn-comparison.json),
[HCCL outputs](qfn-hccl-result.json), [custom outputs](qfn-custom-result.json),
[baseline options](qfn-hccl-options.json), [custom options](qfn-custom-options.json),
and [runtime overlay](runtime-overlay-v2.json).

The first custom attempt was rejected by strict additional-config validation
before worker creation. A narrow recognized pass-through plus an early
dict-or-None guard fixed that integration issue; unknown-key rejection remains
in place. Eighteen regression tests cover this path.

Portable validation: **5,223 tests passed** in the sidecard unit-test suite;
scoped formatting, lint and secret checks passed. Native libraries compiled
against the installed CANN/Torch ABI and passed the device gates above.

## Validation and scope

Every raw run compared complete output bytes with CPU and HCCL on three exactly
representable finite patterns, then retained and checked 90 changed-input calls
per rank with distinct device input/output addresses. All ranks completed the
explicit drain, peer-import closure and resource release protocol. These benign
patterns establish transport correctness, not broad FP16 numerical equivalence
with a model or different reduction orders.

Each exchange writes its local tensor into the peer's imported inbox with SDMA,
records peer READY, waits/resets local READY, then adds **local** input and inbox.
Two alternating inbox/notification slots support reuse. No remote vector reads
are admitted. No CPU payload staging occurs in this path; setup/shutdown use
host coordination. The queued Torch and short-prompt QFN integration gates completed.

Only explicitly registered contiguous FP16/FP32 SUM geometries use the custom
Torch path. Other geometries and other collectives remain on HCCL. This is not
a complete replacement for HCCL. The original eager facade rejects graph capture;
the opt-in graph extension is described below.

The live launcher, qualified QFN weights, cache shape and operator packages are
unchanged. A private copy of the qualified runtime receives only the experimental
communicator overlay plus owner/probe tooling. Short-prompt QFN throughput is recorded above. Broad model quality and
long-context qualification remain open.

## Graph follow-up: native ownership and unobserved replay

The fixed-address recursive-doubling variant adds a device CONSUMED handshake
following each local reduction. Captured calls reuse inbox zero without a host
slot counter. Graph lifetimes use native C++ leases acquired before capture;
plan destruction fails while a lease exists. Only after device drain and every
native graph reset does the cold lifecycle registry release leases and permit
collective teardown. The C ABI relies on the caller's reset-and-drain proof.

Production graph entries cache Torch-NPU's bound native replay method. They
invoke no Python observer callbacks or counters during successful replay.
Diagnostic replay tracking remains an explicit opt-in. Scheduling and attention
metadata updates still use the existing vLLM runtime.

The breakable QFN wrapper now registers ownership before its first native
segment starts. A single segment binds directly to native replay; multiple
segments retain upstream's graph/eager callable sequence. This removes
collective replay observation, not all Python model scheduling. Native C++
implements the collective plan and ownership guard through a C ABI; the SDK
executes the captured DMA, notification and math tasks.

The unobserved N4 hardware gate passed 14 tensor geometries, with captures of
one and three independent SUMs, five changed-input replays per graph, eager
interleaving and seven alternating timed batches of 30 replays. All 28 graphs
reset successfully and all 392 native leases per rank released before plan and
peer-resource destruction. Full exact-byte synthetic checks passed against CPU
and HCCL. These finite-pattern checks do not establish model numerical parity.

| FP16 elements | Custom µs/SUM | HCCL µs/SUM |
| --- | ---: | ---: |
| 2560 | 15.15 | 20.88 |
| 7680 | 17.86 | 26.33 |
| 15360 | 21.86 | 35.51 |
| 30720 | 29.22 | 49.74 |

These are medians for three independent SUMs per captured graph, including
host launch/arrival spread and completed batches. HCCL includes a captured
clone to match out-of-place semantics. The model's existing HCCL path is in-place.
The previous tracked N4 run measured 27.86 µs for the 2560-element FP16 case;
removing observer work reduced the measured host contribution. Runs were
sequential, so this is not an isolated paired estimate of observer cost.

See [unobserved graph results](graphs/n4-unobserved-result.json),
[earlier diagnostic graph results](graphs/n4-result.json), and
[native build receipt](graphs/native-build.json).

The [final recapture gate](graphs/n4-unobserved-recapture-result.json) also
passed. After all 28 captures, every older custom and HCCL graph replayed in
reverse then forward order. Outputs were cleared before replay, and complete
output and unchanged-input bytes were checked. All four ranks retired every
graph and released all 392 leases. Its three-SUM FP16 7680-element median was
18.12 µs custom versus 26.49 µs HCCL. Portable suite: **5,804 passed**.

An earlier attempt at this final sweep stopped at the buffer-clear operation
because the probe launcher omitted the SDK Python path needed by the CANN compiler.
All four ranks failed before final replay, following synchronized timed work.
Only those verified paused processes were terminated; no host/device reset
was used. The corrected environment completed the fresh run linked above.

## QFN graph integration

With the `[3, 6]` capture profile, single-request generation measured **31.09
tok/s custom versus 30.33 tok/s HCCL**, a 2.53% increase across two 512-token
trials. Four concurrent requests exceeded this profile's capture sizes and
mostly used eager decode: 57.79 versus 59.08 aggregate tok/s. The eager warning
applies to those larger batches, not the single-request graph comparison.
MTP uses three query tokens per active request, so four requests need a
12-token capture. The separate `[9, 12]` profile measures that workload.

All four custom workers retired four graph captures and released all 56 native
leases per rank. Production replay observation was disabled throughout; replay
counts are explicitly unobserved. Model outputs are retained for inspection,
but these short explanatory prompts do not establish general model quality.
The original HCCL path is in-place, unlike the cloned HCCL microbenchmark.

See [matched C1 comparison](graphs/qfn-c1-comparison.json),
[HCCL results](graphs/qfn-hccl-c1-result.json),
[custom results](graphs/qfn-custom-c1-result.json), and
[qualified runtime overlay](graphs/runtime-overlay.json). Model weights,
operator packages, cache settings and MTP options were preserved.

The `[9, 12]` profile completed without eager-fallback warnings. Four concurrent
requests measured **62.09 aggregate tok/s custom versus 58.50 HCCL**, a 6.14%
increase across two trials of 128 output tokens per request. Single-request
generation padded to the larger capture and measured 20.09 versus 20.10 tok/s;
larger captures therefore should not replace the smaller profile for that
workload. Only two sizes fit the current qualified event budget.

| Profile | Requests | HCCL tok/s | Custom tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| `[3, 6]` | 1 | 30.33 | 31.09 | +2.53% |
| `[3, 6]` (mostly eager) | 4 | 59.08 | 57.79 | -2.18% |
| `[9, 12]` (padded) | 1 | 20.10 | 20.09 | -0.06% |
| `[9, 12]` | 4 | 58.50 | 62.09 | +6.14% |

Rates include prefill and completed generation on short prompts. Each row is
the median of two trials, without a confidence interval. Different generated
trajectories and MTP acceptance can affect timing; the results do not isolate
individual operator costs. Both profiles preserved the 262144 configured
context limit; this was not a long-context capacity test.

See [matched C4 comparison](graphs/qfn-c4-comparison.json),
[custom C4 outputs](graphs/qfn-custom-c4-result.json), and
[capture/fallback log evidence](graphs/graph-dispatch-evidence.json).
All four workers again reset four captures and released 56 native leases each.
Final device health was OK on all four NPUs, with no NPU processes remaining.
The benchmarks were offline; no serving endpoint was launched.

## Fresh single-request profile and 40 tok/s target

The fresh custom-path run preserved `[3, 6]` graphs, MTP2, four ranks, the
qualified operators/precision and 262144 configured context. Three unprofiled
512-token completions measured **31.11, 31.23 and 31.92 tok/s** (median 31.23).
Two trials preceded profiling and one followed it. A separate request recorded
eight decode rounds on all four NPUs after an eight-step delay; another
128-token request collected worker CPU call profiles. Profiled rates are not
throughput claims. All graphs/leases retired and all workers exited cleanly.

The request-level timings, all-rank NPU timelines and worker CPU profiles give
the following picture. Frontend/engine-core Python time was not separately
sampled. Values below are medians across ranks where appropriate.

| Observation | Measured value | Interpretation |
| --- | ---: | --- |
| Profiled MTP round span | 85.57 ms | Target verification plus two draft passes |
| Recorded compute/copy union | 63.61 ms/round | Overlapping tasks counted once |
| No recorded active task | 22.09 ms/round | Includes dependency waits; not all recoverable |
| Generic dense FP16 matmul | 20.58 summed ms/round | 752 tasks per round; largest named kernel family |
| Cast + layout conversion | 9.35 summed ms/round | Many small tasks throughout the model |
| Routed W4 gate/up + down | 7.22 summed ms/round | Already-specialized expert projections |
| HCCL all-gather | 0.53 summed ms/round | Three calls per round; not the dominant cost |

The custom transport is not counted under HCCL names. It appears as SDMA,
notification/event and Add tasks. This profile does not isolate its entire
critical-path latency. Event-wait durations overlap work and other ranks;
summing them would produce a misleading communication figure. Likewise,
kernel sums and host inclusive times must not be added together.

Cold graph inventory found **14 native segments and 13 eager breaks** in each
target capture: one PLE callback and twelve QSA group-selection callbacks.
Each draft capture has **four segments and three breaks**: causal-position
copy, QSA selection and expert execution. With MTP2 that is 22 native graph
launches and 19 eager callbacks per complete decode round. The trace recorded
exactly 176 native graph launches over eight rounds on every rank.

In the separate 128-token CPU profile, median inclusive callback times were
approximately **328 ms draft experts, 279 ms QSA selection, 152 ms draft
position copying, and 81 ms PLE**. These include device waits. The largest
apparent CPU cost was rejection sampling's `.cpu()` wait for previously queued
device work; replacing that copy alone cannot remove the preceding compute.
Matrix shape attribution also identifies the hyperconnection's two low-rank
projections as a large part of the dense-matmul family, alongside attention and
shared-expert projections.

Priorities supported by this profile:

1. Reduce QSA graph fragmentation and repeated host dispatch. Move dynamic
   selection/metadata preparation into device execution where possible, while
   preserving sparse selection order, causal boundaries and long-context
   behavior. Collapsing segments must preserve the existing mutable metadata
   updates; removing callbacks without a replacement would be incorrect.
2. Make draft expert routing/execution graphable with the existing W8A16 math.
   The current eager callback includes route construction and per-expert
   dispatch. A grouped path with different activation quantization would need
   a separate quality comparison and cannot be claimed as the same-math fix.
3. Optimize small FP16 projections and intermediate layouts, starting with
   the repeated hyperconnection shapes. Target actual serving time as well as
   isolated kernels; earlier component improvements did not reliably increase
   model decode throughput.

Reaching 40 from 31.23 tok/s requires about **22% less time per token** at the
same accepted tokens per MTP round. Relative to an illustrative 85.6-ms round,
that is roughly 19 ms. There is enough work in the measured gaps and small
operators to justify pursuing this target, but the profile does not prove that
those savings are achievable or additive. Forty tok/s has not been reached.

One narrow candidate was tested: copy draft causal positions directly from
pinned host memory into the existing graph buffer, avoiding a temporary device
tensor. The changing-input hardware gate passed 120 copies, including negative
padding and the configured context boundary. Three unprofiled model trials
measured **32.07, 30.41 and 31.42 tok/s**. This overlaps the control range and
does not demonstrate a gain. At the user's request, the candidate is
**promoted as the custom-path research baseline**, with throughput classified
as neutral (median **31.42 tok/s**). The selected remote launcher is
`run-qfn-custom-current.sh`, pointing to `run-qfn-custom-v5.sh` and the exact
hardware-tested `qfn-pinned-position-runtime`. The previous runtime remains
available. See the [promotion receipt](profile-c1/promotion.json) for paths and
hashes. Launches require a fresh collective owner directory and the documented
worker CPU affinity before timing.

The same direct pinned-host copy was backported into the local model callback,
with a CPU-device guard for reference tests. All 14 causal-position tests passed,
including changing metadata, a stable destination buffer, negative padding and
the context boundary. The sidecard suite passed 5,814 tests. Hardware validation
covers the frozen remote runtime; these checks do not qualify all unrelated
changes in the local working tree. Both baseline and candidate outputs vary
between repeats; these short trials do not establish broad model quality.

See the [profile PNG](profile-c1/profile.png),
[per-rank analysis](profile-c1/analysis.json),
[request and graph receipts](profile-c1/result.json),
[candidate comparison](profile-c1/pinned-comparison.json), and
[raw capture provenance](profile-c1/provenance.json).
The reusable harness is `tools/sidecard/qfn_collective_profile.py`; use the
qualified launcher environment and a fresh collective owner directory.
`--throughput-only` excludes both diagnostic captures. Standalone CANN parsing
was run after workers exited because vLLM daemon workers cannot spawn the
profiler parser. The local analysis/plot scripts reproduce the saved report
from the preserved parsed bundle. All four NPUs finished healthy and idle.
