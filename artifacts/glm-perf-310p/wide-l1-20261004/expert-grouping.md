# Expert grouping experiments

These experiments preserve model routes, packed weights, scales, and output
placement. The user first requested offline staging, then authorized NPU use.
No new environment variable, routing approximation, or host route-count
synchronization was introduced.

## Concurrent expert teams

The previous kernel assigned all launched cores to each active expert, then
advanced every core through the expert list. The new optional compile flag
`GLM_W2_GROUPED_EXPERT_LANES=2` or `=4` divides cores into teams. Each team
visits a disjoint subset of active experts, striping that expert's output
tiles across its lanes. Empty experts do not consume a team assignment.

This reduces the number of expert transitions and full pipeline drains per
core when many experts are active. Each team still reuses a decoded weight
tile across that expert's rows. The physical core number continues to index
private GM scratch; replacing it with the team lane would alias workspaces
between concurrently executing teams.

Calls with fewer than 256 total route rows retain the all-core schedule.
When the requested team width cannot divide the launched core count, the
kernel also falls back to all cores. This is not a load-balancing guarantee:
few active experts or highly skewed counts can leave teams idle. The benchmark
must include concentrated routing before any serving promotion.

Three packages compiled from identical sources isolate the schedule:

- `opp-l1-teams-control-v2-20261004`: resident W2/W3/W4, scale cache, large
  groups, with the default all-core ownership.
- `opp-l1-teams2-v2-20261004`: the same options with two cores per expert.
- `opp-l1-teams4-v2-20261004`: the same options with four cores per expert.

All are under `/srv/ai/src/glm-l1-wide-build-20261004/`. The immutable
pre-scheduler `opp-l1-all-resident-v2-20261004` and serving overlap package
are retained as additional controls. The build script saves exact sources,
compile flags, and binary/metadata hashes in each experiment directory.

## Grouping across token chunks

`tools/glm_perf/benchmark_expert_batching_310.py` compares one grouped call
with 64-, 128-, and 256-token chunks over the same 640-token, top-8 workload.
It generates distinct top-8 choices among 288 experts and uses one rank's 72
local experts. Uniform and concentrated top-8 routing are separate cases.

The same packed bank and scales are shared by every comparison. Each chunk
sorts local routes by expert, preserves peer-owned rows, and keeps each token's
top-8 routes together. Reordering outputs back to original route order must
be bitwise equal to the single-call output before timing starts. Measurement
orders alternate across repeats.

Only grouped projection calls are timed. Routing, activation gathers, and
output reordering are outside the timed interval. This measures the cost of
repeating expert work across chunks, not complete MoE or serving throughput.
By default it stays within the retained OPP's 5120-route limit. The explicit
`--max-routes` option permits testing an enlarged package. Raising the server's
prefill size still requires a separate memory and admission-limit experiment.

## Verification

The CPU suite compiles the actual team planner and checks exactly-once tile
ownership across 1..8 cores, team-width requests 0..16, small/large calls,
1..128 output tiles, and empty, sparse, dense, and skewed expert populations.
Chunk-plan tests verify expert boundaries, peer rows, inversion to original
route order, and invalid inputs. Along with the existing route-cap tests,
14 CPU tests passed.

Hardware parity includes W2/W3/W4, canonical and NZ packing, byte/K-tile
boundaries, peer rows, empty experts, 128/129/257-row boundaries, and a mixed
case wide enough to launch all eight cores and activate team scheduling.
`run-expert-team-study.sh` runs these checks and paired projection timings in
separate processes. All four resident/team packages passed 38/38 hardware
checks, and each passed bitwise comparison with the overlap build on the
realistic W4 gate/up and W2 down measurements. The enlarged-route W3 package
also passed the same 38 hardware checks.

## Measured results

### Expert teams: small gains, concentrated-routing regression

The 5120-route case assigns one quarter of its routes evenly across 72 local
experts. Seven timed samples followed two warmups, on device 0, in separate
processes using identical weights, scales, activations, and expert counts.

| Schedule | W4 gate/up | W2 down |
| --- | ---: | ---: |
| Existing overlap | 38.81 ms | 24.62 ms |
| Resident/team control | 38.40 ms | 24.73 ms |
| Two-core teams | 38.19 ms | 24.32 ms |
| Four-core teams | 38.36 ms | 24.54 ms |

This is approximately 0–2% against the matched team control. On a concentrated
640-row single-expert W3 case, two-core teams took 47.67 ms versus 12.29 ms
for the resident control. Fewer cores were available to the only active
expert. These schedules remain experiments, with no serving promotion.

The wide resident W3 plus scale-cache path improved small-group W3 gate/up
from 9.07 to 8.21 ms in the eight-expert prefill probe. The full-K narrow-N
resident schedule for large groups regressed from 4.21 to 12.26 ms. Therefore
the larger-batch package below uses the W3-wide-only package and retains GM
fallback for groups above 128 rows; it does not enable large-group residency,
scale caching, or expert teams.

Records, including source/binary identities and parity checks, are in
`grouping-results/{all-resident,teams-control,teams2,teams4}/`.

### Larger expert batches: substantial isolated projection improvement

`GLM_W2_GROUPED_MAX_ROUTES=20480` is an opt-in host compiler definition.
The main-source default remains 6144. The dedicated
`opp-l1-w3-batch2560-20261004` package overlays the enlarged tiling library
onto `opp-l1-w3-v2-20261004`, leaving its kernel binaries unchanged.
`build-batch-cap-candidate.sh` reproduces the host build and restores the
previous compiler flags afterward.

Each comparison uses exactly the same 2560 tokens, top-8 route selections,
weights, and scales. Uniform routing produced 5122 local routes among 20480
total assignments. The hot case selects the same eight local experts for
every token, with 20480 local routes. Seven timing orders alternate between
ascending and descending chunk size; all output comparisons passed bitwise.

| Projection / routes | Four 640-token calls | Two 1280-token calls | One 2560-token call | Speedup |
| --- | ---: | ---: | ---: | ---: |
| W3 gate/up, uniform | 292.26 ms | 158.64 ms | 91.66 ms | 3.19x |
| W4 gate/up, uniform | 153.88 ms | 89.10 ms | 56.49 ms | 2.72x |
| W2 down, uniform | 97.88 ms | 55.50 ms | 33.96 ms | 2.88x |
| W3 gate/up, hot | 132.18 ms | 120.43 ms | 113.00 ms | 1.17x |

These are total projection times for the same workload, not time per call.
They exclude routing/gathers, KDA, attention, inter-rank collectives, and the
rest of the model. They establish that repeating expert work across chunks
is expensive in this workload; they do not establish a serving TTFT speedup.
All raw samples and package identity are in `grouping-results/batch2560/`;
`grouping-results/summary.json` collects both studies.

## Prepared serving plumbing

Main now accepts the explicit HF override
`"ascend_glm_grouped_max_routes": 20480`. It propagates to each local expert
bank and controls both grouped-call admission and overflow splitting. With
this setting a 2560-token top-8 batch stays in one grouped call. Without it,
the existing limit and splitting behavior remain unchanged. It must be
paired with the enlarged OPP. The kernel cannot infer or negotiate this host
configuration from checkpoint weights.

Bank validation rejects invalid types, nonpositive values, and values above
32768. CPU tests cover bank propagation, 1280/2560-token batches, overflow at
2561 tokens, unchanged weighted outputs, and one shared-expert invocation
per input batch. The complete focused CPU run passed 66 tests; targeted Ruff,
shell syntax, and whitespace checks also passed.

The controlled serving launcher accepts this override as argument 16; omitted
values preserve its previous behavior. `run-serving-variant.sh batch1280`
pairs the enlarged OPP and override with 1280-token prefill. This requires
staging the two Python changes recorded in
`grouping-results/runtime-route-cap.patch` before starting the candidate.

The full-model comparison starts from a fresh overlap server at 640 tokens,
then a fresh candidate at 1280, both on port 8001 with the same checkpoint,
196608 configured context, graph captures, prefix caching, and memory settings.
Each receives the same cold 7269-token retrieval before the quality/short
suite. The fresh control answered correctly with 77.687 s TTFT. Larger KDA
and attention workspace and concurrent-request behavior require this serving
gate; context capacity must not be inferred from the projection-only test.

The initial 1280-token launch with the control's 0.70 cache fraction failed
the startup admission check: 7.40 GiB required versus 5.69 GiB available.
It did not reach inference and did not raise a device allocation OOM. The
control reported 5.96 GiB available. The large increase in required cache
comes from the shared block pool reserving the larger in-flight batch,
including the short-window compressor-state group. The retry
`batch1280-kv94` allocates 0.94 of profiled headroom to cache, retaining 192K
context and all other candidate settings. This changes the memory margin;
device sampling and a successful real request are required before retaining it.
The failed startup log is `grouping-results/batch1280-kv70-startup.log`.

The 0.94 retry also failed admission. Rank 0 logged 7.65 GiB, but a smaller
TP device had only 7.16 GiB. The first retry calculation incorrectly used
rank 0's budget. `batch1280-kv100` uses the complete profiled headroom, which
projects to 7.62 GiB on the limiting rank, while retaining the overall 0.965
memory-utilization setting. This removes the additional workspace reserve
from the fraction setting; it needs actual peak-memory validation. The
second failure is saved as `grouping-results/batch1280-kv94-startup.log`.

## Serving result and NPU release

The third launch passed admission and graph capture. Its reported capacity
was 202366 tokens (1.03 times the configured 196608-token request). The cold
retrieval computed all 7269 prompt tokens with zero cached tokens and returned
the exact expected answer. Both short suites completed five 256-token replies.

| Metric | Overlap / 640 | W3 resident / 1280 |
| --- | ---: | ---: |
| Cold retrieval TTFT | 77.687 s | 70.236 s |
| Short c1 decode | 3.962 tok/s | 4.052 tok/s |
| Short c4 aggregate decode | 10.848 tok/s | 7.964 tok/s |

The 9.6% prefill reduction is an actual serving result. C4 regressed 26.6%,
so this package is not an overall serving promotion. These are single suite
passes, not repeated statistical estimates. The full quality suite was
repeated on the control unnecessarily; it reproduced the same 17/20 result.
The candidate received retrieval and short tests, not another full quality
gate. Sampled peak usage left 435 MiB on the tightest device; full context and
long concurrent requests were not validated.

The candidate server (PID 2189347) and four workers were stopped following
the user's instruction to use the NPUs for another experiment. New work below
is code, CPU testing, and compilation only. No later NPU execution occurred.

## Next offline batch

### Separate resident W3 prefill from decode

`GLM_W2_GROUPED_L1_W3_PREFILL_ONLY` retains the previous W3 schedule when a
grouped call has at most 32 total routes, covering the current top-8 graph
captures for one and four decode tokens. W2/W4 retain their overlap schedule.
Larger calls use the resident W3 candidate. This isolates a suspected cause
of the C4 regression; recovery has not yet been measured.

The canonical and NZ kernels compiled successfully. CPU checks compile the
actual dispatch helper and test both sides of the 32-route boundary. Packages:

- `/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-w3-prefill-v2-20261004`
- `/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-w3-prefill-batch2560-20261004`

The latter overlays the already-built 20480-route host tiling library. Binary
and source hashes are saved in `grouping-results/prefill-only-*.json`.

### Fuse inverse routing, weighting, and summation in FP32

The opt-in HF override `ascend_glm_fp32_route_combine: true` selects
`npu_w2_route_combine_310`. Its inputs are sorted FP16 expert outputs,
INT64 inverse-route indices, original-order FP32 top-k weights, and INT64
local-group ends. It returns FP32 `[tokens, hidden]`; shared-expert addition
and the surrounding collectives retain their existing positions.

Each core owns disjoint token/channel tiles. It reads each contributing FP16
row, weights and accumulates it in FP32 on chip, and writes one output tile.
No atomics, host count readback, full FP32 route tensor, or reordered FP32
route tensor are needed. The two removed FP32 temporaries total 320 MiB at
1280 tokens, top-8, hidden 4096 (640 MiB at 2560). These are tensor-size
calculations; actual peak-memory and speed reductions remain unmeasured.

The kernel checks local route boundaries and zero weights before reading
rows, so grouped projections can skip peer-row initialization. Tests poison
peer rows with NaNs, including nonzero weights on peer-owned routes. This
path does not introduce the FP16 output rounding of the earlier CANN combine
trial. Its FP32 summation order can differ from `torch.sum`; hardware tests
compare against an FP64 oracle with an FP32 error bound. It cannot be enabled
together with `ascend_glm_fused_route_combine`.

The kernel, host tiling library, and op API library compiled with CANN 9.1;
the PyTorch adapter also passed C++ syntax checking. The build is
`/srv/ai/src/glm-l1-wide-build-20261004/build-route-combine-20261004`.
It has not been installed into the serving runtime. Full extension rebuild,
OPP packaging, and hardware tests are still required before enabling it.

The focused CPU suite passed 56 tests. Deferred hardware coverage is in
`tests/e2e/nightly/310p/single_node/ops/test_w2_route_combine_310.py`: precision,
empty/partial/all-local groups, channel tails, argument rejection, and graph
replay with changed routing. The alternating-order latency and peak-allocation
benchmark is `tools/glm_perf/benchmark_route_combine_310.py`.

Next authorized device pass: test the new combine operator directly, measure
the projection dispatch on C4 route shapes, then test a full-model candidate.
Use the existing serving results as the control; do not rerun a known baseline
suite merely to recreate it.

### Explicit x86 CPU affinity

Neither serving run pinned CPU workers. Both saved logs report
`CPU binding skipped: non-ARM CPU detected.` The launcher sets
`OMP_NUM_THREADS=1`, which limits thread count but does not set affinity.
The current host exposes 32 physical cores / 64 logical CPUs in one NUMA
node; SMT siblings are `i,i+32`. Both NPU PCI devices report NUMA node `-1`,
so the available sysfs data does not establish per-card CPU locality.

`tools/glm_perf/worker_affinity.py` stages a separate affinity experiment.
It defaults to a read-only plan, accepts explicit per-process CPU lists,
checks that PIDs belong to the selected server tree, and applies masks to
every existing thread only with `--apply`. It saves prior masks before any
change, verifies each result, and supports restoration while rejecting
recycled process IDs. New threads inherit their creator's mask. Apply after
worker initialization and graph capture, then record the masks with the run.

One candidate partition for this host reserves cores 0–3 for frontend/engine,
uses six physical cores per TP worker (4–9, 10–15, 16–21, 22–27), and leaves
28–31 spare. Include each core's SMT sibling in the corresponding mask.
This is an unmeasured partition, not an established optimum. The 15 CPU
tests use mocked processes/syscalls; no running process affinity was changed.
Keep this experiment separate from kernel changes when measuring its effect.

## Strata lessons applied to GLM

This comparison uses Strata commit `6f32ec070f23ced9f50e704d854d775da52591ab`,
especially its [fused MoE implementation](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_fused.hpp).
Its transferable ideas are reusing unpacked weights across more routed tokens,
shortening scratch lifetimes, and fusing activation processing and reduction.
CUDA integer matrix instructions and its activation rounding are not a direct
implementation for this FP16-activation 310P path. No new hardware runs were
performed for this comparison or the changes below.

### Release consumed activation storage

`_apply_device_grouped` now releases the gathered input after gate/up projection,
the gate/up views after SwiGLU, the activated input after down projection, and
the routed output after combination. This lets the existing stream-aware
allocator reuse storage sooner; it introduces no synchronization, custom pool,
in-place aliasing, or arithmetic change. Actual reuse and graph-pool savings
still need device measurement.

At 1280 tokens, top-8, hidden 4096 and intermediate 2048, the gathered FP16 input
is 80 MiB, gate/up storage is 80 MiB, and the down input is 40 MiB. Previously
these 200 MiB remained referenced through combination and shared-expert work.
These are tensor sizes, not a measured reduction in peak memory. They should
not simply be added to the FP32-combine estimate to predict device headroom.
The persistent shared cache-pool reservation described above is a separate
constraint; earlier frees alone do not change its accounting formula.

Six CPU regression cases cover separate/fused gate-up and torch/CANN/native
FP32 combination. Weak references check release before later stages, including
the base storage owner of fused gate/up views; output and shared-expert behavior
are checked against the same arithmetic. The focused suite now passes 62 tests.

### Order of remaining GLM experiments

1. **Preserve decode while extending resident W3 prefill.** Test the staged
   prefill-only dispatch and native FP32 combine first. The later pipelined
   W2/W4 L1 schedule succeeded; the earlier L1 regression does not invalidate
   residency. The W3 resident/1280 serving candidate improved cold TTFT but
   regressed C4, so it is not an overall promotion.
2. **Increase actual expert reuse within a layer.** Pair larger scheduler
   chunks with the Python route cap and enlarged operator limit. The dispatcher
   already groups the current chunk by expert on device; repeating that sort
   will not remove repeated projection work across chunks. Routing for later
   layers depends on earlier layer outputs, so experts cannot be preselected
   once for the whole model. For groups above the 128-row wide-L1 limit,
   investigate keeping an unpacked weight tile resident across multiple row
   tiles while bounding live accumulators. The tested narrow full-K large-group
   variant regressed; do not relaunch it as a new candidate. Larger batches must
   retain context admission and concurrent-request memory headroom.
3. **Fuse GLM SwiGLU without adding activation quantization.** Preserve the
   current FP16 gate/up inputs, FP32 SiLU and multiplication, and final FP16
   rounding. A fused operator can remove full FP32 activation intermediates.
   Qwen's INT8 packing operator has a different numerical contract. Fusion
   into projection epilogues also needs gate and up tile coordination; merely
   calling a separate activation kernel does not eliminate gate/up GM traffic.
4. **Treat packed integer compute as a distinct experiment.** Strata unpacks
   low-bit weights and uses INT8 matrix arithmetic. Reproducing that approach
   requires an explicit activation-quantization and block-scale accumulation
   design for GLM W2/W3/W4, with separate numerical and performance gates.
   It cannot be presented as a bitwise-preserving scheduling change.

The historical 152.2 GB decoded-write estimate describes the GM fallback with
all local experts active, not measured traffic for every current path. The GM
kernel materializes decoded tiles while iterating over matrices; it does not
hold every full decoded expert matrix simultaneously. Wide-L1 dispatch already
avoids that FP16 GM round trip for supported groups. Future traffic estimates
must distinguish the selected paths and actual expert row distributions.

QSA selection and speculative policy are lower priorities for this GLM study;
the Qwen QSA and MTP2 configuration should not be assumed to describe GLM.
CPU expert offload does not address repeated expansion of already resident
expert banks. Device testing remains deferred, and saved serving controls
remain the reference for the next authorized experiments.

## Implemented and tested Strata batch

The user briefly reauthorized NPU access, then deferred it again after the
operator checks below. All benchmark processes exited; no server was launched
and port 8001 had no listener at handoff. Source changes remain on main with
experimental features disabled by default. All measurements below are isolated
operators, not serving throughput or a new model-quality result.

### Bounded resident row reuse

`GLM_W2_GROUPED_L1_ROW_REUSE`, paired with wide/pipelined L1 and prefill-only
W3 selection, extends residency to expert groups of 129–256 rows. Two independent
128×128 FP32 accumulators occupy 128 KiB of L0C. Each decoded 128×1024 weight
tile stays in L1 while both row tiles consume it. K accumulation order is
unchanged for each output. The epilogue writes each row tile separately through
the existing 96 KiB UB region, preserving the decode tables above that region.
No FP16 decoded-weight GM workspace is used by this path.

The first version allowed arbitrary groups through repeated 256-row windows.
On the saved 2560-token concentrated W3 gate/up case, it measured 175.12 ms
versus the earlier GM-reuse result of 113.00 ms. Balanced routing measured
91.90 ms versus the saved 91.66 ms. Repeated dequantization across windows
therefore outweighed the removed GM traffic for large groups. The final
dispatch keeps the existing GM schedule above 256 rows. The older narrow
full-K L1 experiment was not rerun.

A new boundary benchmark measured four active experts at realistic projection
dimensions in separate candidate/control processes. The control uses the saved
prefill-only W3 package. These are seven-sample medians in one process per
variant, without repeated alternating process order:

| Projection | Rows per expert | Control | Bounded resident | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| W3 gate/up | 129 | 6.854 ms | 6.421 ms | 6.3% |
| W3 gate/up | 192 | 7.659 ms | 7.189 ms | 6.1% |
| W3 gate/up | 256 | 8.796 ms | 8.308 ms | 5.5% |
| W4 gate/up | 129–256 | — | — | 3.0–8.0% |
| W2 down | 129–256 | — | — | 5.7–9.0% |

All 15 output hashes matched across packages. The unchanged 128/257-row
boundaries varied by roughly 0.2–2.3%; no improvement is claimed there.
The 15 mixed-expert hardware regression cases passed bitwise against canonical
GM output, covering 129/255/256/257/513 rows, all three bit widths, empty
experts, singleton neighbors, tails, and peer-owned trailing rows.

### Fused GLM SwiGLU and FP32 route reduction

The new `npu_w2_swiglu_310` consumes contiguous FP16 `[routes,2*intermediate]`
gate/up output, evaluates SiLU and multiplication in FP32 in 16 KiB of UB,
and writes FP16 down-projection input. It does not add activation quantization
or copies of the strided gate/up views. HF override
`ascend_glm_prefill_swiglu: true` enables it for at least nine tokens when the
bank supports combined gate/up. C1/C4 decode retains the previous activation
path; separate projection banks retain the Python activation path.

The first hardware extreme-value test caught a real conversion difference:
AscendC cast generated infinity on FP16 overflow while the serving torch_npu
cast saturated to 65504. Explicit FP32 clamping before conversion fixed this
case. All 13 final hardware tests passed, including aligned tails, 20480 routes,
extreme finite values, invalid-input rejection, and changed-input graph replay.
The numerical gate permits one FP16 relative step plus a subnormal allowance;
this operator is not claimed bitwise identical to framework SiLU.

The previously staged `npu_w2_route_combine_310` passed all 24 hardware tests,
including FP64-oracle error bounds, poisoned peer rows, zero local groups,
invalid inputs, and graph replay. It retains FP32 output and avoids an FP16
rounding step at reduction. It remains a separate opt-in HF override,
`ascend_glm_fp32_route_combine: true`.

Alternating original/fused measurements used identical inputs, nine timed
samples each, top-8 routing, intermediate 2048 and hidden 4096. Route reduction
used 25% local rows. Peak numbers are PyTorch allocated-byte deltas for one
operator, including its output and visible workspace; they are not full-model
memory savings and must not be added together as simultaneous peaks.

| Operation | Tokens | Original | Fused | Original peak | Fused peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| SwiGLU | 640 | 2.781 ms | 1.640 ms | 120 MiB | 22 MiB |
| SwiGLU | 1280 | 5.425 ms | 3.216 ms | 240 MiB | 42 MiB |
| SwiGLU | 2560 | 10.709 ms | 6.342 ms | 480 MiB | 82 MiB |
| FP32 route reduction | 640 | 2.909 ms | 0.422 ms | 170 MiB | 12 MiB |
| FP32 route reduction | 1280 | 5.805 ms | 0.739 ms | 340 MiB | 22 MiB |
| FP32 route reduction | 2560 | 11.515 ms | 1.403 ms | 680 MiB | 42 MiB |

Small eager C1/C4 reduction measured about 0.071 ms versus 0.136–0.141 ms.
Those measurements include launch overhead; graph decode speed remains untested.
The custom operators also allocate roughly 2 MiB of runtime workspace, so
small-shape allocation peaks increase despite their lower eager latency.

### Prepared integration and remaining gate

The final package is
`/srv/ai/src/glm-l1-wide-build-20261004/install-strata-glm-bounded-20261004/packages`.
Its host tiler uses `GLM_W2_GROUPED_MAX_ROUTES=20480`; the Python override must
still match it. Default builds retain their previous admission limit.

Main contains both ordinary extension registrations and Meta implementations.
For this experiment a supplemental library registers only the two new
operators alongside the existing runtime extension. Its entrypoint also loads
it in spawned workers; all other qualified operators remain available.
The two runtime Python files were updated with exactly
`strata-results/runtime-candidate.patch`; previous copies are in
`/home/matteius/experiments/glm-w3-20261004/strata-runtime-backup`.
No full extension replacement or default launcher change was made.

`strata-results/serve-combined-candidate.sh` prepares both fusions while
retaining the previous GLM enhancements, graph sizes `[1,4]`, and port 8001.
At staging time it had passed shell syntax checking. The serving
gate below uses 1280-token chunks, route cap 20480, configured context
196608, the prior 1.0 cache fraction and 0.965 memory utilization. Check the
limiting TP rank at admission; reduced operator scratch does not remove the
larger batch's persistent cache-pool reservation. First run the saved cold
7269-token retrieval, then C1/C4 and candidate quality checks. Compare with
saved serving controls rather than launching a known baseline again.

The focused CPU suite passed 110 tests; targeted Ruff and whitespace checks
passed. Sources, package binaries, and supplemental library hashes are in
`strata-results/manifest.json`. Raw timings, initial failure, final test logs,
compile flags, and build/bootstrap scripts are retained beside it. The subsequent authorized serving tests are recorded below; full-length
context remains untested.

## Strata batch serving validation

After renewed NPU authorization, the combined candidate loaded all four ranks
at configured context 196608, 1280-token chunks, cache fraction 1.0, memory
utilization 0.965, and graph sizes `[1,4]`. Saved controls were reused.
Only `model.py` and `w2_dynamic.py` differed from the saved Python runtime;
the other 745 Python files matched. CPU affinity was unchanged (OMP=1, no
explicit worker binding).

| Configuration | Cold 7269-token TTFT | C1 tok/s | Aggregate C4 tok/s | Strict quality |
| --- | ---: | ---: | ---: | --- |
| Saved overlap, 640 tokens | 77.687 s | 3.962 | 10.848 | 17/20 |
| Previous resident W3, 1280 tokens | 70.236 s | 4.052 | 7.964 | Not repeated |
| Bounded residency + FP32 combine + SwiGLU | 68.436 s | Not run | Not run | Stopped: garbled response |
| Bounded residency + FP32 combine | 68.622 s | 3.989 | 10.268 | 16/20 |

Both cold retrievals returned `BLUE-ORCHID-7319` with all 7269 prompt tokens
computed and zero cached. They are single samples, not confidence intervals.
The reduction-only candidate is 11.7% shorter than the 640-token control,
but only 2.3% shorter than the previous 1280-token candidate. C1 is effectively
flat and C4 is 5.3% below the saved overlap control. All five short responses
reached 256 tokens.

The combined quality run was stopped after garbled `instr_upper` output;
8 of 13 completed cases passed. With SwiGLU disabled, uppercase was correct;
quality finished 16/20 with the three known misses (`instr_reverse`,
`instr_first`, `code_slice`) and backticks around the otherwise correct
`code_range` answer. Neither result is promoted. SwiGLU provided no clear
end-to-end TTFT improvement over the reduction-only candidate in these samples.

Minimum sampled free memory during the combined cold run was 474 MiB.
Reported KV capacity remained 202366 tokens, approximately 1.03 times the
configured context. This validates admission and the short retrieval only,
not a full 192K request.

### Queued command lifetime follow-up

Inspection found that `EXEC_NPU_CMD` released its workspace owner before
`cmd.Run()` and captured ACL descriptors holding raw tensor addresses. The
candidate now retains direct tensor arguments and workspace in the queued
handler. A CPU test compiles the actual macro with a deferred queue; it fails
against the old macro and passes with ownership retained. The focused lifetime
and method tests passed 67 cases. This is not yet proof that storage lifetime
caused the garbled serving response.

A separate supplemental binding overrides only the grouped projection and
registers the two new operators with the revised command macro. Its first
load rejected a stale four-argument grouped adapter against the runtime's
five-argument schema, before inference. The adapter was refreshed from main
and rebuilt. The original serving library was not overwritten.

Raw serving logs, exact launchers, quality/short JSONL, and cold results are in
`strata-results/`. Numerical and hardware lifetime diagnostics follow before
any further serving promotion.

The exhaustive finite FP16 gate sweep (three fixed up-values, 65536 bit
patterns each, nonfinite gates replaced by zero) found only 3, 2, and 2
native/framework mismatches, all within one FP16 rounding step. The framework
explicit Exp/Div formula reproduced those same mismatches. This rules out a
large systematic cast discrepancy on that sweep, but does not establish the
cause of the serving failure or full-model equivalence.

Final ownership-binding hardware validation passed **53/53** tests: 14 SwiGLU
cases (including immediate temporary-input release and same-size allocator
churn), 24 FP32 combine cases, and 15 mixed-expert resident/GM parity cases.
The ownership fix has not had a full-model serving test; it is not claimed to
resolve the earlier corruption. Targeted Ruff checks passed, and the new CPU
regression was formatted. No defaults were changed and no candidate was committed.

At the user's request all four NPUs were released after the check. Final
`npu-smi` reported no running processes on either physical card, all four
AI Core utilization values were zero, and port 8001 had no listener. The
servers remain stopped. No further device tests are scheduled without renewed
authorization. The next useful serving test is the retained-ownership binding
with the same combined flags, to separate lifetime effects from the tiny
arithmetic differences; reuse the recorded performance controls.

## Next CPU-only batch: isolate prefill reduction

After releasing the NPUs, main gained the opt-in HF override
`ascend_glm_prefill_fp32_route_combine: true`. It enables the native FP32
reducer for calls with at least nine tokens and uses the existing torch
reduction and initialized peer outputs for one through eight tokens. This is
a host-shape threshold, not scheduler phase detection: a short prefill retains
the old path, and a decode batch above eight would use the new path. The
qualified server uses at most four requests and captures sizes `[1,4]`.

The existing all-token `ascend_glm_fp32_route_combine` mode remains available
for comparisons. All-token FP32, prefill FP32, and CANN reduction modes are
mutually exclusive; CPU offload disables the device option. Config is passed
through nested text config, geometry, and expert-bank construction. No new
environment variable, synchronization, or default change was added.

`strata-results/serve-prefill-reduce-candidate.sh` stages this mode with the
retained-ownership binding, bounded resident package, and SwiGLU disabled.
It has not been copied into or launched on the NPU host. Hardware performance
is unmeasured. The 105-test focused CPU run passed, covering exact baseline
output at decode sizes, dispatch and missing-extension behavior at 8/9 tokens,
640-token prefill, poisoned peer rows, conflicting modes, and offload.

This isolates the prefill changes from the all-token reducer used in the
10.27 tok/s C4 run. It does not establish that the reducer caused that run's
5.3% regression. Compare the staged candidate with the saved cold retrieval,
C1/C4, and strict-quality records on the next authorized hardware window.

## Prefill-only serving: admission and command retention

Renewed hardware authorization allowed the staged prefill-only reducer run.
The first server (PID 2612425) failed 196608-token admission: available cache
was 4.61 GiB against 7.4 GiB required. All other serving settings were held
constant. No prompt was served. Only the two intended Python runtime files
differed; 745 other files matched the previous run.

The retained-ownership supplemental binding introduced a memory-retention bug:
completed runtime queue slots kept their captured tensor owners. A standalone
chained grouped-projection probe isolated it without loading the model. All
local output values remained correct, but the `owned` binding kept 7,902,220,288
allocated bytes after the 64-call segment (following 1- and 16-call segments).
The original binding returned to 6,358,528 bytes of live weights/metadata after
each segment, with a 184,617,984-byte peak.

The revised handler is mutable and clears its captured tensor tuple and
workspace after the ACL launch and descriptor cleanup. This retains arguments
until their queued launch executes and releases them before queue-slot reuse.
The corrected binding returns to the same 6,358,528-byte allocation after all
three segments. Its peak for the longer segments is bounded at 362,877,440
bytes in this probe. This is a memory regression fix, not a measured speedup.

The CPU macro test now retains the completed queue entry and requires owners
to expire before recycling that entry. A new hardware regression repeatedly
chains grouped projections and verifies allocated memory returns to its warm
baseline. The revised binding passed 54 hardware tests, including numerical
parity, temporary input churn, poisoned peers, and graph replay.

Artifacts: `strata-results/binding-memory-comparison.json`, the three raw
binding-memory logs, `release-hardware-tests.log`, `release-binding-hashes.txt`,
and `prefill-reduce-1280-admission-failed.log`. The previous binding binary is
preserved separately; the corrected library is under
`build-strata-bindings-release-20261004`. PID 2655812 retries serving with
`serve-prefill-reduce-release-candidate.sh` and unchanged context/memory settings.

### Serving outcome after the retention correction

The corrected binding still failed admission at 0.965 memory utilization:
rank 0 reported 7.91 GiB available, but the limiting rank had 7.31 GiB against
7.4 GiB required. Increasing utilization to 0.968 admitted configured context
196608 and graph sizes `[1,4]`; reported capacity was 200104 tokens and graph
capture allocated 0.40 GiB. The first real cold prefill then failed with a
482 MiB allocation OOM in `mhc_post_torch`'s residual-mixing `einsum`.
Admission alone was insufficient. The initial commentary attribution to KDA
was corrected after reading the full traceback.

The next trial kept 196608 context, returned utilization to 0.965, reduced the
scheduler budget to 1024, and allocated 0.9 of profiled headroom to cache. It
admitted 207530 tokens of capacity. However, the sparse/Mamba cache-group
alignment actually scheduled **640-token chunks**, plus the 229-token tail;
1024 was only the configured budget. The source is
`patch_mamba_block_aligned_split.py`, which rounds intermediate sparse chunks
to the resolved common block boundary. Thus this trial did not retain the
intended larger actual expert batches.

That request completed prefill, then generation stopped progressing. The
sampled NPU 1 memory reading reached 46765/46765 MiB; no explicit OOM appeared
in this run's log. The stall's cause remains unproven. The server was stopped
rather than treated as a successful gate. No completed retrieval answer,
new quality score, or C1/C4 measurement is claimed for this hardware window.
Logs are `prefill-reduce-mem968-oom.log` and
`prefill-reduce-1024-stalled.log`. No candidate was promoted.

The command-ownership changes are now behind the compile flag
`GLM_EXPERIMENTAL_OP_API_TENSOR_OWNERS`; default source builds retain the
established command adapter behavior. The supplemental binding build scripts
explicitly request this experimental branch. The measured binary was built
before the guard was added, with the same retained/released-owner behavior;
its hash remains recorded. The guarded branch passed the CPU macro test.
Further full-model graph and memory validation is required before making it
default. No hardware build or test was started after NPU usage was deferred.

### mHC post-mixing memory experiment

The OOM identified a concrete large intermediate outside the expert kernel.
`tools/glm_perf/benchmark_mhc_post_310.py` compares the existing `einsum` with
stream-by-stream mixing using multiply/add or in-place `addcmul_`. Both keep
one output-sized mixing term live instead of the general einsum's larger
workspace. This is an isolated benchmark, not installed serving dispatch.

Unrestricted random FP32 inputs changed about 38% of FP16-rounded outputs,
so those results cannot justify substitution in the model. The qualified
`ascend_glm_mhc_fp16_state` path actually carries FP16-rounded values in FP32
tensors; a second completed probe explicitly reproduced that input contract.

| Tokens | Existing einsum | Streaming addcmul | Existing allocated peak delta | Candidate peak delta |
| --- | ---: | ---: | ---: | ---: |
| 640 | 5.756 ms | 4.163 ms | 280.31 MiB | 90.00 MiB |
| 1024 | 9.120 ms | 6.811 ms | 448.50 MiB | 144.00 MiB |
| 1280 | 11.359 ms | 8.378 ms | 560.63 MiB | 180.00 MiB |

At 1280 tokens this is about **26% less operator time** and **68% lower
allocated peak delta**, measured across seven alternating-order samples.
It is not an end-to-end speedup or a physical-free-memory guarantee. Maximum
FP32 absolute difference was 9.54e-7; 964 of 20971520 outputs differed after
FP16 rounding (about 0.0046%). It is not bitwise equivalent. C1/C4 eager
latencies regressed, so any serving integration should initially target large
prefill calls only and retain the current decode path. Graph and serving
quality gates are still required.

The benchmark math has 13 CPU checks covering leading dimensions, mixing
orientation, input preservation, dtypes, and invalid empty streams. Combined
with the revised command macro test, the final CPU check passed 14 tests.
Raw measurements are `mhc-post-raw-fp32.json` and `mhc-post-fp16-rounded.json`.
The latter probe completed before the user's deferral. All NPU processes
exited, the memory sampler was stopped, and port 8001 is closed. Work is now
CPU-only until renewed authorization.

### Prefill mHC integration staged; hardware deferred

The opt-in HF override `ascend_glm_prefill_mhc_post` now propagates from the
packed model into the fused mHC post/pre operator. With FP16 state rounding
also enabled, four-stream calls with at least 640 tokens use the measured
streaming `addcmul_` mixer. Default configuration, BF16 state, small tails,
and decode retain the existing post mixer. The standalone final post op is
unchanged. Selection uses host shape metadata; inputs are not modified, and
residual rounding still happens before the next pre calculation.

CPU checks passed: 17 new production math/dispatch tests plus 13 existing
benchmark tests (30 total). Focused Ruff checks and formatting checks passed.
The existing worker-package tests could not collect in this CPU environment
because `_build_info` is absent; the new tests load the actual patch against
narrow upstream stubs without device discovery. NPU parity, serving quality,
peak memory, and end-to-end performance remain untested for this integration.

The subsequent hardware grant was withdrawn before any server was stopped,
any source was deployed remotely, or any NPU test was launched. The inspected
Qwen server on port 8001 was left untouched. Next hardware work must use the
established grouped binding, avoiding the unqualified command-ownership
binding, and must await renewed authorization.
