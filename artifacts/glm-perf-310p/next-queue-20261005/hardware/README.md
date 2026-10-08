# GLM queue hardware run — 2026-10-05

**Closed without a promoted performance change.** Qwen had already stopped
when this hardware session began. The user later revoked NPU access; all GLM
server processes and test controllers were stopped. Runtime files and raw
results are in `/home/matteius/experiments/glm-next-queue-20261005` on
threadripper. Later permission for an isolated completed-pool measurement is
recorded separately; it does not resume this launch queue.

## Fixed serving configuration

- Selective packed W3 checkpoint, TP4, MTP1, full decode graphs `[2,8]`.
- Port 8001; 311040 configured context, still not full-length validated.
- Initial scheduler batch 640; histogram route counting; existing native mHC
  and live kpool scorer; prefix caching enabled with cache resets on switches.
- CPU affinity applied to all workers using the existing four disjoint masks.
- Resident switches verify unchanged rank/PID/weight-storage identities.

## Isolated findings

| Experiment | Evidence | Current decision |
| --- | --- | --- |
| Larger expert batch | W2/W3/W4 at 10240 and 20480 routes match smaller calls exactly | Operator gate passed; two memory-admission failures; final launch cancelled |
| Adaptive teams, 2 and 4 lanes | Each passed 20/20 baseline output hashes, including empty/mixed/decode/hot routes | No material projection gain; not selected for serving |
| Batched Q/K norm | Exact NPU results at rows 1/2/4/8 on all four ranks | No established serving gain |
| Batched selector epilogue | Exact indices including padding/ties and capacity 77760, all ranks | Initial c4 gain did not survive baseline variability |
| Native gate/beta | Native compilation and 32 exact cases independently, then 32 exact cases per resident rank | No serving gain |
| FP16 unpermute | Exact outputs at rows 2/8/640 and hidden width 4096, all ranks | Initial c4 gain did not survive baseline variability |
| Combined indexer projection | FP32 differences on real target/draft weights on all ranks; examples reach 2.29e-5 | Failed exact projection gate; not admitted to serving |
| Direct route token IDs | Exact metadata/gathers at rows 2/8/640/1280 on all ranks | Division variant captures; no serving gain |
| KDA constants per head | Four production safe-gate shapes match baseline outputs/carries; deterministic replays | Gate step faster, whole KDA call flat; not selected for serving |
| Skip safe-gate Cube stage | Same safe-gate output/carry hashes and replay checks | Whole KDA timing flat; not selected for serving |

Adaptive-team timings are isolated projection calls, not model throughput.
For W3 gate/up with 72 experts and uniform local routes, baseline/2-lane/4-lane
medians were **80.08 / 79.95 / 79.84 ms**. With concentrated routes they were
**8.49 / 8.54 / 8.92 ms**. A separate `strace` discovery probe confirmed CANN
opened the intended adaptive package's kernel object; its instrumented timings
are excluded from performance results.

For 640 tokens and 16 heads, KDA gate-cumsum medians were **0.939 ms baseline**
and **0.839 ms with per-head constants**. Full KDA was **50.305 / 50.406 ms**;
the skip-Cube candidate was **50.427 ms**. These are initial sequential isolated
measurements, not paired serving wins. Safe-gate cases included dense input,
chunk tails, empty varlen sequences, initial carries and repeated execution.

The non-safe-gate full-KDA probe returned non-finite results on the **baseline
and both candidates**, including a repeat with smaller negative gates. Its
standalone gate-cumsum output matched. This path is not GLM's production mode;
no non-safe full-KDA qualification is claimed. The initial nested chunk-index
list was also rejected by the binding; the corrected probe uses its flat-list
contract. Failed probes remain in the remote logs.

## Controller correction

An invalid baseline-restoration label caused worker validation exceptions and
stale executor replies before any serving measurements. The diagnostic server
was restarted. `ResidentClient.switch` now validates controls locally before
any RPC; three regressions cover invalid names and baseline/source combinations.
The focused harness suite passes 19 tests. Subsequent baseline restoration uses
`candidate="baseline", source=""`; experiment labels are stored separately.

The discarded first launch log is `serve-queue10-baseline.log`. Comparisons use
`serve-queue10-baseline2.log`, API PID 1679112 at the start of this phase.

Native builds use isolated build/package outputs and preserve the qualified
runtime packages. Temporarily patched build inputs were restored byte-for-byte.
`build-results.json` records package paths and flags. The gate/beta manifest
records library/binary hashes; its load receipts show all four rank validations.

## First resident serving sweep

All timed short requests reached 256 output tokens, MTP1 and full graphs stayed
on, and switches retained the four worker PIDs and weight-storage digests.
These are sequential first-pass measurements, **not confirmed speedups**.

| Candidate | c1 tok/s | Aggregate c4 tok/s | Cold 8K TTFT (s) |
| --- | ---: | ---: | ---: |
| Baseline | 5.019 | 11.202 | 90.180 |
| Batched Q/K norm | 4.930 | 11.409 | — |
| Batched selector epilogue | 4.894 | 12.300 | — |
| FP16 unpermute | 4.951 | 12.486 | 89.393 |
| Native gate/beta | 4.922 | 10.598 | — |
| Baseline repeat | 4.951 | 12.661 | 89.961 |
| Direct route IDs, division | 4.951 | 10.992 | 89.913 |
| Selector + FP16 unpermute | 4.872 | 11.300 | 89.623 |

The repeated baseline's c4 result also increased, so initial selector/unpermute
uplifts cannot be attributed to their code. No further baseline runs are queued.
Generated text varies between
requests despite temperature zero; MTP acceptance and device telemetry are
recorded rather than assuming identical decode work from identical token caps.

The direct route-ID candidate passed eager numerical checks but failed full
model graph capture. The harness restored baseline without reloading weights.
The scalar right-shift path caused a synchronous host/device copy during
capture (`aclrtMemcpy`, 107030). Integer floor division captures successfully on
all ranks and completes the serving suite. Its cold TTFT is unchanged. A new
hardware regression covers changed permutations and gathers on graph replay.

## Change in test scope

At the user's direction, further baseline measurement stopped. The final
baseline pass completed c1/c4, the 20-case gate (**17/20**, same three misses)
and the tool-call check, but its 24K request was cancelled. It is not a completed
24K baseline measurement. The pending automatic reload controller was stopped
before changing the server; the 1280-token candidate was then launched directly.
Subsequent performance comparisons use the already completed baseline results.

The route-ID correction also passed **9/9 isolated NPU graph-replay cases**
(rows 2/8/640, top-k 1/3/8, changed permutations and exact gathered outputs).

## Larger-batch memory admission

1280 tokens and 311040 context require **6.36 GiB** of cache admission. The
640-token profile's memory fraction does not carry over: the compressor's
sliding-window manager reserves pages touched by the larger in-flight chunk,
and those virtual IDs consume the shared global block pool. The physical
attention page size remained 640 tokens in both worker configurations.

- Fraction 0.75: rank 0 reported 5.57 GiB; admission minimum was 5.56 GiB.
- Fraction 0.88: rank 0 reported 6.53 GiB; admission minimum was 6.08 GiB.
- The second profile implies at least **6.91 GiB** of unfractioned headroom
  on its lowest rank. The final attempted launch explicitly budgeted **6.45 GiB**
  (6925634765 bytes), above the 6.36 GiB admission requirement and below that
  inferred headroom. It was cancelled during loading when hardware access was
  revoked. Neither admission nor runtime memory safety was validated for this
  budget. The rank-minimum discrepancy needs investigation before another
  launch; no larger-batch serving result exists.

Both rejected launches exited before serving any requests. Logs are
`serve-batch1280.log` and `serve-batch1280fit.log`; the explicit-budget attempt
is `serve-batch1280fixed.log`.
