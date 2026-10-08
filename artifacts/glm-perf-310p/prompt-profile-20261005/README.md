# GLM prompt processing: profile review and column-cache qualification

## Status

The previous failed profiler session released the NPUs at
2026-10-06 00:52:24 UTC. Offline analysis and CPU/native compilation followed.
The user's subsequent **Proceed** authorized hardware qualification against the
new fused server. The corrected, full-chunk-only column cache passed the NPU
score, complete KDA and serving checks below. It is **deployed on port 8001**
after one controlled package restart, with the original Sinkhorn fusion restored.
Cold prompt TTFT fell 4.65% at 8K and 4.70% at 20K in this comparison.

The final API PID is **2511127**, with workers
2512274/2512653/2513147/2513520. The server is unpaused, context remains 311040,
MTP1 and full decode graphs remain enabled, and all four graph/native receipts
are clean. `final-score-cache-status.json` records generation
`02a63bf4ee334fc9a1ae8baf64b4a43d` and the exact original fused source digest.

The patch is applied to shared source behind the default-disabled
`GLM_KDA_SCORE_CACHE_COLUMNS` compiler flag. The isolated serving package
enables that flag. The profile attribution still uses the saved October 4
trace; no successful fresh profiler capture is claimed.

### Subsequent decode baseline

The subsequent port-8001 launch used `completed_pools_sinkhorn`, context 311040,
API PID 2330991 and workers 2332905/2333449/2333970/2334336. This session checked
its receipts and measured two cold prompts before the controlled kernel-package
restart. The restart preserves its command, vendor ordering, host library paths,
MTP1, full graphs, batch 640, and exact fused candidate source.

The [Sinkhorn comparison](../sinkhorn-resident-20261005/README.md) reports
5.1814 tok/s c1 and 12.3173 aggregate tok/s c4. One resident rollback measured
11.8742 aggregate tok/s, a 3.73% aggregate difference. The c4 per-request median
was 3.1246 fused versus 3.1486 control (-0.76%); generated text and prefill
batching differed. This establishes a modest single-comparison aggregate gain,
not a per-stream coding improvement. The fused dispatch only admits 1–8 rows,
so it does not optimize the 640-token prompt chunks targeted here.

`fused-baseline-candidate.py` preserves the exact subsequent candidate source;
its SHA256 matches the saved four-worker receipts. `fused-baseline.json` records
the native resource identity and comparison statistics. Future prefill
comparisons must retain this fused decode baseline and restore its exact source.
The earlier profiler scripts preserve the failed run against `completed_pools`
and must not be used to restore that older configuration onto the new server.

## Observed latency and profile attribution

The preceding [live coding metrics](../coding-metrics-20261005/README.md) recorded
20,643 uncached input tokens, 260.74 seconds of prompt processing, zero queue
time, 4.1 output tokens/s and 276.42 seconds total for 65 output tokens.
Prompt processing consumed approximately 94% of total latency.

The saved FP16 trace covers a 13.43-second task envelope and two 640-token
chunks. Reprocessing its four-rank kernel CSVs gives:

| Task family | Per-rank summed task time |
| --- | --- |
| Packed expert projections | 5.23–5.35 seconds |
| Nine KDA stage launches per call | 3.31–3.32 seconds |
| Third KDA launch within those nine | 2.700–2.704 seconds |

The third launch represents approximately **81.5% of KDA task time** on every
rank. Its median duration is roughly 39.7 ms, repeated 68 times per rank.
The sequence matches the host order `0,6,7,8,1,4,2,3,5`; consequently stage 7,
vector score finalization, is the inferred mapping. Private stage values are
absent from the CSV, so ordinal-to-stage attribution is an inference.
Stage 7 combines vector score computation and the triangular solve; the trace
does not split their individual costs.

Task durations may overlap. These numbers are attribution, not independent
wall-time savings or today's bottleneck percentages. Runtime changes since
October 4 include native mHC post, live decode scoring and completed-pool writes.

Evidence: `historical-prefill-tasks.json` and
`historical-prefill-attribution.json`. Raw CSVs remain under the recorded remote
`profile-current-fp16-640-20261004` paths. The existing category/collective
summary is in `../wide-l1-20261004/profile-current-fp16-summary.json`.

## New candidate: retain KDA score columns in UB

`ComputeRawAqkAkkVector310P` computes K×K and Q×K in separate passes. For each
causal pair, it reloads the same key vector and cumulative gate vector, then
waits for both transfers. At 64 rows and 128 channels, that repeats thousands
of small transfers for immutable inputs.

The staged `GLM_KDA_SCORE_CACHE_COLUMNS` build flag:

- Copies a full chunk's key and gate columns into the existing 128 KiB vector
  arena, beginning at byte 81920, beyond the triangular solve's scratch.
- Uses one key transfer for head-major inputs, or one per row for the strided
  sequence-major layout. Gates always use one contiguous transfer.
- Reuses readonly local views for each causal pair in both existing passes.
- Retains the original FP16 arithmetic, FP32 reduction order, score stores,
  dot-product synchronization, triangular solve, carries and stage boundaries.
- Falls back for every partial chunk, other chunk sizes, and inputs that exceed
  the byte-capacity guard. A full chunk fits for aligned 16–192 channels;
  256 channels use the original path. It adds no persistent device allocation.

For the production 64×128 shape, the existing score scratch ends well before
the cache starts, and the two cached banks end at byte 114688. Complete operator
qualification covers final state, initial carries, tails, and repeated execution.

| Per chunk/head, 64×128 | Existing | Candidate |
| --- | ---: | ---: |
| Global input-copy calls, head-major | 8,576 | 258 |
| Global input-copy calls, sequence-major | 8,576 | 321 |
| Logical input-copy bytes | 2,195,456 | 98,304 |

These are verified source/simulation counts, **not HBM traffic counters or
measured acceleration**. Arithmetic and remaining barriers still cost time.
Other KDA stages and packed expert projections remain substantial work.

The source header SHA256
`36707989c3126c62dff360c80b3d189b85d490a5fc61a21c7a84b4e531e97fb0`
matches the original deployed `kda-persistent-scores-opp` header exactly. All six
packaged kernel source/header files also match the isolated build's starting
source. `kda-score-cache-columns.patch` records the applied source change;
`full-kda-build.json` records the enabled package and binary digests.

## Validation completed without devices

```bash
git apply --reverse --check \
  artifacts/glm-perf-310p/prompt-profile-20261005/kda-score-cache-columns.patch
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest --noconftest -q \
  tests/ut/glm_perf/test_kda_score_column_cache.py
```

**Two tests passed**, including 38 C++ simulation cases, each covering three
different query/gate head pairs. They execute the actual score and reduction
methods extracted from the staged source, and check:

- Byte-identical Aqk/Akk outputs, preserved inputs, FP16 intermediates and
  original FP32 reduction order.
- Empty/full/tail chunks, 16/64/128/192/256 channels, both public input layouts,
  nonzero batch/head/start offsets, changing input data, NaNs and signed zero.
- Bounds of the existing UB arena, preservation of the solve scratch area,
  and delayed copy visibility at wait events.
- Exact copy/byte counts and fallback behavior for other chunk sizes.
- Identical preprocessed score-method tokens when the flag is absent.

Six existing gate/scheduling simulation tests also passed during this work.
Ruff, shell syntax and ShellCheck passed on the new Python/build files.

`prepare-native-score-probe.py` exports the actual score methods into a
fixed-shape standalone probe. `build-score-probe.sh` successfully compiles both
variants with CANN 9.1.0 Bisheng, `dav-m200`. Their logs and source/object hashes
are saved locally; the objects remain under the remote study directory.
The initial relocatable objects lacked argument metadata and were rejected
before kernel execution. `build-score-probe-rtc.sh` produces executable binaries
with the CANN runtime compiler, used by the subsequent standalone checks. The
probe is **not the full KDA pipeline**; its aligned copy adapter is limited to
the fixed 16-head, 64-row, 128-channel geometry.

## Hardware qualification and cold baseline

Standalone score checks passed **32/32 cases across four devices**, with exact
FP32 bit patterns, unchanged inputs, finite/nonzero results, zero causal upper
triangles, and independent zero-gate/diagonal checks. Alternating launch-order
medians show approximately 1.34× score-loop speedup. See
`score-probe-device-{0,1,2,3}.json` and `native-rtc-sha256.txt`.

The complete KDA sweep covers 1/16/63/64/65/134/640 tokens, both layouts,
initial carries, an empty sequence, five 640-token gate/layout configurations,
and repeated execution. All **17 production safe-gate cases** match the original
attention outputs, FP32 final carries and all exported intermediates bit for
bit. An independent unchanged-package control also matches those references.
Full 640-token calls fall from about 50.6 ms to 39.8 ms. This is approximately
21% less operator time; the request-level reduction below is much smaller.

Two earlier revisions admitted partial chunks and failed complete-operator
parity, including actual attention/carry differences at 63 tokens. Moving the
cache alone did not fix them. Their sources, binaries, and failing results are
retained in remote `rejected-cache-32k` and `rejected-cache-80k-tails` directories;
local receipts preserve the failed comparisons. They were never deployed.
The final revision retains the original tail path.

The two nonsafe diagnostic cases produce nonfinite and nondeterministic outputs
in the unchanged baseline as well. They are recorded separately and are not
claimed as qualified; production GLM uses safe gating.

| Cold input tokens | Baseline TTFT | Candidate TTFT | TTFT reduction | Candidate input tokens / TTFT second |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 96.82 s | 92.32 s | 4.65% | 88.74 |
| 20643 | 244.52 s | 233.02 s | 4.70% | 88.59 |

`cold-fused-baseline.json`, `cold-score-cache.json` and the before/after metrics
snapshots record these completed API requests. Each interval contains exactly
one completed request, zero cached tokens, zero preemptions and less than
0.1 ms of queue time. Prometheus prefill-time deltas are 96.806/244.316 s for
the baseline and 92.307/232.815 s for the candidate.

Input is synthetic Python code, truncated to exact token counts through
`/tokenize`. The candidate reuses each baseline nonce and identical text;
candidate token digests are saved. Requests produce one output token and are
distinct from the earlier real coding request. Each size has one comparison
across the package restart; there is no confidence interval. CPU kernel
compilation overlapped part of the baseline timing. The interval logger's
bursty prompt-throughput figure is not the end-to-end rate reported here.

Prefix reuse passed: repeating the 20K prompt reused **20480 tokens**, computed
163 and completed in **3.69 s**. This verifies the existing prefix cache;
the column-cache patch did not introduce prefix caching.

All **six serving checks** passed: c1, four concurrent c4 streams and a tool
call. The 256-output-token tests measured **5.095 tok/s c1** and **12.111
aggregate tok/s c4**, with c4 per-request median 3.084 tok/s. The earlier fused
figures were 5.181 and 12.317; this unpaired decode comparison is approximately
1.7% lower and does not establish a decode improvement. The cold 20K wait is
still almost four minutes, so broader prompt-processing work remains.

`optimization-summary.json` consolidates operator qualification, metric deltas,
package digests, prefix reuse, serving results and limitations.

`launch-qualified-kda.py` requires complete operator qualification before
stopping a server. It replaces only the existing KDA vendor slot, retains host
library paths and every other vendor's priority, reapplies the original fused
source, and requires a completed warmup inference. Startup/fusion failure
triggers restoration of the original package and fusion.

## Fresh capture failure

The bounded trace attempted early and approximately 16K-context windows using
the resident worker harness. Initial profiler replies exposed the preexisting
unequal response-queue offsets; a profiler nonce/status read was added so the
start/stop operation would be sent once and confirmed on all ranks.

Stopping the first aborted profiler session and recapturing graphs reported
an invalid allocator stream and fatal `HcclAllreduce` errors at 00:44:47.
A later recapture returned clean graph receipts, but the driver's task queue
remained in `CAN_EXIT`. The subsequent prompt stuck in sampled-token event
synchronization on all inspected workers; no iteration completed. This is a
profiler/runtime failure, not a measured slow prefill or a score-cache result.

The API health endpoint and clean graph receipts did not prove runtime recovery.
The poisoned workers were stopped rather than retained for further measurements.
`capture.log`, `failed-start*.log`, `blocked-rank0-stack.txt`,
`rank-pid1894308-profiler-errors.log`, `profiler-error-evidence.json`,
`capture-failure.json` and `release.json` preserve the failure/release evidence.
The large raw error log stays remotely; the local file contains an excerpt.
The profiler scripts are failure reproductions, not a qualified capture recipe.

## Remaining serving checks and optimization work

1. Use a fresh runtime and qualify the profiling lifecycle before another long
   request. Avoid graph recapture while the profiler is active. Check driver
   errors and a completed inference request, rather than health/status alone.
2. Preserve this qualified kernel/fusion combination as the next comparison
   baseline. Python candidate changes continue through the resident harness;
   replacing a CANN package requires the controlled restart workflow here.
3. Separately revisit larger actual expert batches. Their measured isolated
   reuse gains remain promising, but 1280 requires matching 10240-route host
   and Python limits, runner allocations and cache admission. Previous admission
   failures and bundled decode regressions remain unresolved; this candidate
   does not require that scheduler change.
