# Qwen TP6 quality repair and recovery, October 10, 2026

This report distinguishes individual regression fixes from serving profiles.
Passing a component or serial benchmark does not qualify a configuration for
concurrent coding sessions. Final recovery measurements are recorded separately
after the hardware probes finish.

## Selected server

The selected profile is TP6/EP6 on all three cards, with MTP2, capture buckets
`[3, 18]`, six request slots, a 262,144-token context limit, and fresh/cached
1024×1024 image processing. Its two matched decode sets have medians 29.118
and 28.992 tok/s; combined median 29.066 tok/s is 50.9% above the historical
TP4 reference. This is a measured comparison to the recorded baseline, not a
fresh paired two-card experiment or a promise for every live workload. The
final hardware guard retains ordinary completion boundaries outside 310P; its
enabled 310P path passed fresh/cached images and another cache-isolation run.

The serving alias is `qwen38-flash-next` at `http://192.168.53.187:8001/v1`.
The final controller source is archived verbatim in `selected-profile`.

The cache planner reports approximately 1.62M tokens total. Six short active
sessions are hardware-tested; six completely occupied 256K contexts are planner
capacity, not a completed stress test. A larger exact-arithmetic probe passed
at 80,028 prompt tokens: 203.727 seconds cold, then 1.863 seconds with 80,000
cached tokens. This checks a large cache restore, not general coding accuracy.

The thermal log recorded no holds or shutdowns during final qualification.
Maximum sampled temperatures and sensor status are in
`selected-profile/thermal-summary.json`. Cooling hardware changed, so lower
temperatures are not attributed solely to these code changes.

To restart the admitted profile on the host:

```bash
bash /srv/ai/src/qwen-quality-fast-correct-tp6-20261010/start-server.sh
```

The launcher waits for API readiness, admits the native manifest, restores the
frozen resident candidate, recaptures, checks text/code/images, and releases
maintenance. This bootstrap is syntax-checked; native admission and the switch
protocol were hardware-tested, but a separate cold run of the combined wrapper
was not repeated. Follow `bootstrap.log` and require `bootstrap-ready.json`.
The native assets remain at their checksummed host paths; their binaries are
not vendored into this report. Canonical file hashes match the deployed snapshot.

## Changes and current result

The reproduced mixed-load stall is repaired. Later Qwen MTP steps now own
persistent, separately refreshed block tables across graph capture and replay.
The native residual candidate was requalified only after that repair. All
qualified profiles completed mixed prefills, six simultaneous decoders,
text/math/code, fresh/cached 1024×1024 images, long-prompt requests, and 18 exact
arithmetic answers across six cached sessions.

Other changes preserve per-step CPU query boundaries, use one-row-per-request
capture geometry for later proposals, trim padded QSA host metadata without a
device read, pack draft result rows before CPU readback, and avoid redundant
block-table clone/padding copies. Synchronous Qwen draft replay uses stream
ordering instead of a host completion barrier; other families, targets with mutable graph tasks,
asynchronous scheduling, DCP and ENPU retain their existing boundaries.

Resident mode-only changes preserve weights, captured graphs and prefix
checkpoints when source digests match. Source replacement still uses a drained
recapture transaction. Uncertain mutation outcomes remain paused.

| Qualified TP6 profile | Median 256-token decode tok/s | Cold 23,410-token TTFT (s) | Cached TTFT (s) |
| --- | ---: | ---: | ---: |
| Persistent draft tables | 27.502 | 62.390 | 1.963 |
| Plus all-row native residual | 28.281 | 56.711 | 1.884 |
| Plus stream-ordered draft replay | 28.722 | 56.673 | 1.885 |
| Plus reduced table copies | 28.609 | 56.734 | 1.875 |

Each short-decode median uses six requests rotating three synthetic prompts
with 256 output tokens; warmup is excluded. The recorded two-card reference is
19.262 tok/s, making the strict 50% target 28.893 tok/s. The best result in that intermediate table
is 49.1% higher; the subsequently qualified empty-update path below exceeds
the target in both measurement sets. These are comparisons to a historical
reference, not a fresh paired two-card run. Thermal conditions and affinity
also differ. The small difference between the last two profiles does not
establish a copy-reduction speedup.

Native component admission compared 70 cases per rank, 420 total, byte for byte
against the original residual arithmetic. Serving receipts are in the
`mtp2-table-lifetime`, `mtp2-table-native`, `mtp2-draft-stream-order` and
`mtp2-table-copy-budget` directories. The independent table ownership is the
root-cause repair; the earlier boundary and copy experiments alone did not
resolve the stall.

## Compact greedy draft selection

A Qwen-only candidate projects local LM-head logits, masks padding, and gathers
one FP32 score/token-ID pair per rank. Shard offsets and token IDs remain exact
through INT32 and FP32 within the bounded vocabulary; rank/local ties preserve
the original first-index argmax. Probabilistic draft sampling, vocabulary
remapping, added embeddings and separate LM-head parallelism retain the full
logits path. Target sampling is unchanged.

The first hot-swap did not exercise the new method because `_runnable` retained
a startup-bound function. A corrected caller resolves the replacement method;
all six workers then matched 18 live local-reduction results each against the
full-vocabulary argmax. The original compact path and the variant avoiding a
direct FP32-to-INT64 cast passed the complete serving/cache-isolation gates.
Their six-request medians were 28.195 and 28.608 tok/s, respectively. They reduce
logit transport but do not establish a throughput improvement over the earlier
stream-order result. Receipts are in `mtp2-local-argmax` and
`mtp2-local-argmax-corecast`.

## Three-draft comparison

Three drafts, with captures `[4, 24]`, passed text, code, fresh/cached images,
mixed prefills, six simultaneous 512-token decoders, long cold/cached prompts,
and all 18 arithmetic answers. The short-decode median was 27.762 tok/s.
Six-decoder rates were 13.05–14.76 tok/s; the planner reported 1,581,123 cache
tokens and 6.03 times 256K concurrency. The extra proposal cost did not improve
throughput, so the selected depth returns to two. Receipts are in `mtp3`.

## Empty target-update candidate

The 310P runner visits an update stream before FULL target replay even when
Qwen's fixed QSA/GDN capture has registered no mutable attention tasks. A
candidate skips that empty round trip and uses replay-stream ordering instead
of host completion. It requires Qwen MTP, synchronous scheduling, DCP1, no
ENPU, 310P hardware, and an initialized empty task registry with matching capture keys.
Missing registries or any populated task/parameter list retain the existing
update and completion boundaries. CPU regression tests exercise both sides of
those decisions. All six real-weight workers confirmed eligibility with zero handles and zero
parameter records. The candidate passed the complete serving gates, including
six simultaneous decoders, mixed prefills, images and cached arithmetic. Its
first six-request median was 29.118 tok/s. A second set on the final source
reached 28.992 tok/s. Both exceed the 28.893 target; the combined twelve-request
median is 29.066 tok/s, 50.9% above the recorded two-card reference. Cold
23,410-token TTFT was 56.909 seconds and its cached repeat 1.885 seconds.
Receipts are in `mtp2-empty-target-updates` and `selected-profile`.

The runner change is necessary because that class owns the update-stream
round trip; an operator patch cannot remove it. The graph wrapper separately
keeps the host boundary for actual mutable task updates. No environment
variable or global mutable state is introduced.

## Failed qualification and root-cause limits

On the all-row native HC residual profile, six clients overlapped one decoding
request with five cold 2,048-token prompts. The engine initially reported two
running and four waiting requests, then stopped making progress. Worker stacks
waited in the draft-token CPU-copy completion event. The shared-memory executor
later warned that no broadcast block was available for 60 seconds. Temperatures
were below the thermal hold threshold.

Kernel driver logs at 21:00:03 also reported `Va is not alloced` in
`devmm_chan_check_va_d2h_process` for workers 1050355 and 1050358. The addresses
and timestamps are preserved in [mixed-driver-errors.txt](mixed-driver-errors.txt).
This supports investigating invalid addresses and buffer lifetimes on the
device-to-host path; it does not identify which tensor or allocation was invalid.

This initial evidence identified only the completion boundary. Subsequent
producer-stream isolation, described below, located an earlier QSA gather fault
and led to the persistent-table repair. Reducing draft depth was a recovery
experiment, not the root-cause fix. HTTP `/health` remained successful during the
stall, so qualification requires actual completions and engine RPC responses.

Device-position and shared-position-cache hot-swap experiments also stalled.
Frozen source and stack evidence are retained here for offline investigation;
those candidates are excluded from the recovery profile. The initial all-row native
dispatch API was reverted after mixed-load failure; its replacement was later
qualified with the persistent-table repair.

## Serial measurements from rejected profiles

These numbers are observations, not a production recommendation:

| Profile | Median short decode tok/s | Mixed-load qualification |
| --- | ---: | --- |
| Repaired MTP2 graph | 25.136 | Not established independently |
| Repaired MTP2 direct draft | 23.710 | Not established independently |
| All-row native, first restart | 25.981 | Not established |
| All-row native, final restart | 26.236 | Failed: worker event wait |

The final all-row native profile completed a cold 23,410-token prompt with
56.864 seconds to first token and 26.837 decode tok/s. Repeating the prompt used
23,296 cached tokens, took 1.880 seconds to first token, and decoded at 29.084
tok/s. The historical same-length cold prefill was 59.688 seconds. Prompt headers
and thermal conditions differ; this is not a controlled speedup measurement.

The older TP4 short-decode reference is 19.262 tok/s. Its 50% improvement target
is 28.893 tok/s on the matching short-decode workload. A cached long-prompt
measurement above that threshold does not meet the target.

## Draft-depth strategy

TP6 distributes each token's model computation across six chips. It does not
provide six independent future tokens: this MTP implementation recursively
generates proposals, then the target verifies the proposed sequence together.
Draft depth should be selected using accepted output per unit of elapsed time,
including proposal, verification, rejection, and graph padding costs. Larger
depths also consume more verification rows per request.

The fallback uses one speculative token and graph buckets `[2, 12]`. The
qualified repaired profiles use two speculative tokens and buckets `[3, 18]`.
Six drafts are a valid later benchmark candidate, after depth one and two pass
correctness and mixed-load gates. Chip count alone does not determine depth.
See the [primary speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html).

## One-draft recovery results

The baseline operators, with optional native candidates excluded, passed:

- Marker, arithmetic, Python function, and fresh/cached 1,024-pixel image probes.
- One decoder overlapping five cold 2,048-token prompts. All clients completed;
  serial decode was 23.77 tok/s, mixed decode was 12.81 tok/s, and the maximum
  observed output-chunk gap after the cold prompts arrived was 0.71 seconds.
- Six simultaneous decoders, each producing 512 tokens. Metrics confirmed six
  running and zero waiting; individual decode rates were 12.92–13.71 tok/s.
  Steady aggregate logger intervals reported approximately 78–84 tok/s.
- Six standardized single-request, 256-output-token measurements: median
  23.365 tok/s. This is below the earlier repaired MTP2 serial measurement.
- A cold 23,410-token prompt: 62.353 seconds to first token, 20.814 decode tok/s.
  Its cached repeat reused 23,296 tokens, reached first token in 1.824 seconds,
  and decoded at 21.777 tok/s.

The cache planner reported 1,661,434 tokens and 6.34 times concurrency for a
262,144-token request. Six short concurrent sessions are hardware-tested; six
fully occupied 256K contexts are planner capacity, not a completed stress test.
The one-draft profile is a validated recovery option, not the selected optimum.
The repaired two-draft baseline and subsequent native candidates are now
qualified separately, as recorded above.

Receipts and launch commands are in [mtp1](mtp1/). The launcher retains the
94°C hold, all-six-at-or-below-85°C resume, and independent 96°C cutoff.

## Two-draft readback diagnosis

The repaired two-draft baseline reproduced the mixed-load event wait and
unallocated-address errors with every optional native candidate excluded.
Those receipts are in [mtp2-baseline-failure](mtp2-baseline-failure/). An
experiment retaining QSA eager position destinations also failed the mixed
case and was rolled back; it is not part of the proposed fix.

The draft producer returned a transposed `[request, draft]` view. With more
than one request and draft, that source is noncontiguous. The runner posts
its CPU copy on a separate stream and waits for the copy's event before
returning draft IDs to the scheduler.

The installed `torch_npu` is `2.13.0.rc1`, source revision
`462f4e18a8303698aca3997edfcc5dd3b1707968`. Its
[device-to-host copy implementation](https://github.com/Ascend/pytorch/blob/462f4e18a8303698aca3997edfcc5dd3b1707968/torch_npu/csrc/aten/common/CopyKernel.cpp#L207)
materializes a temporary contiguous source for a strided input, then queues
the device-to-host transfer. A temporary-buffer
lifetime failure in that path was investigated. Packing the source on the
producer stream did not resolve the serving stall, so the conversion path
has not been established as its cause. The observed invalid address has
not been matched to an allocation by a backend allocator trace.

Small standalone copies for one, two, and six rows produced correct values
with both layouts. The strided process timed out during cleanup (exit 124);
the packed control completed and exited normally (exit 0). A standalone
per-copy failure is therefore not claimed. The regression workload remains
the full server's overlapping prefill and decode.

The narrow candidate packs on the producer stream and returns that tensor to
the runner, which retains it until the CPU completion event is consumed.
The maximum packed result for six requests and two drafts is 96 bytes. This
does not change token values, dtype, sampling, or proposal depth. It passed serial text and image requests but failed the same mixed-load
qualification. The candidate is not a fix for the serving stall. Diagnostic
barriers around the producer stream and CPU readback were used to isolate
the fault; they are instrumentation, not production optimizations.

## Producer-stream fault isolation

With a producer-stream barrier before posting each draft readback, earlier
copies completed. The failure then surfaced at that producer barrier, before
the current readback was submitted. The device runtime reported error 507011
and named `QsaGatherValueNzV310` as the faulting AI Core kernel. The device log
and diagnostic boundaries are in [producer-diagnosis](producer-diagnosis/).
The CPU-copy wait was a downstream symptom in this reproduction.

Later draft block tables were cloned/padded while building metadata outside
capture. A captured gather therefore could retain a temporary table address
without owning it or refreshing its pages on replay. A candidate now owns
independent table storage for each later Qwen draft step, sized for the largest
configured capture bucket, and updates it before execution. CPU regressions
verify lifetime after the source metadata is discarded, stable addresses
across buckets, fresh pages, cleared padding, and independence between steps.
The candidate passed real-weight two-draft mixed serving, six simultaneous
512-token decoders, text/math/code and fresh/cached 1024×1024 images, a cold
23,410-token prompt and its cached repeat, and 18 exact arithmetic answers
across six sessions with cached repeats. The six-sample short-decode median
was 27.502 tok/s; cold long TTFT was 62.390 seconds and cached TTFT 1.963
seconds. Receipts are in [mtp2-table-lifetime](mtp2-table-lifetime/).

The first isolation prompt asked for a long session marker and included an
instruction to ignore background text. A request returned a refusal; that
failed probe is retained rather than counted as a pass. Neutral arithmetic
requests then passed all 18 exact answers; repeats reused 2,560 tokens each.

A direct-draft comparison failed with a separate shape mismatch: six actual
attention rows were combined with an 18-row residual. It is not a successful
control for excluding graph-specific gather failures.

## Offline validation

The eleven scoped CPU suites passed: 293 tests, 20 warnings. Compact draft
argmax coverage includes padding, ties, large IDs, six shards, fallback modes,
and first/later caller geometry. Scoped Ruff checks
are recorded with the final delivery. Ordinary unit-test collection is blocked
locally by missing upstream `vllm.third_party.flash_linear_attention`.
Repository-wide `format.sh ci` failed on existing unrelated lint and spelling
issues; incidental unrelated edits were restored. No full-suite pass is claimed.

Private Kilo prompts, source, and model responses are excluded from this report.
The original long-context gibberish incident was not reproduced on the old
profile; its root cause remains unknown.
