# Six-chip Qwen hardware qualification

Test host: `matteius@192.168.53.187`, three Ascend 310P cards, six chips.
Source base: `f25e70c0943681c76ca253a973bb031256405ea1`, with the GDN
corrections delivered alongside this report. Evidence includes failed attempts;
process exit, token-budget completion, and startup are not quality gates.

## Findings and corrections

1. The FwdO ping-pong scheduler added the second lane to a base head index.
   With nine value heads, that lane can cross a chunk boundary and read head
   nine instead of head zero of the next chunk. Decode each complete linear
   task, and drain the previous valid stage without scheduling an exhausted
   lane. Both architecture scheduler implementations were corrected; arch22
   has host scheduler coverage only, not hardware qualification.
2. The 310P FwdH scheduler paired adjacent heads across an odd head boundary.
   Nine-head shards now schedule one trained head per core, with a no-work
   second lane. Six- and twelve-head geometry retains paired scheduling.
3. Causal-mask vector writes started at unaligned UB column addresses. The
   vector instruction rounded the address down, overwriting valid causal
   entries. Aligned row addresses and explicit lane masks preserve the exact
   upper triangle. Fixed both mask initialization and the epilogue tail.
4. Every FwdH core loaded, converted, and stored every head's initial state.
   For the 128-wide state, only the scheduled consumer initializes its head.
   No global barrier was added. Other state widths retain legacy behavior.
   This removes redundant logical copies; hardware bus-byte and thermal
   improvements have not been measured independently.
5. The real-weight layer benchmark assumed shared-expert divisibility by TP
   size and copied parameters outside inference mode. It now supports uneven
   six-chip shared slices and replicated shared experts under inference mode.
   The capacity probe accepts any positive concurrency, including six.

A partial OPP package can register an operator whose binary it does not
contain. The output-only build passed prefill math but shadowed the convolution
binary lookup. The final isolated `qwen_gdn_unified_v5` package builds
`chunk_fwd_o_vllm`, `chunk_gated_delta_rule_fwd_h`, `causal_conv1d_v310`, and
`recurrent_gated_delta_rule_v310` together. The convolution ABI requires the
matching newly built Python binding. Neither the baseline vendor nor its
binding was overwritten. Earlier graph harness runs also compared capture
outputs before explicitly replaying the graph. The corrected harness checks
both the first replay and a replay after changing inputs; those early failures
remain in the evidence archive.

## Kernel qualification

The final unified package passed all **42 hardware checks**:

- Eighteen constant-input FwdO cases inspect every output row, at 6/9/12 value
  heads, 64/128/2,560 tokens, and batches one/two.
- Nine full GDN cases compare both output and warm final state against an
  independent CPU reference: 4/12, 3/9, and 2/6 key/value heads; 128 and 2,560
  tokens; and variable lengths 128/192. Relative L2 error must be below 0.001.
- Eight compact convolution/recurrent state checks cover ranks zero/five,
  untouched cache slots, eager execution, and changed-input graph replay.
- Seven streaming projection checks cover parity, empty expert peers,
  uneven expert boundaries, bounded down windows, and graph replay.

The initial nine-head 2,560-token case produced NaNs. The corrected prefill
output relative L2 is about 0.0005 and final-state relative L2 about 0.00035
in the separate diagnostic run. The formal suite uses the same strict
thresholds for every geometry; tolerances were not relaxed.

Three host C++ tests compile the actual scheduler headers and check addresses,
complete task coverage, odd-head boundaries, pipeline draining, variable
lengths, and unique initial-state ownership. Batch counts include 1/2/3/4/6.
An additional 61 CPU tests cover the layer loader, capacity client, six-chip
partitioning, and transfer accounting.

## Projection performance

Real checkpoint weights, TP6 rank four, seeds 1024/1025, ABBA ordering, three
repeats, with bitwise output parity:

| Tokens | Baseline projection | Streaming v2 | Result |
| --- | --- | --- | --- |
| 128, seed 1024 | 4.371 ms | 5.206 ms | Streaming slower |
| 128, seed 1025 | 4.543 ms | 5.410 ms | Streaming slower |
| 2,560, seed 1024 | 38.438 ms | 38.426 ms | Essentially tied |
| 2,560, seed 1025 | 37.813 ms | 37.822 ms | Essentially tied |

These timings cover routed projections, not shared experts, HCCL, or the
complete model. Keep the baseline projection. The streaming implementation
needs lower launch/control overhead before its smaller tile transfers produce
an end-to-end benefit.

A second comparison used the weights already loaded in all six serving workers,
with no restart, weight replacement, or production-forward change. The v11
producer skips two UB clears and one extra V-to-MTE2 event pair only for full
32-row tiles, whose bytes are completely overwritten by operand DMA. Partial
rows retain padding clears. Fifty host checks, including alternating full/tail
expert tiles, pass. Both streaming variants matched baseline outputs bitwise on
each rank and both seeds. Six-case eager native parity also passed per variant.

Means of the two seed-specific medians below are descriptive summaries of three
ABCCBA cycles, not whole-model throughput or confidence intervals:

| Rank | 128 baseline / v10 / v11 (ms) | 2,560 baseline / v10 / v11 (ms) |
| --- | --- | --- |
| 0 | 5.363 / 6.112 / 6.069 | 47.434 / 52.525 / 52.458 |
| 1 | 4.932 / 5.566 / 5.637 | 42.236 / 43.797 / 43.773 |
| 2 | 4.163 / 4.873 / 4.929 | 37.032 / 36.707 / 36.645 |
| 3 | 4.515 / 5.359 / 5.317 | 40.499 / 42.007 / 41.886 |
| 4 | 4.706 / 5.640 / 5.596 | 38.536 / 39.316 / 39.155 |
| 5 | 4.298 / 5.037 / 5.023 | 37.484 / 37.541 / 37.461 |

The full-tile change is not a demonstrated performance improvement. The v11
candidate remains experimental and was not selected for serving. Its frozen
candidate digest is `cab2a1df2fa957d442ee9a8ddd1e32e593afc9f1cbec8f18433f415dfef51538`;
the namespace is `qwen_streaming_v11`. Weight storage digests, baseline forward
selection, and graph state were preserved across the loaded-worker experiment.
Appended native registrations remain resident until process exit.

## Stage attribution and next changes

A separate diagnostic timed baseline, two-window streaming, and single-call
full-output streaming on the same loaded layer and routes. All three matched
bitwise on six ranks. Event intervals were collected separately from the
ABCCBA wall timings; they include submission gaps and are not additive physical
pipeline counters. Profiler collection also ran separately from those timings.

At 2,560 tokens, rank zero owned 6,169 local routes, compared with
4,614/3,313/4,233/3,827/3,444 on ranks one through five. Synthetic router load
imbalance explains part of the rank skew; these counts are not evidence about
hardware bandwidth or representative layer-wide routing in normal service.

| Rank-zero stage | Baseline (ms) | Streaming v11 (ms) |
| --- | --- | --- |
| Device dispatch | 2.580 | 2.617 |
| Input packing | 1.359 | 1.359 |
| Packed route gather | 0.578 | 1.063 |
| Gate/up | 24.654 | 25.368 |
| Builtin SwiGLU | 0.938 | 0.938 |
| Hidden packing | 3.300 | 3.301 |
| Down | 12.106 | 7.683 + 7.677 |
| Finalization | 1.124 | 0.795 + 0.795 |
| Output-window copies | None | 0.120 + 0.122 |

Removing windowing in the diagnostic improved rank-zero streaming wall time
from 52.079 to 50.752 ms, still behind baseline's 46.927 ms. Single-call full
streaming beat baseline slightly on ranks two/four/five, but remained slower
on the critical rank. It also restores full routed-output allocation; it is
not admitted as the bounded serving candidate. At 128 tokens all streaming
variants remained slower. This is grouped prefill evidence, not small routed
decode evidence.

The rank-zero device trace recorded 39/45/37 tasks for baseline/windowed/full
streaming, including event markers. Windowing added a second finalizer, two
view-copy kernels, and repeated casts. No explicit DMA-copy tasks appear in
this isolated partial. This does not measure kernel-internal MTE traffic or
exclude transfers in the complete serving pipeline. `HostToDevice` flow labels
in the Chrome trace are dependency arrows, not copied-byte measurements.
The worker is a daemon, so inline parsing was rejected; the preserved raw data
was successfully parsed by a separate offline process. The stage controller's
final display raised a missing-`seed` error after all checks and server resume;
its saved result and after-state receipts contain six passing worker responses.

Prioritize these follow-ups:

1. Optimize the down-projection schedule using actual tile/route geometry.
   Single-call down is still 14.717 versus 12.106 ms on rank zero; output copies
   alone do not account for the regression. Prove slot lifetimes before removing
   producer/Cube/vector fences.
2. Keep baseline gather where the native local gather costs more. Compare
   complete gather plus projection costs and varied learned route distributions;
   avoid device readbacks for runtime selection.
3. Fuse bounded down finalization/store while preserving the qualified FP16
   route boundary and original FP32 addition order. Measure full memory lifetime
   and bus counters instead of assuming fewer tensors means less traffic.
4. Profile the distinct small routed decode path, including HCCL, shared expert,
   graph padding, FP16 LM head, and the absence of TP6 MTP. Grouped streaming
   changes cannot establish higher C1 decode throughput.
5. Continue cold-prefill GDN/WY attribution after the corrected math gate.
   Require paired whole-model cold TTFT and decode improvements, image checks,
   cache correctness, and sustained thermal behavior before promotion.

## Full-model qualification

The first TP6 attempt passed short text and cold/cached images, but generated
incoherent text on a cold 23,000-token prompt. Its capacity-client `passed`
field describes token-budget completion only. That attempt failed the quality
gate and the four-chip baseline was restored before continuing kernel work.

| Initial profile | Serial decode median | Cold 23K TTFT | Quality |
| --- | --- | --- | --- |
| Retained TP4, MTP2 | 19.262 tok/s | 61.438 s | Coherent |
| Initial TP6, no MTP | 18.838 tok/s | 61.906 s | Long prompt failed |

The initial TP6 planner reported 1,374,660 cache tokens, approximately 5.24
256K windows. Six request slots do not establish six simultaneous full 256K
contexts. Full-window capacity and long-context model accuracy need separate
validation. TP6 currently disables MTP and keeps the image encoder in data
mode. The live GDN head counts are 9/9/9/9/6/6; shared expert widths are
107/107/107/107/106/106. Uniform nine-head cache reservation remains in place.

The corrected TP6 build passed three short text checks, cold/cached images,
a coherent cold 23,000-token prompt, six simultaneous distinct image labels,
and six branches of a cached prefix. Every branch reused 3,840 cached tokens
and returned the expected distinct answer. Closing an active streaming client
released its slot and returned running/waiting counts to zero.

The corrected 23K prompt had zero cached tokens and a **60.003-second TTFT**.
The retained TP4 value was 61.438 seconds. This is one cold-prompt sample per
profile with different serving configurations, not an established speedup.
The mixed prompt-length gate (1,024/1,536/2,048/2,560/3,072/4,096 tokens,
256 output tokens each) observed six running requests, **21.635 seconds of
common decode overlap**, and zero preemptions. End-to-end aggregate output was
26.819 tokens/second including prefill. A decode-only ten-second server logging
interval reported 69.6 tokens/second; this interval is not an end-to-end rate.
All six previews were coherent Python testing guides.

Serial decode median was **18.756 tokens/second**, compared with 19.262 for
retained TP4/MTP2, about 2.6% lower. The configurations differ in TP size,
MTP, shared slicing, and LM-head execution, so this is a serving-profile
comparison, not an isolated kernel speed measurement. The six-chip profile
keeps baseline projections for its greater session capacity; streaming v2
remains disabled. Capacity and correctness qualification do not satisfy the
performance acceptance target: repeatably higher decode tokens/second and lower
cold-prefill time to first token. Neither profile should be called a completed
streaming performance optimization based on these measurements.

## Thermal protection and evidence

All-six-chip admission was monitored. Standalone kernels abort at 90°C.
Serving holds active work and queues new requests at 94°C, resumes only when
all six chips are at or below 85°C, and has an independent 96°C shutdown
watchdog. Thresholds were not weakened for a benchmark. Kernel qualification
peaked around 72°C. Later, overlapping external client traffic and the first
full-size image probe reached 94°C. The first pause RPC timed out; the controller
retried successfully, held requests with all AI cores idle, and resumed at 85°C
about 148 seconds after the successful hold. No watchdog shutdown occurred.
The image client timed out during the hold. After the user paused Kilo, a fresh
1024×1024 image passed both uncached and cached processing, taking about 7.4
and 6.0 seconds. These checks do not establish sustained thermal safety, and
pause time must remain included in user-visible request latency.

`validation/` contains compressed raw test, build, serving, thermal, benchmark,
and failure logs with SHA-256 receipts. `launch/` retains the isolated
launch/build scripts and a compressed helper archive. They use host-specific
paths and are not replacements for the production launcher. Package receipts bind the tested source, binary, and Python binding.

The required repository-wide `bash format.sh ci` was run and failed on
existing unrelated lint/format issues. The raw failure log is retained.
Unrelated auto-format changes were restored only in the isolated worktree;
the user's shared checkout was left untouched. All task-scoped automatic hooks and the manual Markdown check passed;
compressed logs are included with the final delivery evidence. The final receipt
records six healthy workers, baseline projections, clean graph state, zero
checkpoint spills/restores, and idle admission resumed after the diagnostic.
Text and fresh/cached image smoke passed again after native registration.
