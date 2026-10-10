# Live Qwen transfer audit and offline prefill fairness fix

The six-chip max-context server was first audited read-only. After the user
authorized cutover, the paced scheduler was deployed and passed real-weight
text, image, prefix/cancellation, and mixed-request checks. Checkpoint
spill/restore thrashing is absent in the available counters. Thermal holds and
long synchronous prefill steps explain much of the earlier latency. A short
physical trace now measures DDR bandwidth and copy task time; causal thermal
attribution and per-copy direction/payload bytes remain incomplete.

## Evidence and limits

The audited original runtime was
`/srv/ai/src/qwen-performance-mtp-tp6-v2-20261010-8fce905`, API PID 39853,
workers 43943–43948, model `qwen38-flash-next`. It retains six 262,144-token
slots, TP6/EP6, MTP2, graphs `[3,18]`, FP16 PLE, and image processing up to
1,048,576 pixels. The active scheduler is `PrefixMambaBoundedScheduler`.
See [launch configuration](launch-config.json).

The two [first](interactive-transfer-status.json) and
[second](interactive-transfer-status-repeat.json) status snapshots were taken
at 12:40:47 and 12:43:28 UTC on October 10. Their counters are cumulative
since worker startup, not measurements limited to those three minutes.
Every rank and all three state groups report zero spills, restores, host
checkpoints, and device archive hits. Each group retains 32 checkpoints.
The primary pool has 104 usable slots; archive capacity varies by rank.

| Rank | Archive slots | Recorded barrier calls | Total barrier host seconds |
| --- | ---: | ---: | ---: |
| 0 | 82 | 668 | 0.676 |
| 1 | 61 | 668 | 0.701 |
| 2 | 99 | 668 | 0.669 |
| 3 | 60 | 668 | 0.685 |
| 4 | 82 | 668 | 0.691 |
| 5 | 95 | 668 | 0.713 |

The prefix ledgers record zero submitted H2D, D2H, and D2D checkpoint bytes.
They continue counting with detailed events disabled. These are logical
payloads and host barrier durations; do not sum parallel ranks into wall time.
They exclude model kernels' cache reads/writes, attention CoW outside the
tiers, graph-internal copies, PLE rows, and most runner metadata. The runner
ledger is null and module ledgers are empty: their opt-in diagnostic candidate
is not active. This is **not evidence of zero total memory traffic**.

[Collected logs, thermal samples, and source hashes](live-audit.json) and the
[derived summary](summary.json) preserve the evidence. Seven audited source
files match this worktree byte for byte. The live prefix-state file differs
only by lacking the later paced-scheduler string recognition; that difference
does not alter the active bounded scheduler's accounting. No spill/restore or
ACL graph fallback warnings were found in the collected server log.

## What accounts for the slow requests

A thermal hold from 12:29:34.464 to 12:31:01.625 lasted 87.161 seconds.
It overlaps approximately 87 seconds of each pasted request's decode interval:
102.682 seconds for `chatcmpl-a147d42cf1f7387a` and 173.211 seconds for
`chatcmpl-9496cc656022b22e`. Request log boundaries have one-second precision.
Their reported 1.8 and 1.5 tok/s include cooling time.

After resumption, generation briefly reached 21.1 tok/s with one running
request, then fell to 0.2–0.4 tok/s with two requests as cache use increased.
Completed long prefills achieved roughly 300–330 computed tok/s. With the
active 2,560-token budget, an individual mixed step can take about eight
seconds. A decoding request advances only between synchronous steps.
This is an evidence-supported scheduling explanation, not a per-kernel trace.
Prompt throughput may read zero until a prefill finishes; the later prompt
counter burst is not an instantaneous thousands-of-tokens-per-second rate.

The controller also logged a pause RPC timeout at 12:34:10, with pause ownership
set, and a resume at 12:35:41. The start acknowledgment is ambiguous, so that
interval is excluded from the exact hold calculation. A later read-only check
found the server unpaused, zero running/waiting requests and preemptions, and a
59°C maximum. The server was left unchanged.

## Transfer and barrier inventory

| Area | Current behavior and audit priority |
| --- | --- |
| Route-count temporary | The live target uses the default `compare` count mode. At 2,560 tokens, ten routes and 86 local experts, it materializes a 25,600 × 86 comparison result: about 2.10 MiB before reduction, per MoE layer and rank. The existing fixed-output histogram candidate avoids this matrix. Prioritize a same-shape parity/performance A/B; do not assume its NPU histogram kernel is faster. |
| PLE host rows | `ple_layer.py:gather_embeddings` and `model.py:_PLEInjection.forward` gather demand-paged FP16 rows and upload them on every invocation, including graph replay callbacks. The 2,560-element embedding implies 5,120 logical bytes per scheduled token per rank, or 12.5 MiB per 2,560-token invocation. This is a source-derived estimate, not measured bus traffic. CPU hash inputs avoid an ID readback in the normal path. Measure row payloads, page-fault time, staging, and callback wait separately before changing transport. |
| Runner metadata | `_310p/model_runner_310p.py:_prepare_inputs` uploads positions, sequence lengths, query boundaries, slot/block tables, accepted-token information, and PLE history. Some capacity-sized copies remain. Measure payload sizes and dependencies before coalescing or shrinking them. Preserve stable capture addresses and request ownership. |
| Sampling and MTP | Sampled-token delivery requires host results. Accepted-token event waits are explicit. Async computed-token round trips are conditional; the live scheduler is synchronous. `moe.py` has a local-route count `.item()` for large W8 grouped batches; distinguish that conditional draft path from the target W4 path. |
| Target W4 routing | `w4_moe.py:forward` selects device routing or grouped routing for this backend. Its host `.cpu().tolist()` loop is a compatibility fallback, not the selected normal route. Grouped activation packing, permutations, temporary route buffers and finalization still generate device memory traffic. Trace branch selection and payloads. |
| Resident weights and state | Resident expert weights still travel through device DDR and on-chip memory during computation. GDN recurrent state and attention KV loads/stores occur inside kernels and are absent from host-copy ledgers. Profile these alongside explicit memcopies; residency does not imply zero bandwidth. |
| Collectives | Target and draft MoE, attention and logits use HCCL. Six-rank communication and waits need separate attribution. Host enqueue duration and logical tensor bytes are not measured wire traffic or collective completion time. |
| Prefix lifetime | Recorded layout/invalidation/retirement barriers are small here, and no checkpoints spilled. Attention CoW remains separately scoped. Retain ownership barriers until a trace and correctness gate prove a replacement safe. |
| Images | Cold vision encoding and input uploads are additional work; cached images differ. Small decoder chunks do not preempt a full encoder kernel or remove its compute/cache admission requirements. |

Temperature alone cannot identify an unnecessary transfer. The earlier v14
profile counted 13,415 `MEMCPY_ASYNC` tasks and 575 CPU H2D/16 D2H calls on
rank 0, but lacked payload bytes and used a different, non-MTP configuration.
It motivates a fresh trace; it cannot establish current bandwidth or heating.

## Offline scheduling change

`QwenPrefillPacedScheduler` now supports an opt-in `decode_only_steps` cadence
after mixed prefill work. It orders decoders first and reserves their full
speculative token demand. Decode-only turns defer both running and waiting
local prefills through the qualified upstream scheduler, including when its
DP capacity override would otherwise admit them. Configuration, capacity flag,
and surviving FCFS order are restored after scheduling, including exceptions.
Only a matching completed step advances the cadence.

Image-bearing requests no longer disable pacing for their entire lifetime.
When decoder image chunking is allowed, the normal paced budget applies;
upstream encoder admission remains unchanged. When atomic image spans are
required, only an overlapping span widens that mixed step, within the original
global budget. Consumed/future images do not bypass pacing, and widening one
span does not recursively admit later images. Oversized atomic spans still
require sufficient configured token/encoder budgets.

Pure prefill retains the original throughput budget. Defaults retain the prior
zero decode-only cadence. No request tokens are removed, checkpoint ownership
is unchanged, and the 94°C hold / all-chip 85°C resume policy remains intact.
This improves scheduling opportunities; it does not establish lower total
memory bytes, lower temperature, or a hard inter-token latency bound. Giving
decoders more turns can increase concurrent cold-prefill completion time.

The isolated CPU suite passed **105 tests**. It exercises the real subclass
against a scheduling boundary stub, pacing policy, image spans, MTP reservation,
cadence progress, empty/repeated outputs, and exception restoration. It also
covers strict Ascend configuration registration and exact histogram parity
at the six-chip 2,560-token/85–86-expert/ten-route geometry. The focused CPU
scheduling stub alone does not qualify real KV allocation or NPU execution;
the live checks below supply a bounded integration gate.
The standard UT invocation failed during parent conftest import because this
host lacks `vllm.third_party.flash_linear_attention`; dependency versions were
not changed. See the test receipts in this directory.

## Authorized live cutover and checks

The previous service had zero running/waiting requests and a 57°C maximum.
Its engine was drained and stopped once: worker hot-swap RPCs cannot replace
the engine's scheduler class. A complete copy of the qualified runtime plus
the pacing/configuration modules was staged at
`/srv/ai/src/qwen-performance-paced-tp6-20261010`. The first startup attempt
rejected the unregistered `qwen_prefill_pacing` key before loading weights.
The new strict `AscendConfig` field and validator correct that configuration
gap; its regression tests reject unknown options and invalid cadence/budgets.

The retry is serving as API PID 440842, engine 443387, workers 444210–444215.
[Cutover receipt](cutover.json), [launcher](start-server.sh),
[full command](start-paced-tp6.sh), and
[post-cutover audit](post-cutover-audit.json) preserve provenance.
The capacity planner reports **1,607,731 cache tokens / 6.13×256K**. Six
slots, images, MTP2, graphs `[3,18]`, precision, affinity, the 4-GiB reserve,
and 94/85 control remain configured. This is planner capacity, not a full
six-window stress or quality test. The old max-context launcher remains
available for rollback.

Real-weight checks passed:

- Arithmetic, sorting, capital, and fresh/cached image checks.
- Fresh 1024×1024 image OCR/shape identification: 3.967 seconds; cached repeat:
  1.070 seconds. Both returned the expected answer.
- All six prefix branches reused 3,840 tokens and returned correct answers;
  cancellation drained, graphs remained clean, and all ranks had zero
  checkpoint spills/restores.
- During a concurrent 2,271-token cold prefill, the 384-token decoder achieved
  **17.4 tok/s** in server timing. SSE content-event gaps were **87 ms median,
  638 ms p95, 674 ms maximum**, with a **71°C** peak and no thermal hold.
  One SSE event can contain multiple accepted tokens. The concurrent prefill
  took 23.316 seconds to its first content event: improved decode fairness
  deliberately trades concurrent prefill latency. This is one controlled pair,
  not a matched baseline speedup or a 50% throughput qualification.

See [mixed probe](mixed-decode-probe.py) and
[raw receipt](mixed-decode-result.json), plus the image/prefix receipts here.
The selected profile uses 128-token initial/minimum mixed chunks, a 200-ms
best-effort target, a 640-token adaptive maximum, and eight decode-only turns.
Pure prefill keeps the original 2,560-token budget. The minimum chunk can
exceed the target duration; it is not a latency SLA.

## Physical transfer trace on the new runtime

All six ranks started and stopped profiling successfully. The bounded request
contained a 456-token cold text prefill and 12 generated tokens, and took
2.562 seconds with profiling enabled. The server was resumed afterward;
parameter-storage digests were unchanged, graphs remained clean, and no
profiler is left active. The projection candidate remains baseline, while
the paced scheduler is active.

Raw captures total about 705 MiB and remain under
`/srv/ai/src/qwen-prefill-fairness-20261010/transfer-profile-v16`.
All six ranks were parsed offline, with database fingerprints in the
[all-rank summary](all-rank-transfer-summary.json). Each parser reported
40 incomplete memory records; copy direction and payload sizes were not
exported in the task tables. Ranks 0–3 report similar mean DDR rates.
Ranks 4–5 contain implausible DDR counter values, so their bandwidth
summaries are excluded rather than averaged into a claimed measurement.
Copy counts remain similar across all six ranks (8,770–8,818 tasks).

Measured rank-0 results over this mixed trace:

| Measurement | Result |
| --- | ---: |
| DDR samples / sampled span | 133 / 2.625 s |
| DDR read sample mean / peak | 49.4 / 95.0 GB/s |
| DDR write sample mean / peak | 21.9 / 58.4 GB/s |
| Explicit asynchronous copy tasks | 8,787 |
| Summed copy task duration | 70.281 ms |
| Median / maximum copy task duration | 2.318 / 241.875 µs |
| All-reduce operator calls / summed duration | 213 / 500.459 ms |
| W4 INT4 matmul calls / summed duration | 528 / 315.348 ms |
| Cast tasks / summed duration | 18,423 / 126.188 ms |
| Layout conversion tasks / summed duration | 12,248 / 73.914 ms |

These values confirm substantial device memory traffic and many small layout,
cast and copy tasks. Summed durations are **not exclusive wall time**: streams,
communication and ranks overlap, and profiled timing is perturbed. DDR traffic
includes required weights/state and temporary tensors; it does not isolate
CPU link traffic or prove a transfer is unnecessary. Prioritize attribution of
cast/layout chains and collective waits, the route-count matrix, and PLE host
staging. Do not remove ownership barriers based only on counts or temperature.

## Remaining qualification

[Profile and remaining gates](queued-tests.json) do not start jobs automatically.
The smaller mixed chunks and eight-step cadence are now selected. Compare
baseline and candidate using identical models, affinity, cache
state, requests, images and temperature. Measure SSE token-gap percentiles,
cold TTFT, aggregate decode throughput, prefill completion time, and cooling
wall time independently.

Extend the bounded trace to resolve target/draft/HCCL and PLE host callbacks,
copy direction/payload bytes, and recurrent/KV kernel memory access separately.
Run one cold prefill, one cached continuation, and a mixed pair separately;
avoid repeated full-window loads. Retain 94/85 control and abort controlled
tests at 90°C. Do not enable debug launch blocking for performance runs.
Require image quality, six-prefix CoW/cancellation gates, zero checkpoint
spills, and bounded trace output. No benchmark or candidate activates itself.

The original +50% decode target and lower cold TTFT remain unqualified.
