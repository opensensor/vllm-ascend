# Six-chip Qwen MTP and HC qualification

The October 10 test server on `matteius@192.168.53.187:8001` was available as
`qwen38-flash-next`, with TP6/EP6, MTP2, image processing up to 1,048,576
pixels, six request slots, 262,144-token maximum context, and target graph
sizes `[3,18]`. Its selected resident candidate is `fused_native_hc`.
Testing stopped when the operator reclaimed the server, then resumed with
explicit access. No cold-prefill request was submitted in this testing phase.
The later restart with a larger cache is documented separately below.

## Measured result

Identical three chat prompts, 256-token output caps, temperature zero, seed
42, thinking disabled, serial C1, and the same client timing definition:

| Variant | Median decode tok/s | Versus recorded TP4/MTP2 |
| --- | --- | --- |
| Older two-card control | 19.262 | Reference |
| Six-chip MTP2, baseline HC | 26.787 | +39.06% |
| Six-chip MTP2, native HC residual | 27.031 | +40.33% |
| Six-chip MTP2, joined HC projection + native residual | 27.193 | +41.18% |
| Joined HC/native residual, cached W8 PLE trial | 26.128 | +35.65% |
| Joined HC/native residual, FP16 PLE repeat | 26.929 | +39.80% |

The **50% target is not met**; its threshold for this recorded control is
28.893 tok/s. These are initial three-prompt measurements, not repeated paired
qualification or a claim of sustained throughput. The small differences
between HC variants do not establish a repeatable performance improvement.
Output hashes and acceptance rates vary. Client timing accounts for multiple
tokens per SSE chunk; authoritative request timings remain in the server log.
The restart exposed another comparison limitation: the historical MTP launcher's
affinity helper rejected the changed model path, so these workers did not receive
the intended six-core masks. The subsequent restart corrects and records those
masks. A fresh paired two-card control remains necessary.

The older control already uses dynamic-W8A8 LM-head and PLE projection.
The six-chip candidate uses W8 for the head and grouped draft experts, but its
PLE projection remains FP16. The checkpoint and client workloads match, while
runtime precision and draft execution differ; a fully matched precision A/B
is still required. The earlier TP4 cold 23K sample was 61.438 seconds; no new
cold-TTFT result is available for this MTP configuration.

The planner reports **1,173,285 total cache tokens**, approximately 4.48 full
256K contexts. Six request slots do not mean six simultaneous full 256K
contexts. No 256K or 1M quality/capacity test was run in this phase.

## Fixes and validation

- Explicit `mtp_uneven_sharding=true` supports 512 draft experts across six
  ranks with 86/86/85/85/85/85 experts, contiguous offsets, uneven shared
  channels, and vocabulary padding identical to the target. Legacy rejection
  of unsupported nondivisible configurations remains the default.
- The initial MTP attempt loaded real weights and captured graphs, then failed:
  the worker reserved two state windows while bounded retention required four
  plus retained checkpoints. A shared sizing rule now allocates 105 slots for
  six requests/MTP2, including the null slot, and retains 32 cached checkpoints.
  Worker allocation and scheduler retention select the same explicit policy.
- Both existing and new MTP constructor/loader tests passed: **32 CPU checks**
  in the qualified Python/NPU environment. Pool/scheduler checks passed **70**.
  Two stale runner test fixtures were fixed to call the current update helper.
- Eight hardware checks passed for speculative convolution/recurrent state
  views at the actual 9/6 head geometry, accepted-state selection, untouched
  storage, and changed-input graph replay. These are component checks.
- All six workers passed **196 real-HC-weight comparisons each** for joined
  projection plus native residual, at 3/18 rows, with bitwise equality.
  Inputs are synthetic; this is not a complete language-model accuracy gate.
- Real-weight arithmetic, sorting, capital, and fresh/cached image smoke passed.
  Six concurrent common-prefix branches returned the correct answers and each
  reused 3,840 tokens. Cancellation drained. Saved post-branch status reports
  clean graphs and zero checkpoint spills/restores on every rank.
- Fresh/cached 1,024 × 1,024 image processing and OCR passed. The first image
  harness expected the wrong fixture text and its cold answer exhausted the
  128-token cap before OCR. That failure is preserved. The corrected concise
  prompt with a 256-token cap read `FULL IMAGE` in both cases; the multimodal
  cache was cleared before retrying.

Controlled decode and prefix probes reached at most **87°C** and did not
trigger their 90°C abort or the serving thermal hold. The existing 94°C hold,
all-six 85°C resume, and independent 96°C cutoff remain active. An intervening
interactive thermal hold is in the continuous controller log. These short
checks do not qualify sustained thermal operation or prove reduced heat.

## Transfer evidence and controller robustness

A separate real API baseline request with 16 output tokens produced six
successfully parsed, untruncated traces. Dense matmuls, layout conversions,
event waits, casts, and copy tasks are prominent. Rank zero recorded 13,415
`MEMCPY_ASYNC` and 399 `MemcopyAsync` device tasks. CPU tracing also records
575 host-to-device and 16 device-to-host calls; enqueue/dequeue records are
not additional copies. Names alone do not expose payload bytes or prove
redundant transfers. The request includes prefill and profiler overhead;
overlapping task durations must not be added into wall latency.

An earlier, larger profiling attempt filled the root filesystem and produced
truncated data. It is excluded from attribution/performance results. Its own
data was moved to the data volume; no user files were removed. The thermal
controller had also exited when its log write raised `OSError`. It now keeps
enforcing hold/resume through full disks or broken log pipes; a regression
injects repeated write failures and verifies pause/resume/pause behavior.
All later logs and trace data use the data volume. Raw traces remain on the
host in `/srv/ai/src/qwen-performance-evidence-20261010/decode-profile-v14`;
the compact summary is archived here.

## Offline work after the operator handoff

The reversible `cached_ple_w8a8` candidate retains the FP16 parameter and
builds a separate INT8 weight/scale cache using the existing PLE W8 arithmetic.
Only preparation copies weights through CPU; cache hits require no host weight
transfer. Version changes require warmup before capture; cache release requires
cleared graphs. This changes precision. After renewed access, all six workers
passed six comparisons each against the existing W8 policy, including changed
inputs and graph replay. Original parameter pointers and versions remained
unchanged. Text, fresh/cached 1M-pixel images, six prefix branches, and cancellation
passed; saved branch receipts show clean graphs and zero spills/restores.
The W8 trial measured 26.128 tok/s median; the subsequent FP16 repeat measured
26.929. Neither establishes a new gain, and **FP16 PLE was restored**.
It adds about 31.25 MiB per 12,800 × 2,560 projection, plus scales, until released.

The bounded-pool recognizer now also accepts the built-in paced scheduler's
qualified string name; it previously recognized only its parent string or a
subclass object. This extra lookup correction has CPU coverage but was not
installed on the live runtime. The archived source manifest fingerprints the
hardware snapshot before this offline correction, not all final delivery bytes.

Final focused local checks: **86 passed**, plus **44 passed/27 skipped** for
state/scheduler coverage. The skips require the qualified fork's block-hash
API absent in the local vLLM version. Required repository-wide formatting was
run and retains unrelated baseline failures; scoped hooks are recorded.
No full-suite, GPQA, tool-call, cold-TTFT, or sustained thermal pass is claimed.

## Next controlled experiments

1. Continue paired repetitions with the PLE W8 trial available for isolation;
   its first measurement did not improve the retained FP16 serving path.
2. Attribute target replay, draft replay, host callbacks, and HCCL separately.
   Inspect drafter padding before proposing independent capture sizes: its
   first pass can consume verification tokens, so `[1,6]` cannot be assumed safe.
3. Measure the exact 23,410-token cold/repeat pair from a cool idle start.
4. Quantify unused archive allocation before reallocating memory to contexts.
   The current archive reserves roughly 7.37 GiB per worker, while bounded
   retention was sized to fit primary slots. No archive reclamation is shipped
   without scheduler ownership, allocation, graph, and long-history gates.
5. Run sustained C1/C2/C6 and mixed image/prefill trials after cooling work and
   renewed NPU access. Include thermal holds in service wall throughput.

Nothing in this report schedules an automatic server restart or test.

## Maximum-context restart, October 10 at 11:39 UTC

The operator requested the service again after reboot. All six chips were healthy;
device nodes had reverted to root-only access, so their group permissions were
restored for the existing `HwHiAiUser` group. No driver or ECC changes were made.

The retained runtime/model were started with six slots, 262,144-token maximum
context, images up to 1,048,576 pixels, MTP2, graphs `[3,18]`, FP16 PLE, and the
baseline projection dispatch. GPU utilization is 0.955 and KV fraction is 0.95.
The planner reports **1,607,858 tokens / 6.13 full 256K contexts**, compared with
1,173,285 / 4.48 previously. The archive shrank from 7.37 to 1.67 GiB while the
post-capture 4-GiB runtime reserve remains. This uses the existing allocation
policy; no graph-visible state layout or memory ownership was changed.

The old affinity helper rejected the MTP model path. A copied helper corrects
that identity check, verifies the API/engine/worker tree and process start times,
and records all worker-thread masks. The retained masks are six disjoint groups
of six cores. The restart wrapper now selects the corrected helper explicitly.

Startup completed with real weights and graph capture. Planner capacity is not
a sustained six-full-window quality or thermal qualification. Runtime evidence
is retained at `/srv/ai/src/qwen-max-context-20261010`; restart with the
[profile wrapper](launch/start-max-context.sh) only when authorized.

Startup probes passed text, cold/cached 1M-pixel images, six prefix branches,
and cancellation, with clean graphs and zero spills/restores. They peaked at
79°C. Subsequent interactive traffic reached 94°C on the last card and the
controller held requests with caches preserved. The first 23,014-token uncached
user request took 67.842 seconds for prompt computation and decoded at 17.3
tok/s; follow-ups decoded around 16–17 tok/s. These user workloads differ from
the three short benchmark prompts. This profile is **not sustained-thermally
qualified or demonstrated to match the fastest short-prompt result**. The
resident joined-HC/native-residual candidate was not reapplied after reboot;
it previously showed only a small, unconfirmed gain over baseline HC.
