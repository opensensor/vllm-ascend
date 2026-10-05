# Qwen TP4 cold-prefill named trace, 2026-10-05

The isolated 2,560-token native-W4 candidate was launched with the
[`--profiler-config` launcher](../start-batch2560-profile-prefill.sh). One
unique 23,410-token prompt requested one output token, reported zero cached
tokens, and had a 62.980-second TTFT under profiling. The recorder captured
four opening prefill scheduler iterations on all four 310P ranks. That TTFT is
diagnostic and must not be compared directly with an unprofiled service run.
The Qwen service was then stopped; the NPUs were released.

The complete raw and parsed captures remain in the isolated Threadripper
runtime at
`/srv/ai/src/qwen38-prefill-batch256-runtime-20261005/results/named-prefill-20261005`.
The four local rank CSVs and rank-1 host operator CSV were copied for offline
analysis and packed into [`selected-trace-csvs.tar.gz`](selected-trace-csvs.tar.gz)
(35 MB; extract in this directory to rerun the plots). [`summary.json`](summary.json)
contains named-kernel totals and shapes. The charts were made with
[`plot_trace.py`](../../../tools/qwen4exp/plot_trace.py).

## Read the pictures

- [Device occupancy](device-occupancy.png): whether *any* kernel is running
  in each 50 ms bin, across streams. This is the right view for device idle
  time.
- [Task timeline](task-timeline.png): each line is one task, separated into
  thin category lanes. The white space within lanes makes this look sparse
  even when another lane is active.
- [Operator mix](operator-mix.png) and [top kernels](top-kernels.png): summed
  device task durations. Concurrent tasks are counted separately.
- [Host operations](host-ops.png) and [copy-call tail](host-copy-tail.png):
  rank-1 host self time. These sums can include waits and overlap between
  threads; they are not an additive TTFT breakdown.

Across the 26.21-second captured device span, each rank has at least one
kernel running for 25.07–25.13 seconds (95.66–95.88%). The ranks share
roughly aligned no-kernel gaps near 6.9, 13.3, and 19.8 seconds, consistent
with boundaries between the four captured prefill chunks. The trace does not
identify their host cause. The largest gap on a rank is approximately
244–275 ms. There is about 1.08–1.14 seconds of total no-kernel time per
rank in the common span.

## Ranked work in the captured interval

| Device work per rank | Summed duration | Calls per rank | Interpretation |
| --- | ---: | ---: | --- |
| Native W4 projections | 5.46–5.97 s | 384 | Two projections in 48 MoE layers, across four chunks. |
| QSA K/V gather | 3.00–3.10 s | 4,168 | One operator handles both key and value; many small QSA tiles. |
| QSA tiled batch matmul | 1.89–1.90 s | 4,160 | QK and PV tiles. |
| `aclnnInplaceCopy_CastAiCore_Cast` | about 2.01 s | about 35,034 | Device conversion work, distinct from host API self time. |
| Native W4 activation pack | about 0.89 s | 384 | Already uses the built-in SwiGLU path in this service. |

The large `(2560,10240)` conversions contribute about **1.29 seconds** of
the rank-1 device cast total: 1,188 FP32→FP16 tasks take 0.966 seconds and
416 FP16→FP32 tasks take 0.323 seconds. The shape matches the model's
`hc_count * hidden_size` hyperconnection state. The model explicitly casts
that state in grouped RMSNorm and the mix/combine path; this is a source-level
match, not a proven one-to-one mapping from each profiled task to a line of
Python. The FP16 round trip is part of the current numerical contract, so
simply retaining FP32 state may change outputs and double that state traffic.

The rank-1 host operator table attributes **5.279 seconds** of self time to
40,379 `aclnnInplaceCopy` calls. The distribution is concentrated in a tail:

| Host call duration | Calls | Summed host self time |
| --- | ---: | ---: |
| Under 2 µs | 33,376 | 35.6 ms |
| 2–10 µs | 4,868 | 16.8 ms |
| 10–100 µs | 260 | 6.5 ms |
| 0.1–1 ms | 1,307 | 539.0 ms |
| 1–10 ms | 447 | 1,727.9 ms |
| At least 10 ms | 121 | 2,953.6 ms |

The same host table reports about 2.568 seconds of attributed device work
for `aclnnInplaceCopy`. Several 38–44 ms host calls have only microseconds
of attributed device copy work. Thus the 5.279 seconds cannot be interpreted
as host↔NPU transfer time or as time recoverable by removing copies. The
exported operator CSV has no call stacks, shapes, or timestamps for these
host calls. Its row order and neighboring `aten::copy_` entries suggest many
are Torch `.to()`/`copy_` paths, but do not pinpoint the cause of the long
stalls. The saved raw `trace_view.json` *does* have timestamps: matching its
121 calls above 10 ms to rank-1 device kernel intervals shows the device
running for **2,953.03 of their 2,953.60 combined milliseconds (99.98%)**.
Those calls are spread through the four chunks, not concentrated at the
boundaries. The observed host self time is therefore mostly overlapped with
device work; eliminating the host wait alone is unlikely to save that amount
of TTFT.

Rank 1's 4,168 QSA gather calls comprise 2,080 full-tile key gathers
(1.569 seconds), 2,080 full-tile value gathers (1.438 seconds), and eight
one-token calls. The kernel name is the same for both orientations. Its
reported median MTE2 time is about 235 µs, versus about 10 µs of vector time;
the hardware pipeline counters can overlap and should only guide where to
look. Native W4 matmul reports about 93% median Cube utilization in this
trace. These observations favor testing QSA address/layout and memory
traffic changes before assuming the main W4 math needs a new algorithm.

## Next controlled experiments

1. Inspect QSA gather's K and V address generation and layout, and whether
   its repeated per-tile selection conversions can be safely shared between
   the two calls. Group indices, tail counts, page mappings, and cache values
   change with the current request; they cannot be cached across tokens by
   shape alone. Reusing invariant arange vectors or converted metadata
   *within one call* is plausible but needs a measured device-task reduction
   and exact output parity. Fusing K/V could also remove repeated address
   work, but may forfeit existing stream overlap.
2. Correlate the large hyperconnection-state FP32↔FP16 casts with source
   operations before changing them. A fused norm/mix or combine path could
   avoid materializing some intermediates while preserving the current
   rounding boundaries. The slow host-copy tail is a clue to inspect, not
   a standalone 5.3-second target.
3. Compare `ascend_qsa_prefill.parallel_gather=false` with the retained
   parallel-gather setting in an isolated service. The source switch is
   prepared and host tests pass. An earlier isolated QSA layer comparison was
   essentially tied, so this is a service-level overhead experiment, not a
   predicted speedup.
4. Preserve the startup-load observation for later: worker `load_model` totals
   were 242.98, 391.43, 373.52, and 446.53 seconds for ranks 0–3. The
   8.49-second safetensors line quoted in the rank-0 log is only the final
   draft-weight load pass. The current trace does not establish that Mamba
   paging is the cause of the longer per-worker startup.

No new NPU service comparison was run after this capture.
