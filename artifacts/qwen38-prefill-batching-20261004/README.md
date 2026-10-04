# Qwen native-W4A8 grouped prefill batching, 2026-10-04

The one-card test used layer 0, TP rank 0, real Qwen3.8 Flash-Next W4
checkpoint weights, synthetic FP16 activations, and the model's own router.
The 2,048-token input sent 6,525 of its 20,480 routes to this rank. The
largest local expert received 453 rows. Timings exclude router evaluation,
shared experts, collectives, attention, and service scheduling; they include
grouped dispatch, both dependent native W4 projections, activation packing,
and route finalization. Every tested chunk partition produced bitwise
identical layer output.

## Baseline batch sweep

| Chunk tokens | Projection pairs | Layer median |
| ---: | ---: | ---: |
| 512 | 4 | 59.45 ms |
| 1,024 | 2 | 81.03 ms |
| 2,048 | 1 | 59.89 ms |

The repeated [timings](rank0-profiled.json) agree with the first
[unprofiled sweep](rank0.json). In three profiler calls, native W4 projection
tasks summed to 127.59 ms at 512-token chunks, 192.94 ms at 1,024, and
130.44 ms at 2,048. See the [trace summary](trace-summary.json) and exported
[512](chunk-512-kernels.csv), [1,024](chunk-1024-kernels.csv), and
[2,048](chunk-2048-kernels.csv) kernel details. The profiler reported an
"Incorrect schedule" warning at exit, but exported complete three-call
windows with the expected projection counts. Kernel sums attribute work; the
unprofiled medians are the latency measurements.

The native kernel switches from 32-row to 128-row tiles when
`numRows > numExperts * 64`. This rank has 128 local experts, so the switch
occurs between 8,192 and 8,193 total routes. An [819-token call](boundary-819.json)
finished in 21.73 ms; an [820-token call](boundary-820.json) took 37.22 ms.
Their local route counts were 2,608 and 2,612, respectively. The one-token
step and kernel trace identify a schedule cliff, not a benefit from larger
projection batches.

## Isolated tile-switch candidate

The candidate was built from the coherent October 1 source in a separate
directory. Its only kernel source edit raises the switch to
`numRows > numExperts * 128`. The candidate OPP clones the retained
coherent package and replaces only this operator's two compiled objects and
their JSON metadata. Host API, tiler, other operators, and the 20,480-route
limit are unchanged. It does not alter the retained server package.
The candidate source is retained at
`/srv/ai/src/qwen38-batch-switch-128-src-20261004` (edited kernel SHA-256
`7849ba4da0565fc73cc024e6511443094f5911dce0c6d349596afb4297939ffb`).
It was built with `csrc/build.sh --pkg --ops=qwen_w4_a8_int4_matmul_v310
--soc=ascend310p --vendor_name=qwen_w4_batch128 -j8` and installed in the
isolated OPP at `/srv/ai/src/qwen38-batch-switch-128-opp-20261004`.

Paired processes used the same checkpoint, RNG seed, route IDs, and source
snapshot. Their route-ID and final-output SHA-256 hashes matched exactly.

| Chunk tokens | Baseline | Candidate | Candidate change |
| ---: | ---: | ---: | ---: |
| 512 | 60.08 ms | 60.16 ms | +0.1% |
| 1,024 | 80.86 ms | 52.02 ms | -35.7% |
| 2,048 | 59.92 ms | 59.96 ms | +0.1% |

See the [baseline](qwen-w4-baseline-hash.json) and
[candidate](qwen-w4-candidate-hash.json) records. At 1,536 tokens, the
[baseline](qwen-w4-baseline-1536.json) took 49.80 ms and the
[candidate](qwen-w4-candidate-1536.json) took 36.73 ms, with the same output
hash. At 820 tokens, the candidate recovered to 21.90 ms.

The [candidate chunk sweep](qwen-w4-candidate-chunk-sweep.json) then compared
the same 2,048-token input partitioned at 512 through 2,048 tokens. Its best
one-card medians were 51.54 ms at 1,536 and 51.24 ms at 1,638; 1,639 crossed
the new 16,384-route threshold and took 64.18 ms. A 1,536-token model chunk
was chosen for the service candidate because it leaves 1,024 route rows of
margin below the switch and uses a regular chunk size. The one-card gain
against the retained 2,048-token path is about 8.4 ms per MoE layer call
(14.0%); it is not an end-to-end speedup claim.

## Service and Strata implication

The service candidate is a full isolated runtime snapshot with only
`MAX_GROUPED_NATIVE_TOKENS = 1536` changed, paired with the new OPP. It
retains the 2,048-token scheduler batch and every other qualified launcher
setting. Its runtime source is retained at
`/srv/ai/src/qwen38-batch1536-runtime-20261004` (edited model SHA-256
`b45e3d0378f8ef0da6224e85c68e1d2026338f61daffb59c43eae9d8c09c5cfb`).
The baseline service was stopped during startup at the user's
request; no baseline prompts were sent in this round. Existing matched
23,410-token cold-prefill records from the
[CANN finalizer gate](../qwen38-cann-finalize-20261004/README.md) provide a
historical comparison, with the usual cross-run limitation.

The candidate served three matched prompts with 32 generated tokens each.
All reported `cached_tokens = 0`; prompt and generated-text SHA-256 hashes
matched the saved torch arm in every case. Drafted and accepted MTP token
counts also matched. Raw candidate [request records](service-batch1536.jsonl)
include usage, TTFT, decode time, and hashes.

| Case | Saved torch TTFT | Candidate TTFT | Difference |
| ---: | ---: | ---: | ---: |
| 0 | 73.278 s | 69.080 s | -4.198 s |
| 1 | 72.652 s | 68.729 s | -3.923 s |
| 2 | 72.864 s | 68.950 s | -3.913 s |
| Median | 72.864 s | 68.950 s | -3.913 s (5.37%) |

The service reported 1,068,936 NPU cache tokens, 4.08 concurrent requests
at the configured 262,144-token context, and successful decode graph capture.
The candidate was shut down after the measurements; port 8001 was closed.
These results qualify the isolated candidate and support carrying the source
changes forward. A rebuild from the shared main tree, with its other pending
operator changes, still needs a separate NPU gate before that combined tree
is treated as qualified.

[Strata's prefill study](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-28-prefill-speed/README.md)
supports amortizing work across tokens, while also recording a grouped GEMM
that improved in isolation and regressed inside its engine. Our Qwen result
shows why tile schedules and the complete model must be measured before
raising the batch cap. Qwen already uses native packed INT4 expert compute;
the separate prefill SwiGLU-plus-pack experiment remains the direct
memory-traffic follow-up. Its NPU parity and timing gate is pending.
