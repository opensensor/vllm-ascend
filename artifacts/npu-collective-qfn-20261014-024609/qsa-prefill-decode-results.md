# QSA prefill and decode: retained combined candidate

The resident service retains `qsa_gather_fused_geometry_v2_fused`. The user
explicitly accepted keeping the small/noisy decode improvement on October 14.
Model `qwen38-flash-next`, port 8001, all four worker PIDs, and tracked weight
storage remain unchanged. The authenticated resident transactions recaptured
graphs without restarting workers. Final state is unpaused with healthy NPUs.

## Matched serving measurements

All rates below are single-request measurements with profiling disabled.
Decode uses the same 25-token chat prompt and exactly 256 output tokens.
Cold prefill uses exactly 8192 prompt tokens, zero cached tokens and one output
token. HTTP rates include the entire request. The server log's decode rate
uses its 255 inter-token gaps and excludes prefill; it is not pure kernel time.

| Measurement | HC/W4 control | Combined QSA candidate |
| --- | ---: | ---: |
| Cold prefill HTTP prompt tok/s | 415.00, one fresh trial | 414.81 median, three trials |
| Decode HTTP output tok/s | 31.05 median, three trials | 31.06 median, three return trials |
| Request-log decode tok/s | 32.3 median | 32.4 median |
| Request-log decode range | 31.8–32.6 | 31.3–33.6 |

The combined cold trials were 415.50, 414.81 and 413.44 prompt tok/s, taking
19.716–19.814 seconds. An earlier combined decode batch measured median 30.35
HTTP tok/s. The later control and return batches measured 31.05 and 31.06.
The log median improvement is about 0.3%, smaller than run variation; there is
no established end-to-end speedup in this comparison. It also does not show
a consistent regression. The earlier HC median of 412.09 and preliminary
K-only result of 417.79 should not replace the fresh matched control.

Generated text differs between temperature-zero requests within both control
and candidate batches. Speculative acceptance also varies in the log windows;
those windows are not per-request acceptance receipts. Consequently this is a
fixed-budget workload comparison, not identical token/routing replay. No claim
of statistically significant improvement follows from three requests.

## Implemented corrections and hardware proof

The INT32 geometry prototype silently took the original fallback because real
runner positions are INT64. The independent INT64 v2 kernel passed 60 private
exactness/replay cases on each rank, including integer extremes. Model capture
inventory confirmed admission. The subsequent eight-iteration trace contained
112 native geometry calls totaling 196 microseconds, while AiCPU FloorDiv count
fell from 144 to 32. The remaining calls are outside this helper. There is no
Python replay observer. See [geometry evidence](qsa-geometry-v2/README.md).

The apparent V gather regression was dominated by host submission overhead.
Dense V native execution was approximately 369 versus 372 microseconds installed;
sparse V was 419 versus 495. Duplicate Python admission and descriptor work
masked those gains. The corrected wrapper admits once and caches descriptor
block counts while preserving native tensor ownership, stream readiness and
producer queue draining. It repeated 432 physical-byte comparisons and eight
whole-attention checks per rank. Serving enables eight-block K gathering only;
V still uses the installed operator. See [gather evidence](qsa-gather-staging-results.md).

The combined model passed eight boundary, long-input, concurrency and
return-to-short smoke cases. These are correctness smoke checks, not full
quality qualification of the configured 262144-token context.

## Remaining opportunities

The fresh decode profile still puts substantial time in FP16 projections,
layout conversions and casts. A private real-weight GDN QKV/Z concatenation
gate passed all four ranks and reduced the three-row projection pair from
152.51 to 144.73 microseconds in graph execution. It is not serving: one
representative layer was qualified, and packing all layers adds roughly
720 MiB per rank before lifetime overhead. A shared input conversion is a
potential lower-memory follow-up, but has not been implemented or qualified.
See [projection gate](gdn-projection-private-gate.md).

The next gather target is moving metadata admission and preparation into a
native entry while preserving validation and failure ownership. Removing the
producer queue drain is not justified by this profile. Neither candidate is
promised to produce a particular tok/s gain. Broad profiler Free time is also
not equivalent to recoverable host scheduling delay; see
[the interval-union analysis](decode-projection-next.md).

## Receipts and service

- [Matched measurements and request-log lines](qsa-resident-v1/combined-matched-summary-v1.json)
- [Combined model smoke gate](qsa-resident-v1/validation-combined-gather-geometry-v1.json)
- [Final installation](qsa-resident-v1/restore-combined-after-matched-control-v1.json)
- [Final live status and sensors](qsa-resident-v1/combined-serving-status-v1.json)

API: `http://192.168.53.187:8001/v1`, model `qwen38-flash-next`.
Kilo: `vllm-310p-w4/qwen38-flash-next`.
Log: `/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v3/server.log`.

Native resources are resident-run additions, not baked into the launcher.
A future restart requires rebuilding private ownership and requalifying before
applying the payload. Never apply its stored source to fresh unloaded workers.
No service restart, driver reset, unsafe peer read or native unload was used.
