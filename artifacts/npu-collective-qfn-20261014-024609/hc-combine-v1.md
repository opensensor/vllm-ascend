# Exact HC combine candidate on resident QFN

This candidate replaces only the final hyperconnection residual combination.
The injection projection, FP16 divide/sigmoid/scalar multiply, and their
rounding remain unchanged. The new kernel widens the residual, multiplies the
FP32 block update by widened injection gates, adds in FP32 as a separate
operation, and converts to FP16 using the installed 310P saturation behavior.
No FMA, new reduction, or precision change is introduced.

## Ownership and deployment

The immutable bundle is
`/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build`, namespace
`qwen_hc_combine_v1`. It uses the shared argument owner and producer-thread host
queue drain from the qualified W4 v3 bridge. The old native resources remain
owned. Its separate build receipt is `hc-combine-compile-v1.json`; build-time
hardware flags are not edited after runtime gates.

The source-guarded model binding is limited to contiguous, same-device
FP16 residuals, four streams, hidden size 2560, FP32 compute, FP16 output and
64–2560 rows. Short eager decode and every captured call use the original
implementation, even for a cached descriptor. Descriptors are append-only and
bounded; new unsupported shapes fall back before projection. Failed launches
retain operands and their first tensor-bearing traceback instead of retrying.

## Completed private gates

All four ranks passed exact FP16 output-bit comparisons at 68, 640 and 2048
rows, with unchanged input bytes. Each row size covered random values, explicit
no-FMA cancellation, signed zero, subnormals, overflow, infinities and NaNs.
The numerical gate uses real installed NPU operations as its reference.
The syntactically supported 2560-row maximum has not been hardware-qualified.

At 2048 rows, per-rank median synchronized component latency was 3.30–3.33 ms
for the original operations and 0.929–0.935 ms for the fused kernel. Component
latency is not serving throughput. No NPU process or server was restarted.

CPU validation: 76 HC wrapper/binding tests passed, alongside the separate
builder tests. Private receipts are `qsa-resident-v1/hc-load-v1.json`,
`hc-validate-{68,640,2048}-v1.json` and `hc-timing-2048-v1.json`.

## Model qualification and timing

Eight model checks passed: short input, QSA boundary crossing, 8K input,
concurrent requests, and return to short input. Worker PIDs and tracked weights
remained unchanged. These are bounded semantic/integration checks, not a
full-context quality qualification.

All timing requests contain 8192 prompt tokens, zero cached tokens, a unique
early nonce, and one generated token.

| Trial | HTTP seconds | Prompt tokens/s |
| --- | ---: | ---: |
| HC A | 19.863 | 412.43 |
| HC B | 19.879 | 412.09 |
| HC C | 19.949 | 410.64 |

The median is 412.09 prompt tok/s, +4.7% over the preceding W4 schedule+finalizer
median 393.55, +12.2% over the previous finalizer-only winner 367.17, and +16.5%
over the earlier compact-selector control 353.74. These sequential live-service
measurements are not statistical guarantees or decode throughput claims.

A subsequent W4-only control measured 390.75 prompt tok/s (20.965 seconds),
supporting the HC gain after the candidate trials. The combined `qsa_hc_fused`
payload is retained. A final chat request produced a coherent answer; the
service was unpaused with the same four worker PIDs and tracked weights.
All four NPUs were healthy at 75–80°C immediately afterward. Model identity
and port remain `qwen38-flash-next` and 8001; configured context remains 262144.
This does not establish full-context quality.

Receipts: `qsa-resident-v1/validation-hc-fused.json` and
`cold-benchmark-hc-fused-{a,repeat}.json`. A bounded final native profile is
recorded under `qsa-resident-v1/native-timeline-hc-fused-v1`.
All four native profilers stopped once and released ownership. The selected
serving payload was restored before offline parsing. Additional receipts are
`cold-benchmark-hc-control-final.json`, `promote-hc-fused.json` and
`hc-serving-status.json`.

## Final profile

Rank-0 bounded traces for the finalizer-only and combined candidates each
contain 6852 uncached prompt tokens and 16 generated tokens. Prompt nonces
differ, so this is diagnostic evidence rather than identical routing replay.

| Profile metric | Finalizer only | Schedule + finalizer + HC |
| --- | ---: | ---: |
| Device computation | 16.062 s | 14.345 s |
| Communication, not overlapped | 2.835 s | 2.606 s |
| Kernel-free gaps | 0.673 s | 0.723 s |
| Step interval | 19.571 s | 17.674 s |
| Physical allocation calls | 98 | 98 |
| Physical free calls | 0 recorded | 0 recorded |
| Device-wide synchronization calls | 1 | 1 |

The new HC kernel ran 490 times for 0.259 s total; the local-count W4 kernel ran
480 times for 3.750 s. QSA K/V gathering remains 1.359 s across 2070 calls.
These operator sums can overlap and do not add directly to request latency.
The stable allocator counters confirm that these additions did not recreate
the earlier allocation churn. [Hashed profile summary](qsa-resident-v1/native-timeline-hc-fused-v1/rank0-summary.json).

The subsequent [QSA gather staging investigation](qsa-gather-staging-results.md)
passed hardware gates and identified excess host admission overhead. The
[combined candidate](qsa-prefill-decode-results.md) retains this HC path.
