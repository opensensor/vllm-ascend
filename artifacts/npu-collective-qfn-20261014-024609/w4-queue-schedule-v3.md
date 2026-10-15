# Queue-drained W4 scheduling on four resident 310P ranks

The v3 local-count schedule and finalizer together are retained. The service
identity remains `qwen38-flash-next` at `http://192.168.53.187:8001/v1`;
worker PIDs 579046–579049 and tracked weight storage are unchanged.

## Change

The W4 schedule reads the last expert boundary on-device to choose a tile
shape using local work instead of all allocated route slots. It preserves the
frozen INT4 arithmetic and peer-zero output coverage. V3 drains pending host
submissions and shares operand ownership, preventing the earlier allocation
regression. The model binding now supports version 3 with strict versioned
resource ownership and full numerical qualification receipts. During capture
it falls back before descriptor lookup, including already-cached shapes.

The finalizer stays enabled. No other serving settings change in this trial.

## Validation

Both real resident projections at 650 and 20480 allocated routes passed exact
FP16-bit comparison on all four ranks. Cases cover no local routes, balanced
EP4 work, all-local work, and concentration on one expert. Baseline, forced-small
and local-count schedules were compared to the installed operator before the
candidate was bound into the model. The component results do not imply that
one tile shape wins for every expert distribution.

The schedule and descriptor CPU suites passed 82 tests. Eight model requests
passed short-input, QSA boundary crossing, 8K input, concurrent requests and
return-to-short checks, with unchanged weights and workers. These are bounded
integration/semantic checks, not full-context quality or a decode speed claim.

## Uncached 8K measurements

Each request uses an early random nonce, exactly 8192 prompt tokens, zero cached
tokens and one generated token. These sequential timings share the same live
service and are not statistical guarantees.

| Candidate | HTTP seconds | Prompt tokens/s |
| --- | ---: | ---: |
| Schedule + finalizer, A | 20.816 | 393.55 |
| Schedule + finalizer, B | 20.787 | 394.10 |
| Schedule + finalizer, C | 20.816 | 393.54 |

Median is 393.55 prompt tok/s: +7.2% versus the preceding finalizer-only median
367.17, and +11.3% versus the prior fresh compact-selector control 353.74.
The original v1 schedule trial was 274.73 prompt tok/s and remains rejected;
its integration behavior must not be confused with this v3 result.

Receipts are in `qsa-resident-v1/w4-queue-schedule-*.json`,
`cold-benchmark-w4-queue-fused-*.json`, `validation-w4-queue-fused.json`,
and `install-w4-queue-fused.json`.
