# Qwen native W4 built-in SwiGLU plus CANN finalizer, 2026-10-05

The four-Ascend-310P test combined the selected built-in FP16 SwiGLU path with
the opt-in CANN grouped-route finalizer at the 1,536-token expert chunk. The
candidate service is running on port 8001 as
`qwen38-w4-builtin-finalize-candidate`. Its isolated runtime is
`/srv/ai/src/qwen38-builtin-finalize-runtime-20261005`; the launcher differs
from the selected built-in service only by adding
`"grouped_finalize": "cann_v2"` to the W4 model override
([launcher diff](candidate-launcher.diff)). It uses the same
coherent OPP, checkpoint, TP4/EP4, 2,048-token scheduler batch, MTP2, and
two-size decode graph configuration. The model source SHA-256 is
`dd529902ff595eb98feff7638ce721b7382c6b3994add4902fee6a458658588d`.
The [candidate service wrapper](start-builtin-finalize-service.sh) records
the exact runtime and OPP paths.

## Paired real-weight layer

The [one-card layer gate](layer-1536.jsonl) used layer 0, TP rank 0, real
checkpoint weights, and seeded synthetic activations. It switched only the
finalizer within one process, with built-in SwiGLU active in both arms.

| 1,536-token local grouped MoE | Median |
| --- | ---: |
| Built-in SwiGLU + torch finalizer | 34.249 ms |
| Built-in SwiGLU + CANN finalizer | 29.671 ms |

The CANN path was **13.4% faster** for this layer partial. Its maximum absolute
output difference was `1.85e-6`, and relative L2 error was `0.000438`.
The test excludes TP reduction, attention, and service scheduling.

## Cold-prefill service result

Three 23,410-token prompts were matched by SHA-256 against the earlier
[built-in SwiGLU service records](../qwen38-prefill-swiglu-pack-20261004/service-builtin-v2-20261004.jsonl).
Both arms generated 32 tokens per request, and every request reported zero
cached prompt tokens. The candidate records and server timing log are
[here](service-cann-v2.jsonl) and [here](serve.log).

| Case | Built-in TTFT | Combined TTFT | Saved | Text |
| ---: | ---: | ---: | ---: | --- |
| 0 | 67.428 s | 64.863 s | 2.565 s | One opening phrase changed |
| 1 | 67.152 s | 63.861 s | 3.291 s | Exact match |
| 2 | 67.093 s | 64.151 s | 2.942 s | Exact match |
| **Mean** | **67.225 s** | **64.292 s** | **2.933 s (4.36%)** | |

Effective prompt rate, computed as prompt tokens divided by client TTFT,
increased from **348.2 to 364.1 tok/s**. This is an end-to-end cold-prefill
rate, not a kernel rate. Case 0 changed “code excerpts” to “repository excerpt.”
That same latter phrase appeared in the earlier built-in run's cases 1 and 2.
MTP acceptance was 20/24 drafted tokens for each candidate request. The
earlier built-in arm recorded 19/26 for case 0 and 20/24 for cases 1 and 2.
The candidate did not show an obvious semantic regression on these short
continuations, but strict output parity does not hold.

The candidate's cache capacity was 1,068,936 tokens, or 4.08 concurrent
262,144-token requests. Two decode graphs captured. The slowest target-model
worker load took 256.21 s. This service comparison reused a saved built-in
result instead of restarting that arm, so run order and device conditions were
not fully paired. The within-process layer measurement supports the direction
of the gain. There is no sustained decode or broad quality result here.

## Response and thermal checks

The [smoke gate](smoke.json) passed six of seven cases, including the tool
call. Its sole failure was reversing `ASCEND` as `DNESCA`; the older Qwen
baseline made the same error. A [thinking-enabled request](thinking-smoke.json)
answered `17 × 23` with `391`, ended normally, and reported 26 reasoning
tokens. The [temperature trace](thermal.jsonl) sampled all four NPUs during
the three long prompts and smoke requests; its peak reported temperature was
**76°C**. The saved built-in service run has no matched thermal trace, so this
does not establish a thermal improvement.

The CANN finalizer remains an explicit `grouped_finalize=cann_v2` experiment
in source. The combined candidate is live on port 8001 for use and further
quality checks. The previously selected built-in service can be restored with
`/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/start-builtin-service.sh`
after stopping this session (`qwen38_builtin_finalize_20261005`).
