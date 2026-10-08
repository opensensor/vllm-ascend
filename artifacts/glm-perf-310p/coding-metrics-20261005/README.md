# Live coding request metrics

Read-only inspection of the GLM server on port 8001, captured on
2026-10-06 at approximately 00:38 UTC (October 5 in America/New_York).
No inference request, control transaction, candidate change, or restart was
performed for this inspection. The observed requests came from the existing
client at 192.168.50.211.

## Completed request

Request `chatcmpl-ad60edf928667ab6` finished at 00:35:45 UTC with reason `stop`.

| Measurement | Observed value |
| --- | --- |
| Input tokens computed | 20,643 |
| Input tokens cached | 0 |
| Prompt processing | 260.737 seconds |
| Effective prompt throughput | 79.2 tokens/s |
| Time to first token | 260.778 seconds |
| Output tokens | 65 |
| Decode time, 64 inter-token gaps | 15.640 seconds |
| Decode throughput reported by request logger | 4.1 tokens/s |
| Queue time | 0.0 ms |
| Total request time | 276.417 seconds |

TTFT is the difference between the Prometheus TTFT sums before and after this
request, 414.89117407798767 minus 154.1136453151703 seconds. The count increased
from 22 to 23. The sum difference agrees with the per-request prompt timing.
The request spent approximately 94% of its total time processing its prompt.

Iterations 347–379 cover exactly 20,643 prompt tokens: 32 chunks of 640 and
one final chunk of 163. The logged iteration times sum to 245.872 seconds;
they do not cover every part of the request's 260.737-second prefill interval.
The median full-chunk iteration took 7.518 seconds. The first took 6.362
seconds, and the last full chunk took 7.878 seconds.

Speculative counters increased by 28 accepted / 37 drafted tokens between
the prior live snapshot and this capture, approximately 75.7%. The subsequent
request shows no decode in the captured log. This is a counter difference,
not an instrumented per-request acceptance receipt.

## Cache and server state

The next request started at iteration 417. Prefix-cache hits increased from
zero to 20,480 tokens, proving that prefix reuse occurred. Three additional
640-token prefill chunks appear before the log's last iteration at 00:36:10.
The snapshot does not contain a completed timing record for that request;
it establishes neither its final latency nor why iteration logging stopped.

There were no queued requests during the observed work and the preemption
counter remained zero. KV usage during prefill reached approximately 31.5%.
The later snapshot reported zero running requests and zero waiting requests.
All four 310P3 devices reported healthy; their zero AICore utilization was an
idle snapshot and is not a utilization measurement during the completed work.
The API health endpoint subsequently returned success.

## Optimization implications

The earlier 12.79 tokens/s benchmark was aggregate throughput across four
requests. Its per-request median was 3.21 tokens/s. Neither that aggregate nor
the short-prompt 2.77-second TTFT represents this coding request's experience.

Prompt processing is the first priority for this workload. The live scheduler
uses 640-token chunks. Prepared larger-batch experiments require matching
expert route capacity and workspace allocations; raising a launcher number
alone is not a qualified optimization. The existing candidate queue documents
those constraints and earlier decode regressions.

For decode, the existing
[resident trace](../projection-nz-resident-20261005/README.md) identifies
packed expert projection as the largest individual task family on all four
ranks. It also records many normalization reductions and associated launches.
Those overlapping task sums are not wall-time shares or promised speedups.
This inspection did not collect a new prefill operator trace or establish its
current operator-level critical path.

Raw evidence: `metrics.txt`, `server-log.txt`, `npu-smi.txt`, `snapshot.json`,
and parsed request/chunk measurements in `summary.json`.
