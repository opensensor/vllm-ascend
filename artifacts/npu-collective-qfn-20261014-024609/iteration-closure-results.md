# QFN iteration boundary: acceptance reuse and commit planning

## Retained runtime change

The resident service now reuses the sampled-token host observation when the
310P CPU Mamba alignment fallback needs raw accepted-token counts. Previously,
bookkeeping copied sampled IDs to the CPU, then alignment copied a separate
NPU count tensor back. The candidate derives counts from the first observation
and removes the second readback. Counts precede vocabulary/discard filtering,
are tied to the original sampled tensor and request order, and expire after one
state-update call. Reordered or unsupported batches retain the original path.

This is an intermediate dependency cleanup. Sampling still returns to the host,
the CPU Mamba state-copy fallback still runs, and the captured model's eager
callback count has not changed. The whole-model distributed DMA graph remains
unfinished. Graph/device completion events and state-copy ordering remain in
place; no sampling decision, precision, or reduction order was intentionally
changed.

- Model: `qwen38-flash-next`
- API: `http://192.168.53.187:8001/v1`
- Kilo: `vllm-310p-w4/qwen38-flash-next`
- Context configuration: 262144; full-context quality remains unqualified
- Candidate: `qsa_combined_acceptance_snapshot`, reuse enabled, verification off
- Target captures: general and prefix variants at `[3, 6, 9, 12]`
- Workers unchanged: 1638646, 1638647, 1638648, 1638649
- Log: `/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v4/server.log`

No server restart occurred during this work. Resident maintenance used temporary
scheduler pauses; final status is unpaused. All profilers are stopped/released.
Final recorded NPU temperatures were 73/72/78/76°C, with all devices healthy.

## Measurements

| Measurement | Reuse disabled | Reuse enabled |
| --- | --- | --- |
| C1 HTTP output tok/s, median of six 256-token requests | 30.08 | 30.33 |
| Same requests, server decode tok/s median | 31.50 | 31.75 |
| C4 aggregate HTTP tok/s, one four-request burst per mode | 63.75 | 63.66 |
| C4 per-request SSE decode estimate, median | 16.99 | 17.27 |

The roughly 0.8% C1 median difference is within observed variation. C4 aggregate
is effectively unchanged. Retain this as a removed dependency with neutral
measured throughput, not a demonstrated acceleration or progress to 40 tok/s.
The two C1 rounds bracketed candidate/control trials; each round had three
requests. HTTP includes short prefill; server decode excludes it. C4 is aggregate
throughput, not the C1 rate. Tests with reused prompt caches are not cold-prefill
benchmarks, and their shorter validation times must not be reported as a gain.

## Correctness evidence and limitation

A diagnostic mode compared the candidate's raw CPU acceptance counts against the
original NPU count tensor on every state update. All four ranks passed **214
comparisons each**, covering ten short, boundary-crossing, 8K, concurrent mixed
context, and return-to-prefix requests. Diagnostic mode deliberately retains the
old readback and is excluded from performance evidence. Ten more model checks
passed with verification disabled and the duplicate readback removed.

Full generated-token parity could not serve as a reliable acceptance gate:
identical baseline requests already produced different greedy token sequences.
Three warm baseline repeats produced three distinct outputs. With cache and
state reset between requests, baseline repeats also diverged; one candidate
repeat exactly matched the first baseline while another differed. These receipts
are retained. This does not establish a new candidate regression, but neither
does it establish full end-to-end numerical equivalence. The source of baseline
nondeterminism remains unresolved.

The combined host regression suite passed **200 tests**, including 21 acceptance
snapshot cases, nine queue-recovery cases, twelve tensor commit-plan cases, and
the existing profiler/graph selection suites. Scoped repository hooks passed;
the complete repository suite and secret-scan hook were not run.

## Next graph component implemented offline

Follow-up: the planner and a fused state-copy kernel have since passed
[private NPU graph qualification](mamba-commit-graph-results.md). The following
paragraphs describe the earlier CPU-only milestone; model integration remains
pending.

`tools/qwen4exp/mamba_commit_plan.py` expresses aligned Mamba commit decisions
entirely as tensor operations: destination block, accepted-token bias, copy
predicate, and next accepted count. Tests compare it with the current scalar
fallback across block sizes 1/4/16/128/256, near-boundary positions, rejection
counts, and full-context positions. Tests also prohibit host value observation
through `item`, `tolist`, `cpu`, or tensor truth testing.

This component is **CPU-tested only** and is not installed in the service. The
next hardware step must capture/replay changing inputs, then integrate predicated
state copies into owned KV/recurrent buffers. Inactive rows must not access copy
addresses. Input arithmetic and active-row geometry require admission before
submission. Native notifications must order producer completion and state commit
before dependent graph work can consume the next inputs. A tensor decision plan
alone does not implement a state-copy kernel or autonomous graph continuation.

## Control-path repair

An attempted diagnostic revision initially reused the same control source digest,
so the resident patch session kept the old callable signature. The controller now
includes the candidate module's SHA256 in its source identity. Calling the new
signature on the old function exposed an upstream executor behavior: it raises
on the first failed rank without consuming later ranks' replies.

Repeated ordinary status reads can appear healthy while maintaining those FIFO
offsets. Unique prepare generations measured exact reply debts of 0/1/2/3.
A bounded cold repair suppressed exactly that many subsequent control replies
while the executor consumed the queued old replies. Inference was paused and
only repair/read-only probes were submitted during this operation. Each temporary
worker enqueue hook restored itself, and a unique-generation all-rank check
verified complete retirement. Model weights, native graphs and workers stayed
resident. The recovery code has nine CPU tests, including unequal debts,
idempotence and restoration of existing instance methods.

This exceptional recovery intentionally uses the prepare factory to install a
short-lived control-reply hook; it must not be used as a normal model patch or
run against undrained inference. The underlying upstream executor error path
still needs a permanent drain-on-error fix. Ordinary candidate RPCs return error
receipts instead of raising, and no recovery hook remains on the execution path.

## Evidence and reproduction

Evidence is under `iteration-closure-v1/`: `final-serving-status.json`,
`benchmark-summary.json`, `benchmark-request-log.txt`, both model validation
receipts, baseline repeatability/parity failures, queue-debt/repair/retirement
receipts, and resident switch controls. Individual unprofiled benchmark JSON
files are under `qsa-resident-v1/*acceptance*.json`.

`qfn-acceptance-switch.py on|off --label UNIQUE` toggles the retained experiment
through a drained cold RPC without recapturing graphs. `verify` enables the
slow numerical diagnostic and must not remain enabled for speed measurements.
The controller reads the private token from `resident-live-v4/control.token`;
credentials are excluded from the saved artifacts. A future restart still needs
the prior combined native qualification before this candidate can be installed.
