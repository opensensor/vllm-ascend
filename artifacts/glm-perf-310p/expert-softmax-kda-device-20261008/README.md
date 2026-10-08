# GLM expert, QSA and KDA device qualification

This continues the [offline phase](../offline-expert-softmax-kda-20261008/README.md).
All three changes compile on the serving CANN 9.1.0 SDK and pass device
qualification. The final serving comparison is recorded below separately from
operator timing. This batch preserves arithmetic boundaries and checkpoint
values; it does not change model precision or claim a language-quality score.

## Changes and operator measurements

| Candidate | Parent | Device change | Measured example |
| --- | --- | --- | --- |
| Expert v992 | Decode v984 | Cube/vector readback events | Decode evaluated through the serving comparison |
| Expert v993 | Prefill v985 | Cube/vector readback events | W3 gate/up: 15.269 to 14.920 ms |
| Expert v994 | Prefill v985 | Strided FP16 route store | W3 down: 7.703 to 7.300 ms |
| Expert v995 | Prefill v985 | Both changes | W3 down: 7.709 to 7.140 ms |
| QSA v996 | Shared-cache v991 | Head max/sum and delta exponentials batched | 640-row sparse: 28.146 to 23.964 ms |
| KDA matrix package | Qualified row-batch package | Matrix repeats for full-row score construction | Complete 640-token BSND operator: 17.751 to 10.894 ms |

Expert timing uses actual checkpoint layers 10/W3, 11/W4 and 33/W2,
640 rows and A4. The combined v995 down stage improves all three examples by
7.4–7.8%; its gate/up improves by 2.3–3.6%. KDA's example is 38.6% less
operator time. These are complete kernel/operator measurements, not model
throughput predictions. QSA adds 256 B UB/core; the other changes add none.

Each of v992–v995 passed 30 independent fixtures, 12 real-weight fixtures and
60 matched parent boundaries with three changed graph replays per boundary.
The real-weight cases cover A4/A8 and rows 1, 8, 9, 16, 17, 24, 25, 31, 32
and 640. Bounded admission also passed on all four serving ranks.

QSA passed 36 exact comparisons against both the frozen v991 binary and the
installed production operator, including heads 1/3/12, dense/sparse inputs and
three changed graph replays. Dense cases also use an independent reference.
Bounded serving admission passed 24 cases on each of four ranks.

KDA passed 17 finite cases, both layouts, all 12 outputs, unchanged inputs and
zero byte mismatches. Changed-input graph replays passed at 64 and 640 tokens.
Two existing unsafe fixtures are nonfinite in both parent and candidate; they
are excluded from the finite qualification count. The staged header admits
only its exact qualified parent hash and retains earlier KDA fusions and tails.

## Serving comparison

| Workload | Parent v984/v985/v991 | Combined v992/v995/v996 | Retained v984/v995/v991 |
| --- | --- | --- | --- |
| Cold 640 tokens | 5.142 s | 5.037 s | 4.980 s |
| Cold 1,280 tokens | 10.822 s | 10.515 s | 10.485 s |
| Cold 6,400 tokens | 56.160 s | 53.479 s | 53.954 s |
| C1 median, full generation | 10.177 tok/s | 9.726 tok/s | 10.164 tok/s |

The retained selection measured 3.93% less cold 6,400-token time.
Its C1 median changed by -0.13%. The combined selection regressed
C1 by 4.43% and was not retained.
The isolated QSA gain is not an established model-throughput gain.

The final protocol warms the worker before recording, clears prefix state for
every request, uses identical synthetic token IDs in separate fresh worker processes with
the same checkpoint, SDK and serving configuration. Each 6,400-token case is repeated twice. C1 uses three
128-prompt/128-generation requests and measures all 127 generation gaps through
completion, rather than selecting fast iteration ticks. MTP remains enabled.
These token-ID workloads measure speed; the separate text smoke checks API
execution only. Model quality remains for the user's evaluation.

All these serving selections use the new KDA package. Consequently their comparison
covers the expert/QSA additions together for v6 and expert prefill alone for
v7; KDA's individual benefit is
established at operator level. The earlier QSA-only serving run did not show
an end-to-end win and is retained in the receipts. Improvements from different
measurements must not be added together.

## Graph lifecycle limitation

First source selections in fresh workers capture and serve. Repeated changes
failed while capturing graphs with a 242 MiB allocation error at the current
0.965 reservation. One failure occurred on the third selection, and another
on the second selection after the parent workload completed. Releasing retired
projection scratch and inactive descriptors was insufficient to prevent it.
The precise remaining graph/allocator lifetime issue is unresolved. The
combined candidate has served correctly when selected first in fresh workers.

The final protocol therefore compares the completed parent workload from v5
with the same warmup and requests for candidates in fresh processes (v6 and v7).
The fully combined v6 selection regressed C1, so the final v7 selection keeps
decode v984 and QSA v991 while retaining prefill v995 and the new KDA package.
All three implementations remain available for experiments; decode v992 and
QSA v996 are qualified but unselected. This is a process-separated comparison;
timing drift and the small sample count limit the model-throughput conclusion.
The original checkpoint, 311,040 context limit, four sequence slots and memory
reservation remain. Failed switches and each recovery receipt are archived;
no failure is counted as a successful benchmark. The final health receipt
checks all four ranks, clean graphs, zero native fallbacks and an unpaused API.

## Artifacts and validation

[SUMMARY.json](SUMMARY.json) gives qualification counts and selection details.
[serving-config.json](serving-config.json) records the final launch command and
configuration. [RUNBOOK.md](RUNBOOK.md) gives the build and recovery procedure.
[README.zh.md](README.zh.md) provides the Chinese summary.

`qualified-builds.tar.gz` contains the frozen compiled candidates and provenance.
`measurements/gate-receipts-and-protocols.tar.gz` records initial qualification;
`measurements/final-receipts-and-protocols.tar.gz` adds complete serving results,
failures, recoveries and the final protocol. SHA256SUMS covers the artifacts.
Build-time metadata correctly says evaluation had not yet occurred; later gate
receipts establish device and serving results without rewriting provenance.

The isolated GLM CPU suite passed 1,587 tests. Changed-file hooks pass. Full
repository hooks have inherited failures, preserved in this directory's
checks and the offline phase; unrelated root Qwen edits were kept outside this worktree and commit.

Feature scope follows the existing deployment: TP4, MTP one draft token,
FULL_DECODE_ONLY capture sizes 2/8, chunked prefill 640 and prefix caching.
Logs additionally show PIECEWISE prefill capture. EP and flashcomm1 are not
newly enabled or qualified by this batch. Multimodal is disabled for this
text-only deployment. No context/concurrency expansion is claimed.

Four concurrent 128-prompt/128-generation requests also completed on the
retained selection, exercising graph size 8 without recapture. That functional
check has no paired C4 speed comparator.
