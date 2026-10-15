# Device-owned PLE history and lookup inputs

This closes a correctness dependency for future queued execution. It does not
complete host-independent iteration or remove the host-paged embedding lookup.

## Implemented path

`ResidentPLEHistory` bootstraps history from the existing runner when a request
layout first becomes eligible. For unchanged, uniform MTP2 decode rows, an NPU
graph advances it from device acceptance. Acceptance A appends the current
target's first input and the first A-1 sampled tokens; the final accepted token
becomes the next query input. Rejected drafts never enter history.

The second revision publishes a device packet containing both the next history
and next query IDs. The current host-paged lookup consumes a mirror of that
packet, rather than CPU sampled-token placeholders. This mirror still requires
a blocking D2H copy. Request turnover, prefill, unsupported geometry, or a failed
input fast-path bootstraps through the existing path. Uncertain device submission
retains owners and fails instead of silently falling back.

The history owner remains attached to the native commit owner. Its graphs and
buffers outlive replay. All changes use drained resident control; model name,
port, weights, and worker PIDs remain unchanged during these installations.

## Hardware evidence

- Private history graphs passed 32 changing replays and 128 history checks on
  each of four NPUs with history width three. A separate rank-0 gate passed the
  model's actual history width two (`ngram_size=3`).
- Revision 1 passed 290 live history comparisons per rank, then the same
  ten-request suite with comparisons disabled.
- Revision 2 passed 289 live history-and-query-ID comparisons per rank. Its
  comparison-disabled ten-request serving suite also passed.
- Cases include short prompts, the 2,048-token chunk boundary, 8,192 input
  tokens, four concurrent mixed-length requests, and return to a single request.
  Text checks are semantic smoke tests, not a model-quality evaluation.

Receipts live under `resident-live-v6/requalification/ple-*`. The retained
engine-control transaction was also tested in the engine process, including
unpaused and incorrect-digest rejection; it did not change the scheduler.

## Performance

With diagnostics disabled:

| History path | Median C1 HTTP tokens/s | C4 aggregate HTTP tokens/s |
| --- | ---: | ---: |
| Revision 1 enabled | 30.119 | 62.384 |
| Disabled | 30.014 | 64.078 |
| Revision 2 enabled | 30.124 | 64.811 |

C1 uses three 256-token trials. C4 uses four simultaneous 256-token requests.
Both include prompt/API overhead; C4 also includes request tails. This does not
show a clear speedup. Revision 1's C4 result was slightly slower; the final
revision is roughly neutral compared with the disabled control. There is no
claim of pure device throughput or deterministic quality equivalence.

Revision 2 is retained enabled with verification off. The server is unpaused,
all four native profilers are absent, and both the engine and worker groups
remain healthy. Final sensor readings were 80/79/85/84 C with all devices
reporting OK. The tests stopped after qualification; the service remains up.
The focused host suites total 108 passing tests. Ruff and the affected-document
Markdown checks pass; the full repository suite was not run.

## Remaining dependencies

The synchronous engine still observes model output before scheduling another
iteration. Host compact-cache mapping/admission, normalized-count readback,
metadata preparation, PLE lookup, and eager model graph callbacks remain.
Device ownership of PLE history and query IDs is necessary groundwork; it is
not sufficient to remove the async guards. Queued-step cache ownership,
termination/cancellation, and output retirement still require qualification.
