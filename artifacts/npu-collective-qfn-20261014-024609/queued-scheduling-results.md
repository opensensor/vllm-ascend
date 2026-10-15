# QFN scheduling changes qualified on four 310P ranks

The resident service now overlaps engine scheduling and output observation with
model execution, consumes unchanged-batch draft IDs on device, starts the PLE
packet readback earlier, and stages compact Mamba tables without blocking device
conversions. These changes are retained in `resident-live-v6`.

This is **not full CPU bookkeeping removal**. CPU metadata construction, host
PLE lookup, draft callbacks, request lifecycle, and graph submission remain.
The current implementation does not autonomously schedule a device graph loop.

## Service identity and retained state

- API: `http://192.168.53.187:8001/v1`
- Model: `qwen38-flash-next`
- Kilo: `vllm-310p-w4/qwen38-flash-next`
- TP4/EP4 on two Ascend Duo cards; MTP2; captures `[3,6,9,12]`.
- Context setting remains 262,144; quality checks in this work reach 8K.
- Candidate name: `qsa_combined_queued_observer`, with subsequent drained controls.
- Engine: `QueuedBalanceScheduler`, async scheduling, queue depth 2.
- Worker: synchronous history ownership; the guarded bridge resolves optimistic
  engine schedules from completed local results. Worker async-history flags
  remain disabled deliberately.
- Queue bridge, output observer, device drafts, early PLE readback, and compact
  table staging are enabled. Transition/history verification is disabled after
  prior qualification. All native and CPU profilers are released.

The [final receipt](resident-live-v6/final-scheduling-status.json) reports health
200 and an unpaused engine. Worker PIDs and weight storage stayed unchanged.
Unused next-input/history packets may remain retained after a request finishes;
these are snapshots guarded by request identity, not pending execution. The
observer and device-packet consumer hooks have no unretired work.

## Implemented changes

1. `tools/qwen4exp/queued_engine.py` preserves the existing scheduler and cache
   manager while enabling upstream async placeholder accounting and a two-batch
   engine queue. `queued_schedule.py` creates worker-local corrected schedules;
   the engine retains its optimistic schedule for rejection rollback. New,
   resumed, recompute, and changed-layout requests retain guarded handling.
   Finite cache-recycling admission bounds and unsupported connector/parallel
   configurations are rejected rather than silently reinterpreted.
2. `output_observer.py` starts device acceptance/commit, copies sampler output on
   a separate stream, and runs existing CPU output bookkeeping while the draft
   executes. It joins before publishing results or permitting request movement.
   At the final inspection, every rank reported 3,092 observations and 3,092
   completions before join. CPU work still occurs, but it did not delay the join
   in those admitted iterations. Unsupported sampling modes keep the fallback.
3. The queued bridge bypasses the draft-ID CPU readback on unchanged MTP2
   batches. CPU draft slots are shape reservations only. Both target input and
   PLE hooks must consume the matching device packet before model execution.
   Missing packet consumption fails rather than feeding placeholders to a model.
   Each rank recorded 2,373 uses at final inspection.
4. `ple_history_runtime.py` can enqueue the host lookup packet copy immediately
   after the history graph, before next-step metadata uploads. Consumption waits
   for that copy's event. Host PLE hashing still reads real device-produced IDs.
   Final counters show 1,349 produced packets and 1,302–1,305 already-ready
   observations per rank; counts also include packets discarded at request
   boundaries and must not be interpreted as identical denominators.
5. `resident_candidates/qfn_compact_table_staging.py` replaces two exact,
   source-checked `torch.as_tensor(..., device=...)` seams with retained pinned
   buffers. The active prefix path formerly performed three blocking conversions
   per iteration. Host-buffer reuse waits only for the prior DMA completion;
   device copies remain ordered on the consuming stream. Existing mappings,
   allocation decisions, and scheduler-owned block IDs are unchanged. Each rank
   recorded 2,415 staged copies at final inspection.

## Unprofiled serving measurements

C1 is the median of three 256-output-token HTTP requests, including short prefill
and API overhead. C4 is aggregate throughput over one finite simultaneous batch
of four 256-token requests, including prefill and request tails. These are small
sequential trials, not a confidence interval or a pure device decode benchmark.

| Configuration | C1 HTTP tokens/s | C4 aggregate HTTP tokens/s |
| --- | ---: | ---: |
| Fresh synchronous-engine baseline, new scheduling controls off | 29.671 | 64.334 |
| Queue + observer + guarded device drafts | 31.321 | 66.420 |
| Above + early PLE readback | 31.807 | Not separately measured |
| Retained, including compact-table staging | 32.056 | 69.179 |

The retained measurements are about 8.0% higher at C1 and 7.5% at C4 than the
fresh baseline. Small incremental differences remain sensitive to run variation.
The isolated early-readback comparison measured 31.231 off versus 31.807 on.
The retained C4 median per-request SSE decode estimate was 18.917 tokens/s;
this differs from aggregate batch throughput.

Evidence is in `qsa-resident-v1/`:

- `decode-benchmark-scheduling-all-off-v1.json`
- `concurrency-scheduling-all-off-v1.json`
- `decode-benchmark-device-drafts-on-v1.json`
- `concurrency-device-drafts-on-v1.json`
- `decode-benchmark-ple-readback-{off,on}-v1.json`
- `decode-benchmark-compact-tables-on-v1.json`
- `concurrency-scheduling-final-c4-v1.json`

Do not compare validation-suite elapsed times as cold-prefill performance:
prefix-cache state differs between suites. No new cold-prefill gain is claimed.

## Profiling evidence

The [native trace summary](resident-live-v6/scheduling-profile-summary.json)
compares queue/observer/device drafts enabled and disabled, before the later PLE
and compact-table changes. Four ranks were captured; ranks 0 and 3 were parsed.
Each trace contains 104 graph launches: eight iterations with 13 launches each.
The summary uses six complete interior iterations.

| Rank | Disabled iteration ms | Enabled iteration ms | Disabled no-recorded-task ms | Enabled no-recorded-task ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 86.961 | 82.606 | 15.440 | 10.983 |
| 3 | 86.558 | 83.342 | 13.609 | 10.109 |

Unrecorded-task intervals are not proof of device idleness. Summed CPU API waits
can overlap device execution and other threads; they are not additive latency
savings. The next-iteration boundary still contained approximately 7 ms without
recorded device tasks, motivating the subsequent CPU profile.

The [CPU comparison](resident-live-v6/scheduling-cpu-summary.json) covers one
short prefill plus 51 decode iterations in each case. Before compact staging,
the remap path called blocking `torch.as_tensor` 156 times per rank; after the
change it called it zero times. Rank 0 remap cost fell from 1.746 to 0.752 ms per
call, and total input preparation from 2.935 to 2.033 ms. Across ranks, the new
input preparation cost was 1.927–2.033 ms. Instrumentation adds overhead, so these
are attribution measurements rather than promised serving latency reductions.
Raw CPU profiles and receipts are retained under `resident-live-v6/cpu-scheduling-*`.
Raw native traces remain on the hardware host under
`resident-live-v6/native-timeline-scheduling/`.

## Validation and limitations

- 109 focused host tests passed for queued accounting, observation, transition,
  history, and compact staging. Another 48 engine/server tests passed. Ruff
  passed on changed implementation and test files. The full repository suite
  was not run.
- Separate ten-case hardware suites passed for the queue, real observer, device
  drafts, early PLE readback, and compact tables. These check short requests,
  2,030/2,100-token graph boundaries, 8,192-token inputs, mixed C4 contexts, and
  return to short requests, with output length and basic semantic checks.
- The retained configuration passed twelve serving checks: budgets 1/2/3,
  penalties, logprobs, uneven C4 completion, JSON-schema output, stream abort,
  and a successful request after abort. See
  [serving receipt](resident-live-v6/requalification/scheduling-serving-gate-retained-v2.json).
- The first observer trial excluded INT32 sampler output and exercised no
  observations. Only the corrected observer-v2 trial qualifies that path.
- Forced cache-pressure preemption was not hardware-tested. Later recompute
  chunks have a host regression test. Full-context quality and broader model
  quality are not established by these serving checks.
- No service restart or model reload was required. Changes were applied only
  after draining through the authenticated engine/worker control paths.

## Remaining work toward CPU-independent execution

The next measured target is attention metadata construction, approximately
2–3 ms per iteration in the CPU diagnostic. It needs stable device-owned
positions, sequence lengths, slot mappings, and accepted-count ownership across
request boundaries. The host-paged PLE table and draft position/QSA/expert
callbacks also remain dependent work. Moving only scheduler bookkeeping to a
background thread does not close those graph breaks.

A full implementation still needs bounded device execution windows, cache
admission and recurrent-state ownership, termination/cancellation, output
observation independent of progress, and coordinated failure retirement. This
work enables actual engine overlap and removes measured host waits; it does
not satisfy that final autonomous-loop contract.

## Restore and inspect

On the hardware host, client scripts run from `resident-source-v1` with that
folder prepended to `PYTHONPATH`, using the QFN venv. Private control tokens stay
on the hardware host. Preserve existing receipts by using fresh labels.

Starting from the qualified v6 device-history candidate, the restoration order
is queued installation, observer installation/INT32 refresh, device-draft
installation, queue on, observer on, device drafts on, early readback on, compact
tables on. Relevant controllers are `qfn-queued-control.py`,
`qfn-observer-control.py`, `qfn-observer-refresh.py`,
`qfn-device-drafts-control.py`, `qfn-ple-readback-control.py`, and
`qfn-compact-table-control.py`. The live compact-table candidate also requires
`vllm_ascend/_310p/host_staging.py` in the pinned runtime.

These are retained hot controls for the current run, not a claim that every
historical launcher automatically restores the new configuration. The final
inspection script is `qfn-scheduling-final-status.py`; its receipt intentionally
preserves pending snapshots instead of resetting counters or state.
