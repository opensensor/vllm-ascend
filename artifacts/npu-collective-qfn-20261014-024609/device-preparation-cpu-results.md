# Device input preparation and removal of unused commit uploads

The resident QFN service retains both changes on four Ascend 310P ranks. The
public identity remains `qwen38-flash-next` at `http://192.168.53.187:8001/v1`.
CPU elimination is **not complete**. The existing Mamba cache policy and recurrent
math are unchanged; cache replacement remains deferred per the user's direction.

## Implemented changes

`tools/qwen4exp/decode_preparation_graph.py` consumes the retained next-step
device packet for unchanged, exact-size MTP2 decode batches of 1–4 requests.
It supplies positions, computed counts, sequence lengths, fixed query/request
geometry, and speculative draft IDs through one private graph per batch size.
It preserves the existing model-visible buffer addresses and model graph captures.

The qualified host path now reuses fixed index arrays and speculative metadata,
avoids the redundant CPU token gather, removes the duplicate previous-position
calculation, and avoids repeated geometry uploads. It retains compact page
remapping, host mirrors needed by other consumers, checkpoint correction, and
request lifecycle handling. Request changes, prefill, actual multimodal inputs,
prompt embeddings, unsupported sampling consumers, and unsupported shapes keep
the original path. A multimodal-capable model can still use this path for a
text-only request.

`tools/qwen4exp/host_commit_metadata.py` removes four legacy metadata uploads per
step. The existing native commit and qualified CPU fallback both consume these
host columns, while the native owner stages its own packet. The legacy GPU
columns therefore have no reader in this qualified configuration. Installation
checks both the exact staging source and the original CPU fallback underneath
the resident wrappers. The host packing and its validation remain unchanged;
foreign contexts retain their original staging path.

Both are isolated resident tools rather than a blanket change to other models
or backends. Their source checks deliberately reject an unreviewed runner.

## Validation and recovery

- 221 focused host tests passed; scoped Ruff passed.
- The corrected preparation graph passed 194 exact comparisons on each of four
  ranks. These check positions, sequence lengths, request/query geometry,
  draft IDs, and speculative index tensors against the original runner.
- Verification and fast-mode model suites each passed ten cases: short inputs,
  2,030/2,100-token boundaries, 8K context, mixed C4 contexts, and return to short
  requests. This is not a full-context or broad model-quality evaluation.
- The preparation configuration passed all twelve serving cases. After removing
  the four commit uploads, all twelve passed again: budgets 1/2/3, penalties,
  logprobs, uneven C4 completion, JSON schema, abort, and generation after abort.
- Full repository tests were not run.

The first preparation gate exercised only fallback because admission incorrectly
excluded the entire multimodal-capable model. Those v6 receipts are not fast-path
qualification. After correcting that guard, verification caught INT32 cached
logits indices where the runner requires INT64. The assertion propagated through
the worker and stopped v6 at server timestamp 18:48:38. The cumulative index dtype
was corrected and regression-tested. No successful qualification is claimed for
that failed version.

The service was restarted under the user's existing restart authorization into
`resident-live-v7`. Native owners, W4/HC/gather/geometry kernels, general/prefix
capture coverage, transition/history, queue, observer, device drafts, compact
staging, GDN graphs, and consolidated admission were restored and requalified.

Pure-metadata verification now restores the independently computed original
buffers and disables the candidate on a completed comparison mismatch, before
model execution. It records the error and prevents promotion. Native submission
failures still fail closed rather than replay uncertain work. Both paths have
host regression tests.

## CPU profile evidence

Each trace is one 128-output-token HTTP request with profiling enabled. The old
trace has 53 model iterations; the final trace has 52. Counts below are rank 0.

| Operation | Prior retained v6 | Preparation graphs | Plus unused-upload removal |
| --- | ---: | ---: | ---: |
| Model iterations | 53 | 53 | 52 |
| Original speculative-metadata constructions | 52 | 1 | 1 |
| Previous-position calculations | 106 | 55 | 54 |
| `CpuGpuBuffer.copy_to_gpu` calls | 1,271 | 914 | 689 |
| Mean input-preparation cumulative time | 1.925 ms | 1.650 ms | 1.656 ms |
| Mean legacy commit staging cumulative time | 0.056 ms | 0.057 ms | 0.013 ms |

The single remaining speculative-metadata construction is a fallback step.
The earlier progress update saying zero was based on a filtered listing that
excluded single calls; the table above corrects that count.

Normalized host-buffer upload calls fell from about 24.0 to 13.3 per iteration,
roughly 45%. This is a reduction in host dispatch calls, not a claim of 45% lower
CPU time, transferred bytes, or inference latency. Generic tensor copies are
not all represented by `CpuGpuBuffer.copy_to_gpu`. Profile timings include
instrumentation and waits; nested timings must not be added as independent costs.
The older reference precedes the recovery restart, so use the operation counts
as the stronger evidence of removed work.

Raw profiles are retained under `resident-live-v7/cpu-scheduling-*`. The
[comparison](resident-live-v7/cpu-preparation-comparison.json) includes all ranks
and distinguishes same-named functions by source line.

## Unprofiled serving measurements

C1 is median HTTP output throughput from three 256-token requests. C4 is one
finite batch of four simultaneous 256-token requests, including prefill/tails.
All rows below use the restored v7 service. These are small trials without
confidence intervals, and the final upload-removal trial is not a separate A/B.

| Configuration | C1 HTTP tok/s | C4 aggregate HTTP tok/s |
| --- | ---: | ---: |
| Preparation graphs off | 31.164 | 65.786 |
| Preparation graphs on | 31.804 | 64.647 |
| Retained, including unused-upload removal | 32.509 | 66.511 |

The isolated preparation A/B was mixed at C4. The user explicitly requested
retaining verified CPU reductions even when short throughput measurements do not
show a clear gain. Both changes remain enabled.

## Current service and remaining CPU dependencies

[Final inspection](resident-live-v7/final-cpu-reduction-status.json) reports health
200, unpaused serving, no dirty model graphs, and all CPU/native profilers released.
Preparation recorded 1,430 replays and 194 verified replays per rank. The commit
staging owner recorded 595 calls and 2,380 avoided uploads per rank.

- API PID: 2167990; engine PID: 2168713.
- Worker PIDs: 2169038, 2169039, 2169040, 2169041.
- Model/Kilo entry: `qwen38-flash-next` / `vllm-310p-w4/qwen38-flash-next`.
- Context setting: 262,144; TP4/EP4; MTP2; general/prefix captures `[3,6,9,12]`.
- Stable hardware log: `/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/server-current.log`.
- Actual log: the same root followed by `resident-live-v7/server.log`.

The earlier long qualification/benchmark run reached 89 C on the hottest chip,
with all devices reporting OK. Load ended and temperatures returned to 66–72 C
before the final upload-removal qualification. No persistent stress workload or
profiler is left running.

Remaining dependencies are per-step CPU schedule/lifecycle work, compact-state
mapping/admission, checkpoint preprocessing and accepted-count synchronization,
common attention metadata, PLE host lookup, and graph submission. The accepted
counts cannot simply bypass the host upload: checkpoint preprocessing can reset
them after moving recurrent state. Moving that normalization and pre-copy decision
together is required for the next safe removal of the count round trip.

The next larger boundary is a pre-admitted execution window with device-owned
positions, state indices, normalized counts, and termination status. That must
preserve page ownership and fallback/cancellation semantics. Moving work to a
CPU background thread or native CPU wrapper alone does not meet that requirement.

## Restore controls

The restoration procedure is `restore-v7-driver.py`, following launch with
`start-qfn-engine-control-candidate.sh --run-root NEW`. Its v7 paths preserve v6
failure evidence; create fresh paths/labels for another restart. Qualification
is `qualify-preparation-v7.py`, then
`qfn-host-commit-metadata-control-v7.py on --label FRESH` and the serving gate.
Current inspection uses `qfn-cpu-reduction-final-status-v7.py`.

These controllers use the private token on the hardware host. No token is
archived here. Hot controls are not automatically restored by historical launchers.
