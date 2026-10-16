# TP6 mixed profiling and shared-RAM prefill PLE

The October 16 resident server now gathers prefill PLE rows from the same
read-only RAM table used by native decode. The previous prefill path still
read checkpoint shards row by row through a thread pool. Hashing, pinned
upload, projection, native decode transport, scheduler and captures are retained.
Each worker adds a virtual read-only mapping of the existing 102,400,491,520-byte
table; this does not allocate another physical table or pin it again.

## Hardware results

All six 310P workers passed 96 private comparisons: ten row-gather cases and
six complete PLE cases per worker. Checks cover empty output, first/last rows,
duplicates, random IDs, mixed request boundaries, EOS and actual loaded PLE
weights. Complete outputs match bit for bit. The table below reports the median
of six per-rank medians, with six alternating timing samples per implementation.
These are synchronized complete PLE calls, including transfers and projection,
not complete-model throughput measurements.

| Tokens | Checkpoint path | Shared RAM | Reduction |
| --- | ---: | ---: | ---: |
| 65 | 14.726 ms | 9.024 ms | 38.7% |
| 131 | 23.747 ms | 10.423 ms | 56.1% |
| 134 | 21.824 ms | 10.479 ms | 52.0% |
| 256 | 42.396 ms | 15.779 ms | 62.8% |
| 640 | 80.197 ms | 25.564 ms | 68.1% |
| 2048 | 266.948 ms | 63.118 ms | 76.4% |

The live binding additionally checked four real prefill gathers per rank against
the original reader, returning the reference during those checks. All 24 passed.
Four API smoke requests returned nonempty outputs and eight completion tokens
each. A mismatch disables the alternative for that binding and keeps returning
the original reader's output. No model-quality or 256K accuracy claim follows
from these exact local comparisons.

The first private attempt found that NumPy's empty indexed array had zero
strides unsuitable for a dtype view. The implementation now returns an ordinary
empty Torch tensor, covered by a regression test. Two subsequent harness attempts
used the wrong HC input dimension/name and were rejected before projection.
The successful gate uses the loaded layer's `hc_hidden_size` (10240).

## Controlled end-to-end A/B

After the user released the server for exclusive NPU tests, an unprofiled
ABBAAB sequence compared the checkpoint reader (A) and shared gather (B).
Each phase ran two isolated uncached prefill requests and one mixed pair:
a 512-token decoder, followed after eight content updates by a new prefill
request. Prefill requests generated sixteen tokens. Every prompt used a fresh
leading nonce; all measured requests reported zero cached prompt tokens.
Existing unrelated prefixes were preserved. No other user requests ran during
the measurement window.

| Measurement | Checkpoint | Shared RAM | Trials per path |
| --- | ---: | ---: | ---: |
| Median isolated prefill TTFT | 6.657 s | 6.384 s | 6 |
| Isolated prefill TTFT range | 6.608–6.692 s | 6.333–6.405 s | 6 |
| Median prompt tokens / TTFT | 491.9 / s | 513.9 / s | 6 |
| Median mixed prefill TTFT | 25.992 s | 25.451 s | 3 |
| Median maximum decoder content-update gap | 0.703 s | 0.687 s | 3 |

The isolated prefill median improved **4.1%**. Prompts contained 3272–3282
tokens; the normalized rate also improved. Mixed-prefill ranges overlap
(A: 25.804–26.235 s; B: 24.754–26.045 s), so the approximately 2% median
improvement is tentative. No pure-decode throughput gain is claimed. Streaming
updates can contain multiple speculative tokens; their gaps are not individual
token latencies. These short prompts do not establish long-context scaling.

The controller restored the enabled path after every failure or completed
sequence where the underlying control transaction allowed it. Final status
shows HTTP health 200, unpaused serving, 97 shared-gather calls per rank, four
successful shadow checks per rank and no failures. Worker PIDs and all twelve
captured graphs per worker remain unchanged. See `ab-summary.json`,
`ab-requests.json`, and `ab.py`; full event timestamps remain on the host.

## What the live profile established

A bounded host profile captured 423 execute calls per rank, with no dropped or
failed spans. It recorded shapes and counts only, without adding device waits.
There were 217 three-token and 176 six-token target forwards in `FULL` graph
mode. Twenty-one 134-token and four 131-token mixed forwards used `NONE`.
Schedules confirm these are 128 prefill tokens plus two or one three-token
decoders. Median mixed forward host duration was approximately 0.66 seconds.
The prefill gather span was approximately 15–16 ms; native decode's host
produce/fetch call was approximately 0.11 ms. Its enclosing lookup also waits
for the existing Torch stream, so lookup time is not a DMA-only measurement.
Nested spans overlap and must not be summed as independent device costs.

The synthetic workload overlapped a live user request: its uncached 3275-token
prompt had 51.36 seconds TTFT, and the 512-token decoder's largest content-update
gap was 0.926 seconds. This is a contention observation, not an isolated baseline
or an end-to-end before/after comparison for the retained change.

## Collective experiments rejected

Larger native owners were rejected before allocation: measured free P2P pools
were 2,088,960–2,097,152 bytes, below the 3,731,616–6,414,624 bytes required even
for the 131-token FP16 case. Ordinary free device RAM does not establish P2P
allocation capacity. The private gate now derives inbox memory from each rank's
actual schedule degree; the TP6 binomial root has three peers.

A second experiment reused existing fixed owners and concatenated exact chunks,
requiring no additional peer buffers. It passed the representable finite SUM
patterns but changed rounding on general random inputs. It was also slower:

| Rows / dtype | Chunked native | Retained in-place HCCL |
| --- | ---: | ---: |
| 131 / FP16 | 1454 us | 720 us |
| 131 / FP32 | 1564 us | 831 us |
| 134 / FP16 | 1532 us | 708 us |
| 134 / FP32 | 1543 us | 850 us |

These are median cross-rank completion spans from five timing pairs of three
calls, including arrival skew and submission. Inputs for timing were zero to
keep repeated in-place SUM finite. Random-input maximum differences were
0.01171875 for FP16 and 9.54e-7 for FP32. The chunked alternative remains a
private diagnostic and is not installed in serving. HCCL fallbacks are retained.

## Reproduction and ownership

Use `tools/qwen4exp/ple_shared_gather_gate.py:engine_source()` to build the
private gate, then the authenticated controller in the evidence directory.
After successful qualification, build an `enable` or `disable` source with
`tools/qwen4exp/ple_shared_gather_control.py:engine_source(action)` and pass
`--apply` to that controller. Its existing deployment paths target resident-v8;
review them before reuse on another server. The controller reads its token
locally and never exports it. Enabling requires the qualified worker-owned
mapping from the gate; it is not a generic cold-start configuration change.
After a restart, requalify and reapply explicitly.

All operations require `pause?mode=wait&clear_cache=false`; requests complete
before mutation. There is no graph retirement, reset or cache flush. Scheduler
prefix hashes, KV storage metadata and Mamba tier metadata matched across the
private tests and hook installation/removal. Graph identities remained stable
through the profile. Each worker retains twelve graph entries. The API PID
remained 605287 and resident generation remained unchanged.

`resident_status` now exposes `shared_prefill_ple` with installed, failed,
checked, remaining and call counters. Disabling restores the exact instance
attribute state while retaining the mapping owner for safe lifetime handling.
The PLE implementation is in `tools/qwen4exp/ple_shared_gather.py`; the temporary
profiler is `tools/qwen4exp/tp6_mixed_profile.py`. The profiler was removed.

Evidence is in `artifacts/qfn-tp6-mixed-20261016/`. Raw host spans and controller
sources remain under the same artifact name on `/srv/ai/storage/artifacts`.
The initial profile and private gates are separate from unprofiled serving.
Focused host tests and changed-file lint are required before pushing; the
repository-wide check has existing failures recorded in the validation receipt.

## Validation limits

87 focused host tests passed (26 QFN profiler/gather/chunk tests and 61 private
collective tests). The complete hardware PLE comparisons, live shadows and
unprofiled A/B requests are separate evidence. The broad local UT conftest
cannot initialize because this workstation lacks the expected
`vllm.third_party.flash_linear_attention` module. Changed-file checks pass.
The whole-repository `bash format.sh ci` still reports existing Ruff, spelling,
Clang format, Markdown and forbidden-import failures outside this change.
Formatter-only edits to unrelated files were reverted. Fork Actions remained
disabled before pushing.
