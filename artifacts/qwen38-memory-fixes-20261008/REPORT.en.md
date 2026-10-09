# Qwen 310P: offline fixes for all P1 memory findings

All ten P1 areas from the [audit](../qwen38-memory-audit-20261008/REPORT.en.md)
have concrete fixes or opt-in mitigations with CPU regression coverage. No
server was launched, stopped, hot-swapped, or reconfigured. No NPU workload was
run. These changes are **offline validated, not hardware qualified**.

## Changes and remaining hardware gates

| Finding | Implementation | Default / gate |
|---|---|---|
| A01 QSA selected K/V | Honor explicit batched query tiles; fixed-shape shared group union avoids dynamic output planning and duplicate group reads. | Default remains 64-query batched gather. Both alternatives require NPU timing. |
| A02 MTP counts | Explicit grouped W8A8 draft path can stay inside capture for at most eight rows; larger batches retain eager dispatch. | Off; requires grouped quantized weights and preparation before capture. |
| A03 PLE rows | Bounded persistent pinned host rows, asynchronous H2D, and copy-completion ownership before host reuse. | Off; configure staging capacity explicitly. |
| A04 GDN plan | Snapshot CPU scheduler bounds on fresh metadata and use them to build the per-step plan. | Active; legacy callers without the matching mirror retain their fallback. |
| A05 Mamba counts | Read raw sampled tokens once for bookkeeping, reuse raw acceptance before discard filtering, consume the snapshot once in matching request order. | Active for synchronous Qwen MTP/PLE only. Async and mismatched snapshots retain fallback. |
| A06 FULL replay | Opt-in completion events separate prior replay completion, host task mutation, update completion, and replay readiness. | Stream synchronization remains default. Events need hardware ordering and throughput gates. |
| A09 Compact tables | One pinned host stage per group copies directly into the persistent device table; removes the temporary device tensor and its D2D copy. | Active for compact mapped tables. |
| A19 Grouped MoE routes | Route temporaries expire per chunk; explicit logical scratch budget bounds chunks; optional fixed-output histogram counting avoids the route/expert comparison matrix. | Per-chunk lifetime fix is active; existing chunk size and comparison counting remain default. |
| A22 GDN WY | Reuse the existing contiguous key operand; fuse layout conversion with FP32 casting for value, gate, beta and expanded keys. | Active; accumulation and recurrent state remain FP32. |
| A24 HC operands | Explicitly reuse a narrowed normalized operand between down and injection projections when their operand dtypes match. | Off; FP32 gated mean, gradient path, and mismatched-dtype fallback are preserved. |

Also fixed the nonuniform GDN state-index branch's undefined `seq_lens` and its
bound-method `is_pinned` check. Mamba copies now bypass cloning for disjoint
contiguous views, skip exact self-copies, and retain cloning for overlap or
strided views. No native kernels, weights, selection policy, image processor,
cache capacity, or server launcher were replaced.

The runner owns sampled-token snapshots because it has the authoritative
request order and bookkeeping lifetime. It owns graph ordering because it
submits parameter updates and target replay. The shared Mamba fallback remains
compatible with callers that do not supply a snapshot. All storage and event
state belongs to a layer or runner instance; no new environment variables or
mutable module globals were introduced.

## Logical bounds, not measured DDR or thermal results

At TP4 geometry and a 2,560-token chunk:

- Explicit batched QSA tile 8 reduces selected K/V scratch from 129 MiB to
  16.125 MiB. This changes peak scratch, not total per-query KV traffic.
- Fixed union tile 8 has capacity 4,104 groups, about 16.047 MiB selected K/V
  scratch. It reads each distinct selected group once, while still allocating
  and masking unused capacity. Poor overlap can increase QK/PV work by roughly
  eight times. It must beat batched gather before promotion.
- PLE capacity 2,560 reserves 12.5 MiB of pinned host rows per rank. The same
  12.5 MiB still crosses H2D per full chunk; pinning removes pageable staging
  and enables asynchronous submission, not the required transfer itself.
- A 128-MiB logical MoE route budget derives a conservative smaller chunk;
  see [bounds.json](bounds.json) for its exact size. The bound excludes native
  GEMM workspaces, weights, caller buffers and allocator fragmentation, and
  is not a limit on total device memory. Smaller chunks can hurt throughput.
- CPU operator tracing counts five legacy FP16 WY layout clones versus two
  after the fix at local K/V heads 4/12. The key, value and beta copies removed
  total about 10.06 MiB per layer/chunk; NPU lowering remains to be measured.
- Matching HC projection dtypes share a 50-MiB FP16 saved operand instead of
  retaining the 100-MiB FP32 normalized operand, and avoid its second narrowing.
  The gated mean still consumes FP32 values. Total peak or graph-pool allocation
  reduction has not been established.

These bounds do not establish physical memory-bus traffic, heat attribution,
new serving throughput, or cooling behavior. Required stream waits, overlap
protection and FP32 state were retained. The existing 94°C hold / 85°C resume
controller is still deferred; its hardware cooling gate remains outstanding.

## Validation

The focused fork suite passed **307 tests**, with one existing `msmodelslim`
dependency skip. See [offline-tests.log](offline-tests.log). Tests execute real
production logic, including immutable mixed-partition metadata, state-copy
aliasing, acceptance/discard ordering, snapshot expiry and exceptions, pinned
buffer lifetime, graph readiness and exception cleanup, route ownership,
causal-tail masks, HC precision, and production 4/12-head GDN WY geometry.

The workstation's normal unit-test bootstrap cannot load with its installed
vLLM: `vllm.third_party.flash_linear_attention` is missing. A direct MTP test
import also requires unavailable `torch_npu`. The focused tests use CPU-only
imports and AST isolation for hardware-coupled functions. Neither isolated
functions nor mocked events establish hardware correctness. The entire
repository/NPU test suite was not run under the offline restriction.

The source changes were separately applied to a local copy of the frozen
serving runtime and compiled. Its QSA policy returns two values; current fork
policy returns three. The [compatibility patch](audited-runtime-compatibility.patch)
adapts that single context line while preserving the runtime's two-value API.
The runtime source checks passed **75 tests**, with one main-only policy test
deselected. Patch/source fingerprints are recorded in
[runtime-validation.json](runtime-validation.json) and
[runtime-tests.log](runtime-tests.log). The main-only QSA three-value policy test
was deselected there; the runtime lacks that earlier parallel-gather option.
The patch has not been deployed. Upstream `SamplerOutput` was verified to be a
dataclass in the pinned vLLM commit, so its replacement preserves logprobs.

Required `bash format.sh ci` failed on existing repository-wide Ruff, spelling,
Clang, Markdown and forbidden-import issues, plus a spelling false positive in
a retained patch hash. It changed 144 unrelated files
only in the isolated check tree; those edits were discarded. Focused owned-file
hook results are recorded separately in [scoped-checks.log](scoped-checks.log).
Exact patch/fingerprint/log receipts are checked separately with spelling
hooks skipped for opaque hashes and verbatim diagnostics; [raw-checks.log](raw-checks.log) records that scope. Python compilation and scoped Ruff checks
passed. No dummy or real-weight serving gate was attempted.

## Deferred validation order

[queued-profiles.json](queued-profiles.json) contains disabled partial overrides,
not a launcher or permission to start a server. Preserve the image-enabled
configuration and merge only the particular fields being tested.

1. Validate the default semantic fixes against the last qualified runtime with
   real weights, MTP rollback, mixed requests, prefix reuse and an image request.
2. Gate PLE pinning, HC sharing, smaller batched QSA tiles and MoE budgets
   individually, collecting all-rank timing, copy and temperature receipts.
3. Gate grouped MTP separately; W8A8 activation quantization changes the draft
   numerics and may change acceptance and end-to-end throughput.
4. Test fixed QSA union and graph completion ordering separately. Exercise both
   capture sizes, row reordering, cancellation and repeated replay.
5. Combine only passing candidates, then run sustained mixed image/text load
   with the thermal hold controller. Do not infer a thermal pass from smoke.

Future promotion requires NPU accuracy, stream/capture lifetime, all-rank
throughput, and sustained thermal evidence. There is no new capacity or 1M
context claim. The English and [Chinese report](REPORT.zh.md) describe the same
changes and limits.
