# GLM: next decode and cold-prefill opportunities

## Scope

Initial offline source and saved-profile review, 2026-10-05, used no NPU
requests or server changes while the user tested. The subsequent stall report
led to diagnosis and a bundled relaunch, recorded below. Decode tok/s and
cold TTFT have equal priority.

Starting point: native live kpool scoring, MTP1, graph sizes `[2,8]`, native
mHC **prefill** post, overlap W2/W4 OPP, 640-token chunks, 311,040 configured
context. The preceding resident comparison measured 4.872 tok/s c1 and
10.794 aggregate tok/s c4, with 17/20 quality. These are existing results,
not new measurements. Full-length context has not been tested.

## Priority and concrete implementation targets

| Priority | Decode work | Cold-prompt work |
| --- | --- | --- |
| First | Finish selector fusion: query rotation/casts, mask/index expansion, live-length selection | Tile prefill scoring and fuse the 32-head reduction before writing device memory |
| Second | Specialize native mHC for 1/2/8 rows; fuse KDA input normalization/gates | Increase actual expert batch to 1280 with both route caps raised, using the qualified mHC workspace reduction |
| Third | Reduce W3 unpack traffic on small expert groups; preserve reuse across verifier rows | Complete resident W3 pipeline and activation/combine fusion with existing lifetime fixes isolated |
| Later | Qualify MTP2 graphs by accepted tokens per millisecond | Reduce KDA stage traffic after profiling the current runtime |

These priorities are based on source structure and earlier measurements.
Current full-model critical-path attribution requires a future authorized
trace. No new serving speedup is predicted from the arithmetic below.

## 1. Decode: the selector still has removable work

`sparse_attn_indexer_kpool.py::_select_tokens_fixed` now uses native live
scoring, but calls `select_kpool_groups` and `expand_kpool_groups` separately
for every verifier row. At c4 this means eight top-k calls, eight full-width
validity masks, eight masked score copies, and separate expansion/copies.
The native scorer already initializes unused scores to negative infinity.

A CPU meta-tensor audit of the actual helpers found:

- Query rotation: seven adds, seven subtracts, seven stacks, normalization,
  and three dtype copies across FP32/BF16/FP16.
- Selection/expansion for eight rows: eight top-k calls, 24 masked-fill calls,
  24 arange tensors, eight concatenations, plus casts and elementwise operations.

`offline-audit.json` contains the full ATen census. Views are included in its
raw counts; ATen counts are **not NPU launch counts or timings**. Buffer clear
and final index-buffer copies are outside the census.

Stage independently:

1. Fuse Hadamard butterflies and casts into a small native preparation op,
   preserving intermediate BF16 rounding and FP16 edge behavior. Putting
   the rotation inside every scoring core would duplicate work; compare that
   with preparing each query once.
2. Give native scores a selection path that avoids remasking an already
   masked full-capacity tensor and fuses tail/padding expansion.
3. Implement live-bound exact selection so short requests do not scan all
   77,760 configured pools. Keep graph shape fixed while reading live bounds
   on device. Radix or hierarchical selection needs an explicit tie contract.

Simply batching top-k is not proven equivalent on 310P. The current path
deliberately preserves per-row shapes and boundary-tie behavior. Negative
padding indices, duplicate padding aliases, invalid pages, request changes,
and threshold crossings must remain in the gate. A dense short-context bypass
can avoid scoring altogether, but changes score-sorted index order; even when
the selected set is identical, attention accumulation can change. Qualify it
as a distinct numerical candidate rather than calling it bitwise equivalent.

## 2. Cold prompts: eliminate head-sized score intermediates

The native scorer currently supports at most eight graph rows. Prefill still
uses `kpool_ops.py::score_and_select_kpool_tokens`, which calls `score_kpool`:

1. Rotate queries, round to BF16, convert to FP32.
2. Convert the gathered pooled keys to FP32 for each query subchunk.
3. Materialize FP32 `[queries,32,pools]` matmul logits and apply ReLU in place.
4. Materialize another equally sized tensor for head weighting, then reduce.

For a 640-token chunk near each live context length:

| Live context | Query subchunk | Score calls | Writes for logits + weighted logits | Largest single head tensor |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1024 (640 used) | 1 | 0.3125 GiB | 160 MiB |
| 32768 | 256 | 3 | 1.25 GiB | 256 MiB |
| 131072 | 64 | 10 | 5.00 GiB | 256 MiB |
| 311040 | 26 | 25 | 11.865 GiB | 246.8 MiB |

Formula: `2 * 640 * 32 * floor(live_tokens/4) * 4` bytes. This is logical
tensor-write volume per indexer call, not peak resident memory or measured
HBM traffic. It excludes reads, ReLU writes, query/key conversion, top-k,
and other attention work. The actual prompt length, rather than configured
capacity, controls prefill. The existing dense bypass applies below its
selection budget and is not included in this table.

Target a query-tiled Cube scorer: load each key tile once for a small group
of queries, then ReLU, weight, and reduce heads on chip. Write only reduced
`[queries,pools]` scores. At 311K this final score volume is about 190 MiB for
640 queries, versus 11.865 GiB for the two head tensors. Keep scores tiled
across query rows to bound peak workspace before existing selection.

Increasing the existing decode kernel's eight-row limit alone is insufficient:
it currently loops over rows and reloads keys for each one. The prefill design
must reuse keys across queries, handle per-query causality and request page
boundaries, and preserve BF16 rounding. This targets long-prompt scaling as
well as current 8K TTFT. It does not eliminate necessary sparse-attention work.

## 3. Larger real expert batches: coordinate all limits

Routing already groups all available rows by expert and processes each expert
as a group. The duplication occurs across scheduler/operator chunks.
Historical four-times-640 versus one-times-2560 W3 projection measurements
were 292.26 versus 91.66 ms for uniform routes, but only 132.18 versus
113.00 ms for concentrated routes. These exclude the rest of the model.

The current comparison launcher accepts `--batch 1280` but passes no
`ascend_glm_grouped_max_routes` override and retains the overlap OPP. Main's
Python and host defaults are 6144 routes: 768 top-8 tokens. Raising scheduler
budget alone can split 1280 into 768+512; an older deployed host cap can also
reject admission. A valid experiment needs at least **10240 routes in both
Python and the compiled host adapter**, plus source/binary identity checks.

The scheduler also rounds intermediate chunks to the cache block boundary;
the earlier 1024 budget actually ran 640-token chunks. Check actual iteration
tokens and operator group sizes, not just launcher arguments.

Keep decode's kernel path unchanged in this experiment. Prior 1280/resident
W3 serving improved cold TTFT but regressed c4 by 26.6%; those simultaneous
changes did not isolate the cause. Native mHC post now reduces a previously
identified prefill workspace pressure, but does not prove 1280 will fit at
311K context. Measure real cold-prefill peak memory and c1/c4 separately.

### Local correction prepared

`expert_core_groups.h::AllowResidentW3Prefill` used a 32-route cutoff from
the pre-MTP four-token graphs. MTP1 c4 has eight verifier rows and **64 routes**,
so that candidate would accidentally select resident W3 during decode.
The cutoff now derives from four requests × two verifier rows × top-8.
The CPU C++ regression covers all four request counts and boundaries 63/64/65.

Validation: `python -m pytest --noconftest -q
tests/ut/deepseek_w2/test_expert_core_groups.py` — **1 passed**, including the
existing exhaustive expert-team ownership checks. No CANN rebuild or NPU
qualification was performed. This compile-gated candidate is not enabled in
the running overlap package; the correction is not a claim about its speed.

## 4. Fusion for both phases

- **mHC:** the native post dispatch deliberately starts at 640 tokens. Decode
  still uses the previous post mixer and round trip. Benchmark a small-row
  specialization under graphs; do not lower the cutoff based only on prefill
  results. The CPU streaming alternative previously regressed small batches.
  Next fuse the pre-stage residual reduction and normalization; the existing
  Sinkhorn loop is already native, so replacing it again duplicates work.
- **KDA decode:** `_run_recurrent` launches q/k normalization, layout/casts,
  gate sigmoid/scaling, and beta sigmoid before the recurrent op. Fuse those
  into input preparation or the consumer while retaining accepted-state slot
  selection and speculative rollback. The prepared gate weight constants
  already exist and must stay out of graph-capture casts.
- **KDA prefill:** the old FP16 profile attributed 3.449 s across two chunks
  to KDA/convolution versus 5.349 s to grouped packed projections. These are
  overlapping task sums from an older server, not today's bottleneck shares.
  Audit actual loaded OPP stages before porting any newer source schedule.
- **Expert activation/combine:** staged fused SwiGLU and FP32 route reduction
  already have operator measurements and a problematic serving history,
  including command-lifetime fixes. Requalify them separately with the current
  adapter; do not bundle those old candidates and count them as new work.

`disable_recompute=False` in KDA is not evidence of redundant inference
recomputation. The adapter uses it to omit exported intermediates; flipping
it creates additional tensors. Internal state required by later stages is
still necessary even when those public outputs are disabled.

## 5. MTP: measure marginal benefit, preserve full graphs

Saved native short-suite metric intervals 11:35:37–11:37:57 reported
601 accepted / 655 drafted tokens, approximately 91.8%. These logger windows
cross request boundaries and are not an exact per-request acceptance metric.
They support investigating verification cost rather than assuming poor
acceptance is the main issue.

MTP2 is not a launcher-only change: full-graph validation currently requires
MTP1, the selector admits at most eight rows, and c4 MTP2 needs twelve verifier
rows. The fixed `[2,8]` capture sizes do not qualify that profile. Revisit
selector bounds, graph sizes/HCCL budget, state slots, rejection rollback,
route guards, and memory together. Use expected committed tokens divided by
draft + verification + scheduling time; higher acceptance alone is not speed.

## Future hardware gate

Coordinate hardware access with the user. Keep the current server available.
Use one-variable comparisons, c1 and c4 256-token decode, cold/warm 8K plus a
longer cold prompt, actual expert group sizes, peak memory, MTP acceptance,
and the existing quality/tool gates. No full reload just to repeat an already
qualified baseline. Identify the new decode critical path before attributing
any expected model-level gain to individual operators.

## Bundle prepared after the user reported a stall

The offline review was interrupted by the user's KiloCode request stalling.
The user requested that new improvements be bundled before any restart.
Read-only diagnosis found device page faults at 11:51:40 and an AI Core bus
error at 11:51:45, after 26 completed 640-token prompt chunks (16640 tokens).
Worker stacks wait at the MTP draft-copy completion event; that wait does not
identify the earlier failing operation. The faulting address was an unmapped
reserved device address, not a reported Python allocation OOM. Collected bbox
data also contains older OOM records; those must not be attributed to this run.
Evidence is in `../kpool-live-score-20261005/stall-115140/`.

The runtime bundle contains three focused Python changes:

1. **Continued-prefill page layout:** pass physical shared KV page views and
   the logical KV head count, matching the existing decode contract. The old
   prefill path passed the narrower logical view to a kernel that calculates
   page stride from shape. CPU regression demonstrates correct page addressing
   without a cache copy, including nonzero storage offsets. This is a concrete
   discrepancy, but not yet a hardware-proven explanation for the reported
   fault.
2. **Prefill scratch reuse:** multiply head weights into the private logits
   allocation. This removes the second head-sized allocation (up to 256 MiB),
   not the multiplication or its writes. Widen pooled keys once per chunked
   scoring call instead of once per query subchunk. Matmul shapes and BF16
   query rounding are unchanged.
3. **Decode selection:** native scores already have inactive lanes set to
   negative infinity. Skip rebuilding their full-width mask/copy, retaining
   the same per-row top-k shapes and index/rank padding checks.

The separate W3 prefill-only threshold fix is staged in source and requires
a future CANN build; it is not part of the current overlap binary.

Validation: **41 CPU tests passed**, including input preservation, exact CPU
score equality, query-chunk equivalence, one key conversion per bank, tied
scores, invalid top-k padding, physical page addressing, existing native
scorer dispatch, and expert-core ownership. Targeted Ruff checks passed.
`offline-audit.json` describes the pre-bundle source. No NPU parity or new
throughput result is claimed for this bundle yet.

Bundled relaunch: API PID **471919**, startup completed **12:14:29 UTC**,
port **8001**, configured **311040**, MTP1, full decode graphs `[2,8]`.
Capture completed in four seconds and recorded 0.55 GiB. The three runtime
source hashes match `bundle-source-hashes.json`; worker affinity is recorded
in `affinity-public311k-bundle.json`. No inference request was sent to this
launch by the agent; the user will retry KiloCode. Startup alone does not
establish long-prompt recovery or a throughput improvement.

## Repeated stall and offline isolation, 2026-10-05

The user's bundled-server replay stalled again after iteration 25 at
14:29:23 UTC. The request had processed 26 chunks of 640 tokens (16,640).
Before the hardware deferral, the driver log again reported unmapped
reserved address `0x12db37200000`, this time for workers 474123, 474508
and 474876. **The previous bundle did not fix the stall.**

Hardware use and restarts are deferred at the user's request. Subsequent
work used local CPU tests and read-only access to existing files; no new
NPU operation, server launch, or inference request was made.

One hypothesis to isolate is pooled-key top-k. The original dump includes
`0x44000` and two addresses separated by `0xaa0000`. Those values are
compatible with scratch dimensions involving a padded 4,352-column
score tensor and 640 rows. This arithmetic does not identify the kernel:
the normal score-memory cap also splits 640 queries with 32 heads and
4,320 pools into 485/155-row score calls. No operator name or actual input
manifest has yet established that top-k caused the fault. KV addressing
and queued tensor lifetimes remain possible causes.

Prepared code, **not deployed or promoted**:

- `tools/glm_perf/probe_kpool_prefill_boundary_310.py`: one-shape process
  probing top-k independently, then the complete score/selection path.
  It does not load model weights. Unique-score top-k checks exact CPU
  indices; score/selection checks causal, distinct pools, not full model
  quality or numerical parity.
- `tools/glm_perf/trace_device_ops.py`: opt-in per-operation submission
  and completion logging with tensor storage addresses, sizes, strides,
  and offsets. It logs no tensor contents. A boundary synchronization
  separates earlier work from the first traced operator. Synchronization
  can hide a lifetime race, so a successful trace is not a correctness gate.
- `tools/glm_perf/resident_candidates/kpool_bounded_topk.py`: reversible
  experiment limiting top-k to 32 query rows per call, preserving the full
  candidate column set. Default runtime dispatch remains unchanged.
  CPU tie/order parity does not establish NPU tie/order parity.

CPU validation: **54 tests passed** across the new diagnostics, existing
score-workspace suite, and native scorer dispatch suite. The new tests cover
widths 4096/4160/4320/4352/77760, exact index parity, tied scores, 640-row
batches, partial final batches, strided inputs, and trace failure ordering.

Next hardware gate, only after permission to use devices resumes:

1. Run isolated top-k shapes around the boundary with rows 640, 485, 155,
   32 and 8, using one process per shape. Use an external timeout and stop
   the sweep after the first device error. Start with the default path;
   then compare `--rows-per-call 32`.
2. Repeat the score/selection probe without tracing, then with `--trace`.
   Trace timing is diagnostic and must not be reported as throughput.
3. If isolated probes pass, trace a bounded full-model prefill at the
   failing boundary, retaining full decode graphs. Attribute the first
   failing operation before changing kernel binaries or allocating more KV.
4. Require an actual prompt beyond the failed boundary before declaring
   recovery. Neither CPU parity nor startup readiness qualifies the fix.

Example command for that future gate (not executed):

```bash
timeout --signal=TERM --kill-after=10s 120s python -m tools.glm_perf.probe_kpool_prefill_boundary_310 \
  --device 0 --rows 640 --pools 4320 --stage topk
```

## Hardware attribution and draft-table correction

After the user released the NPUs, isolated top-k and score/selection at
640 rows and 4320 pools both passed three repeats. The row-bounded top-k
candidate was **not enabled**.

An instrumented 20K retrieval then reproduced the device fault during the
**draft forward at 16640 tokens**. The target forward completed. CANN named
the faulting kernel explicitly:
`QsaSparseAttentionV310_69bafdd660f83d102d2ce54286e7ad4e_0`.
The program start address matched the original dump. Synchronizing at the
operator boundary surfaced the error instead of leaving workers waiting
indefinitely at the later MTP event.

Root cause: `_draft_block_table_width` treated the MLA cache specification's
physical page count as the width of the runner's expanded kernel-block
table. The indexer branch already handled expansion, but the MLA fallback
did not. For 311040 configured tokens:

- 486 scheduler pages, 640 tokens per page;
- 20 kernel blocks per scheduler page, 32 tokens per kernel block;
- required runner table width: **9720**, incorrectly cropped to **486**;
- QSA converts that cropped table back to physical pages: only **25**
  entries, covering **16000 tokens**;
- the next 640-token prefill crosses its final entry and reads an invalid
  device address. This explains the repeatable boundary without relying on
  score-workspace or top-k hypotheses.

The fix multiplies the physical page count by the runner table's
`blocks_per_phys_block`. A host-metadata check in continued MLA prefill also
rejects a table shorter than the visible context before QSA dispatch. It
adds no device-to-host read or synchronization.

Validation completed before the public restart:

- **46 CPU regressions passed**: draft width/cropping, failure guard, prefill
  physical pages, and MTP MLA graph metadata.
- Native QSA probe using the production metadata methods passed **3/3
  repeats** at visible lengths **16000, 16001, 16640, and 311040**, with zero
  maximum error against the constructed reference. This is an operator
  addressability test, **not a full-model 311K prompt test**.
- Broader existing proposer/MLA tests failed during collection because this
  venv lacks `fla_npu`; no pass is claimed for that invocation.
- Targeted Ruff and `git diff --check` passed.

Evidence and the focused runtime patch are in
`../kpool-live-score-20261005/boundary-isolation/`. The corrected public launch
is `serve-public311k-mtp-width-fixed.log`, API PID **1072210**, port **8001**,
311040 configured context, MTP1 and full decode graphs.

The cold full-model retrieval completed successfully at 16:36:38 UTC:
**20487 prompt tokens, zero cached, exact answer `BLUE-ORCHID-7319-5`**.
Client TTFT was **219.952 s**; server prompt time was **219.444 s**, or
**93.4 prompt tok/s**. It crossed the former 16000-token failure boundary
and returned 13 completion tokens. The short answer is not a replacement
for the established 256-token decode benchmark. Server request timing
reported 4.9 decode tok/s; client streaming timing reported 6.46 tok/s,
using a different interval and token denominator.

The launch retains live pooled-key scoring, native mHC prefill post, MTP1,
full decode graphs `[2,8]`, CPU affinity, and 640-token prefill chunks.
The prior qualified live-selector run measured 89.842 s cold 8K TTFT and
4.872 c1 / 10.794 aggregate c4 decode tok/s. That selector optimization
targets decode; it does not implement the separate native prefill scorer
or larger expert-batch experiment. No prefill speedup is claimed for this
metadata repair. The public server remains running on port 8001. A full
311040-token model prompt remains untested.
