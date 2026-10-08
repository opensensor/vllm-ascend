# Packed GLM MTP candidate

Date: 2026-10-04. Implemented on the shared main checkout. The initial CPU
stage deferred NPU use. The user subsequently authorized hardware testing.

## Current status: MTP repetition fixed and clean startup qualified

Latest: [repetition-fix-14](repetition-fix-14/README.md) identifies a gate-cache
lifetime defect. KDA decode lazily created weight-derived operands during graph
capture; another capture reused their storage without producing its contents.
An actual layer-0 gate became constant -2.5, prematurely decaying recurrent
memory. The fix prepares those operands after loading, before capture, with an
uncached fallback that never persists a forward intermediate.

With MTP1 and full graphs, the prepared resident candidate restored **17/20**
strict quality, with all 20 answers terminating normally; four concurrent
bounded generations and the tool-call case also passed. **12 targeted CPU
regressions and 2 native graph tests passed.** A clean ordinary serving startup
repeated the same results. The fixed MTP1/full-graph server is running at
**192.168.53.187:8001**, with the existing **32768-token diagnostic cap**. See
the linked study for final status and artifacts.

The resident harness qualified in [resident-13](resident-13/README.md) enabled
these comparisons without repeated weight loading. Its earlier 12 failing
requests are historical evidence from before this gate fix.

`mtp-graphs-07/` includes the MLA live-metadata fix and completed full decode
graph capture for `[2, 8]` on TP4. The revised NPU test, using the actual MLA
metadata builder, passed. End-to-end repetition is **not fixed**: arithmetic
failed on its first request and passed on immediate repeat; retrieval repeated
until its token limit. `hardware/smoke-graphs-07.jsonl` records those results.
These outputs do not qualify a throughput improvement.

The original metadata defect was real: MTP1 graphs have more tokens than
requests (2/1 or 8/4). Empty padding concatenations detached positions, slots
and page tables from the scheduler's live storage. The fix preserves those
views when no token padding is needed. Actual padding and explicitly unpadded
drafting keep their existing contracts.

- CPU regression: **2 failed, 3 passed** before; **5 passed** after.
- `hardware/graph-mla-live-metadata.log`: actual-builder NPU graph replay passed.
- `hardware/kda-reference.log`: speculative KDA outputs and every intermediate
  state match an independent CPU recurrence with acceptance counts `[2, 1, 2]`.
- `hardware/conv-reference.log`: speculative convolution graph replay matches
  an independent CPU convolution across acceptance changes, request movement,
  and an inactive request; state contents match exactly.

`mtp-graphs-08/` repeated the failure. Its experimental metadata audit
attached to `ACLGraphWrapper`, while this runtime uses
`BreakableACLGraphWrapper`; it produced no trace and is not evidence about
live metadata. That server was stopped for the user's hardware handover.

### Shared physical page corruption found by CPU tracing

The GLM planner aliases MLA and KDA cache tensors over the same physical page
pool; scheduler groups allocate distinct block IDs. The runner incorrectly
reshaped the shared KDA backing as dense components across all blocks.
Consequently, a KDA write for one block could overwrite a physical page owned
by attention or another KDA group. This affects aligned MTP caches; private
compact live-state caches keep their existing dense layout.

- `tests/ut/glm_w2/test_mtp_shared_state_pages.py` executes the runner's actual
  reshape branch: **6 failures and 1 pass before; 7 passes after**. The six
  failing cases write convolution/recurrent state at different block IDs and
  detect changes in neighboring physical pages.
- The runner now creates page-strided shared GLM state views. Native convolution
  and recurrent kernels address these views with the physical first-dimension
  stride. No per-token state staging or transfer is added.
- Native page-stride tests: **8 passed on NPU**, including independent CPU
  references, offset views with guard regions, and graph replay with changing
  acceptance counts, active requests, and state slots.
- Build artifacts: remote `glm-l1-wide-build-20261004/opp-mtp-pages-20261005`.
  Convolution's generated ACLNN ABI now includes `stateStride`, so the matching
  rebuilt torch binding is required. The old package and binding remain saved.
- `mtp-graphs-09/` captured full graphs (0.56 GiB) and passed **6/8** smoke
  requests. Single-request retrieval and subtraction still looped to 128
  tokens; first/repeated arithmetic and all four concurrent requests passed.
  See `hardware/smoke-graphs-09.jsonl`. This is **not** a repetition fix or a
  qualified speed result. The server was stopped for a bounded diagnostic
  retry (`mtp-graphs-10/`) using the actual `BreakableACLGraphWrapper` capture
  and replay methods.

Context remains an explicit **32,768-token test cap**, not a capacity
measurement. Worker affinity uses disjoint CPU groups with SMT siblings.

### Replay isolation and hardware pause

`mtp-graphs-10/` audited the actual target and draft graph wrappers. Across
48 bounded replay records, captured device metadata retained the live addresses
and values for positions, query boundaries, accepted counts, state slots and
block tables. The CPU indexer length metadata differed, but the fixed device
selector does not consume that field. This check does not establish cache
payload correctness.

`mtp-graphs-11/` also checked graph input addresses, then bypassed both target
and draft graph replay while retaining MTP. Retrieval and subtraction still
repeated; a nonstreaming retrieval request also repeated. Thus graph replay
and streaming alone do not explain the regression. This diagnostic bypass is
not an eager serving benchmark or a qualified performance configuration.

The user identified enabling MTP as the onset of repetition and subsequently
deferred NPU usage. Hardware requests and launches are paused. The existing
diagnostic server has not been promoted; its replay control was left at
`direct-both`. Remaining offline work targets verification/state rollback and
cache ownership. Native prefill writes into page-strided state and the
single-request, two-query attention path still need hardware qualification
when NPU use is authorized again.

## Hardware qualification history

The user requested skipping the eager serving comparison and proceeding with
full graphs. The initial eager startup loaded the target and packed draft
weights, then exited on a stale cache-spec equality check: a four-token pool
now needs a five-token retention window with MTP1. No eager requests ran.
The cache-spec check now allows the larger retention window.

The second real-weight launch uses port 8001, TP4, MTP1, full decode graphs
with capture sizes `[2, 8]`, synchronous scheduling, aligned Mamba caching,
a 640-token scheduler budget, and four sequences. Workers are pinned to
disjoint four-core L3 groups with their SMT siblings; the original masks
and applied thread bindings are saved in `mtp-graphs-02/affinity.json`. **32,768 is an explicit
initial test context cap**, not a measured capacity limit. Draft context is
clamped to this target setting. Early config alignment reports 384 tokens;
after real cache specs are available, workers select 640-token attention blocks.

Graph changes:

- The native recurrent kernel accepts a fixed per-request state-slot table.
  It directly reads the previous accepted slot and writes current output
  slots, eliminating the Python carry promotion and dynamic slot compaction.
  Existing flat-slot callers retain their original behavior.
- The opt-in `ascend_glm_mtp_full_graph` profile enables device-only decode
  selection using fixed cache bounds, current positions and request page
  tables. Both target and draft receive it. Prefill retains batched selection.
  This removes the host selection break but scans the configured context
  bound; its serving cost still needs measurement.
- The multi-cache-group proposer admits packed draft graphs with this profile.
  Initial graph scope is MTP1, up to four requests, and at most eight input
  tokens per capture. Other GLM draft profiles retain their existing behavior.

Validation so far:

- `cpu-graph-tests.log`: **64 passed** in the focused CPU suite.
- `hardware/graph-components.log`: **9 passed** on the matching runtime,
  including cache-spec tests and bitwise recurrent parity with/without graph
  capture. Replay changes acceptance counts, shortens queries, moves request
  rows, and supplies inactive padding.
- `hardware/graph-selector.log`: **1 passed**, checking captured selector
  replay with changed positions, dense/sparse boundaries, and request pages.
- The local cache-spec test cannot collect with the installed CPU vLLM
  (missing `kv_cache_spec_registry`); it passed in the matching remote runtime.

The native operator package is
`/srv/ai/src/glm-l1-wide-build-20261004/opp-mtp-state-20261005/packages/vendors/custom_transformer`.
It must precede the existing W3 OPP stack. Build logs and the first startup are
under `/home/matteius/experiments/glm-w3-20261004/mtp-hardware-01/`.
The first graph startup (`mtp-graphs-02/`) reached memory profiling after
loading 34.7258 GB of target/draft weights per worker and sharing the target
embedding and output head. It then failed because the proposer compacted
indexer rows for subsequent draft steps even with MTP1. A focused regression
check covers the fix: indexer reuse/compaction is enabled only with more than
one draft token. Retry logs and memory are in `mtp-graphs-03/`.
The next startup passed memory profiling: 5.49 GiB cache at fraction 0.70,
with an admission estimate of 96,981 tokens. It reached full graph capture and
exposed the MLA wrapper's single-query guard. The native QSA operator already
accepts multi-query request boundaries; the GLM decode wrapper now supplies
the current device boundaries and per-query causal pool/tail lengths.
`hardware/graph-mla.log` records **2 passed**: argument plumbing and an actual
NPU graph replay test against causal attention after changing request mapping.
The next startup (`mtp-graphs-04/`) reached draft graph metadata construction
but called the 310P CPU-only slot mapper with device positions. The packed
graph profile now computes secondary-group slots on device and copies them
into preallocated per-step buffers. Its first isolated graph check exposed
an unsupported `Range` dispatch; the mapper now reuses the proposer's existing
row-index buffer. The revised NPU graph check passed with changed request
boundaries, positions and page tables. The test also sets the same non-JIT
operator mode as the serving runtime.

- `cpu-slot-prelaunch-tests.log`: **6 passed**, including actual metadata-hook
  execution, stable buffer addresses, shortened batches and invalid slots.
- `hardware/cpu-proposer-prelaunch.log`: **19 passed** in the matching runtime,
  with NPU visibility disabled; existing non-packed proposer paths remain covered.
- `hardware/graph-slots-preallocated.log`: **1 passed** on NPU graph replay.
- Launcher defaults now agree on four requests for graph captures `[2, 8]`,
  and startup checks that the recurrent-state OPP package exists.

The user deferred launches during the offline preparation, then explicitly
re-authorized hardware. The next full serving attempt is `mtp-graphs-05/`.
That attempt passed secondary slot mapping and failed inside the draft's first
indexer forward: lazy FP32 conversion of its NZ weight dispatched to ACL Cast
during capture. Packed target and draft loading now materialize both FP32
indexer projection copies before capture. `hardware/graph-indexer-preloaded.log`
records **1 passed**, capturing the actual indexer forward with NZ weights and
without an eager indexer warmup. Changed-input graph replay matches eager.
`cpu-indexer-preload-tests.log` records **29 passed**, including cache refresh
after loading. The full focused CPU suite before this last fix recorded
**66 passed** in `cpu-prelaunch-tests.log`. Retry: `mtp-graphs-06/`.
Startup succeeded in attempt 06; request correctness and throughput remain
unqualified as described above.

## Status

This replaces the packed draft registration stub with an experimental adapter
around the existing GLM predictor. It is **not hardware-qualified**, is not
promoted, and has no measured throughput or acceptance result. Existing
non-speculative serving remains the default.

A read-only inspection of the exact selective-W3 checkpoint manifest at
`/srv/ai/models/GLM-5.3-Flash-selective-W3-310p` found 1,753 layer-45 tensors:
288 experts × six packed tensors, plus 25 attention/router/shared-expert/norm
and projection tensors. There is one prediction layer, pool size four, and
no independent `shared_head.head.weight`. Tensor contents were not loaded or
validated against the full model in this stage.

## Implemented

- Select `Glm5NextW2MTPModel` for a packed target, including when target dict
  HF overrides do not propagate to the draft. Flatten the GLM text config and
  refresh the draft registry metadata without mutating the target config.
- Construct the shipped MLA draft layer while suppressing FP8 expert-bank
  allocation. Reuse the existing mixed W2/W3/W4 resident packed MoE banks.
- Stream draft expert tensors through the existing placement path and dense
  tensors through the shipped MTP loader. Require complete local banks and
  non-shared parameters, both shared-expert gate/up halves, and reject duplicate
  tensors. The filtered reader skips backbone tensors and peer draft experts.
- Remove absent shared allocations after loading; the proposer binds the actual
  target embedding and LM head. A supplied draft head retains ownership.
- Use FP32 PyTorch MTP input normalization on 310P rather than invoking the
  Triton-only normalization kernel. Register GLM's tuple output in the proposer
  and include the packed architecture in the multi-cache-group selector.
- Preserve compressor-state pools over a speculative rejection window. State
  pages remain one pool wide; the sliding window grows by the draft count, and
  writes retain the earliest possibly accepted pool as well as later pools.
- Promote the accepted KDA recurrent carry before packing current token rows.
  Previous acceptance can exceed the current query length: the native kernel
  indexes its carry inside the current packed row list, so passing the old
  count directly could fail on shortened verification calls. Promotion gathers
  from the full slot table, ignores inactive rows, and preserves rejected
  suffix states. Native convolution retains its existing acceptance-offset
  handling; aligned state-copy machinery remains in use.

The CPU-stage KDA promotion described above has been superseded by the native
fixed-stride state-index interface during graph qualification. The compact-live-state
guard is retained.

## CPU validation

`cpu-tests.log`: **60 passed** across seven focused files, covering:

- Nested/dict config normalization and registry refresh.
- Constructor reuse and restoration of the scoped FP8 factory override.
- Packed W3/W4 banks, fused shared MLP loading, missing/duplicate tensors,
  absent and owned heads, and filtered draft reads.
- Position-zero normalization, recycled hidden states and input preservation.
- Recurrent acceptance selection with shorter queries, reordered requests,
  inactive padded rows, and poisoned rejected suffixes; actual wrapper argument
  checks against a stand-in recurrent operator.
- Actual pool-writer execution across rejection at a pool boundary, compared
  with compression of the clean accepted sequence.
- Existing package/import and KDA host checks.

Device dependencies are stubbed where required; the dense-loader test executes
its shipped methods from the source AST. These tests do not establish native
kernel parity or scheduler/preemption correctness. The full existing
`tests/ut/models/test_glm5next_cache_config.py` could not collect because the
installed CPU vLLM lacks `register_all_kvcache_specs` expected by this checkout.
It must run in the matching development environment.

## Original CPU-stage hardware plan (superseded by graph-first request)

Use the established packed-W3 runtime and OPP/binding, port **8001**, and add:

```text
--enforce-eager
--no-async-scheduling
--mamba-cache-mode align
--speculative-config '{"method":"mtp","num_speculative_tokens":1,"draft_load_config":{"load_format":"glm_w2_filtered"}}'
```

Remove the decode-graph compilation override for this initial profile. The
adapter rejects graph mode, live-only KDA mode, asynchronous scheduling,
pipeline parallelism, host MLA, and more than one prediction layer. It accepts
1–7 draft tokens (the verifier kernel supports at most eight total tokens),
but start with one draft token. Qualify two only after one works.

Before measuring speed:

1. Confirm loaded packed draft architecture, complete weights, shared head,
   native 310P normalization dispatch, and cache-group geometry.
2. Check deterministic output and state against speculation off, including
   zero/all/partial draft acceptance, pool and KDA block boundaries, shorter
   final queries, request-row movement, completion/reuse, and preemption.
3. Check actual memory admission and peak use. The extra draft layer, retained
   pools, and speculative recurrent slots change the budget; the previous
   192K serving capacity is not an MTP capacity claim.
4. Measure accepted tokens per verification, draft and verification latency,
   total c1/c4 decode throughput, quality, cold TTFT and memory. Compare against
   the same eager runtime with speculation off; saved graph results are useful
   context but are not a matched MTP comparison.
5. Only then replace eager state promotion with a graph-compatible operation
   and qualify capture sizes within the existing HCCL capture budget.

No dummy gate was used; the real-weight serving gate is in progress. No commit or promotion is claimed;
this is staged work in the shared checkout alongside other ongoing changes.

## 简要交接

已完成 packed GLM MTP 草案适配、权重加载、共享输出头、310P 输入归一化及
拒绝草案后的状态恢复代码；60 项 CPU 检查通过。用户已授权 NPU 测试，
用户要求直接测试 full decode graphs；已补齐原生状态索引、设备端选择及多 token MLA 路由。
完整服务已完成图捕获并启动，显式设置 32K 测试上限，但三次冒烟请求均出现重复。
已离线复现并修复 MLA 空填充导致的位置、写槽及页表副本不随图重放更新的问题；
修复前两项失败，修复后五项 CPU 测试通过。用户暂停 NPU 使用，硬件回归待授权。
按用户要求保留 8001 实验服务；运行版本尚未包含最新修复，未发布。
