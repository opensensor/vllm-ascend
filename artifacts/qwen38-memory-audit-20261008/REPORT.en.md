# Qwen 310P memory transfer and barrier audit

[Chinese companion](REPORT.zh.md). This audit is offline. The server remains
stopped; no NPU workload, profiler capture, pause/resume transaction, or model
reconfiguration was run. The 94°C hold / 85°C resume controller remains staged
for a future authorized launch, with the independent 96°C cutoff retained.

## Conclusions

Zero Mamba spill warnings do **not** mean zero memory transfers or barriers.
The failed serving profile still has recurring host boundaries in MTP expert
routing, PLE row staging, GDN chunk planning, Mamba accepted-token handling,
and sampled-token delivery. It also repeatedly materializes substantial
device-local QSA, GDN, MoE and HC intermediates and performs TP collectives.

The strongest memory-traffic candidate is **QSA prefill K/V materialization**.
The strongest avoidable host-stall candidates are **MTP routing readbacks**,
**GDN plan readbacks**, and **duplicate accepted-token boundaries**. The target
graph wrapper also synchronizes the host before replay. These are source-level
findings supported where possible by older hardware traces; they do not identify
the cause of the October 8 thermal shutdown or measure its energy distribution.

## Scope and provenance

The inventory scans every Python and C/C++ source file under four frozen roots:

It covers **6,281 files and 56,161 candidate sites**, with zero parse errors.
Manual dispatch review covers **36 serving and adjacent areas**. Inventory
counts include inactive paths and are not runtime operation counts.

- Fork `2e5f07071`: `vllm_ascend`, `csrc`, and `tools`. Uncommitted experiments
  and other agents' changes are excluded.
- Deployed plugin `/srv/ai/src/qwen38-prefix-bounded-runtime-20261008`, copied
  through read-only SSH, including its runtime tools and generated headers.
- OpenSensor vLLM `3ab5dda29`, checked against the remote source hashes. Two
  differences from the clean commit were captured: generated `_version.py` and
  `model_executor/kernels/mhc/torch.py`. Qwen uses its own HC implementation;
  the latter is outside this serving model's path.
- The deployed native HC residual bridge, kernel source and manifest.

The deployed plugin differs from the committed fork in 26 Python files,
including `models/qwen4_exp/model.py`. Therefore findings below use deployed
source positions and dispatch, rather than assuming every fork optimization
was deployed. See [runtime differences](runtime-differences.json),
[upstream provenance](upstream-provenance.json),
[source hashes](source-manifest.json) and [coverage](coverage.json).
The runtime source archive preserves original bytes for later review.

`inventory.jsonl.gz` enumerates candidate calls with root, path, line, enclosing
function, category, excerpt and source SHA256. It includes inactive models and
diagnostic tools for coverage. A candidate is not a measured transfer:
`.item()` on a CPU tensor is a host read, `.to()` may be a no-op, a view need
not allocate, and an event wait can order device work without blocking the host.
Native results are lexical sites, not a complete control-flow proof.

## Ranked findings

### 1. QSA repeatedly gathers the same selected K/V into per-query scratch

Deployed `models/qwen4_exp/ops/qsa_batched_attention_310.py:132–350` translates
selected paged K/V into NZ scratch, tiles queries by 64, runs QK, widens scores
to FP32, applies softmax, then runs PV. K/V gathering uses two side streams;
buffer-reuse events protect the previous consumer and the next writer.
This is **NPU-local traffic**, not a host offload.

With TP4, each rank owns six query heads and one KV head. At the full
2,048-token budget, the finite tail and NZ alignment produce 2,064 scratch
tokens. K plus V scratch payload is:

```text
2 × 1 KV head × 256 dimensions × 2,064 tokens × 2 bytes
= 2,113,536 bytes per query
```

A 64-query tile uses **129 MiB** of selected K/V scratch. A fully sparse
2,560-token chunk writes **5.039 GiB** of logical K/V scratch payload per QSA
layer; twelve target QSA layers give **60.469 GiB per rank per chunk**.
There are **960 K/V gather calls** for that one-request chunk. This is a
logical payload estimate, not measured DDR traffic: cache reuse can reduce
physical reads, early contexts and partial chunks differ, and this excludes
scores, indexer work and other intermediates. See
[byte estimates](logical-byte-estimates.json).

The older four-rank prefill trace puts `QsaGatherValueNzV310` at
**11.02–11.28% of summed task time**, with 4,168 calls per rank. That supports
investigation but is not a speedup or heat estimate.

Candidate: fuse selected-page access with attention, or reuse selected groups
across a query tile with an exact per-query causal mask. The existing opt-in
group-major path uses `torch.unique` and data-dependent output shapes, so it
can trade scratch traffic for implicit host coordination. It is not a qualified
drop-in replacement. Removing gather stream waits would create buffer races.

### 2. MTP still dispatches experts through the host on every draft forward

`models/qwen4_exp/mtp.py:43–58, 206–282, 311–315` selects local routes with
`torch.nonzero`, sorts them, computes `bincount`, then calls `counts.tolist()`
before dispatching per-expert W8A16 matmuls. Graph capture explicitly inserts
this as an eager callback and copies the result into a stable output buffer.

The serving overrides omit `mtp_expert_execution`, so the deployed default is
**`w8a16_routed`**, not the device-grouped branch. The count vector has 128
INT64 entries: **1 KiB per rank per draft forward**, plus dynamic route-size
coordination. With MTP2 this repeats for the two draft forwards. The payload is
small; the forced dependency boundary and many small expert launches matter.

The target W4 path differs: small decode uses device-routed kernels; larger
prefill uses device-grouped dispatch. Its host-routed `.cpu().tolist()` branch
is not selected by this backend. Do not attribute that fallback to live W4.

Candidate: qualify the existing `w8a8_grouped` MTP branch or implement a fixed
shape device-routed W8A16 schedule. W8A8 changes activation quantization and
needs acceptance, numerical, text/tool/image and sustained-use gates. Its
graph eager callback must also be reviewed; changing only the config does not
prove all host dispatch disappears.

### 3. GDN chunk-plan caching still performs a device readback per metadata group

`_310p/ops/fla/gdn_310.py:151–175` builds a host plan from
`cu_seqlens.to(torch.int64).cpu()`. Memoization prevents repeating this in
every layer of the same metadata group. The target has **36 GDN layers in
three Mamba cache groups**, so the boundary can recur once per group's metadata
and distinct query-boundary tensor each prefill step, including mixed views.
It is not one drain per layer, and it is not eliminated entirely.

Candidate: build the same padding plan from scheduler-owned CPU query lengths
or boundaries, and retain the current tensor identity/lifetime checks. Match
the speculative and non-speculative partitions exactly. This can be tested
offline before any hardware gate, without changing the recurrent dtype.

### 4. Mamba align postprocessing adds another accepted-token D2H boundary

The 310P patch selects the tensor-copy fallback in
`patch/worker/patch_mamba_utils.py:177–265, 393–405`. It copies
`num_accepted_tokens_gpu` into a CPU tensor with default blocking behavior,
then decides whether aligned state must be copied. It may perform
`dst_state.copy_(src_state.clone())` for convolution and recurrent tensors.

Separately, synchronous sampled-token bookkeeping reads generated tokens back
to the host, and the next `_prepare_inputs` synchronizes the accepted-token
event and sends accepted counts back to device. These are distinct boundaries
even though the metadata payload is tiny. The clone plus copy is **D2D state
traffic**, triggered by state advancement/alignment or preprocess copying,
not a prefix spill. The clone also protects potentially overlapping copies.

Candidate: reuse an already completed accepted-token host snapshot where its
semantics and request row ownership agree, or move align decisions and copying
to a fixed-shape device operator. Track original counts separately from the
postprocess reset-to-one value. Eliminate clones only after proving disjoint
storage or providing overlap-safe copy semantics.

### 5. FULL target graph replay retains a host stream drain

`compilation/breakable_aclgraph.py:74–91` calls
`torch.npu.current_stream().synchronize()` before FULL replay when ENPU is off
and the EAGLE-style draft exception does not apply. The 310P runner already
orders the update stream with main-stream waits at
`_310p/model_runner_310p.py:1090–1112`; the final host drain remains.

The pinned config treats MTP as EAGLE-style for this flag, so the draft wrapper
exception must be considered. The target barrier is confirmed; asserting three
identical wrapper drains per MTP2 iteration would be incorrect.

Candidate: replace the host drain only when mutable graph task parameters,
previous replay, update submission and current replay have an explicit safe
ordering contract. A stream event alone does not establish the safety of a
host-side task-parameter update. Keep an accuracy and concurrent replay gate.

## Remaining areas reviewed

Paths below are relative to the deployed `vllm_ascend` source unless marked
`vllm` or `native_hc`. Detailed source anchors are in `reviewed-areas.json`.

| Area | Phase and traffic/barrier | Finding and disposition |
| --- | --- | --- |
| Prefix tier retirement | Layout changes and checkpoint retirement; device-wide drain | `_update_states` drains before slot reassignment; bounded retirement shares one drain across groups. Necessary writer protection; further `invalidate`, CoW `copy`, and admissions can request additional drains. Batch them only after proving no intervening writes. |
| Prefix device archive | Eviction/reuse; checkpoint D2D copies and optional D2H/H2D | `prefix_mamba_state.py` still has device archive/swap and host spill/restore paths. A checkpoint is 9,744,384 bytes per group. Bounded retention reduces pressure; it does not remove these branches or count every D2D copy. |
| Compact Mamba tables | Every preparation; host temporary → NPU temporary → persistent NPU table | `model_runner_310p.py:617` uses `torch.as_tensor(mapped, device=...)` then `copy_`. Reuse bounded pinned staging and copy once into stable storage; validate table tails and request remapping. |
| Token/position/slot metadata | Every preparation; many small H2D copies | Persistent `CpuGpuBuffer` staging is mostly already present. CPU `.numpy()`/`.tolist()` here often reads a host mirror. PLE history buffers use ordinary CPU allocations; avoid treating `non_blocking=True` alone as proof of asynchronous staging. |
| Embedding staging | Image/prompt embedding preparation; pageable H2D | The large FP16 CPU staging buffer intentionally disables pinning after a reproduced AVX2 zero-fill crash. A blanket pin-memory change would reintroduce that failure. |
| Sampled output | Every synchronous step; NPU → CPU | `vllm/v1/sample/rejection_sampler.py:267–299` parses `output_token_ids.cpu().numpy()`. Required for scheduling/history and streaming in this mode. Logprobs add readbacks only when requested. Async Qwen MTP/PLE scheduling is currently rejected; do not simply enable it. |
| QSA logical positions | Eager graph segment; small CPU → NPU copy | Host lengths generate logical positions and a callback copies them into stable device storage. A per-forward cache shares compatible derivations. CPU `.item()` on those lengths is not an NPU synchronization. |
| QSA selection | Each sparse layer; compressed-key gather, FP32 scores, sorting | Visible-page bounds already exclude unused cache suffixes. GEMM scoring copies selected compressed keys and widens them. Stable sort preserves ties; changing selection or sharing across different layer caches requires numerical validation. |
| QSA selection graph callback | Each sparse graph step; D2D copies into fixed selection buffers | Eager selection is intentional because visible context grows. No whole-model eager fallback warning is required for these callbacks. Capture/callback execution counts need runtime measurement. |
| QSA ND conversion fallback | Conditional; NZ → ND, reorder and full visible-page copies | `token_major` materializes visible K/V when direct NZ gather is not used. Default 2,048-budget long sparse prefill uses NZ gathering. Avoid blaming the ND fallback without branch evidence. |
| QSA cache updates | Every layer; cache writes and compression scratch | Native index-cache and K/V writes remain device-side. Reference per-row Python copies are separate fallback paths. |
| MoE sorting/counting | Large prefill; routes, two sorts, count comparison matrix | Default `compare` counts 25,600 routes against 128 local experts: 3,276,800 comparisons per chunk/layer. Existing histogram mode is an experiment; check AI Core/AI CPU dispatch and exact group ends. |
| MoE routed activation staging | Large prefill; quantized route expansion and output scratch | Input activation packing is already shared per token before route gathering. Full route geometry includes peer-owned sentinel routes. Gate/up output is 62.5 MiB; down output is 125 MiB per full MoE chunk/rank, before finalization. Compact routes only with a device-side shape/ownership contract. |
| MoE finalization | Prefill; FP16 combine then FP32 result | `cann_v2` avoids the reference full FP32 route-output materialization but widens combined results for existing TP reduction. It changes rounding; do not revert this benefit accidentally. |
| Shared expert side streams | Conditional; events and deferred collectives | The serving setting is `tp_sharded`, so overlap/deferred shared-expert stream branches are inactive. Their waits are not evidence of live stalls. |
| GDN WY preparation | Prefill; transposes, contiguous buffers, FP32 transforms, inverse scratch | This remains substantial Torch device work around native state/output kernels. Grouped Gram already avoids triplicating keys. Fuse layout/precision transitions with exact FP32 recurrence and production head geometry preserved. |
| GDN recurrent state layout | Prefill/mixed; advanced indexing, multiplication, transpose copies | Initial state is gathered, masked, transposed to kernel layout and returned. Final state and outputs are transposed/materialized again. Decode updates state in place. Different phase paths need separate attribution. |
| HC mix/norm | All target layers; FP16↔FP32 activation materialization | Norm/mix still widens HC activations and narrows GEMM operands. Static affine and NZ projection weights are cached; `_linear` keeps NPU weights in storage dtype. No recurring full-weight FP32 conversion is selected there. |
| Native HC residual | All supported combine calls; GM↔UB copies, kernel pipeline barriers | Bridge uses `aclrtLaunchKernelWithArgsArray` on the current NPU stream, without explicit host memcpy or host synchronization. Kernel uses `PIPE_ALL`/`PIPE_V` ordering and numerical clipping. These barriers are within AI Core, not CPU PCIe transfers. Shape tiling metadata is copied once per new shape. |
| Native W4/GDN/QSA kernels | Selected operator calls; GM/L1/L0/UB movement and pipeline events | Reviewed fork sources contain DMA loads/stores and paired producer/consumer flags. Resident weights are reread by kernels even without offload. Narrow barriers or double-buffer only with dependency proofs. External coherent OPP binaries are not proven identical to current fork kernel sources. |
| TP/HCCL | Every target decoder block; inter-device collective traffic | 48 attention plus 48 MoE reductions give 96 decoder reduction sites per target forward, plus embedding/head/draft traffic. TP4 has no CPU expert-dispatch all-to-all in the custom W4 branch. Standard communicator and alternate PyHCCL allocation paths differ. |
| LM head/PLE projection preparation | Startup; quantization, CPU/device copies, NZ layout | Dynamic W8 weights are prepared once and retained. Draft embedding/head sharing is logged. These load-time copies cannot explain repeated steady-state traffic by themselves. |
| Image encoder and embedding cache | Uncached image prefill; pixels/metadata staging, vision scratch | Vision work happens when encoder inputs are scheduled, not every decode token. Cached embeddings are reused. CPU grid metadata avoids per-layer device scalar reads; image processing remains enabled. |
| Attention block zeroing/CoW | Allocation or shared-prefix mutation; D2D writes/copies | Zeroer writes newly allocated pages; CoW copies selected blocks and avoids duplicate aliased storage. Required initialization/isolation; record bytes per block rather than counting it as offload. |
| Sleep/weight offload/KV connectors | Optional features; potentially bulk CPU/NPU/network movement | CPU offload, KV transfer, PP/DCP/PCP, dynamic EPLB, FlashComm1, and cache parallelism are not configured in the failed profile. Inventory sites remain, but this audit does not assign their traffic to the incident. |
| Capture/admin/profiling | Startup or explicit control; drains, allocations and optional cache clears | Recapture, native validation and reset contain deliberate synchronizations. They are separate from ordinary inference. Benchmark timing barriers are excluded from serving attribution. Thermal hold uses `keep` without cache clearing or weight offload. |

## A latent fallback defect, reproduced offline

`_flatten_state_indices` in deployed `gdn_310.py:80` reads local `seq_lens`
before assigning it when `ndim=2`, uniform-state mode is false and capture is
inactive. An extracted-function CPU test reproduced `UnboundLocalError`.
The same fallback spells `flat_cpu.is_pinned` without calling the method,
so its intended pinning condition cannot work as written. Uniform MTP decode
bypasses this branch. These defects do not establish the incident's cause;
repair and regression-test them before relying on the variable-shape fallback.
See [offline proof](offline-latent-gdn-proof.json).

## Existing hardware evidence and its limits

[Profiler receipts](historical-profiler-receipts.json) recompute the saved
October 5 four-rank exports. Prefill cast/layout/copy tasks account for
**12.37–12.72%**, collectives **9.09–10.92%**, and native W4 projections
**19.88–21.96%** of summed task time. Rank 1 records 12 `Event::synchronize`
calls totaling **3,642.51 ms host self duration**, eight `aten::nonzero` calls
totaling **433.94 ms**, 128 explicit H2D records and eight D2H records.
Names and totals do not uniquely attribute them to individual source sites;
copy records omit byte sizes. Nested and cross-stream records can overlap.

These captures predate the bounded scheduler and current native HC selection.
They are evidence that transfers and barriers existed, not current-profile
performance measurements. Earlier system DDR exports also contain implausible
values for devices 0/1 under the stated MB/s units; PCIe transaction-class
rates do not directly equal application memcpy payload. Those exports are not
used to claim a measured byte rate or heat cause.

During the incident, a long prefill chunk could delay existing decode for
several seconds. At 330 prompt tok/s, 2,560 prompt tokens alone represent about
7.76 seconds of work before other step costs. That is consistent with low
streamed generation during mixed prefill, but it is not a measurement of memcpy
stall time. Fewer windows can improve latency contention without removing
single-request steady-state heating. Smaller chunks do not guarantee lower
total energy or throughput.

## Fix order and deferred measurement plan

1. Remove GDN plan readbacks using exact host metadata; coalesce compact-table
   staging; repair the latent variable-shape fallback. Prove request ownership,
   padding and buffer lifetime offline.
2. Qualify a device-routed/grouped MTP path and consolidate accepted-token
   handling without changing rejection or Mamba state-selection semantics.
3. Instrument QSA selected-payload bytes, MoE route staging, Mamba D2D copies,
   and graph/retirement waits from host-owned sizes and control events. Add
   phase labels without new `.item()` reads or timing synchronizations.
4. Reduce QSA scratch duplication and fuse GDN/HC layout transitions. Keep
   recurrent state FP32 and existing numerical/causal contracts.
5. Revisit graph drains after the mutable-task ordering contract is proven.
   Do not strip correctness barriers based solely on their count.

[Deferred plan](profiling-plan.json) separates cold prefill, warm-prefix reuse,
C1/C2/C3 decode, mixed prefill/decode and fresh/cached images. It requests short
all-rank captures, memcpy direction/bytes where available, event intervals,
device-local MTE counters, HCCL tasks and host CPU sampling. Temperature and
thermal-controller timestamps must share the clock. Neither summed times nor
source byte formulas substitute for an actual transfer counter.

The plan is disabled and starts no server. Hardware tests require a later
authorization. CANN/PyTorch allocator internals, driver DMA, proprietary
operator binaries, implicit dynamic-shape synchronization, and physical heat
attribution remain measurement gaps. This is a source audit with historical
trace analysis, not a clean bill of health for sustained serving.

## Validation and reproduction

Eleven offline regression checks passed: eight scanner and three profiler
receipt checks. The tests cover provenance,
comments/strings, function scope, native templates, missing/invalid source,
candidate categories, self-duration accounting and invalid profiler values.
The historical analysis and logical formulas ran
offline; no inference throughput or thermal result was generated.

```bash
python -m pytest --noconftest -q tests/ut/qwen38_1m/test_memory_audit.py
python artifacts/qwen38-memory-audit-20261008/reproduce_evidence.py
python -m tools.qwen4exp.memory_audit \
  --root fork=/path/to/frozen-fork \
  --root runtime=/path/to/deployed-plugin \
  --root vllm=/path/to/remote-vllm-source \
  --root native_hc=/path/to/native-hc-source \
  --output /path/to/new-inventory
```

Scoped checks and the required repository-wide `bash format.sh ci` result are
saved alongside this report. The repository-wide check fails on existing Ruff,
spelling, Clang, Markdown and forbidden-import issues. Unrelated automatic
formatting changes were confined to the isolated check worktree. Raw source/trace evidence is preserved without
spelling edits. The audit changes only analysis tooling, tests and reports.
