# PRD: Qwen3.8-Flash-Next 1M Context on Four Ascend 310P Chips

| Field | Value |
| --- | --- |
| Status | Draft; implementation and Ascend validation pending |
| Date | 2026-09-15 |
| Product | Local Qwen3.8-Flash-Next inference on Atlas 300I Duo |
| Target host | Two Atlas 300I Duo cards, four independent 48 GB Ascend 310P chips, 256 GB DDR4-3200 |
| Initial checkpoint | W8A8_DYNAMIC routed experts, FP16 non-expert tensors and host PLE |
| Primary workload | One text request with up to 1,048,576 total tokens |
| Delivery repositories | vLLM plus vLLM Ascend; ModelSlim conversion is an upstream dependency |

## 1. Problem statement

Qwen3.8-Flash-Next has a native 262,144-token context and is designed to be
extended to approximately one million tokens. A reported CUDA deployment serves
a one-million-token request on one 96 GB RTX PRO 6000 using an NVFP4 model, FP8
KV cache, host-offloaded FP8 PLE, chunked prefill and a customized SGLang runtime.

Our target has a similar aggregate amount of accelerator memory, but it is split
across four independent 310P memory domains. The planned W8A8 model must span all
four chips. QSA has two KV heads, so ordinary TP4 duplicates KV storage, and the
existing upstream Qwen4Exp runtime has CUDA and AMD implementations but no Ascend
implementation. The current 310P runtime also cannot materialize Qwen4Exp's QSA
side-cache specifications.

The product must therefore provide a native Ascend Qwen4Exp execution path and a
memory architecture that fits weights, cache and runtime workspaces on every
chip. Aggregate capacity alone is not an acceptable fit calculation.

## 2. Goals

1. Serve one complete Qwen3.8-Flash-Next text model across four 310P chips from
   an OpenAI-compatible endpoint.
2. Validate real W8A8 weights at 8K, 128K, native 262K, 512K and finally
   1,048,576 total tokens without silent truncation.
3. Preserve Qwen4Exp semantics for GDN, QSA/indexer, four-stream gated residual,
   routed and shared experts, PLE and long-context position encoding.
4. Keep full PLE tables in host memory and fetch only requested rows to the
   accelerator.
5. Fit a single one-million-token request with measured per-chip headroom for
   runtime workspaces and allocator fragmentation.
6. Make interrupted loading and long-prefill failures diagnosable and
   repeatable, with durable benchmark and accuracy artifacts.

## 3. Non-goals for the first 1M release

- Two independent model replicas. The W8A8 model requires all four chips.
- Concurrent independent one-million-token requests.
- Matching the CUDA system's throughput or 335-second time to first token.
- A 128 GB Mooncake pool. It does not fit safely alongside FP16 PLE and normal
  process headroom in a 256 GB host.
- Pipeline parallelism. Qwen4Exp PLE currently requires pipeline parallel size 1.
- Multimodal, MTP, prefix-cache sharing and ACLGraph as prerequisites for the
  first functional one-million-token gate. Each is a later, explicit gate.
- Quantizing the PLE table in the first correctness milestone.

## 4. User scenarios

### 4.1 Primary

A local user submits a document or repository corpus approaching one million
tokens and asks a retrieval or synthesis question. The server accepts the full
input, reports its actual token count, completes chunked prefill, and returns a
non-empty answer without preemption or process restart.

### 4.2 Secondary

- Native 262K coding and document-analysis sessions.
- Several shorter requests sharing the same four-chip instance.
- Repeated requests that reuse compatible hybrid prefix state after the base
  runtime is correct.

## 5. Fixed inputs and assumptions

### 5.1 Model

- Source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Architecture: 48 decoder layers: 36 GDN and 12 QSA.
- Hidden size: 2,560.
- QSA: 24 query heads, 2 KV heads, head dimension 256.
- QSA indexer: 4 query heads, 1 key head, dimension 128, compression ratio 4,
  selection budget 2,048 tokens.
- MoE: 512 routed experts, 10 selected per token, plus one shared expert.
- PLE: 51.2 billion n-gram embedding parameters in one PLE layer.
- Native context: 262,144 tokens. Longer contexts require an explicit, validated
  RoPE/YaRN configuration.

The 128 examples used during GPTQ conversion are calibration inputs. They are
unrelated to context slots, KV blocks, QSA's 2,048-token selection budget, or the
one-million-token serving limit.

### 5.2 Hardware

- Two Atlas 300I Duo cards.
- Four independently allocated 48 GB 310P memory domains.
- 256 GB DDR4-3200 host memory.
- Actual free NPU bytes, HCCL topology, PCIe generation/link width, NUMA layout,
  host channel population and sustained bandwidth must be recorded on the target
  machine. Marketing capacities are planning values only.

### 5.3 Checkpoint

The initial checkpoint keeps routed experts in W8A8_DYNAMIC and non-expert main
model tensors in FP16. PLE stays FP16 on the host. MTP remains floating point and
is excluded from the first text-only capacity gate.

At PRD creation, the full conversion had committed 9 of 49 durable sequential
stages. The canonical current status is the conversion run's `progress.json`,
not this snapshot.

## 6. Capacity model

All binary figures below are planning estimates and must be replaced with actual
allocator measurements. The checkpoint accounting excludes MTP unless stated.

| Component | Aggregate | Ideal per chip | Notes |
| --- | ---: | ---: | --- |
| Accelerator-resident W8A8 main model | 123.25 GiB | 30.81 GiB | Requires balanced TP4/EP4 placement |
| Host FP16 PLE | 95.37 GiB | N/A | One shared host copy is required |
| Logical BF16 QSA K/V at 1M | 24.00 GiB | 6.00 GiB | Assumes sequence sharding, no TP duplication |
| Conventional TP4 BF16 QSA K/V | 48.00 GiB | 12.00 GiB | Two KV heads are duplicated across four ranks |
| Conventional TP4 8-bit QSA K/V | 24.00 GiB | 6.00 GiB | Requires new C8-capable QSA kernels |
| BF16 compressed indexer history | 0.75 GiB logical | 0.75 GiB if replicated | One row per four tokens; raw ring is small |

The conventional TP4 BF16 design consumes approximately 43.56 GiB per chip
before MTP, activations, communication buffers, graph capture, temporary full
weights or allocator fragmentation. It is rejected as the one-million-token
configuration.

Two primary candidate layouts remain:

| Layout | Estimated model + persistent 1M cache per chip | Decision |
| --- | ---: | --- |
| TP4 + 8-bit main QSA KV + 8-bit/BF16 indexer | 37.2-37.6 GiB | Candidate A |
| TP4 + QSA-aware DCP4 BF16 main KV + replicated BF16 indexer | 37.56 GiB | Candidate B |
| TP4 + host main KV + device hot buffers/indexer | To measure | Fallback C |

Each candidate must leave at least **8 GiB measured free per chip after model and
persistent cache allocation**, followed by a successful worst-case prefill run.
If graph capture requires more headroom, the measured requirement supersedes the
8 GiB floor.

Host planning must reserve at least 48 GiB for the OS, server processes, pinned
transfer buffers and filesystem activity. FP16 PLE therefore leaves roughly
112 GiB for host KV and optional prefix state under the 256 GB nominal capacity.
A prefix-cache pool may be introduced only from measured remaining memory.

## 7. Product requirements

### R1. Reproducible environment

- Pin vLLM, vLLM Ascend, torch-npu, CANN, Transformers, ModelSlim and checkpoint
  revisions in every result artifact.
- Do not upgrade Transformers as part of this adaptation.
- Run the server directly from `/workspace` in the target container; verify
  imports resolve to `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend`.
- Record firmware, driver, chip inventory, NUMA mapping and HCCL topology.

### R2. Architecture registration and loading

- Select an Ascend Qwen4Exp implementation without importing CUDA, ROCm or
  Triton-only modules.
- Register both text-only and conditional-generation architecture names while
  allowing the first gate to disable multimodal inputs.
- Map fused source expert tensors and ModelSlim W8A8 scale/offset tensors
  explicitly. Reject missing, extra, duplicate and incompatible tensors.
- Stream and shard weights so no rank materializes the full expert bank.
- Report per-rank weight bytes by component and fail startup when placement is
  imbalanced beyond 5% without an approved reason.

### R3. Parallel topology

- The base topology is one instance with TP4 across all chips.
- Routed experts should use EP4 if supported by the quantized MoE kernels and if
  measured throughput improves; correctness must also pass with the chosen
  production topology.
- Shared experts, router, LM head, gated residual and two QSA KV heads require
  explicit sharding/replication rules.
- No code may treat the four chip memories as one pooled allocation.
- Cross-card versus within-card collectives must be identified in traces.

### R4. W8A8 execution

- Execute all 73,728 routed expert projections using the native 310P W8A8 path.
- Support dynamic INT8 input activation scales and the checkpoint's weight
  scales/offsets without round-tripping the full experts through FP16 storage.
- Keep attention, router, shared expert, PLE projection, gated residual and LM
  head in their required FP16/FP32 types for the first release.
- Validate fused MoE numerical output against the ModelSlim eager QDQ reference.
- Avoid device `tensor.item()` calls in hot paths.

### R5. PLE host lookup

- Allocate exactly one logical FP16 PLE table on the host, not one copy per rank.
- Define ownership when four workers are separate processes. Acceptable designs
  include shared memory, read-only mapped pages with verified physical sharing,
  or explicit sharding with collective assembly.
- Pin or register only the transfer regions justified by measurement; do not
  assume 95.37 GiB can all be pinned safely.
- Implement exact Qwen n-gram hashing, EOS boundaries and history maintenance.
- Batch, deduplicate and asynchronously prefetch requested rows to the correct
  rank without synchronizing the decode loop on each row.
- Prove exact row identity against the source checkpoint and numerical parity of
  the PLE layer at short context.
- Record PLE host bytes, page faults, transfer bytes, hit rate and lookup latency.

### R6. GDN and hybrid state

- Implement Qwen4Exp's 36 GDN layers with the correct convolution and recurrent
  state shapes, dtype and request lifecycle.
- Preserve state across chunked prefill, decode, preemption, block reuse and
  request completion.
- State copies and slot remapping must support all four ranks and must not alias
  different requests.
- Test chunked versus unchunked results at lengths where both are feasible.

### R7. QSA and indexer

- Implement an Ascend QSA prefill and decode backend with no Triton dependency.
- Maintain full logical K/V history while attending only to the indexer's
  selected 2,048 tokens.
- Implement raw index-key ring state and compressed index-key history at one row
  per four tokens.
- Match Qwen Q/K normalization, partial rotary dimensions, output gate, causal
  behavior, block selection and selection-count semantics.
- Support chunked prefill at 4,096 tokens initially; make chunk size configurable.
- Add deterministic kernel tests against the eager reference for boundary
  lengths, partial compression groups, repeated blocks and the final token.

### R8. One-million-token cache strategy

Candidate A and Candidate B must be prototyped at cache-spec and kernel level
before choosing the release layout.

**Candidate A: C8 QSA cache**

- Store main K/V in signed 8-bit form with documented scale granularity.
- Quantize and dequantize in fused cache-write/read paths.
- Establish long-context retrieval and short-context logit quality relative to
  BF16 cache.

**Candidate B: QSA-aware DCP4**

- Shard main QSA K/V over the sequence dimension across four ranks.
- Keep the indexer history replicated unless a correctness-equivalent distributed
  top-k design is demonstrated.
- Convert selected global positions to owning ranks and local slots.
- Exchange only selected K/V rows for sparse attention; do not all-gather the
  complete one-million-token cache during decode.
- Preserve deterministic top-k and output reduction across ranks.

**Fallback C: host sparse KV**

- Keep full main K/V in host memory and retain the indexer plus bounded hot
  buffers on device.
- Transfer selected rows asynchronously and expose hit/miss metrics.
- This path must be designed specifically for 310P and Qwen4Exp; current Ascend
  sparse offload support for other models and A3/A5 is not sufficient evidence.

The decision record must compare memory, accuracy, TTFT, decode throughput,
collective traffic, implementation risk and compatibility with prefix caching.

### R9. Long-context position handling

- Preserve the source 262,144-token native mode unchanged.
- Make the extension configuration explicit in deployment metadata; never
  silently raise `max_model_len` without the corresponding RoPE behavior.
- Validate 512K and 1M positions against a trusted Qwen reference implementation.
- Reserve requested output and speculative tokens inside the advertised context
  window. The 1M test target is 1,048,060 input tokens plus up to 512 output and
  four draft tokens.

### R10. Scheduling and serving

- Support OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints.
- Disable input truncation for validation and report actual accepted input tokens.
- Use chunked prefill so a one-million-token request does not require full-sequence
  activations.
- Set the initial one-million-token admission and execution concurrency to one.
- Preemption must either resume with correct hybrid state or fail the request
  explicitly; silent state loss is forbidden.
- Return health only after a real inference path is ready. Startup without a
  successful request is not a passing result.

### R11. Prefix caching

Prefix caching is a later gate because Qwen4Exp state includes K/V, GDN, PLE
history and QSA raw/compressed index state.

- A cache key must bind token IDs, model/checkpoint revision, quantization,
  RoPE configuration and all relevant execution settings.
- Save and restore every hybrid-state component at one consistent token boundary.
- Verify cold and replay outputs agree and report reused token counts.
- Size the host pool from measured free RAM. Do not copy the CUDA deployment's
  128 GiB pool by default.

### R12. MTP, multimodal and graph execution

- After base text correctness, validate the source MTP layer with fixed-step
  native speculative decoding. Long-context target and draft configurations must
  agree on the effective maximum position.
- Add multimodal only after one text-only 1M run passes; require a real image
  request and account for visual tokens.
- Attempt ACLGraph after eager correctness. If QSA, dynamic host lookup or 310P
  graph limitations prevent capture, retain eager as the supported path and
  document evidence.
- Try FlashComm1 with EP because this is MoE; use it only after correctness and a
  measured benefit.

### R13. Observability and failure artifacts

Each run must write machine-readable status and a human-readable log containing:

- source, code and environment revisions;
- topology and per-rank device identity;
- accepted prompt/output token limits;
- per-rank weights, persistent caches, workspaces, free memory and peak memory;
- host PLE/KV/prefix bytes, pinned bytes, RSS, swap and NUMA placement;
- prefill chunks completed, tokens per second and current layer;
- QSA selected-row transfer and collective bytes;
- GDN/QSA/PLE cache lifecycle events and prefix reuse;
- first fatal error and rank, without reducing it to a generic worker death.

Long runs must preserve enough state or deterministic inputs to reproduce the
failure. A failed 1M test must not invalidate lower-context release gates.

## 8. Validation and release gates

Dummy weights validate allocation and control flow only. Every milestone requires
a real-weight gate before it can be marked complete.

| Gate | Configuration | Required result |
| --- | --- | --- |
| G0: component parity | CPU/CUDA reference vs Ascend ops | GDN, QSA, PLE and MoE unit parity; cache boundary tests pass |
| G1: startup | Dummy then real weights, TP4, eager, 8K | `/v1/models` 200 and non-empty text response; no missing weights |
| G2: quantized correctness | Real W8A8, TP4/EP selection, 8K | Stable deterministic responses and accepted quality delta |
| G3: baseline capacity | 128K, one request | Full prompt accepted and response completes with memory report |
| G4: native context | 262,144, one request | Retrieval suite passes with no truncation/preemption |
| G5: extended context | 524,288, one request | Position and retrieval tests pass; cache strategy remains within budget |
| G6: one million | 1,048,060 input, 512 output reserve, concurrency 1 | 8/8 distributed retrieval records correct; non-empty response |
| G7: shorter concurrency | Approximately 8K input + 1K output | Concurrency sweep 1/2/4/8; report throughput and memory |
| G8: optional features | MTP, prefix reuse, multimodal, ACLGraph, FlashComm1 | Each feature gets an independent real-weight pass/fail record |

The normal adapter baseline of 128K with 16 simultaneous sequences is not a
reasonable first capacity gate for this model and hardware: it represents more
than two million cached tokens before duplicated QSA heads and workspaces. It is
replaced by 128K at concurrency one plus the separate 8K concurrency sweep. The
reason and measured limits must appear in the final report.

### 8.1 Accuracy criteria

- Short-context component comparisons use the eager reference with tolerances
  set per dtype and operation before seeing the final result.
- The converted model's held-out perplexity report must be complete and finite.
- Quantized text evaluation must meet the quality threshold chosen from the
  completed before/after conversion report; the threshold may not be relaxed
  after inspecting Ascend output.
- Long-context validation includes retrieval targets near the beginning, quarter,
  middle, three-quarter and end of the prompt.
- G6 requires all eight deterministic retrieval records correct.
- Run at least one adversarial no-answer case to detect spurious retrieval.

### 8.2 Initial performance criteria

Correctness and memory fit are the first release blockers. Performance is still
measured at every gate.

- G6 initial TTFT ceiling: 900 seconds.
- G6 target TTFT: 600 seconds; stretch target: 400 seconds.
- Corresponding initial prefill floor: approximately 1,165 input tokens/second.
- Post-prefill single-stream decode floor: 10 output tokens/second without MTP.
- No sustained swap activity during serving.
- No per-token host transfer of the full QSA history or full PLE table.

These thresholds may be revised once G4 produces real 310P measurements, but a
revision requires a recorded product decision rather than an undocumented flag
change.

## 9. Implementation plan

### Phase 0: freeze artifacts and measurement harness

- Complete and verify the ModelSlim W8A8 checkpoint.
- Capture manifests for source and converted weights.
- Add memory-accounting and long-context dataset generators.
- Record hardware topology and effective per-chip free memory.

Exit: reproducible checkpoint plus deterministic 8K/128K/262K/512K/1M inputs.

### Phase 1: architecture and real-weight load

- Isolate Qwen4Exp common model code from CUDA/AMD implementations in vLLM.
- Add Ascend model selection and explicit W8A8 loading.
- Bring up TP4 text-only eager execution at 8K.

Exit: G1 and G2 pass with real weights.

### Phase 2: PLE and GDN

- Implement shared/sharded FP16 host PLE lookup.
- Port PLE projection/convolution and GDN recurrent state.
- Validate chunk boundaries and request lifecycle.

Exit: PLE and GDN component parity plus end-to-end 8K regression pass.

### Phase 3: QSA BF16 correctness

- Implement QSA projection, indexer side caches, sparse selection and BF16 sparse
  attention kernels on 310P.
- Validate ordinary TP4 at short contexts even though it cannot fit 1M.

Exit: QSA parity and G3 at 128K if memory permits.

### Phase 4: choose the 1M cache architecture

- Prototype C8 QSA cache and QSA-aware DCP4.
- Benchmark at 128K and project/measure 1M allocations.
- Select Candidate A or B; retain host offload as the fallback.

Exit: decision record, at least 8 GiB measured post-allocation headroom per rank,
and G4 at native 262K.

### Phase 5: context expansion

- Add explicit long-context position configuration.
- Pass 512K before attempting 1M.
- Tune chunked prefill and bounded workspaces.

Exit: G5 and G6 pass.

### Phase 6: operational features

- Short-context concurrency sweep.
- Hybrid prefix caching with a measured host-memory pool.
- MTP, multimodal, ACLGraph and FlashComm1 gates.
- Publish deployment tutorial, test configuration and compact runbook.

Exit: G7, chosen G8 features, signed commits and reproducible artifacts.

## 10. Work packages and dependencies

| ID | Work package | Depends on |
| --- | --- | --- |
| WP0 | Conversion completion, manifest and quality report | None |
| WP1 | Hardware/topology/memory probe | Target host |
| WP2 | Qwen4Exp common/Ascend architecture dispatch | None |
| WP3 | W8A8 fused expert loader and MoE execution | WP0, WP2 |
| WP4 | Host PLE ownership, lookup and prefetch | WP1, WP2 |
| WP5 | Qwen GDN state and kernels | WP2 |
| WP6 | QSA BF16/indexer/cache kernels | WP2 |
| WP7 | C8 QSA cache prototype | WP6 |
| WP8 | QSA-aware DCP4 prototype | WP1, WP6 |
| WP9 | Long-context RoPE/YaRN and test corpus | WP2 |
| WP10 | Capacity ladder and performance tuning | WP3-WP9 |
| WP11 | Hybrid prefix cache | WP4-WP6, WP10 |
| WP12 | MTP, multimodal, ACLGraph, EP/FlashComm1 | WP10 |

## 11. Current feature status

| Feature | Status at PRD creation | Required first gate |
| --- | --- | --- |
| ModelSlim W8A8 checkpoint | In progress | Complete manifest and verification |
| Qwen4Exp architecture in upstream vLLM | CUDA/AMD implementation exists | Separate reusable common code |
| Qwen4Exp on Ascend/310P | Unsupported | Real 8K text request |
| W8A8 on 310P | Framework capability exists; model unvalidated | Expert parity and real request |
| TP4 | Framework capability exists; model unvalidated | Per-rank weight and output validation |
| EP4/FlashComm1 | Applicable MoE features; unvalidated | Optional measured gate after TP4 |
| PLE host offload | Missing for Ascend Qwen4Exp | Exact lookup and end-to-end parity |
| GDN | Generic hybrid support exists; Qwen semantics unvalidated | State lifecycle tests |
| QSA/indexer | Missing on Ascend | BF16 component and request parity |
| C8 QSA cache | Missing | Candidate A prototype |
| QSA-aware DCP | Missing | Candidate B prototype |
| Sparse host KV on 310P/Qwen | Unsupported | Fallback prototype only if needed |
| MTP | Weights exist; runtime unvalidated | Post-1M optional gate |
| Multimodal | Source weights exist; deferred | Real image request after text gate |
| ACLGraph | Deferred; 310P/QSA compatibility unknown | Attempt after eager correctness |
| Hybrid prefix caching | Missing | Cold/replay state parity |

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| W8A8 model imbalance or unsupported scale layout | Startup failure or OOM on one rank | Explicit tensor accounting and per-rank load tests |
| BF16 QSA cache does not leave runtime headroom | Cannot reach 1M | C8 and DCP prototypes before long runs |
| C8 cache harms selection/attention accuracy | Retrieval regression | Keep indexer BF16 initially; compare against BF16 at every gate |
| DCP sparse exchanges dominate decode | Unusable generation speed | Exchange selected rows only; profile within-card and cross-card traffic |
| 95 GiB PLE causes NUMA/page-fault stalls | High TTFT/TPOT | Shared allocation, NUMA placement, batched lookup and prefetch metrics |
| 256 GB host is consumed by PLE plus caches | OOM or swap | Fixed host reserve, no default 128 GiB prefix pool, fail-fast accounting |
| FP16 non-expert math differs from BF16 reference | Quality loss | Component tolerances and real held-out comparison |
| Long-context RoPE setting is wrong | Plausible but incorrect output | Explicit config plus position/reference tests before retrieval tests |
| Hybrid state is incomplete on preemption/reuse | Silent corruption | State inventory, deterministic replay and fail-closed behavior |
| 310P kernels lack required primitives/performance | Schedule slip | Eager reference path, bounded custom ops and milestone-based go/no-go |

## 13. Go/no-go rules

- Do not start a 1M run until 512K passes and allocation predicts the required
  per-chip headroom.
- Do not claim model support from dummy weights or server startup alone.
- Do not claim 1M support if input truncation, cache eviction, preemption or state
  recomputation reduces the effective retained context.
- Stop Candidate A if its fixed short-context accuracy threshold fails.
- Stop Candidate B if selected-row communication makes the decode floor
  unattainable after one focused optimization pass.
- Invoke host KV fallback only if both device-resident candidates fail memory or
  correctness gates.
- A functional 262K release remains valuable even if 512K or 1M is blocked.

## 14. Deliverables

1. Minimal Qwen4Exp changes in vLLM and backend-specific changes in vLLM Ascend.
2. Unit tests for tensor mapping, PLE, GDN, QSA, C8/DCP cache and hybrid state.
3. NPU end-to-end test configuration with real accuracy values.
4. Model tutorial covering the supported capacity and exact launch command.
5. Hardware and memory report for every capacity gate.
6. Cold/replay prefix-cache report if prefix caching is enabled.
7. Feature matrix covering ACLGraph, EP, FlashComm1, MTP and multimodal.
8. Real-weight logs and machine-readable results; dummy evidence is labeled
   separately.
9. Signed commits following the repository contribution rules.

## 15. References

- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next FP8 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- [Reported two-replica RTX PRO 6000 deployment](https://www.reddit.com/r/LocalLLM/comments/1wgzf2b/2_rtx_pro_6000_blackwell_server_edition/)
- [vLLM Ascend context-parallel design](context_parallel.md)
- [vLLM Ascend layerwise and sparse KV offload design](layerwise_and_sparse_kv_cache_offloading.md)
- [vLLM Ascend feature matrix](../../user_guide/support_matrix/feature_matrix.md)

## 16. Open decisions

1. Does the target expose approximately 46 GiB or the full nominal 48 GiB to
   each worker after firmware reservations?
2. Can the 310P QSA kernel maintain acceptable accuracy with C8 main K/V while
   retaining BF16 index keys?
3. Is QSA DCP selected-row communication faster than local C8 dequantization on
   the actual two-card topology?
4. Can four worker processes share PLE physical pages reliably, or should table
   rows be partitioned with an explicit owner?
5. What long-context RoPE/YaRN parameters will be treated as the authoritative
   one-million-token configuration for this checkpoint?
6. What held-out perplexity delta does the completed W8A8 conversion establish as
   the frozen accuracy threshold?
7. Which CANN/vLLM Ascend release will be the supported deployment baseline?
