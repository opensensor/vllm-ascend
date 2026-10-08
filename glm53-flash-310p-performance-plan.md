# Plan: GLM-5.3-Flash Performance on Four Ascend 310P Chips

**Generated**: 2026-10-01  
**Mode**: Swarm-ready dependency plan

## Overview

Implement the performance PRD in small, measurable candidates around the existing GLM adapter. First build a reproducible parser-aware quality and timing suite, then measure the exact current checkpoint on all four ranks. Develop a grouped W2/W4 projection candidate and a graph-safe kpool writer in separate work areas. Qualify each against the same eager baseline before promotion. Investigate 8K prefill with the full trace, prove resident-versus-host MLA equivalence at 16K, then add a bounded hot-page policy and qualify longer contexts. Treat MTP-1 as an optional later experiment. Integrate only candidates that pass operator parity, answer quality, serving throughput, memory, and stability gates in the [PRD](docs/source/developer_guide/Design_Documents/glm53_flash_310p_performance_prd.md).

The current source already has the GLM model, NZ-packed grouped W2/W4 operator, compact live KDA state, physical-stride QSA, pooled indexer, and synchronous host MLA implementation. Do not rebuild these from scratch. The prior 2.236/4.476 tok/s and 210.3-second 8K prefill results are historical, parser-free measurements, not the gate-A baseline for new candidates.

**2026-10-01 execution update:** The user permitted direct NPU operator checks while deferring another vLLM serve, then deferred all further NPU use. A filtered 32-token request completed after linking the matching 12-argument QSA extension, but produced incoherent output. Repeated greedy one-token requests varied at the prefill output, before QSA decode. The active filtered and baseline launchers had forced experimental FP16 mHC state rounding, which differs from the golden BF16 math; that override is removed. Direct NPU checks confirmed BF16 rounding bits, fresh convolution cache reset, 26-token flash-attention parity and determinism, focused KDA chunk parity, and mHC Sinkhorn parity. Whole-model coherence and first-token stability remain unverified with the corrected launchers. The mixed W4/W2 checkpoint also differs from the NVIDIA NVFP4 GPU golden, so checkpoint-level equivalence cannot be inferred from operator tests.

## Assumptions

- This is a coding plan only. No NPU run starts until the user releases the four devices. Source work and CPU tests can proceed beforehand.
- Keep the existing mixed W2/W4 checkpoint and its precision map. No conversion, public checkpoint release, vision work, or new environment variable is required for this pass.
- Use the existing `Glm5NextW2ForCausalLM` text path, TP/EP=4, `glm47` reasoning parser, and `poolside_v1` tool parser. Fix parser or template assumptions only when the actual served checkpoint disproves them.
- The current `artifacts/glm-profile-20260930/post-fusion-20260930/serve-compact-kda-8001.sh` is a launch record tied to paths on the NPU host. Port and path substitutions are recorded in each run manifest. The known-good source and OPP build are frozen before candidate work is promoted.
- The source baseline for this plan is committed HEAD `5fdfb93908e796856e5040251c955ab2eb1a0cd9`; unrelated working-tree edits are excluded. T5 must use this snapshot, or record a newer deliberately chosen known-good hash before any candidate is compared.
- Gate B/C/release numbers are targets, not guarantees. A failed target produces a bottleneck analysis and a recorded no-go or revised experiment, never an unreported target change.

## Prerequisites

- Repo dependencies, `pytest`, `ruff`, CANN 9.1/310P build tools, and `pre-commit`/`markdownlint` for `bash format.sh ci`.
- The real `GLM-5.3-Flash-W4through32-noclip-310p` checkpoint, tokenizer and chat template revision, four 48 GiB 310P chips, and an isolated known-good custom OPP package for hardware tasks.
- Access to the saved four-rank raw traces identified in `artifacts/glm-profile-20260930/post-fusion-20260930/README.md`; if unavailable, recapture them on the first authorized NPU run.
- Enough host RAM to reserve the declared host MLA history for long-context tests; report the allocation calculation before attempting a tier.
- Every worker is sharing a codebase: edit only owned files, do not revert unrelated edits, and report the exact paths changed. Use separate worktrees/build directories for native candidates and baseline measurement. Do not commit as a task requirement; any eventual implementation commit follows the repository's signed Conventional Commit rule.

## Dependency Graph

```text
T1 ──> T5 ──> T6, T7, T8, T9
T2 ──> T4 ──> T6
T3 ──> T7, T8
T9 ──> T10 ──> T11
T7 + T12 ──> T13
T6 + T7 + T8 + T11 + T13 ──> T14
T5 ──> T15
```

`T12` is an early read-only MTP feasibility audit; `T13` starts only after both `T7` and `T12`. A conditional task closed with a documented no-go satisfies its downstream dependency. Hardware tasks remain waiting, not failed, while devices are reserved.

## Tasks

### T1: Parser-aware quality and serving workload suite

- **depends_on**: []
- **agent_type**: worker
- **ownership**: `tools/glm_perf/__init__.py`, `tools/glm_perf/suite.py`, `tools/glm_perf/workloads.json`, `tests/ut/glm_perf/test_suite.py` (new paths only)
- **description**: Extend the behavior of `probe-coherence.py` in a reusable CLI without editing the historical artifact. Provide at least 20 fixed arithmetic, instruction, code, and retrieval cases; fixed seeds and scoring rules; 25-token short 256-output cases at one/four streams; 8K/32K retrieval; and parametrized four-window tiers. Use streamed timestamps for first/last token and per-request/aggregate decode, and record parser-separated reasoning/final content, raw markers, finish reason, token counts, p50/p95, early EOS, request settings, and tool-call success. Keep the 32-token fault smoke distinct from speed measurement. Store JSONL plus a summarized result and reject malformed/incomplete records.
- **acceptance_criteria**:
    - A deterministic offline fixture exercises timing arithmetic, concurrent fairness, early EOS, parser leakage, and tool-call scoring.
    - The CLI can target any base URL/model and writes one machine-readable row per request without requiring an NPU during development.
    - A quality pass requires completed final answers and zero raw thinking markers in final content; HTTP success alone does not pass tool calls.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_perf/test_suite.py`
    - `python3 -m tools.glm_perf.suite --help`
- **status**: Completed
- **log**: Parser-aware streamed JSONL suite has 20 fixed quality cases, tool-call scoring, short one/four-stream decode, 8K/32K retrieval, context tiers, and fault smoke. Tokenizer calibration now brings local raw 8K/32K/128K/256K prompts within 7 tokens of target and records tokenizer identity; served usage outside 5% fails the tier. `python3 -m pytest -q --noconftest tests/ut/glm_perf/test_suite.py`: 8 passed; CLI help and Ruff check/format pass. Server qualification waits for T5.
- **files edited/created**: `tools/glm_perf/__init__.py`, `tools/glm_perf/suite.py`, `tools/glm_perf/workloads.json`, `tests/ut/glm_perf/test_suite.py`

### T2: Isolated expert projection measurement harness

- **depends_on**: []
- **agent_type**: worker
- **ownership**: `tools/glm_perf/operator_bench.py`, `tests/ut/glm_perf/test_operator_bench.py` (new paths only)
- **description**: Generalize the existing `artifacts/glm-profile-20260930/bench-nzpacked-candidate.py` into a repeatable known-good/candidate operator harness. Compare separate processes and OPP paths with source and binary SHA-256, equal warmup/synchronization, 8 and 32 routed rows plus a prefill case, the actual fused W4 gate/up `[4096,4096]` and W2 down `[4096,2048]` projection shapes (`[output,input]`), zero local routes, peer-owned rows, and repeated experts. The older 2048-output W4 microbench is diagnostic only. Record medians, distribution/variance, workspace bytes, packed-code identity, output dtype, and dtype-specific parity or predeclared reduction-order bounds. Never load baseline and candidate implementations of the same operator name in one process.
- **acceptance_criteria**:
    - CPU-only tests verify case generation, output comparison, hash and summary schemas; NPU timing is deferred.
    - The harness explicitly identifies the OPP binary used and separates operator latency from serving throughput.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_perf/test_operator_bench.py`
    - `python3 -m tools.glm_perf.operator_bench --help`
- **status**: Completed
- **log**: Separate-process OPP/source/binary hash and output comparison harness covers 8/32 decode and 416-row prefill, W4 fused gate/up and W2 down, zero/singleton/repeated/mixed/peer routes, latency variance, and a labeled workspace estimate. It requires one isolated OPP root and a compiled grouped binary path; actual ACL workspace remains a hardware measurement. Route bound is 5120. `python3 -m pytest -q --noconftest tests/ut/glm_perf/test_operator_bench.py`: 5 passed; CLI help and Ruff check/format pass. NPU timing is T4/T6 work.
- **files edited/created**: `tools/glm_perf/operator_bench.py`, `tests/ut/glm_perf/test_operator_bench.py`

### T3: Make kpool writes capture-safe without changing selection

- **depends_on**: []
- **agent_type**: worker
- **ownership**: `vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py`, `tests/ut/glm_w2/test_kpool_ops.py`, `tests/ut/glm_w2/test_kpool_graph_buffers.py`
- **description**: Define CPU reference cases first, then replace both boolean advanced-indexing sites in `_write_pools` with bounded device-side slot/mask operations and fixed-shape metadata suitable for graph capture. Cover partial pools, completion across chunks, multiple requests, zero-work/padded decode rows, reused cache blocks, and preemption/state reset. Preserve kpool scoring, headwise ReLU, APE, causal tail, selected groups, and physical storage mapping. Keep eager as the default until T7.
- **acceptance_criteria**:
    - CPU tensors show exactly the same state/key writes and selected indices as the pre-change reference across the listed cases.
    - No device `.item()`, `nonzero`, boolean gather/scatter, dynamic allocation tied to selected-count, or host sync is added to the capture path.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_kpool_ops.py tests/ut/glm_w2/test_kpool_graph_buffers.py`
    - `ruff check vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py tests/ut/glm_w2/test_kpool_ops.py tests/ut/glm_w2/test_kpool_graph_buffers.py`
- **status**: Completed
- **log**: Fixed-shape masked storage writes replace both boolean-indexed sites; CPU reference parity passed with 20 tests and 1 unrelated import-dependent test deselected. `ruff check`, `ruff format --check`, and `git diff --check` passed. Full local suite cannot collect without `torch_npu`/`fla_npu`; 310P `index_add_` support and capture/replay remain T7 gates.
- **files edited/created**: `vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py`, `tests/ut/glm_w2/test_kpool_ops.py`

### T4: One fixed-shape grouped W2/W4 kernel candidate

- **depends_on**: [T2]
- **agent_type**: worker
- **ownership**: `csrc/gmm/w2_grouped_blocked_dequant_matmul_v310/**`, `csrc/gmm/w2_blocked_dequant_matmul_v310/op_kernel/w2_blocked_dequant_matmul_v310.h` (shared tile/Cube implementation), `tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_blocked_dequant_matmul_310.py`, `tools/glm_perf/operator_bench.py` and `tests/ut/glm_perf/test_operator_bench.py` (T2 is complete; extend the case matrix for singleton routes)
- **description**: In an isolated native build, prototype one fixed-shape decode schedule that visits active groups and reduces per-active-expert dequant/Cube tile traffic. Handle singleton and repeated-expert groups separately; preserve signed packed W2/W4 codes, NZ order, 32×32 scales, 4096-input W4 tiling, the existing projection output dtype/rounding contract, and zero peer-owned outputs. Predeclare error bounds if accumulation order changes. Do not repeat the already slower resident-L1 or binary-boundary-scan designs unchanged. Keep the existing kernel selectable as the baseline; do not change the serving method in `w2_dynamic.py` yet.
- **acceptance_criteria**:
    - Operator ABI and weight format stay compatible with the real checkpoint, or an explicit isolated ABI version is supplied without replacing the known-good OPP.
    - Hardware parity covers W4 gate/up and W2 down, 8/32 rows, empty/singleton/repeated/peer routes, and prefill shape before timing is trusted. The standalone W2 operator that shares the edited header also retains its parity.
    - Source notes explain expected tile/workspace traffic and exact candidate binary identity.
- **validation**:
    - `python3 -m pytest -q tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_blocked_dequant_matmul_310.py` (310P, after release)
    - `python3 -m tools.glm_perf.operator_bench --help` (CPU/import smoke)
- **status**: Isolated native build complete; NPU parity deferred by user
- **log**: Opt-in singleton-NZ L1 schedule plus repeated-group GM schedule implemented behind `GLM_W2_GROUPED_CANDIDATE`; default build remains known-good. Shared W2 header reuses invariant decode tables across active experts. Real fused W4/W2 e2e geometries cover singleton→repeated→singleton mixed groups, peer rows, and standalone regression. Harness enforces one isolated OPP root and compiled grouped binary identity; separate-process baseline/candidate bitwise comparison is a required hardware gate. Source/traffic/build notes added. The candidate package built and installed in isolated Threadripper paths; installer and object hashes are in `CANDIDATE.md`. Five harness CPU tests passed on the target host and all 25 operator tests collect. The user deferred NPU parity and timing.
- **files edited/created**: `csrc/gmm/w2_blocked_dequant_matmul_v310/op_kernel/w2_blocked_dequant_matmul_v310.h`, `csrc/gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/w2_grouped_blocked_dequant_matmul_v310.cpp`, `csrc/gmm/w2_grouped_blocked_dequant_matmul_v310/CANDIDATE.md`, `tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_blocked_dequant_matmul_310.py`, `tools/glm_perf/operator_bench.py`, `tests/ut/glm_perf/test_operator_bench.py`

### T5: Freeze and measure the exact eager baseline

- **depends_on**: [T1]
- **agent_type**: local
- **ownership**: `tools/glm_perf/analyze_trace.py`, `tests/ut/glm_perf/test_analyze_trace.py`, `artifacts/glm-perf-310p/baseline/**` (new paths only)
- **description**: When devices are released, launch the pre-edit committed source snapshot from the assumptions in an isolated worktree with a frozen known-good OPP package and the recorded parser-enabled flags; hash the OPP before starting the server. First check current kpool/router/combine and compact KDA state on NPU; if an AICore fault appears, identify and repair that first failing transition in a separate local fix before proceeding. Run the T1 suite at 256 tokens, 8K and 16K, one/four streams, and tool calls. Complete all four-rank trace attribution, separating prefill, decode, collective arrival wait/transit, and the per-step critical path. Record weight/allocator/workspace/graph/HCCL/fragmentation HBM ledger, host RSS, code/OPP/checkpoint hashes, launch arguments, prompt/response records, and fault logs. Do not infer new speed from the old 32-token run.
- **acceptance_criteria**:
    - A versioned gate-A artifact has parser-separated completed answers, four-rank timelines, memory per rank, and the exact build identity.
    - The 8K prefill duration is divided into model kernels, transfer, and API/scheduler time; all subsequent comparisons use this same baseline workload and a contemporaneous known-good server.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_perf/test_analyze_trace.py`
    - `python3 -m tools.glm_perf.analyze_trace --help`
    - `python3 -m tools.glm_perf.suite --help` (then run its 1/4-stream and 8K/16K hardware cases with the frozen server)
- **status**: QSA binding validated in serving; golden mHC rounding restored in launchers; vLLM serving deferred by user
- **log**: Four-rank trace analyzer and CPU fixtures added; 3 tests, CLI help, Ruff check/format pass. Saved 30 September profiler export has parsed kernel CSVs for only two ranks per capture. An isolated committed source snapshot, exact launcher, checkpoint metadata, and 464 OPP file hashes are frozen in the baseline artifact. Attempt 1 loaded weights in 315.24 seconds then failed on a missing Torch binding; the historical binding was linked and hashed. Attempt 2 loaded weights in 269.79 seconds (274.11–274.31 seconds total `load_model`) then failed on sparse MLA selecting the generic attention backend. A GLM-only selector fix and four CPU regressions are staged. Attempt 3 used that selector, loaded weights in 286.70 seconds (290.91–291.56 seconds total `load_model`), and opened the API; its first request failed because the shared 310P GDN builder omitted nested metadata consumed by W2 KDA. Attempt 4 used the W2-only builder, loaded weights in 299.33 seconds (303.28–303.58 seconds total `load_model`), and completed prefill, then failed on the first decode token because the 310P MLA override had an obsolete call signature. The override now accepts the shared MLA result object; 24 focused 310P MLA and W2 builder CPU tests pass on the target host. A later filtered-loader run passed that Python call but exposed an older 11-argument QSA extension linked into both isolated source copies; both now link the matching 12-argument extension, with a SHA preflight in the launch scripts. No completed answer or all-rank trace exists. The user deferred NPU retesting.
- **files edited/created**: `tools/glm_perf/analyze_trace.py`, `tests/ut/glm_perf/test_analyze_trace.py`, `artifacts/glm-perf-310p/baseline/README.md`

### T6: Qualify or reject the grouped-kernel candidate in serving

- **depends_on**: [T4, T5]
- **agent_type**: local
- **ownership**: `vllm_ascend/_310p/quantization/methods/w2_dynamic.py` only if a selectable integration is needed; `artifacts/glm-perf-310p/expert-candidate/**` (new)
- **description**: Rebuild cleanly and compare candidate versus contemporaneous known-good using T2 and T1, same checkpoint and launch flags. Check packed bytes and FP16 operator outputs first, then 8/32-row and prefill medians, then one/four-stream full serving and quality. Record repeat variance and new critical path. Leave known-good selected if any promotion rule fails.
- **acceptance_criteria**:
    - Promotion requires reproducible at least 15% isolated relevant decode-projection reduction and at least 10% matched end-to-end decode improvement, with no answer, prefill, HBM, or context regression.
    - Otherwise artifact records `revise` or `reject`, exact deltas, and next bottleneck; no unqualified candidate is enabled by default.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm5next_w2_grouped_bank.py`
    - `python3 -m pytest -q tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_blocked_dequant_matmul_310.py` (310P)
    - `python3 -m tools.glm_perf.suite --help` (then run the frozen matched hardware suite)
- **status**: Waiting for T4 native/NPU parity and T5 baseline
- **log**: No candidate package or serving comparison has been run; the 15% operator and 10% serving promotion gates remain open.
- **files edited/created**:

### T7: Capture and replay decode graphs as a separate candidate

- **depends_on**: [T3, T5]
- **agent_type**: local
- **ownership**: `artifacts/glm-perf-310p/graph-candidate/**`; `vllm_ascend/_310p/model_runner_310p.py` and `tests/ut/_310p/test_model_runner_310p.py` only if a minimal GLM-specific graph integration is required
- **description**: Review the runner change architecturally before editing it. Preflight `has_npu_triton_conv1d_update()` from `models/glm5next/ops/causal_conv1d.py`, because its PyTorch fallback calls `.item()` and stalls FULL capture; classify this separately from kpool failures. In an isolated candidate launch, confirm the kpool writer no longer triggers `aclnnNonzeroV2` under FULL_DECODE_ONLY capture; test exact-binary capture smoke, stable pointers, and state replay across one/four requests, padding, preemption, block reuse, and long decode. Compare eager and replay recurrent state, selected history, logits, final answers, memory, and sustained throughput. Keep host MLA eager-only and do not bundle MTP into this test.
- **acceptance_criteria**:
    - Capture and replay complete with no stale KDA/kpool state, wrong selection, AICore fault, or quality regression.
    - Graph memory is charged in the per-rank ledger; promote only with a measured serving gain that leaves the context gate viable.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_kpool_ops.py tests/ut/_310p/test_model_runner_310p.py`
    - `python3 -m tools.glm_perf.suite --help` (then run eager-versus-graph matched hardware cases)
- **status**: Waiting for T5 baseline and 310P
- **log**: T3's CPU-safe writer is prepared, but NPU `index_add_` support, convolution fallback preflight, graph capture/replay, recurrent-state parity, and HBM cost remain unverified.
- **files edited/created**:

### T8: Trace-guided 8K prefill and kpool optimization

- **depends_on**: [T3, T5]
- **agent_type**: worker
- **ownership**: `vllm_ascend/models/glm5next/kpool_ops.py`, `tests/ut/glm_w2/test_kpool_ops.py` only after T3 has finished, `artifacts/glm-perf-310p/prefill-candidate/**`
- **description**: Use T5's 512-token chunk attribution to pick one prefill critical-path change. Optimize kpool scoring only if score/selection is measured there; otherwise keep this task read-only and record the actual higher-priority operation for the integration task. Preserve 2048-token sparse budget, four-token groups, headwise ReLU, causal tail, selection parity at 2048/2049 and page-reuse boundaries, and physical-stride QSA addressing. Compare model prefill and long-context decode, not just an isolated cache copy.
- **acceptance_criteria**:
    - A candidate has fixed operator error bounds before benchmarking, CPU selection parity, NPU selected-group parity, and a matched end-to-end 8K prefill or long-context gain without shifting cost into host transfers.
    - If kpool is not on the traced critical path, artifact explains the no-change decision and names the measured bottleneck.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_kpool_ops.py`
    - `python3 -m tools.glm_perf.suite --help` (then run matched 8K/32K hardware cases)
- **status**: Waiting for T5 all-rank 8K attribution
- **log**: The saved parser exports are incomplete across ranks; no prefill kernel has been selected for optimization or benchmarked.
- **files edited/created**:

### T9: Prove host MLA equivalence at resident 16K

- **depends_on**: [T5]
- **agent_type**: local
- **ownership**: `artifacts/glm-perf-310p/host-16k/**`, `tests/ut/glm_w2/test_host_kv.py`; `vllm_ascend/models/glm5next/host_kv.py` and `vllm_ascend/_310p/attention/mla_v1_310.py` only for an isolated 16K correctness repair
- **description**: Run the existing opt-in synchronous host path, still eager, against the exact all-NPU 16K checkpoint and T1 prompts. Validate selected-page remapping and real-weight logits/final outputs; measure host RSS, NPU peak, transfer bytes, decode, and prefill. Add CPU regression cases for page reuse, request completion/preemption, scheduler block reuse, and four-long-request admission. Fix an equivalence bug locally before any hot-cache optimization; do not assume CPU unit coverage proves NPU correctness.
- **acceptance_criteria**:
    - Host and resident outputs agree within predeclared logit/operator bounds and pass the same final-answer suite at 16K.
    - The artifact states the host-path latency/capacity penalty and the exact memory reserve or startup failure for the configured request count.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_host_kv.py`
    - `python3 -m tools.glm_perf.suite --help` (then run resident-versus-host 16K hardware cases)
- **status**: Waiting for T5 baseline and 310P
- **log**: Real-weight resident-versus-host logits, quality, transfer cost, and memory have not been measured. T10 cannot start until this equivalence gate passes.
- **files edited/created**:

### T10: Add a bounded resident hot-page policy

- **depends_on**: [T9]
- **agent_type**: local
- **ownership**: `vllm_ascend/models/glm5next/host_kv.py`, `vllm_ascend/_310p/attention/mla_v1_310.py`, `tests/ut/glm_w2/test_host_kv.py`, `tests/ut/_310p/attention/test_mla_v1_310.py` after T9 is complete
- **description**: Add page identity/versioning from scheduler IDs, reuse selected resident 32-token pages across decode calls, and batch nonblocking H2D copies with explicit event/wait ordering before QSA. Audit the MLA caller's synchronous NPU-to-host row and selection-metadata copies, batch them where safe, and expose any unavoidable wait in timing metrics. Handle writes to resident pages, block-ID reuse, preemption, evictions, multi-request unions, bounded prefill segments, pinned-memory failure, and no-space startup diagnostics. Keep compressed indexer and live KDA state on NPU. Add transfer bytes, hits/misses, overlap, hot-cache capacity, and host RSS counters; distinguish NPU-to-host transfer, H2D transfer, and API/scheduler waits.
- **acceptance_criteria**:
    - CPU tests prove identical selected rows and remapped tables to T9's synchronous reference for reuse, eviction, and reused scheduler IDs.
    - Async path has a documented data-ready event and never lets attention consume an incomplete copy; HBM hot-cache size stays bounded by the declared concurrency. Per-row copies and synchronization are removed where possible, and remaining waits are measured rather than assumed away.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_host_kv.py tests/ut/_310p/attention/test_mla_v1_310.py`
    - `ruff check vllm_ascend/models/glm5next/host_kv.py vllm_ascend/_310p/attention/mla_v1_310.py tests/ut/glm_w2/test_host_kv.py tests/ut/_310p/attention/test_mla_v1_310.py`
- **status**: Waiting for T9 equivalence
- **log**: No hot-page source change was made; eviction/transfer policy depends on a validated synchronous 16K reference.
- **files edited/created**:

### T11: Qualify context tiers and host transfer economics

- **depends_on**: [T10]
- **agent_type**: local
- **ownership**: `artifacts/glm-perf-310p/host-context/**` (new)
- **description**: Compare optimized host mode with T9's synchronous reference and the resident baseline, in this order: 16K parity, one 32K, four 32K, one 128K, four 128K, then four 256K stretch. Record per-rank HBM and host RSS, transferred bytes, hit rate, overlap, prefill, decode, errors, and complete final answers. Stop before a tier whose modeled host or hot-cache allocation is unsafe. Keep graph/MTP disabled for host mode until separately supported.
- **acceptance_criteria**:
    - Each promoted tier has full-answer quality, no truncation, clear startup admission/failure, and a measured capacity/latency statement.
    - Release target requires four 128K windows and the PRD's throughput/prefill/soak gates; 256K is a measured stretch, never inferred.
- **validation**:
    - `python3 -m tools.glm_perf.suite --help` (then run each authorized context tier against a frozen launch)
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_host_kv.py`
- **status**: Waiting for T10 and 310P
- **log**: No 32K/128K/256K hardware tier was attempted while devices are reserved.
- **files edited/created**:

### T12: Audit MTP-1 feasibility before implementation

- **depends_on**: []
- **agent_type**: explorer
- **ownership**: `artifacts/glm-perf-310p/mtp-audit/README.md` only; source inspection is read-only
- **description**: Inspect actual checkpoint MTP tensors, the `Glm5NextMTP` shared-head fallback, live-only KDA cache restrictions, 310P graph dispatch, quantized expert mapping, and host-MLA eager restriction. State whether an eager, resident-MLA MTP-1 experiment can preserve rollback/recurrent state. Enumerate exact files and tests needed; give a no-go if the checkpoint or cache contract lacks a safe path. Do not weaken the cache guards during the audit.
- **acceptance_criteria**:
    - The audit resolves the PRD's “one MTP layer” claim against the actual checkpoint and current source, including the absent-own-head case.
    - It distinguishes a technically valid MTP proposal from a quality/throughput win and records a reversible enablement plan or no-go.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/models/test_glm5next_mtp_rotation.py tests/ut/models/test_glm5next_cache_config.py`
    - `rg -n 'speculative decoding|has_own_lm_head|GLM_HOST_KV' vllm_ascend/models/glm5next vllm_ascend/_310p/model_runner_310p.py`
- **status**: Completed
- **log**: Read-only inspection of the exact served checkpoint's 75,575-tensor index (SHA-256 recorded in artifact) confirmed 1,753 layer-45 MTP tensors, one declared MTP layer, and no own MTP head. Source audit found an unwired W2 MTP stub and live-KDA/host-MLA speculation guards; current build is a no-go. Static source search and Markdown lint passed. Unit-test collection is deferred locally: installed vLLM is stale; matching vLLM reaches missing `fla_npu`.
- **files edited/created**: `artifacts/glm-perf-310p/mtp-audit/README.md`

### T13: Optional MTP-1 candidate and hardware comparison

- **depends_on**: [T7, T12]
- **agent_type**: local
- **ownership**: `vllm_ascend/models/glm5next/mtp.py`, `vllm_ascend/models/glm5next/cache_config.py`, `vllm_ascend/_310p/model_runner_310p.py`, `tests/ut/models/test_glm5next_mtp_rotation.py`, `tests/ut/models/test_glm5next_cache_config.py`, `tests/ut/_310p/test_model_runner_310p.py`, and `artifacts/glm-perf-310p/mtp-candidate/**`; revise the plan before editing any other path T12 identifies
- **description**: If T12 finds a safe resident-MLA path and T7 passes graph replay, implement the minimal MTP-1 state/rollback change with explicit model-runner and cache architectural review. Keep host MLA excluded. Test deterministic output parity, proposal acceptance, verifier cost, accepted tokens per draft, full generated tokens/s, memory, and one/four-stream service. If either prerequisite yields a no-go, close this task as `Skipped (documented no-go)` with no code changes.
- **acceptance_criteria**:
    - Candidate either improves matched one/four-stream throughput with quality/state parity and acceptable HBM, or remains disabled with measured reason.
    - No guard is removed solely to make startup pass; preemption and rejected-draft state restoration have regression tests.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/models/test_glm5next_mtp_rotation.py tests/ut/models/test_glm5next_cache_config.py tests/ut/_310p/test_model_runner_310p.py`
    - `python3 -m tools.glm_perf.suite --help` (then run matched MTP-off/on hardware cases if feasible)
- **status**: Skipped (documented no-go)
- **log**: T12 inspected the exact served checkpoint and found the current W2 draft registration is a stub while compact live KDA rejects speculation. MTP-1 remains disabled; no guard or runner was changed. Reopen only after a reviewed W2 draft loader and rollback-safe KDA state design, then T7 graph parity and NPU availability.
- **files edited/created**: None

### T14: Integrate, validate, and make the release decision

- **depends_on**: [T6, T7, T8, T11, T13]
- **agent_type**: local
- **ownership**: `artifacts/glm-perf-310p/release/**`, final reproducible launcher under `examples/`, and this plan's status/log fields; reconcile any shared runner/test changes locally
- **description**: Review parallel diffs and merge only individually promoted candidates into one clean build. Resolve any interaction between expert, graph, prefill, and host modes; graph and host remain separate launch profiles unless proven compatible. Run the full GLM unit suite, targeted NPU operator tests, T1 quality/workload ladder, one/four-stream comparisons, tool calls, 8K/32K/128K context tiers, per-rank memory ledger, and an extended mixed-load/no-AICore-fault run. Record source/OPP/checkpoint hashes, exact launch, rollback command, gate A/B/C/release results, and `adopt/revise/reject` for every candidate. Run `bash format.sh ci` after all edits, including Markdown.
- **acceptance_criteria**:
    - An exact promoted build passes the PRD quality and stability gates, and each numeric target is reported as pass or fail from complete 256-token requests; early EOS is separate.
    - Release artifact states recommended configuration, validated maximum concurrency/context, memory headroom, known limitations, and rollback. If release target fails, artifact gives a measured no-go and next bottleneck rather than labeling the service ready.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2 tests/ut/models/test_glm5next_mtp_rotation.py tests/ut/_310p/test_model_runner_310p.py`
    - `python3 -m pytest -q tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_blocked_dequant_matmul_310.py` (310P)
    - `python3 -m pytest -q tests/` (full local suite, with required 310P hardware available)
    - `bash format.sh ci`
    - `python3 -m tools.glm_perf.suite --help` (then run the documented exact-build hardware ladder and soak)
- **status**: Waiting for hardware gates T6–T11
- **log**: No candidate has passed NPU parity or matched serving gates. Final launcher, release recommendation, full suite, and `bash format.sh ci` remain deferred until candidate decisions and a stable hardware build exist.
- **files edited/created**:

### T15: Qualify an opt-in GLM W2 safetensors reader

- **depends_on**: [T5]
- **agent_type**: local
- **ownership**: `vllm_ascend/model_loader/glm_w2_safetensors.py`, `vllm_ascend/__init__.py`, `tests/ut/glm_w2/test_glm_w2_safetensors_loader.py`, `tools/glm_perf/audit_loader.py`, `artifacts/glm-perf-310p/loader-candidate/**`
- **description**: The Qwen EP filter does not match GLM's `_codes` and `_scale` tensors. Keep the default loader untouched and offer an opt-in safetensors iterator that skips nonlocal decoder W2 expert tensors before `get_tensor`, using the model bank's TP-contiguous ownership, and skips superseded local expert tensors using the checkpoint index. Preserve final local experts, vision, dense layers, and unwired MTP. First prove exact checkpoint-header coverage on CPU, then compare default and filtered startup, physical reads, local parameter digests, full answers, tool calls, and memory on the same frozen build when the user releases NPUs again.
- **acceptance_criteria**:
    - All four TP ranks retain the exact required local tensor names and reject unsupported checkpoint/source geometry.
    - Startup improvement is measured from matched default and opt-in runs; 102.239 GiB of skipped `get_tensor` payload per rank, including superseded local experts, is treated as theoretical until disk reads and wall time confirm it.
    - Any missing weight, output mismatch, higher peak memory, or no material startup gain leaves the default loader selected.
- **validation**:
    - `python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm_w2_safetensors_loader.py`
    - CPU checkpoint-header audit, then matched NPU startup and answer suite after device release
- **status**: Filtered request decoded after QSA ABI fix; golden math and parameter parity remain unverified; vLLM serving deferred
- **log**: Eight CPU audit/loader tests passed for the initial peer filter. Full checkpoint-header review found 62,208 peer tensors / 107,017,666,560 payload bytes (99.668 GiB) per rank skipped before `get_tensor`, including 7,776 superseded expert entries absent from the logical index; every one of the 33 mixed shards remains needed. The audit identified another 2,592 superseded **local** entries / 2,760,376,320 bytes (2.571 GiB) per rank. The candidate now uses the index to skip these too; five focused loader CPU tests pass, including an overlay-value regression. One filtered run loaded all shards in 238.67 seconds on rank 0, versus 299.33 seconds in one default run. The 60.66-second difference is a single uncontrolled comparison; skip counters, live bank contents, and completed answers remain unverified. The filtered request reached first decode and failed because an older 11-argument QSA extension was linked despite a 12-argument Python call. The matching binding is now linked and preflight checked, with NPU retesting deferred. See `artifacts/glm-perf-310p/loader-candidate/README.md`.
- **files edited/created**: `vllm_ascend/model_loader/glm_w2_safetensors.py`, `vllm_ascend/__init__.py`, `tests/ut/glm_w2/test_glm_w2_safetensors_loader.py`, `artifacts/glm-perf-310p/loader-candidate/README.md`

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
| --- | --- | --- |
| 1 | T1, T2, T3, T12 | Immediately, on disjoint files; T12 is read-only except its audit artifact |
| 2 | T4 | T2 complete; isolated native build |
| 3 | T5 | T1 complete and NPUs released; frozen known-good source/OPP |
| 4 | T6, T7, T8, T9 | T5 complete and each task's other dependencies complete, including T3 for T8; separate candidate worktrees/OPP packages and serialized NPU slots |
| 5 | T10 | T9 passes 16K equivalence |
| 6 | T11, T13 | T10 complete for T11; T7 and T12 complete for T13; use separate NPU windows |
| 7 | T14 | T6, T7, T8, T11, and T13 closed, including documented no-go outcomes |
| Independent | T15 | CPU audit can proceed now; matched startup waits for T5 and device release |

## Integration Strategy

- Baseline measurement and each native candidate use isolated source snapshots and OPP directories. Record binary and source SHA-256; never compare two kernels with the same registered name inside one Python process.
- Workers edit their owned paths directly and report changed paths. `T3` finishes before `T8` touches `test_kpool_ops.py`; `T9` finishes before `T10` touches `test_host_kv.py` or host MLA source. `T7` and `T13` serialize any runner edits. The local integration owner resolves final test/code conflicts without overwriting unrelated work.
- Source work may continue while hardware is reserved; mark T5–T11/T13/T14 `Waiting for NPU availability` in their logs if necessary. If a task hits a parity, fault, or memory blocker, record the input, binary identity, failing case, and no-go/revision decision. Dependent work waits for the fix or closes conditionally; it does not bypass the gate.
- Keep one promoted change per measured serving comparison. Do not combine expert, graph, host, and MTP candidates before each individual candidate's gate is decided.

## Testing Strategy

- CPU: deterministic request/timing/score fixtures, packed-code and kpool reference parity, graph metadata reuse, host-page remapping/eviction, startup/admission failures, and focused regression tests for every code change.
- NPU operator: GLM geometry and 8/32 decode rows plus prefill, exact packed-code parity, declared FP16/error bounds, current/peak workspace, repeated/zero/peer routes, and stable medians with variance.
- NPU service: parser-enabled complete answers; at least 256 generated tokens for sustained rates; 25-token one/four stream, 8K/32K retrieval, four 32K then 128K contexts, per-request p50/p95, TTFT, output token counts, HBM/host RSS, and tool-call success. Include short smoke only for rapid fault detection.
- Promotion: isolated 15% and matched 10% expert thresholds; graph and host require parity plus positive end-to-end gain/capacity; all candidates preserve quality and no AICore faults. Full release gates are the PRD's B/C/release table.

## Risks and Mitigations

- **Risk**: NPU reservation delays all real speed and quality conclusions.  
  **Mitigation**: Complete CPU/source tasks now; keep hardware tasks waiting with exact commands and a frozen baseline build.
- **Risk**: Stale OPP objects or cross-rank build mismatch produce false parity/timing.  
  **Mitigation**: Isolated clean packages and source/binary hashes for each rank and run.
- **Risk**: A faster projection or QSA call does not improve service because the critical path moves.  
  **Mitigation**: Require matched end-to-end timing, all-rank attribution, and a recorded reject/revise decision.
- **Risk**: Graph capture reuses stale kpool/KDA state or changes selected history.  
  **Mitigation**: CPU reference cases before capture, eager/replay state/logit comparison, and preemption/reuse tests; keep graph off on failure.
- **Risk**: Host history raises capacity but synchronous or premature page copies worsen decode or read stale data.  
  **Mitigation**: Prove synchronous 16K equivalence before caching; explicit page version/event ordering and tiered NPU qualification.
- **Risk**: MTP conflicts with live-only KDA cache or lacks a valid checkpoint head.  
  **Mitigation**: Read-only feasibility audit; conditional implementation and no-go path without weakening guards.
- **Risk**: Four long windows exceed HBM or host RAM despite single-request success.  
  **Mitigation**: Per-rank allocator ledger and pre-start capacity model; step through 16K, 32K, 128K, then 256K stretch.

## Reviewer Notes

- One read-only subagent review found a missing T3→T8 dependency, host MLA D2H synchronization outside the original T10 ownership, an unrecorded baseline snapshot, an additional KDA convolution graph blocker, dtype wording, and unclear MTP audit ownership. The plan now freezes committed HEAD for baseline, includes the MLA caller in T10, adds the convolution preflight, corrects dependency and ownership fields, and redraws the graph from those fields.
- Local review corrected the real fused W4 gate/up benchmark shape to `[4096,4096]`; the older 2048-output microbench remains diagnostic only. The live workspace's unrelated Qwen GDN edits and artifact directory were not touched.
