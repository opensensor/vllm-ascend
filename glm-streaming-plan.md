# Plan: Upgraded GLM expert streaming architecture

**Generated**: 2026-10-09
**Mode**: Swarm-ready dependency plan

## Overview

Implement the complete streaming expert path shown in the
[architecture diagram](artifacts/glm-perf-310p/scale-layout-live-20261009/STREAMING.md):
input quantization/routing → resident weight reuse → gate/up projections →
SwiGLU/requantization → compact handoff → down → stable route combination →
shared expert and rank reduction. Process every expert and K32 group in bounded
tiles. Redesign the operand, readback and vector-consumer schedule inside both
projections; validate the entire model before promotion.

Execution began on 2026-10-09 with `$parallel-task glm-streaming-plan.md`.
T1–T11 are complete. The final v1111 passed native and actual-input gates, but
serving trials rejected promotion because c4 generation rate regressed 6.8%;
qualified v1021 was restored. T12/T13 are complete within their bounded
diagnostic/offline scope; T14 integration and qualification are in progress.
Task logs below distinguish these results from promotion.

The starting code is commit `b7d8b8f88`, with qualified prefill v1021 and decode
v984 as the reference. Recheck actual runtime state when execution begins.
The fresh v1021 trace attributes rank-zero gate/up 15.258 s, down 7.013 s,
preparation 1.442 s, route reduction 1.361 s and unoverlapped communication
11.652 s. Device-free time is 0.154 s. These are attribution counters, not
independent recoverable latency. These starting-profile measurements preceded implementation. Final serving
results are recorded in T10/T11; the candidate was not promoted.

## Assumptions

- Preserve the permanent mixed W2/W3/W4 checkpoint and FP16 scale marker
  `fp16_storage_fp32_compute_v1`. W2/W3 rebuild INT4 codes on core; W4 loads
  prepared codes directly. Do not introduce FP16 weight banks or model-wide
  requantization.
- Preserve independent K32 factors, FP32 addition order, qualified FP16
  projection/output rounding and stable route order. FMA, larger quantization
  groups and approximate nonlinearities require a separate numerical plan.
- Target A4 bulk prefill initially. Keep qualified sparse/A8 behavior explicit
  and retain decode v984. Decode and mixed workloads are regression gates;
  this plan does not promise faster decode from unchanged decode kernels.
- Keep external N128/M32 route layouts and the compact GM gate/up→down
  boundary. Internal microtiles may change only through the T1 contract.
  Down needs intermediate columns produced by multiple cores.
- Keep context 131072, token budget 1280, MTP1, graph sizes 2/8, TP4 and cache
  settings fixed during comparisons. The separate 1296-token scheduling
  worksheet and prefill graph experiments are outside this candidate.
- Existing L1 weight reuse and fused SwiGLU/requantization are retained.
  Zero scalar instructions is not a target; move tensor operations to
  vector/Cube units and reduce scalar issue/address/event overhead.

## Prerequisites

- Repo root: `/run/media/matteius/20TB-drive/vllm-ascend`; preserve concurrent
  Qwen edits. No task may reset, revert or overwrite unrelated work.
- CPU tools: Python, pytest, C++17 compiler, pre-commit; existing GLM CPU tests.
- NPU host: `matteius@192.168.53.187`, SSH with `-F /dev/null`, installed CANN
  and `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`. Source
  `/usr/local/Ascend/ascend-toolkit/set_env.sh` before NPU Python commands.
- Real checkpoint:
  `/srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008`.
  Four loaded ranks previously contain 43 banks × 72 experts per rank.
- API historically `http://127.0.0.1:8001`; actual launch/dispatch must be read
  again. Freeze the current qualified replacement source, baseline build
  provenance, binary hashes and loaded storage identities before any swap.
- Read installed SDK headers for 310P UB/L1/L0 capacities, copy alignment,
  event support and event-ID limits. Do not infer 310P compatibility from a
  newer CANN/910/950 example or upgrade the toolkit to make an example compile.
  Huawei's [static tensor guide](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00019.html)
  describes ownership and forward/backward synchronization for double buffering;
  installed-version support and actual overlap remain verification requirements.
- Artifact root for execution: `artifacts/glm-perf-310p/streaming-upgrade/`.
  Each task writes only its own `Tn/` subdirectory; final integration owns
  the root report. Commands below run from repo root unless stated otherwise.
- T6/T7 stage a frozen source checkout and bundle in an isolated directory on
  the NPU host, preserving these relative paths. Run hardware commands from
  that checkout's root with the NPU Python environment; do not run against an
  unstaged live source tree. Record local-to-remote paths and source digests.

## Dependency Graph

```text
T1 memory + ABI ──┬── T3 operand producer ──┐
                 ├── T4 product consumer ──┼── T6 kernel/build integration ──┐
                 ├── T5 compact epilogue ──┘                               │
                 └───────────────────────┐                                ├── T8 native gates
T2 reference + workload protocol ────────┴── T7 resident controller ─────────┘       │
        │                                                                          ▼
        └─────────────────────────────────────────────────────────────────────── T9 actual-input gate
                                                                                   │
                                                                                   ▼
                                                                            T10 paired E2E trials
                                                                                   │
                                                                                   ▼
                                                                            T11 final integration
```

`depends_on` fields below are authoritative. T8 also requires the T2 reference;
T9 requires T2, T6, T7 and T8; T10 requires T2, T7 and T9.

## Tasks

### T1: Prove memory lifetimes and freeze the streaming interface

- **depends_on**: []
- **agent_type**: local
- **ownership**: New `tools/glm_perf/glm_streaming_contract.h`,
  `tools/glm_perf/streaming_memory.py`,
  `tests/ut/glm_perf/test_streaming_memory.py`; artifacts `T1/`.
- **description**: Inventory actual v1021 generic and W4 allocations and
  overlapping lifetimes. Define operand/product slots, microtile geometry,
  scale layout and producer/consumer/epilogue signatures. Prove startup,
  steady-state and drain behavior with finite buffers and exact event ownership.
  Inspect installed SDK constants without launching kernels.
- **acceptance_criteria**:
    - Publish `T1/contract.json` and a lifetime/event table covering gate, up,
    down, W2/W3 reconstruction, quantization and sparse tails.
    - Freeze concrete tensor types, byte offsets, valid versus physical paired
    rows and slot states. T6 owns the Cube issue loop; T3 supplies acquired
    operands, T4 consumes completed products, T5 stores finalized outputs.
    Define their call signatures and the allocation/event-ID owner explicitly.
    - Include `decoded_`, `mask_`, factor scratch, scale indices, L1 weight and
    activation caches, L0A/B/C and the gate result that survives up. Current
    product scratch is 64 KiB and the separate accumulator is 16 KiB.
    - Account for alignment and 8 KiB SDK UB reserve; use actual SDK capacity
    limits. All K4096 partials require 1.9375 MiB and must never be materialized.
    - Select only a schedule that fits. If two live slots do not fit, define a
    smaller internal microtile or reject that overlap topology with evidence.
    Do not silently degrade into the old loop and label it upgraded streaming.
    - Header and JSON agree; later task workers request contract changes from
    the coordinator rather than editing T1-owned files.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_memory.py tests/ut/glm_perf/test_compact_w4_scratch.py`
    - New CLI: `python -m tools.glm_perf.streaming_memory --output artifacts/glm-perf-310p/streaming-upgrade/T1/contract.json`
    - Check rejected overcommit and live-alias cases, not only accepted totals.
- **status**: Completed
- **log**: Installed dav-2002 CANN 9.1.0 header capacities verified. Two 32 KiB raw slots reuse decoded scratch; independent 16 KiB conversion plane fits all four projections (generic down spare 4288 B). Device event/performance qualification remains T8. Contract and compact-W4 CPU checks: 52 passed using `--confcutdir=tests/ut/glm_perf`.
- **files edited/created**: `tools/glm_perf/glm_streaming_contract.h`, `tools/glm_perf/streaming_memory.py`, `tests/ut/glm_perf/test_streaming_memory.py`, `artifacts/glm-perf-310p/streaming-upgrade/T1/`.

### T2: Freeze the reference and reproducible workload protocol

- **depends_on**: []
- **agent_type**: worker
- **ownership**: New `tools/glm_perf/streaming_protocol.py`,
  `tests/ut/glm_perf/test_streaming_protocol.py`; artifacts `T2/`.
- **description**: Capture immutable reference metadata and define synthetic,
  real-bank, shadow and E2E workloads. Use existing v1021 sources/provenance and
  read-only runtime queries. Do not allocate NPU tensors or change dispatch.
- **acceptance_criteria**:
    - Produce `T2/reference.json`, frozen qualified replacement source and
    `T2/workloads.json`; include exact build options, source/binary hashes,
    scale marker, runtime settings and checkpoint/bank identity rules.
    - Freeze `T2/evidence-schema.json` for build, native gates, shadow and trial
    reports: contract/reference/source/binary digests, version/namespace,
    supported dispatch, required case coverage, completion/error states and
    artifact locations. T6 implements admission; T8 produces native evidence;
    T7 consumes it. No worker invents its own incompatible report format.
    - Include cold 640/6400/long prompts, 1920-token shadow input, warmed
    identical-prefix controls, c1/c4 decode and mixed prefill/decode workloads.
    Freeze token IDs or tokenizer/rendered-input hashes and generation length.
    - Distinguish scheduled versus completed tokens, TTFT versus generated
    tokens/s, first-use versus steady state, and cache-hit versus cold work.
    - Include natural-language/tool inputs in addition to repeated-token timing
    controls. Exact first-token equality alone is not a quality gate because
    baseline text varied in the prior session.
    - Prescribe identical untimed initialization/graph warmups for both variants
    and identical prompt token IDs within every ABBA cycle. Separate these
    from timed cold requests, clearing prefixes after each warmup.
    - Freeze timing semantics: use request-correlated server first-generated-token
    time when available; otherwise label client time to the first token-bearing
    SSE event as streamed TTFT. Track first visible text separately for
    reasoning/tool responses; never silently substitute it for token latency.
    - Define repeat spread as `max((p75 - p25) / median)` across the two variants.
    A practical promotion gate requires median cold latency improvement greater
    than `max(0.02, 2 * repeat_spread)`. Report this as a repeatability rule,
    not a statistical-significance proof. The 10% objective is distinct.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_protocol.py`
    - New CLI: `python -m tools.glm_perf.streaming_protocol --freeze --npu-host matteius@192.168.53.187 --output-dir artifacts/glm-perf-310p/streaming-upgrade/T2`
    - Verify manifest hashes and inspect that freeze makes no mutation RPCs.
- **status**: Completed
- **log**: Read-only freeze verified four ranks using actual prefill v1021/decode v984 source/operator/binaries, unchanged generation/storage and health 200. Reference digest `960369e7bf514748c5e9d81e1377535289199f5e349f15a365c917393df94c71`; schema covers 438 native and 344 shadow cases. 49 isolated CPU tests passed; combined T1/T2 regression selection passed 101 tests. Global pytest conftest currently lacks an unrelated flash-linear-attention module; `--confcutdir=tests/ut/glm_perf` resolves isolation without environment changes.
- **files edited/created**: `tools/glm_perf/streaming_protocol.py`, `tests/ut/glm_perf/test_streaming_protocol.py`, `artifacts/glm-perf-310p/streaming-upgrade/T2/`.

### T3: Implement bounded operand production and reuse

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `tools/glm_perf/glm_streaming_operands.h`,
  `tests/ut/glm_perf/test_streaming_operands.py`,
  `tests/ut/glm_perf/reduce_kernel_cpu_stubs/check_streaming_operands.cpp`;
  helper-specific `tests/ut/glm_perf/streaming_operands_cpu_stubs/`; artifacts `T3/`.
- **description**: Implement the T1 producer interface for resident weights
  and routed activations. Stage upcoming operands while preserving the complete
  existing expert/column/row ownership and L1 reuse. Do not edit the shared
  Projection implementation; T6 connects the helper.
- **acceptance_criteria**:
    - W4 direct loading and W2/W3 reconstruction preserve code interpretation;
    scales are independent of integer operand packing.
    - Every slot has explicit acquire/release dependencies, startup and drain
    handling; consumers cannot observe a partially copied tile.
    - Account for gate→up activation reuse, multi-batch expert reuse, empty
    experts, odd final K-group batches and row counts 1/4/5/15/16/17/30/31.
    - No device→host route reads, per-element scalar tensor arithmetic, new
    weight GM workspace or blanket deletion of barriers.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_operands.py`
    - CPU address/event model detects overwrite-before-release and unmatched
    events. Real instruction ordering is deferred to integrated T8 gates.
- **status**: Completed
- **log**: Actual-helper CPU event/copy tests passed 346 cases; independent T3/T5 combined selection passed 379. Added contract-approved MTE1_MTE2 backward fence protecting L1 staging in addition to M_MTE1 protecting L0. No extra allocation or PIPE_ALL in helper steady state; frozen API retained. Device legality/overlap remains T8.
- **files edited/created**: `tools/glm_perf/glm_streaming_operands.h`, `tests/ut/glm_perf/test_streaming_operands.py`, `tests/ut/glm_perf/reduce_kernel_cpu_stubs/check_streaming_operands.cpp`, `tests/ut/glm_perf/streaming_operands_cpu_stubs/kernel_operator.h`, `artifacts/glm-perf-310p/streaming-upgrade/T3/`.

### T4: Implement streamed product extraction, scaling and accumulation

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `tools/glm_perf/glm_streaming_products.h`,
  `tests/ut/glm_perf/test_streaming_products.py`,
  `tests/ut/glm_perf/reduce_kernel_cpu_stubs/check_streaming_products.cpp`;
  helper-specific `tests/ut/glm_perf/streaming_products_cpu_stubs/`; artifacts `T4/`.
- **description**: Replace fine-grained consumption with the T1 bounded
  product-slot consumer. Handle complete Cube readback tiles and vector scale
  application for gate, up and down; retain FP32 accumulators over all K.
- **acceptance_criteria**:
    - Preserve each independent K32 dot and the exact operation sequence:
    weight scale × activation scale, then dot × factor, then ordered FP32 add.
    Do not enable FMA, merge group factors or change the quantization group.
    - Preserve N16 fractal interpretation, paired-group ordering and tail masks;
    consume every valid product exactly once.
    - Reuse slots only after all vector readers finish; cover product/decode
    scratch alias boundaries and the accumulator lifetime through the last K.
    - Record expected reductions in issue/readback/barrier counts as a model,
    not measured performance. Tests exercise cancellation-sensitive sums.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_products.py`
    - CPU exact FP32 reference and ownership model cover changing tiles,
    adversarial scales, zero/signed-zero and sparse/dense boundaries.
- **status**: Completed
- **log**: 124 CPU tests and two ASan/UBSan probes passed, including all row counts, two scale layouts and cancellation-sensitive ordered FP32 sums. Final consumer uses direct strided casts and individual borrowed descriptors, preserving the frozen public API. v1111 complete native/shadow gates passed. Removing ProductViews return reduced gate/up scalar attribution from 57.434 to 31.800 ms in the controlled diagnostic; exact spill attribution remains unproven.
- **files edited/created**: `tools/glm_perf/glm_streaming_products.h`, `tests/ut/glm_perf/test_streaming_products.py`, `tests/ut/glm_perf/reduce_kernel_cpu_stubs/check_streaming_products.cpp`, `tests/ut/glm_perf/streaming_products_cpu_stubs/`, `artifacts/glm-perf-310p/streaming-upgrade/T4/`.

### T5: Implement the compact epilogue tile handoff

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `tools/glm_perf/glm_streaming_epilogue.h`,
  `tests/ut/glm_perf/test_streaming_epilogue.py`,
  `tests/ut/glm_perf/reduce_kernel_cpu_stubs/check_streaming_epilogue.cpp`;
  helper-specific `tests/ut/glm_perf/streaming_epilogue_cpu_stubs/`; artifacts `T5/`.
- **description**: Use the T1 dead-scratch interval to assemble bounded
  quantized-output tiles and issue larger aligned copies into the unchanged
  routed-down ABI. Retain existing vector SwiGLU and exact quantizer arithmetic.
- **acceptance_criteria**:
    - Preserve FP16 projection and post-SwiGLU boundaries, K32 max/scale and
    round-to-nearest code behavior; do not change scales to approximate math.
    - Produce identical packed hidden codes and v1021 dense group-major/sparse
    scale layouts, including defined padding for all replayed row counts.
    - Prove tile-store ownership across output cores; keep compact GM handoff
    explicit. Reuse dead scratch only after its final projection reader.
    - Handle zero activation, tails and successive 31→5→4→0-local route counts
    without stale padding; report DMA call/byte tradeoffs.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_epilogue.py tests/ut/glm_perf/test_group_major_down_scales.py`
    - Compare packed bytes and FP32 scale bits, not only dequantized tolerances.
- **status**: Completed
- **log**: Exact frozen API and byte-layout preservation validated; final strided staging passed 90 CPU/sanitizer tests. At 31 rows modeled GM calls/bytes fall 126/8064 to 2/4096; final local staging uses 32 copies rather than 124 with the same 3968 local bytes. v1111 full native/shadow gates passed. Quantizer/scales unchanged; staging alone did not materially improve event medians.
- **files edited/created**: `tools/glm_perf/glm_streaming_epilogue.h`, `tests/ut/glm_perf/test_streaming_epilogue.py`, `tests/ut/glm_perf/reduce_kernel_cpu_stubs/check_streaming_epilogue.cpp`, `tests/ut/glm_perf/streaming_epilogue_cpu_stubs/kernel_operator.h`, `artifacts/glm-perf-310p/streaming-upgrade/T5/README.md`.

### T6: Integrate the complete native kernel and append-only build

- **depends_on**: [T1, T2, T3, T4, T5]
- **agent_type**: local
- **ownership**: `tools/glm_perf/glm_fused_moe.cpp`,
  `tools/glm_perf/build_reconstruction.py`, `tools/glm_perf/glm_fused_moe.py`,
  `tools/glm_perf/fused_moe_control.py`; new
  `tools/glm_perf/build_streaming.py`, `tests/ut/glm_perf/test_streaming_build.py`,
  `tests/ut/glm_perf/test_streaming_dispatch.py`; artifacts `T6/`.
- **description**: Connect producer/consumer/epilogue through one Projection
  schedule for both gate/up and down. Integrate compile-time bounds, option
  validation, resource construction and manifest admission in one owned change.
- **acceptance_criteria**:
    - A single integrated candidate covers all expert stages. Gate/up and down
    must both use the upgraded bulk path; unchanged stages remain explicitly
    qualified. Unsupported/sparse/A8 dispatch is recorded, never hidden.
    - Build takes T1 contract and frozen T2 reference options. Reserve a unique
    unused version at execution time; copy/hash all helper headers and reject
    partial pairs, incompatible options, stale contracts and overwritten dirs.
    - Runtime manifest requires matching complete gates and actual binaries.
    No new environment variables, mutable global state or broad model-runner
    patch. External packed bank/route ABI stays unchanged.
    - Standalone probe construction is permitted with matching build/contract
    digests before T8 evidence exists; it cannot install into the live server.
    Live admission alone requires complete T8 gates under the frozen T2 schema.
    Reject missing coverage, copied reports and changed headers/binaries.
    - Publish the actual combined prologue/steady-state/drain instruction and
    event schedule for gate, up and down. Recheck helpers' lifetime proofs
    against this integrated order, including L0C readback, gate survival during
    up, `decoded_`/`mask_` aliases, quantizer scratch and event reuse per expert.
    - Keep decode v984 resource selection intact and preserve shared scratch
    capacity/return-output lifetime. Whole-rank memory envelope includes
    simultaneously loaded old/new resources and verification scratch.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_build.py tests/ut/glm_perf/test_streaming_dispatch.py tests/ut/glm_perf/test_build_reconstruction.py tests/ut/glm_perf/test_fused_moe.py`
    - New CLI: `python -m tools.glm_perf.build_streaming --contract artifacts/glm-perf-310p/streaming-upgrade/T1/contract.json --reference artifacts/glm-perf-310p/streaming-upgrade/T2/reference.json --build-dir artifacts/glm-perf-310p/streaming-upgrade/T6/build`
    - Compile on the NPU host with its installed SDK; inspect provenance and
    compiled bounds. This command builds only and never contacts the server.
- **status**: Completed
- **log**: Both generic/W4 gate/up and down integrate the two-slot producer/consumer and compact handoff; exact arithmetic and sparse/A8 reference paths remain explicit. Actual-method schedule tests passed 736 cases (48 inapplicable down-cache skips), detecting eight injected faults and a fixed direct-W4 staging regression. Integration selection passed 879 tests. Installed dav-2002 build v1102 completed with frozen contract/reference/source/binary evidence. Whole-rank device correctness/performance remains T8 onward.
- **files edited/created**: `tools/glm_perf/glm_fused_moe.cpp`, `tools/glm_perf/glm_fused_moe.py`, `tools/glm_perf/fused_moe_control.py`, `tools/glm_perf/build_reconstruction.py`, `tools/glm_perf/build_streaming.py`, `tests/ut/glm_perf/test_streaming_build.py`, `tests/ut/glm_perf/test_streaming_dispatch.py`, `artifacts/glm-perf-310p/streaming-upgrade/T6/`.

### T7: Prepare a guarded resident swap and E2E controller

- **depends_on**: [T1, T2]
- **agent_type**: worker
- **ownership**: New `tools/glm_perf/streaming_resident.py`,
  `tools/glm_perf/resident_candidates/streaming_experts.py`,
  `tests/ut/glm_perf/test_streaming_resident.py`; artifacts `T7/`.
- **description**: Reuse ResidentClient, qualified replacement composition,
  native launch audit and prompt-progress logging. Prepare phase-specific
  commands for gated install, identical-input shadow comparison, trials,
  attribution and restoration. Unit-test with fake RPCs; no live swaps here.
- **acceptance_criteria**:
    - Require expected generation, complete T8 gate hashes and four matching
    acknowledgments; preserve unrelated normalization resources such as
    `sinkhorn`, checkpoint storage, startup settings and decode v984.
    - Check idle before and after acquiring maintenance. If a user request is
    active, abort without switch/recapture/cache clearing. Restore only when
    this invocation changed or may have changed dispatch; reopening maintenance
    depends on ownership. Never cancel user work or reset the device to recover.
    - Record mutation intent before reset/apply RPC submission, even if no
    receipt returns. Complete receipts permitting reconciliation may trigger
    restoration; an unresolved timeout must not trigger another collective.
    Prove the prior operation completed before querying ranks or restoring.
    If bounded reconciliation cannot establish a safe transition, keep owned
    maintenance/paused state, report recovery failure and halt. Never resume
    partially acknowledged dispatch or remove another controller's maintenance.
    - Keep controller actions serial, timeouts bounded and logs unambiguous.
    A failed/partial HCCL acknowledgment cannot be treated as successful swap.
    - Clear retired graph/reference scratch only after its lifetime ends;
    preserve required shadow resources until comparison completes.
    - Define the new CLI used below; reject trial/promote commands missing their
    gate reports. All tools default to dry-run/status, not mutation.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_resident.py tests/ut/glm_perf/test_resident_harness.py tests/ut/glm_perf/test_native_launch_audit.py`
    - Fake-RPC faults include active requests, admission race, pre-acquisition
    failure, partial apply with/without receipts, apply timeout with an
    outstanding collective, stale replies and failed recovery. Verify call order
    and absence of another collective/resume after an unresolved timeout.
- **status**: Completed
- **log**: 69 controller/harness/audit CPU tests passed. Guarded primitive RPCs preserve unresolved-operation ownership; complete shadow and ABBA workflows tested. Sealed audit/progress dependencies fix older worker packages. Actual experimental v1102 load and 344 native plus 344 whole-MoE comparisons passed; one cold pair regressed (75.512 versus51.042 seconds), so the controller restored v1021 and released maintenance. That experimental run did not qualify promotion; final v1111 subsequently completed T8/T9 and was rejected by T10 serving controls.
- **files edited/created**: `tools/glm_perf/streaming_resident.py`, `tools/glm_perf/resident_candidates/streaming_experts.py`, `tests/ut/glm_perf/test_streaming_resident.py`, `artifacts/glm-perf-310p/streaming-upgrade/T7/`.

### T8: Qualify real native kernels and replay before live installation

- **depends_on**: [T2, T6, T7]
- **agent_type**: local
- **ownership**: New `tools/glm_perf/streaming_probe.py`,
  `tests/ut/glm_perf/test_streaming_probe.py`; artifacts `T8/`.
- **description**: Run serialized standalone hardware validation of the
  complete candidate using synthetic and real checkpoint banks. Reuse existing
  fused probe/reference machinery; add changed-input/replay and lifetime cases.
- **acceptance_criteria**:
    - Test W2/W3/W4 × A4/A8 with actual supported dispatch recorded; A4 bulk
    gate and down both select streaming. Check K4096 and K2048 reductions.
    - Cover expert row counts 0/1/4/5/15/16/17/30/31/32/62/63, top-k8,
    72-expert banks, balanced/skewed/all-peer routes and 640/1280 chunks.
    - Change activations, scales, weights, route IDs/counts and replay graphs
    repeatedly. Compare output bits with frozen v1021 on identical inputs;
    require finite real-model outputs and no uninitialized padding.
    - Include many repeated launches to exercise slot drain/reuse and watchdog
    detection. No dangling events, memory overrun, silent fallback or partial
    gate report can pass manifest admission.
    - Stress multiple live slots, a final partially filled slot and successive
    experts using the exact T6 instruction schedule, not only isolated helpers.
    - Record source/checkpoint/binary hashes and measured per-stage metrics.
    Hardware is leased only by the coordinator when user serving is idle;
    standalone NPU work must not overlap live inference.
    - Before allocation, prove the standalone peak while the server remains
    loaded: resident allocations, second process/runtime context, one bank at
    a time, candidate/reference scratch, outputs and verification buffers.
    Hold maintenance against new public admission and coordinate loopback
    clients too. Idle utilization does not free the roughly 40 GiB/rank already
    resident. If required coverage cannot fit, stop before allocation and record
    a blocked hardware gate; do not unload the server or shrink required cases.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf/test_streaming_probe.py`
    - New CLI on the NPU host: `python -m tools.glm_perf.streaming_probe --build-dir artifacts/glm-perf-310p/streaming-upgrade/T6/build --reference artifacts/glm-perf-310p/streaming-upgrade/T2/reference.json --checkpoint /srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008 --output artifacts/glm-perf-310p/streaming-upgrade/T8/gates.json`
    - Device tests use the SDK/runtime environment above. CPU tests alone do
    not validate instruction events or overlap.
- **status**: Completed
- **log**:
    - Implemented the bounded, supervised native probe; selected diagnostics
    passed exact bits, finite outputs, changed graph replays, poisoned scratch
    and all-peer routes under fresh memory admission and maintenance leases.
    Selections did not qualify promotion. Final v1111 passed all 438 required
    native cases, including changed replay/lifetimes, finite exact output bits,
    poisoned scratch and real checkpoint banks. Identity-bound evidence is in
    `T10/qualified-v1111/native-gates.json`; the original native report is retained
    alongside the path-normalized companion. Host producer now records absolute
    build/report paths to prevent working-directory-dependent admission.
- **files edited/created**:
    - `tools/glm_perf/streaming_probe.py`, `tests/ut/glm_perf/test_streaming_probe.py`,
    `artifacts/glm-perf-310p/streaming-upgrade/T8/`.

### T9: Qualify the actual loaded full-model input path

- **depends_on**: [T2, T6, T7, T8]
- **agent_type**: local
- **ownership**: Artifacts `T9/`; report faults to the owning code task.
- **description**: Use the guarded controller to install only after memory
  and standalone gates pass, then shadow candidate/reference on identical
  actual activations, banks and routes at every MoE layer on all four ranks.
- **acceptance_criteria**:
    - Cover all 43 banks/rank, both 1280 and 640 chunks; for the frozen
    1920-token two-chunk control require 344 finite, bitwise-equal comparisons.
    Check each rank's bank/precision/geometry coverage rather than counts alone.
    - Validate shared-expert addition, route combine, HCCL results and existing
    decode graph replay with changed inputs. Preserve prefix/KDA checkpoint
    behavior; cold and warmed-prefix requests complete and stay finite.
    - Verify actual operator namespaces, binary hashes and stage submissions;
    no test may infer dispatch only from a status version string.
    - Remove shadow instrumentation before timings. Restore the qualified
    reference on any failure; retain all failure provenance.
- **validation**:
    - New CLI on the NPU host: `python -m tools.glm_perf.streaming_resident shadow --base-url http://127.0.0.1:8001 --reference artifacts/glm-perf-310p/streaming-upgrade/T2/reference.json --gates artifacts/glm-perf-310p/streaming-upgrade/T8/gates.json --output-dir artifacts/glm-perf-310p/streaming-upgrade/T9`
    - Read-only status: `python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 status`
- **status**: Completed
- **log**:
    - User explicitly authorized an early experimental live swap before the
    full native matrix. v1102 passed 344 native and 344 whole-MoE bitwise/finite
    shadow comparisons across all 43 banks, both chunk sizes and all four ranks.
    Actual loaded binary identities were checked. The experimental cold trial
    regressed, so the controller restored v1021 and released maintenance.
    Remaining warm/decode/complete native gates were not claimed as passed for
    v1102. Final v1111 subsequently passed 344 native and 344 whole-MoE finite,
    bitwise shadow comparisons with all 43 banks/rank, both chunks and verified
    operator/binary identity. Changed c1 decode replay and warmed 1920-token
    prefix/KDA controls passed; shadow instrumentation was removed before T10.
    Evidence: `T9/shadow-v1111/shadow.json`.
- **files edited/created**:
    - `artifacts/glm-perf-310p/streaming-upgrade/T9/experimental-live.json`,
    `artifacts/glm-perf-310p/streaming-upgrade/T9/experimental_live.py`.

### T10: Measure complete cold prefill, decode and attribution

- **depends_on**: [T2, T7, T9]
- **agent_type**: local
- **ownership**: Artifacts `T10/`; no kernel/controller edits during trials.
- **description**: Run the frozen workload protocol with reference/candidate
  alternation, complete requests and fixed settings. Run attribution separately
  after timings; stop profiling and restore ordinary dispatch before promotion.
- **acceptance_criteria**:
    - At least three ABBA cycles for primary cold 6400-token work, yielding
    six measurements per variant; record all samples, medians and variation.
    Clear prefixes between cold requests and confirm reused tokens are zero.
    - Test cold 640 and longer prompts at multiple existing-context depths,
    warmed prefixes, c1/c4 decode, natural-language/tool requests and mixed
    prefill/decode. Record TTFT, completion latency, generation tokens/s,
    token/chunk counts, rank skew/wait, peak memory and actual launch audits.
    - Seek at least 10% lower primary cold TTFT; treat this as an objective,
    not an expected result or mandatory threshold. A smaller gain may qualify
    only under the explicit T2 repeatability rule, with no correctness faults
    and no repeatable >5% decode/mixed-service regression. Record inconclusive
    trials as inconclusive; distinguish first generated token from visible text.
    - Report vector/scalar/Cube/MTE counts for representative stages and
    rank-wide attribution. Do not add overlapping pipe counters, equate all
    HCCL wait with expert imbalance or claim local speedups as E2E savings.
    - Keep v1021 if the integrated candidate lacks a defensible improvement;
    allow at most one profiler-guided revision through owning tasks before
    rerunning their dependent gates. Do not accumulate speculative live swaps.
- **validation**:
    - New CLI on the NPU host: `python -m tools.glm_perf.streaming_resident trials --base-url http://127.0.0.1:8001 --reference artifacts/glm-perf-310p/streaming-upgrade/T2/reference.json --gates artifacts/glm-perf-310p/streaming-upgrade/T8/gates.json --shadow-report artifacts/glm-perf-310p/streaming-upgrade/T9/shadow.json --workloads artifacts/glm-perf-310p/streaming-upgrade/T2/workloads.json --output-dir artifacts/glm-perf-310p/streaming-upgrade/T10`
    - Compare raw serving samples with prompt-progress and completion logs;
    profile capture scope and missing-memory-record limitations are explicit.
- **status**: Completed
- **log**:
    - One experimental cold 6400-token pair measured v1102 at 75.512 seconds
    versus v1021 at 51.042 seconds, with no prefix reuse. This is not ABBA
    promotion evidence. Standalone same-input native profiles identified a
    large scalar regression while vector/Cube work changed little. Direct
    casts reduced extraction overhead; full issue-loop unrolling and cached
    descriptor aggregates regressed and were removed. Further owning-task
    revisions removed returned descriptor aggregates and replaced extraction
    copies with direct strided casts. Final v1111 passed complete native/shadow
    gates and completed the frozen serving protocol. Three ABBA cycles
    completed: reference median 50.9959 seconds, candidate 49.6796 seconds
    (2.5811% lower). Complete serving trials rejected promotion: c4 median
    per-request generation rate regressed 6.8044%, above the 5% limit; reference
    language JSON was truncated at 192 tokens. c1/mixed did not regress and
    candidate language/tool passed. v1021 was restored. Separate full cold 6400
    four-rank PipeUtilization trace was collected after timings; raw exports
    and rank attribution are retained. Scalar remains 8.8–9.4s gate/up and
    4.4–4.7s down per rank; counters overlap and are not recoverable latency.
    CANN reports 74 incomplete memory records per rank; memory admission uses
    the independent allocation proof. See `T10/SCALAR.md` and companions.
- **files edited/created**:
    - `tools/glm_perf/streaming_profile.py`, `tests/ut/glm_perf/test_streaming_profile.py`,
    `artifacts/glm-perf-310p/streaming-upgrade/T10/`.

### T11: Final integration, review and measured promotion

- **depends_on**: [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]
- **agent_type**: local
- **ownership**: This plan's status/log fields; root reports
  `artifacts/glm-perf-310p/streaming-upgrade/README.md`, `REPORT.zh.md` and
  final diagram. Other code edits return to the original task owner.
- **description**: Review integrated code and evidence, verify all contracts
  and decide promotion using T10 criteria. Deliver matching US English and
  Chinese summaries, including failed candidates and the final runtime state.
- **acceptance_criteria**:
    - All tasks' files and dependencies reconcile; no unrelated work is reverted.
    Validate whole pipeline, memory proof, manifests, current generation,
    real weights, changed graph replay and full request behavior.
    - If promoted, all four ranks use the qualified candidate; decode v984,
    loaded weights and startup settings remain intact, profiler/shadow hooks
    are removed, graphs are complete, service health is 200, maintenance is
    closed and the service is resumed. If rejected, the same checks hold for
    v1021 restoration. Never report success with only a prepared module.
    - Reports distinguish CPU tests, standalone hardware gates, actual-input
    parity, live performance and untested options. Include the effective config,
    log paths, hashes, memory delta, numerical contract and measured regressions.
    - No worker commits are required. If the execution request asks for delivery
    commits, the coordinator alone creates the required signed, scoped commit.
- **validation**:
    - `python -m pytest -q tests/ut/glm_perf`
    - `git diff --check`
    - `bash format.sh ci` in an isolated check worktree; record existing
    out-of-scope failures and require scoped checks to pass.
    - Read-only four-rank status plus `/health`, `/is_paused` and maintenance
    state, then one complete real-model request while the coordinator owns the
    NPU test window. Check actual dispatch hashes, not just HTTP readiness.
- **status**: Completed
- **log**: Candidate rejected under frozen promotion rules; final v1021 source
  and generation 3f16017ebb4a49eab6469e2ad5ab1672 agree on all four ranks.
  Decode v984 and loaded checkpoint identity retained; graphs clean, maintenance
  closed, service resumed, health 200 and a complete 8-token real-model smoke
  passed. Final CPU suite 3951 passed/48 skipped in 59.36 s. Scoped hooks and
  git diff --check passed. Repository-wide format was run in an isolated
  worktree with out-of-scope failures retained; immutable legacy reference
  capsule was not rewritten. Host profile paths now resolve absolutely, with
  a different-worker-CWD regression test; native source/binaries remain frozen.
  Matching US English/Chinese reports distinguish achieved parity, measured
  modest prefill gains, rejection, remaining scalar cost and offline proposals.
- **files edited/created**: `glm-streaming-plan.md`, `glm-streaming-plan.zh.md`,
  `artifacts/glm-perf-310p/streaming-upgrade/README.md`, `REPORT.zh.md`, `T11/`.

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
| --- | --- | --- |
| 1 | T1, T2 | Immediately; offline/read-only |
| 2 | T3, T4, T5 | T1 contract frozen; three workers, coordinator integrates decisions |
| 3 | T6, T7 | T6 needs T1–T5; T7 needs T1/T2 and may start earlier |
| 4 | T8 | T2/T6/T7 complete and an idle NPU window is available |
| 5 | T9 | T8 hardware gates pass; one resident controller owns the server |
| 6 | T10 | T9 actual-input gate passes and shadow hooks are removed |
| 7 | T11 | All preceding evidence complete |

Use at most three workers plus the coordinator. Hardware phases are serial,
even when CPU tasks run in parallel. Same-wave write ownership is disjoint.

## Integration Strategy

- Workers edit assigned files directly in their workspace and report changed
  paths, assumptions, validation and blockers. They are not alone in the repo.
- Only T6 edits shared Projection/build/dispatch code; T3–T5 supply helpers
  against the frozen T1 interface. Only the coordinator changes live dispatch.
- Unavailable SDK details block the relevant contract branch. Failed capacity
  proof blocks producer/consumer implementation for that topology. Ask for
  additional information only if code, installed headers and existing artifacts
  cannot resolve it; do not assume missing hardware support.
- If hardware is busy/unavailable, finish offline tasks and leave unrun hardware
  gates pending; preserve the completed historical T8–T11 evidence.
  No elapsed-time assumption substitutes for an idle lease or a passed gate.
- A correctness failure returns to its owning task and invalidates all dependent
  gate reports. New source or binary hashes require new T8/T9/T10 evidence.

## Testing Strategy

- CPU: numerical/order reference, compiled geometry, address ownership, event
  state model, manifests, dispatch and faulted controller RPC sequences.
- NPU: synthetic plus real mixed banks, all stages, multi-batch expert reuse,
  changed graph inputs, sparse/all-peer routes, long repeated launches and
  finite-output checks. Verify exact matching binary provenance.
- Model: identical actual activations across all banks/ranks, shared/rank
  combination, KDA/prefix behavior and complete OpenAI-compatible requests.
- Performance: frozen ABBA requests with actual launch evidence; separate
  attribution run; fixed config, prefix policy, prompt hashes and decode kernels.
- Neither CPU event models nor one isolated INT4 matmul proves real overlap or
  an end-to-end gain.

## Risks and Mitigations

- **Risk**: More simultaneous scratch exceeds UB/L1/L0 or overwrites live data.
  **Mitigation**: T1 proves phase-specific lifetimes and bounds before work is
  integrated; SDK reserve and aligned storage are mandatory.
- **Risk**: Event removal causes CO1 overwrite, deadlock or replay races.
  **Mitigation**: Named ownership/event table, startup/drain tests and real
  changed-input stress; reject cross-target event assumptions.
- **Risk**: Batching changes FP32 sums or hides a quantizer change.
  **Mitigation**: Preserve ordered group factors and rounding; exact real-input
  comparisons. Keep approximate math outside this plan.
- **Risk**: Faster Cube schedule increases vector or transfer work.
  **Mitigation**: Measure complete stages and E2E requests; no promotion from
  theoretical instruction counts or isolated pipe utilization.
- **Risk**: Removing the cross-core handoff duplicates work or needs global
  synchronization the launch cannot provide.
  **Mitigation**: Keep the compact ABI; a different core-ownership design needs
  its own proof and plan.
- **Risk**: Profiling or cleanup interrupts user requests.
  **Mitigation**: One guarded controller, maintenance ownership checks and
  restoration only after actual mutation; test failures before acquisition.
- **Risk**: Concurrent Qwen work collides with global changes or NPU testing.
  **Mitigation**: Narrow GLM ownership, isolated formatting, unique build
  directories and a coordinator-held hardware window.

## Reviewer Notes

- Local pre-review: reviewed Projection/Product/Accumulate/GateUp, shared
  scratch, permanent route layouts, manifest gates and ResidentClient. Tasks
  name newly created CLI modules explicitly; commands using them run only after
  their producer task completes. Existing probe defaults are insufficient for
  640/1280-token, 72-expert and changed lifetime tests, so T8 extends coverage.
- Independent plan-only review by `review_glm_streaming_plan` found the wave
  ownership sound for coordinator + three workers. Its six actionable findings
  were incorporated: concrete Cube-issue/interface ownership; frozen evidence
  schemas; standalone construction versus live admission; loaded-server memory
  feasibility; partial-mutation timeout recovery; combined event schedule and
  operational warmup/timing/repeatability rules. The Chinese companion reflects
  the same revisions. The reviewer made no implementation or server changes.
- Final independent review passed: no remaining execution-blocking dependency,
  ownership, admission or recovery gaps. Execution subsequently completed
  T1–T11, including hardware and serving gates. The measured c4 regression
  rejected v1111 promotion; v1021 restoration was verified.

## US English and Chinese handoff

T1–T11 completed memory/ABI proof, independent helpers, shared-kernel
integration and serialized hardware/full-model/serving validation. v1111 was
rejected after a 6.8% c4 generation regression and v1021 was restored. Continued
T12/T13 work has bounded diagnostic/offline evidence; T14 must repeat its native,
actual-input and serving gates before any promotion.

[Chinese companion](glm-streaming-plan.zh.md).

## Continued scalar reduction experiments

User authorized continued work after the v1111 rejection. The restored v1021
remains the serving reference. All native/serving work remains serialized under
the coordinator; independent helper development may proceed offline. Never
reuse a candidate namespace or interpret a partial diagnostic as promotion.

### T12: Amortize cast mask setup without changing arithmetic

- **depends_on**: [T11]
- **agent_type**: local
- **ownership**: `glm_streaming_products.h`, its CPU stubs/checker/tests,
  immutable builds and diagnostics under `T12/`.
- **description**: Set the 16-lane normal mask once per independent K32 group,
  then use public `Cast<float,int32_t,false>` for its eight strided casts.
  Preserve all buffers, offsets, rounding, barriers and subsequent mask users.
- **acceptance_criteria**: CPU exact-bit/canary/lifecycle tests cover every row
  and both layouts; mask instrumentation proves two setups per pair instead
  of sixteen. Installed dav-2002 builds and same-input native diagnostics
  compare against v1111 and v1021. No live installation from partial evidence.
- **validation**: Actual helper CPU suite; signed-zero/cancellation checks;
  immutable compile; five selected native gates and separate pipe profiling.
- **status**: Completed
- **log**: Cast mask setup was reduced from sixteen to two per pair with exact CPU checks and installed dav-2002 compilation. Five selected native cases passed; this is not the full qualification matrix. On one W4/A4/31-row diagnostic, unprofiled gate/up was 50.432 versus 52.298 ms and down 22.612 versus 23.808 ms against v1021. Scalar attribution was 31.465/15.495 ms versus baseline 16.878/7.437 ms, essentially unchanged from v1111. No live qualification or swap. See `T12/README.md` and `mask-profile-v1120.json`.
- **files edited/created**: `tools/glm_perf/glm_streaming_products.h`, product CPU checker/stubs/tests, immutable v1120 diagnostic build and `artifacts/glm-perf-310p/streaming-upgrade/T12/`.

### T13: Prepare bounded Gather scale replication offline

- **depends_on**: [T11]
- **agent_type**: worker
- **ownership**: New `glm_streaming_scale_gather.h`, helper-specific CPU
  stubs/tests and `T13/` reports. Do not edit T12 products or shared kernel.
- **description**: Implement a public Gather/Brcb helper using expanded indices
  prepared once per batch. Prove retained offsets after the full route cache.
  The coordinator later supplies the caller/initialization integration.
- **acceptance_criteria**: Exact scale bits for all rows, both layouts and
  arbitrary FP32 representations; no route-cache or factor alias; no scalar
  per-K-pair index construction. SDK legality is distinguished from CPU proof.
- **validation**: Actual helper CPU instruction/bit/interval tests and
  sanitizer probes; SDK source verification; English/Chinese reports.
- **status**: Completed
- **log**: Offline public Brcb/Gather helper passed 94 CPU cases and ASan/UBSan, preserving arbitrary FP32 bits and protecting the complete retained routing cache. SDK source support was reviewed; T13 did not compile against the SDK, execute on an NPU or integrate live. See `T13/README.md`, `REPORT.zh.md` and `validation.json`.
- **files edited/created**: `tools/glm_perf/glm_streaming_scale_gather.h`, dedicated scale-Gather CPU checker/stubs/tests and `artifacts/glm-perf-310p/streaming-upgrade/T13/`.

### T14: Integrate and qualify the measured winner

- **depends_on**: [T12, T13]
- **agent_type**: local
- **ownership**: Shared kernel/caller and source provenance; serialized NPU
  tests and `T14/` reports.
- **description**: Test broadcast replacement independently before combining
  it with a winning mask change. Preserve the current K32 checkpoint contract.
- **acceptance_criteria**: Complete memory/alias proof; native matrix and fresh
  actual-input parity before serving comparisons. Promote only under the
  existing repeatability/decode/quality rules; keep v1021 on rejection.
- **validation**: Full native gates, actual-input shadow, paired serving
  measurements and healthy four-rank final state, with matching summaries.
- **status**: Completed
- **log**: Version2 proof/4506 CPU tests and186 NPU helper probes passed.
  Full438 native cases and344 native+344 whole-MoE shadows passed. Three-cycle
  cold6400 median51.114→47.864s (6.36% lower); mixed decode4.18% higher. Complete
  trials rejected promotion because c4 fell8.98%; c1 fell1.18%. Qualified
  v1021/decode984 restored at3fededcf0064437895628520135c0a8a. Scalar objective
  remains unmet. Original language-budget failure preserved. See T14 reports.
- **files edited/created**: `tools/glm_perf/glm_streaming_contract.h`, `tools/glm_perf/streaming_memory.py`, `tests/ut/glm_perf/test_streaming_memory.py`; coordinator owns shared kernel/caller integration and `T14/` evidence.

### T15: Probe scalar overhead from SDK tensor descriptors offline

- **depends_on**: [T13]
- **agent_type**: worker
- **ownership**: New `glm_streaming_products_pointer.h`, dedicated CPU stubs,
  checker/tests and `T15/` bilingual reports. Keep shared kernel unchanged.
- **description**: Prepare a pinned CANN9.1/dav-2002 experiment using compiler
  vector builtins and UB base pointers obtained once through public tensor APIs.
  Remove per-strip tensor slicing in casts and arithmetic. Preserve both slots,
  group order, exact independent FP32 operations and masks. This is a compiler
  experiment, not a portable/public-API implementation; include no private SDK
  headers and document SDK hashes and required target.
- **acceptance_criteria**: Actual helper CPU tests cover every count/layout,
  signed zero, cancellation, mask effects and retained regions. No native or
  serving claims from CPU tests. Root owns later SDK/NPU qualification.
- **validation**: CPU instruction/bit/canary/lifecycle tests, sanitizers and
  read-only installed SDK verification; immutable hardware probe remains root.
- **status**: Completed
- **log**: Offline pointer consumer passed187 CPU cases, 30 ASan/UBSan boundary probes and seven negative faults. It queries seven UB bases once, uses no descriptor slices, preserves16 casts/6 multiplies/2 ordered adds, 10 vector fences and full-mask exit. SDK/compiler call-site hashes are pinned in bilingual T15 reports. No SDK compile, native validation or performance claim; root defers hardware work until current E2E measurements finish.
- **files edited/created**:

### T16: Audit decode routing metadata for the next measured candidate

- **depends_on**: [T11]
- **agent_type**: worker
- **ownership**: `T16/` English/Chinese reports only; no shared source changes.
- **description**: Identify the actual v984 decode bundle and active call chain.
  Audit repeated expert-end GM scalar reads inside output-tile scans, retained
  scratch lifetimes and whether once-per-launch DMA caching is safe for M16.
  Include sparse/A8/W2/W3/graph-replay hazards and a concrete profiling gate.
- **acceptance_criteria**: Source-backed wiring/offset proof and bounded proposed
  experiment; distinguish suspected scalar cost from measured attribution.
- **validation**: Frozen source/provenance review and existing CPU tests only.
  Root retains all NPU access; no compile, runtime mutation or speedup claim.
- **status**: Completed
- **log**: Verified actual v984 M16/wideK128/FP16-scale frozen source and binary wiring. Source derives3456 GM end reads per H4096/I2048/E72 projection pair. Prepared/no-LUT mask writers permit a full M16 end cache at24640..26944 without UB growth. Existing bulk/vector-scale cache flag cannot be reused unchanged. Eight existing admission/CLI tests passed; compiler case excluded because this is a read-only audit. No speedup or hardware claim. See T16 bilingual reports and provenance.
- **files edited/created**:

### T17: Implement the decode boundary-cache helper offline

- **depends_on**: [T16]
- **agent_type**: worker
- **ownership**: New `glm_decode_end_cache.h`, dedicated CPU stubs/checker/tests
  and `T17/` bilingual reports only. Keep shared kernel/builder unchanged.
- **description**: Public-SDK M16 helper loads aligned INT64 expert ends once
  per launch, with at most three safe scalar tail reads, and serves subsequent
  end lookups from retained UB. Preserve metadata bits and all mask neighbors.
- **acceptance_criteria**: All1..288 expert lengths, replay changes, tails and
  monotonic/zero/all-peer boundaries; no out-of-bounds GM DMA or new allocation.
  Retain UB scalar reads rather than claim all scalar work is eliminated.
- **validation**: Actual helper CPU semantics/canaries and sanitizers. SDK
  compilation/native parity/decode profiling belong to root afterward.
- **status**: Completed
- **log**: Offline public-SDK cache helper passed872 CPU tests (all1..288 lengths/replays, seven faults and sanitizers). No allocation; all live INT64 boundaries refresh every launch. Source/builder remains unchanged. SDK compile, native parity/profiling and serving tests remain queued after T14. See T17 bilingual reports.
- **files edited/created**:

### T18: Repair a response-budget-limited quality control

- **depends_on**: [T14]
- **agent_type**: worker
- **ownership**: New `streaming_quality_repair.py`, dedicated CPU tests and
  `T18/` bilingual reports; preserve parent trial artifacts and shared controller.
- **description**: The original reference language control exhausts192 tokens
  with no visible answer. Prepare a prospective768-token check for identical
  language/tool prompts on both variants. Require all parent nonquality gates;
  record unchanged prompt IDs, new response cap and supplemental outcomes.
- **acceptance_criteria**: Never overwrite/relabel failed parent results. Never
  bypass native/shadow/performance/decode gates. Retain candidate only if both
  supplemental quality controls pass and all parent nonquality gates passed.
  Reject a quality failure not explained by the recorded original response cap.
- **validation**: CPU rejection/restoration/prompt-identity tests; root runs
  supplemental real serving requests after original trials release their lease.
- **status**: Completed
- **log**: Offline controller passed36 dedicated and159 related tests. Root
  real-parent dry-run correctly refused v1131 because c4 regression failed;
  no supplemental requests or promotion. Prospective768 design and original
  capped failure remain preserved; later qualifying candidates can use it.
- **files edited/created**:

### T19: Integrate the pointer consumer and qualify scalar reduction

- **depends_on**: [T14, T15]
- **agent_type**: worker integration; coordinator hardware
- **ownership**: Shared streaming consumer call/include, borrowed-descriptor
  dispatch tests and build provenance tests; `T19/` bilingual reports.
- **description**: Substitute the pinned pointer consumer for the v1131
  consumer while preserving prepared Gather, masks, independent group math,
  scratch layout and events. Prepare offline during T14 measurements. Defer
  SDK compilation and all NPU work until those measurements finish.
- **acceptance_criteria**: Frozen build hashes include the actual pointer
  header; caller tests exercise that consumer's real signature. Compile and
  native parity precede timing. Keep measured scalar and E2E results distinct.
- **validation**: Caller/helper/provenance CPU tests, pinned SDK build, full
  native matrix and fresh model shadow before any serving trial.
- **status**: Completed
- **log**: Offline integration passed1070 CPU checks/48 existing skips, plus
  actual-caller/signature sanitizers and lint. Only the include/consumer target
  changed; prepared Gather, aliases, events and ABI are preserved. SDK build,
  full438 native cases and both344-comparison actual-input shadows passed.
  Scalar6.587/2.758ms versus reference16.880/7.436. Nine primary samples show
  partial cold6400 medians51.144→41.594s. User requested coding access; stopped
  benchmark with guarded cleanup, installed1140 on four ranks, healthy/unpaused,
  maintenance/profiler/shadowoff. Full performance qualification remains
  incomplete; no benchmark promotion claimed. Hardware/user handoff complete.
- **files edited/created**:

### T20: Audit RTC optimization and scalar instruction lowering

- **depends_on**: [T11]
- **agent_type**: explorer
- **ownership**: `T20/` bilingual source-backed reports only.
- **description**: Verify the actual RTC device optimization/debug defaults
  and inspect existing immutable binaries/SDK documentation read-only. Separate
  host compiler flags from device flags; identify any credible missed compiler
  setting before attributing all scalar time to source-level descriptors.
- **acceptance_criteria**: Cite installed SDK evidence and frozen compile
  options. Do not infer measured speedups from flags, binary size or CPU tests.
- **validation**: Read-only source/binary/documentation audit. No compiler
  invocation, NPU profiling, server mutation or shared source edits.
- **status**: Completed
- **log**: Verified host-O2/device-argv distinction and frozen ELF identities.
  RTC optimization defaults remain unresolved; SDK's separate release path uses
  O3. Installed disassembler decodes no device instructions. No compile, NPU,
  runtime changes or performance claim. Exact primary-source hashes in T20.
- **files edited/created**: `T20/README.md`, `T20/REPORT.zh.md`, `T20/evidence.json`.

### T21: Explain the concurrent-decode control regression

- **depends_on**: [T14]
- **agent_type**: explorer
- **ownership**: `T21/` bilingual reports only.
- **description**: Analyze complete v1131 trial samples, actual static dispatch
  boundary and available scheduler/speculation logs. Determine which causes
  are supported for slower c4 despite unchanged v984 decode binaries.
- **acceptance_criteria**: Use raw per-request samples and source identity;
  distinguish timing, admission grouping and thermal/scheduling hypotheses from
  measured kernel attribution. Never waive the existing regression threshold.
- **validation**: Read-only artifact/source/log analysis; no server requests,
  native commands, source edits or new benchmarking during parent trials.
- **status**: Completed
- **log**: Exact complete c4 median5.158261→4.694975t/s (8.98145% lower).
  First two candidate groups also had slower engine iterations and lower
  speculative acceptance; third recovered. Pure8-row decode still selects984.
  Same1+3 admission interruption occurs in both variants. Cause unisolated;
  no threshold waiver or new measurements. T21 preserves matched raw evidence.
- **files edited/created**: `artifacts/glm-perf-310p/streaming-upgrade/T21/`.

### T22: Integrate decode boundary caching in an isolated offline candidate

- **depends_on**: [T16, T17]
- **agent_type**: worker
- **ownership**: New isolated decode-candidate builder/source transformation,
  dedicated CPU tests and `T22/` bilingual reports; preserve shared sources.
- **description**: Clone the actual immutable v984 decode source/flags and
  integrate the existing M16 boundary-cache helper. Load expert ends once per
  launch, serve tile scans from UB and refresh every replay. Preserve math,
  dispatch, ABI, all sparse/A8 paths and fixed scratch allocation.
- **acceptance_criteria**: Exact bounded source substitutions and baseline
  identity, no vector-scale feature enabled merely to obtain caching. Freeze
  all inputs/argv for a later unique build; default stays offline/dry-run.
- **validation**: Actual-source transformation/alias/replay regression tests,
  existing helper CPU tests, hash-bound artifact provenance and EN/ZH reports.
  Coordinator alone performs later SDK/native/decode profiling.
- **status**: Completed
- **log**: Isolated v984 source/argv freezer and actual transformed Run passed
  2296 CPU tests; combined cache helper3168 passed13.12s including sanitizers.
  Frozen v1150 inputs ready; no SDK/NPU/live build or version reservation.
  User owns server for coding; no native or speed claim for1150.
- **files edited/created**: `tools/glm_perf/decode_cache_candidate.py`, dedicated
  tests/stubs and `artifacts/glm-perf-310p/streaming-upgrade/T22/`.
