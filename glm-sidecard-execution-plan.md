# Plan: GLM Sidecard Execution and Prefill Optimization

**Generated**: 2026-10-13
**Mode**: Swarm-ready dependency plan

## Overview

Implement the [GLM sidecard execution and prefill PRD](docs/source/developer_guide/Design_Documents/glm_sidecard_execution_performance_prd.md)
through isolated kernel work, serial runtime integration, and exclusive hardware
qualification. Preserve the best qualified CUDA-backbone/Ascend-expert service
while testing the complete Ascend block. First attribute its approximately
83.8 ms execution/packing bucket; the observed input and return DMA medians total
approximately 0.61 ms and are not the primary optimization target.

The proposed release gates are 10% faster median API decode or 20% lower cold
time to first token on each of 2K, 8K, and the unchanged full Code fixture,
with at most 3% regression in the other metric. The absolute objective is
15 API decode tok/s. All releases retain 131,072 total tokens and all tools.
These are targets, not completed results.

Execution began on October 13, 2026 through the parallel-task skill. Task statuses
record validated implementation or research decisions; creating a task or its
command does not qualify an optimization.

## Assumptions

- Allocate one RTX PRO 6000 and one Ascend Duo, physical chips 2 and 3. Keep
  chips 0 and 1 available for unrelated work. Requests remain serialized.
  User override for the specialized-library follow-up: both Duos, chips 0–3,
  are temporarily available for isolated pair probes. The local owner records
  service stops and recovery; the GLM release allocation above is unchanged.
- Use the recovered resident-weight checkpoint and its authenticated native
  bundle. Preserve packed W2/W3/W4 formats, activation quantization, clipping,
  routing, and deferred mHC semantics. No prompt compaction or tool removal.
- The deployed cold-expert architecture is the release baseline. A complete
  block must earn promotion; its current 2K/8K tests do not qualify 128K capacity.
- Parent plus at most three worker agents can run concurrently. Parallelism
  applies to disjoint offline edits; one local owner runs all accelerator tests,
  service transitions, native operator registration, and shared-file integration.
- Workers are not alone in the codebase. Edit owned files directly, report paths
  changed, and never revert, overwrite, stage, or format unrelated research.
  In particular, `tools/glm_perf/` has unrelated dirty native research. Read
  authenticated frozen sources instead of changing those live files.
- Candidate workers supply standalone modules/probes and integration notes.
  They do not edit shared runtime entry points or deploy their candidates.
- Profile findings may reject a candidate before implementation. Record the
  reason and timing bound in its decision receipt; downstream integration uses
  the qualified fallback. A rejected experiment can be completed as research,
  but cannot be counted as an achieved release gate.
- No worker commits are required. The local owner handles delivery commits
  with sign-off if committing under the existing session authorization.

## Prerequisites

- Read [AGENTS.md](AGENTS.md), the PRD,
  [complete-block receipts](artifacts/sidecard-layer-dma-20261013/README.md),
  [layer runbook](tools/sidecard/LAYER_RUNTIME.md), and
  [direct DMA contract](tools/sidecard/pcie/DIRECT_DMA.md).
- T1 must discover current process identities and versions; do not reuse a
  remembered PID. The last restored service was at
  `http://192.168.53.187:8000/v1`, model `glm-sidecard`, with 128K admission.
- On the hardware host, require working CUDA/NPU Python environments, CANN
  compiler and private operator build support, the qualified UAPI 5 bridge,
  scoped passwordless worker launch, and the unchanged checkpoint. T1 freezes
  installed versions. Use their source and version-matched primary documentation
  to resolve any new graph, stream, or operator API assumption.
- T1 creates a new run directory outside the source tree, private fixture
  manifest, public artifact directory, and independently frozen baseline,
  socket-block, and DMA-block specifications. No previous run is overwritten.
- For future hardware commands, T1 supplies shell aliases
  `SIDECARD_RUN_ROOT`, `SIDECARD_PUBLIC_ROOT`, `SIDECARD_PRIVATE_FIXTURES`,
  `SIDECARD_CUDA_PYTHON`, `SIDECARD_NPU_PYTHON`, and `SIDECARD_BASE_URL`.
  These are command parameters, not new runtime configuration variables.
  `SIDECARD_BASE_URL` ends at the host/port; tools add `/v1` themselves.
- The local owner sets `SIDECARD_PHASE_ROOT` to a fresh directory for baseline,
  each candidate, or final qualification, and `SIDECARD_PAIR_ID` per comparison.
  Phase-local `api-pairs.json` lists the original baseline/candidate receipts,
  order, fixture identity, and run manifests. Never overwrite prior phase data.
- Commands run from the repository root unless marked as running from the
  frozen root on the hardware host. Device modules use the pinned interpreter
  and frozen source, preserving its existing CANN environment. Host CPU Python
  runs portable tests with `--confcutdir=tests/ut/sidecard`.
- Commands mentioning new modules or tests below are interfaces that their
  owning task must implement. Their files are not prerequisites today.

## Dependency Graph

```text
T1, T2, T3                 -> T4
T1, T3                     -> T8, T9, T19
T1                         -> T5, T6, T7, T10
T4, T5, T6, T7, T8, T10, T19 -> T11
T4                         -> T20
T10, T11, T20              -> T12
T12                        -> T13, T14, T15
T9, T13, T14, T15           -> T16
T10, T16                   -> T17 -> T18
```

Groups on the left are required dependencies; groups on the right are independent
tasks. The `depends_on` fields are authoritative. Hardware work is serialized
independently of this graph. T19 is the diagnostic-gate task added during review.

## Tasks

### T1: Freeze hardware allocation and reproducible inputs

- **depends_on**: []
- **agent_type**: local
- **ownership**: New public `artifacts/sidecard-execution-<run-id>/preflight/` receipts and private external run/fixture directories; runtime source is read-only.
- **description**: Inspect the live service without interrupting it. Record process/parent identities, card mapping, bridge owners, source/config/binary/checkpoint hashes, toolkit versions, context, and per-device memory. Freeze rollback inputs and fresh benchmark specifications with 128K reservations. Authenticate the frozen native sources used by expert experiments. Preserve private Code content and all 23 tools; publish identities rather than private request text.
- **acceptance_criteria**:
    - A manifest resolves all command aliases, runtime paths, fixture paths, expected limits, and hardware identities. It includes known-good restoration commands and normal bridge-owner counts.
    - Explicitly distinguish inspect-only preflight from later exclusive accelerator ownership. Unknown owners prevent destructive transitions, not offline work.
- **validation**:
    - `git diff --name-only` and `git diff --cached --name-only` establish the protected research inventory.
    - On the hardware host, inspect `/health`, `/v1/models`, process command lines, `npu-smi info`, and `nvidia-smi`; verify only assigned-chip allocation is planned.
    - Manually authenticate copied manifests and native bundle provenance; confirm each new output directory is distinct from the source and previous runs.
- **status**: Completed
- **log**: Inspect-only hardware preflight passed. Healthy expert-only service PID 3244009 has 131072 admission, assigned chips 2/3, and six bridge references; other-card workers remain untouched. Checkpoint/native/CUDA/completion digests and six startup parity receipts authenticated. Original 23820-token/23-tool request verified. Exact rollback inputs and private fixture/specification directories frozen under `/srv/ai/storage/artifacts/sidecard-execution-20261013-215113c`; supplemental four GLM dependency hashes recorded because the original manifest omitted them. No model allocation, service stop, or global hardware change.
- **files edited/created**: `artifacts/sidecard-execution-20261013-215113c/preflight/manifest.json`, `fixture-identities.json`, `aliases.json`, `protected-research-sha256.json`; external private fixture/specification/rollback directories.

### T2: Build a paired API performance harness

- **depends_on**: []
- **agent_type**: worker
- **ownership**: New `tools/sidecard/performance_suite.py`, `tests/ut/sidecard/test_performance_suite.py`; existing benchmark modules are read-only.
- **description**: Add a reusable client for 2K, 8K, and unchanged full Code requests. Reuse existing request/tokenization utilities without modifying them. Implement baseline/reference receipts, five or more paired repetitions, order tracking, confidence intervals, and separate cold TTFT, first-visible-text, API decode, and total latency metrics. Model-forward timing remains a separate field/source.
- **acceptance_criteria**:
    - Collection CLI accepts `--base-url`, `--fixtures`, `--output`, `--repeats`, `--output-tokens`, `--pair-id`, and `--mode baseline|candidate`. A separate `--compare-pairs` input plus `--output` aggregates an authenticated index of actual interleaved pair receipts; it does not require a running server.
    - Count reasoning plus content through output usage; never equate SSE chunks with tokens. Flag coalesced emission, early EOS, missing usage, prefix reuse, incompatible parser/configuration, and failed requests.
    - Statistical comparisons preserve per-fixture results, pairing/order, and uncertainty; no aggregate hides a regression. Synthetic stream fixtures test these failure modes.
    - Cold mode verifies zero cached tokens and records cache policy, startup/gate completion, warmup, service identity, and pair order. The client does not invent a cache-flush endpoint or restart a server itself.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_performance_suite.py`
    - `python -m tools.sidecard.performance_suite --help`
- **status**: Completed
- **log**: Implemented authenticated collection and actual alternating launch comparison with per-fixture bootstrap intervals. Parent verified 46 portable tests passed; worker CLI help and scoped hooks passed. Buffered/special-token emissions are informational and explicitly labeled usage-accounted emission proxies; unmeasurable intervals fail. Added required `--pair-order` and authenticated `buffering_policy` pairing contract. Hardware measurements remain T4.
- **files edited/created**: `tools/sidecard/performance_suite.py`, `tests/ut/sidecard/test_performance_suite.py`.

### T3: Add a graph-compatible execution timeline

- **depends_on**: []
- **agent_type**: worker
- **ownership**: New `tools/sidecard/execution_trace.py`, `tools/sidecard/trace_summary.py`, `tests/ut/sidecard/test_execution_trace.py`; instrumentation edits to `tools/sidecard/layer_stage_profile.py`, `tools/sidecard/stage_dma.py`, `tools/sidecard/layer_actor.py`, `tools/sidecard/layer_math.py`, and `tests/ut/sidecard/test_layer_stage_profile.py`.
- **description**: Add opt-in bounded tracing and integration hooks for both ranks and CUDA. The existing synchronized `StageProfile` remains diagnostic; it cannot stand in for the production graph path. Measure graph-span/device execution, host launch gaps, result packing, final wait, and DMA separately. Join logical identities and calibrated clock domains; attribute collective arrival wait and transit only where supported by the installed profiler.
- **acceptance_criteria**:
    - No per-operator device readback or added global synchronization in the ordinary path. Record trace mode and overhead; leave unobservable graph-internal detail explicitly unattributed.
    - Report a critical path with overlap and matching request/chunk/layer/sequence keys. Do not sum unrelated medians or pair expert-only independent rank counters.
    - Trace output schema and controls are available to T4, T11, and T16. Use runtime JSON options; any truly necessary new environment variable follows centralized repository policy.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_execution_trace.py tests/ut/sidecard/test_layer_stage_profile.py tests/ut/sidecard/test_stage_dma.py tests/ut/sidecard/test_layer_math.py`
    - `python -m tools.sidecard.trace_summary --help`
    - T4 must compare traced/untraced wall time and vocabulary before interpreting attribution.
- **status**: Completed
- **log**: Opt-in bounded event/host traces cover the contiguous CUDA/Ascend block, packing, completion waiting, DMA, and socket prefill overlap. Initial parent 95 targeted and worker 1426 aggregate tests passed. T4 found 22 invalid NPU event durations; collector/summary now mark negative/nonfinite/API-failed intervals unavailable without interrupting cleanup, preserving raw evidence and host times. Parent verified the correction in a 224-test combined trace/lease/transport/math run; worker scoped hooks passed. NPU event timing is explicitly unreliable; graph collectives/expert-only runtime remain unattributed. T4 uses host-monotonic critical-path budgets and same-math vocabulary parity.
- **files edited/created**: `tools/sidecard/execution_trace.py`, `trace_summary.py`, `layer_actor.py`, `stage_dma.py`, `layer_stage_profile.py`; `tests/ut/sidecard/test_execution_trace.py`, `test_layer_stage_profile.py`.

### T4: Measure the current baseline and choose candidate budgets

- **depends_on**: [T1, T2, T3]
- **agent_type**: local
- **ownership**: New baseline/timeline/budget receipts in the run/public artifact directories; runtime source is read-only.
- **description**: Acquire exclusive ownership through the recorded service lifecycle. Establish initial cold 2K/8K API observations for the qualified service and matched socket/DMA complete-block stage measurements, plus traced/untraced device parity and current-build attribution. Run startup gates first, including current admission/cleanup source. Freeze candidate budgets and restore the baseline. T20 retains full Code coverage and the full repeated launch-pair requirement and can run while kernel workers develop offline.
- **acceptance_criteria**:
    - Initial 2K/8K pilot API and matched stage receipts establish trace budgets and same-math parity. Full Code and repeated launch-pair confidence gates remain pending until T20. Architecture quality comparisons do not assume TP2 and CUDA backbone logits are bitwise identical.
    - Use `prompt_prefix_cache: false` for both isolated cold configurations, keep all other settings matched, and require zero cached tokens on every request. There is no public cache-flush endpoint. Preserve the original serving cache configuration for restoration and measure warm-prefix behavior separately.
    - A budget receipt separates measured device/host/packing/collective costs and prioritizes viable candidates. Produce `precision-contract.json` with state/rounding invariants and predeclared numerical limits before arithmetic-change performance gates; workers add shape-specific bounds before device testing, never after a failed comparison.
    - Restoration receipt confirms exact source/configuration, startup gates, expected bridge owners, 128K admission, and a completed real request.
- **validation**:
    - With the recorded baseline service ready: `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --base-url "$SIDECARD_BASE_URL" --fixtures "$SIDECARD_PRIVATE_FIXTURES/manifest.json" --output "$SIDECARD_PHASE_ROOT/$SIDECARD_PAIR_ID-baseline.json" --repeats 1 --output-tokens 256 --pair-id "$SIDECARD_PAIR_ID" --pair-order baseline-first --mode baseline`
    - After the matching candidate launch/gates/warmup, repeat with `--mode candidate` and a distinct `-candidate.json` output. Reverse actual launch order on alternating pairs; complete at least five pairs per fixture.
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --compare-pairs "$SIDECARD_PHASE_ROOT/api-pairs.json" --output "$SIDECARD_PHASE_ROOT/api-comparison.json"`
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.stage_benchmark --config "$SIDECARD_RUN_ROOT/socket/runtime.json"`
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.stage_benchmark --config "$SIDECARD_RUN_ROOT/dma/runtime.json" --reference "$SIDECARD_RUN_ROOT/socket/stage-benchmark.json"`
    - Inspect the joined trace and repeat unprofiled runs. Existing stage benchmark covers 2K/8K only; T2 supplies full Code API coverage.
    - The local owner records fixed warmup and cached-token checks for each launch. Apply this actual alternating lifecycle and fresh phase-local pairing again in T12/T17; a historic T4 baseline cannot substitute for a contemporaneous paired run.
- **status**: Completed
- **log**: Fresh cold 2K/8K expert baseline completed 256 outputs at ~12.2 internal decode tok/s with zero prefix reuse; API proxy receipt retained separately. Socket/DMA/traced complete-block runs passed 192 complete vocabulary comparisons per matching run. DMA measured 9.66–9.78 forward tok/s; observed trace slowdown <0.75% decode/<0.4% prefill, with no dropped/unresolved events. Twenty-two NPU event durations were invalid, so budgets explicitly use matched host-monotonic spans: ~85–87 ms middle, ~9.1 ms prefix/~7.5 ms suffix, ~0.50 ms rank0 DMA total; internal collective/kernel timing remains unavailable. Fixed precision contract published before native device testing. Initial restoration hit Unix socket path length, then original launcher omitted unused archived CUDA build sources; short paths and exact supplemental source copies resolved both. Original public expert service PID 3537158 restored with all five startup parity files plus decode repeat passing, 131072 admission, six bridge refs, completed real request and all 131 archived source files matching. T20 retains full Code/repeated confidence gates.
- **files edited/created**: `artifacts/sidecard-execution-20261013-215113c/profile/` receipts, `precision-contract.json`, `preflight/phase-paths.json`; authenticated external profiling and short output directories.

### T20: Complete repeated paired API baselines

- **depends_on**: [T4]
- **agent_type**: local
- **ownership**: New phase-local paired launch, performance, cache, and restoration receipts; runtime source is read-only and pinned to T4's accepted freeze.
- **description**: Complete the five-or-more actual alternating baseline/candidate launch pairs per 2K/8K/full Code fixture using T2. Run this hardware-only continuation while T5–T9/T19 workers develop isolated code. The local owner retains the sole accelerator lease; later source edits cannot enter these frozen launches.
- **acceptance_criteria**:
    - Preserve T4's cold-cache, warmup, prompt/tool, context, identity, and pair-order contracts. Record all launch windows and compare per-fixture distributions with uncertainty.
    - Publish `api-pairs.json` plus authenticated comparison receipts and exact baseline restoration. If noise is inconclusive, add pairs instead of claiming a gain.
    - T12 cannot qualify candidates before this task finishes. No implementation or release threshold is removed by splitting the long hardware window out of T4.
- **validation**:
    - Collect one baseline and candidate side per actual launch with T2's `--repeats 1 --pair-id <id> --pair-order baseline-first|candidate-first --mode baseline|candidate`, reversing order between pairs.
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --compare-pairs "$SIDECARD_PHASE_ROOT/api-pairs.json" --output "$SIDECARD_PHASE_ROOT/api-comparison.json"`
    - Verify zero reused tokens, source/launch identities, all required fixture coverage, and completed restored-service generation.
- **status**: Blocked
- **log**: First actual launch pair completed all three cold fixtures with 256 outputs and zero cache reuse: expert-only API emission proxy 12.41/12.26/12.65 tok/s and TTFT 9.90/41.08/119.82 seconds for 2K/8K/full Code; current complete-block DMA path 9.62/9.48/7.44 tok/s and 30.50/115.87/532.68 seconds. These are single-pair observations, not confidence-qualified gains. Pair 1 candidate startup failed after assigned chips 2/3 lost heartbeat at 23:16:48 UTC. The fault preceded the startup hold (23:17:18) and first private compile (23:17:46). Controller resumed normally after the 7.65-second compile attempt; both startup and exact original public-service restoration then failed at NPU preparation. Driver reports card 16409 Critical and both devices not working; other card 8217 remains healthy with its existing processes. Controller and failed coordinators exited, device files 2/3 have no users, and bridge refs are zero. No reset/driver changes performed. Recovery exception requested because PRD R6 forbids resets. All remaining launch-pair, confidence, and restoration requirements remain pending hardware recovery.
- **files edited/created**:

### T5: Implement a precision-preserving native decode recurrence

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `csrc/attention/sidecard_kda_v310/`, `tools/sidecard/npu_kda.py`, `tools/sidecard/kda_probe.py`, `tests/ut/sidecard/test_npu_kda.py`.
- **description**: Build an isolated one-row KDA recurrence with FP32 normalization, beta, gate, accumulation, and carry semantics matching `recurrent_kda`. Keep FP16 output rounding at the current boundary. Read the existing recurrent operator as a reference; do not modify its established FP16 input path or sidecard runtime bindings.
- **acceptance_criteria**:
    - Provide operator schema, tiling, adapters, supported shapes, workspace bytes, numerical contract, and binding notes for T11. Inputs/carry ownership and mutation are explicit.
    - Portable reference/probe tests cover changed inputs, nonzero carry, invalid geometry, TP2 head counts, range, and sequential repeated calls. Probe accepts `--config`, `--output`, and `--rank` for later exclusive device tests.
    - Do not cast normalized query/key or beta to FP16 merely to call the existing operator. First-call/setup timings are separated from steady-state recurrence timing.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_npu_kda.py`
    - `python -m tools.sidecard.kda_probe --help`
    - T12 compiles and runs the probe on both chips after registration; this worker performs no accelerator allocation.
- **status**: Completed
- **log**: Isolated opt-in AscendC decode recurrence, tiling, alias-aware adapter, fixed proposed FP32 carry/output tolerances, and authenticated device probe implemented offline. Initial parent verification passed 23 tests; namespace review aligned registration with existing `_C_ascend` and worker verified 26 tests plus scoped hooks/help. Zero external workspace and 4160-byte/core UB are source estimates. Native compile, ACLNN state alias, both-chip numerical/timing and model gates remain T12; no hardware speed claim.
- **files edited/created**: `csrc/attention/sidecard_kda_v310/` (11 files), `tools/sidecard/npu_kda.py`, `tools/sidecard/kda_probe.py`, `tests/ut/sidecard/test_npu_kda.py`.

### T6: Implement fused mHC pre and post kernels

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `csrc/attention/sidecard_mhc_pre_v310/`, `csrc/gmm/sidecard_mhc_post_v310/`, `tools/sidecard/npu_mhc.py`, `tools/sidecard/mhc_probe.py`, `tests/ut/sidecard/test_npu_mhc.py`.
- **description**: Remove broadcast temporaries and short launches with one coherent pre/post kernel package. Preserve FP32 residual range, BF16 rounding locations, norm/output dtypes, mixer orientation, and ordered reductions. Keep native Sinkhorn as the qualified baseline rather than replacing it opportunistically. Leave the legacy FP16-saturating post operator intact.
- **acceptance_criteria**:
    - Provide schemas, tiling, memory/rounding contracts, and T11 binding notes. Cover attention/FFN mixers and one/multiple rows with explicit fallback when a shape is slower or unsupported.
    - Tests include residuals above 65,504, negative gates, changed coefficients, nontrivial combination matrices, and all output fields. Bounds are fixed before device timing.
    - Probe accepts `--config`, `--output`, and `--rank`; CPU/reference timing is never presented as Ascend performance.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_npu_mhc.py`
    - `python -m tools.sidecard.mhc_probe --help`
    - T12 checks real-weight boundaries and steady-state 1/4/32-row timing on both chips.
- **status**: Completed
- **log**: Isolated AscendC pre/post kernels, host/tiling/adapters, prepared-weight policy and real-weight probe implemented offline. Initial parent 31 tests passed; review added default-off CMake options, 310P AIV/AIC metadata fallback and existing-FP32 weight reuse checks, with 36 worker tests/hooks/help passing. Qualified row sets default empty. Rounded fields require exact bytes; raw logits have fixed atol=2e-4/rtol=2e-5. Zero external workspace and pre/post UB 118816/9216 bytes are source estimates; ~3 MiB/layer mixer residency already exists when reused. Native compile, real device equality and serving benefit remain T12.
- **files edited/created**: `csrc/attention/sidecard_mhc_pre_v310/`, `csrc/gmm/sidecard_mhc_post_v310/`, `tools/sidecard/npu_mhc.py`, `tools/sidecard/mhc_probe.py`, `tests/ut/sidecard/test_npu_mhc.py` (24 files).

### T7: Implement deterministic multi-route expert decode

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/native_multi_route/`, `tools/sidecard/multi_route_probe.py`, `tests/ut/sidecard/test_multi_route.py`; authenticated frozen bundle is read-only.
- **description**: Create an isolated native candidate for zero through eight owned contributions at one decode row. Use the authenticated frozen bridge/kernel ABI and resident prepared banks. Avoid 17-row padding through fixed contribution slots and deterministic ordered combination; preserve activation quantization, clipping, and mixed packing formats.
- **acceptance_criteria**:
    - No unordered floating-point atomics, omitted cold contributions, live `tools/glm_perf/` edits, or weight repacking. Duplicate IDs remain separate route contributions.
    - Provide a build manifest, ABI compatibility checks, kernel-factory/binding notes, workspace sizes, fallback contract, and a `--config`, `--output`, `--rank` probe.
    - Test zero/one/eight routes, all route positions, peer-owned IDs, duplicate IDs, non-unit and zero weights, both 144-expert ownership ranges, and incompatible bundle provenance.
    - Quantify coverage beyond the rejected single-contribution graph's 7.2% observed calls.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_multi_route.py tests/ut/sidecard/test_native_bank_release.py`
    - `python -m tools.sidecard.multi_route_probe --help`
    - T12 compares all resident layers and changed activations with the qualified padded path before timing.
- **status**: Completed
- **log**: Authenticated original 36-source/9-helper bundle prepared with six consistent fixed-slot admission patches. Candidate retains original bridge, expert-major duplicate-slot reduction, clipping, banks, and original actor fallback. Parent verified 40 portable tests including bank release; worker scoped hooks/help passed. Actual prepare-only receipt is uncompiled. 100% R1 shape eligibility is not measured traffic coverage or speed. T11 must dispatch R1 before original actor padding and preserve deterministic short-prefill padding; T12 owns compilation and all-layer device gates.
- **files edited/created**: `tools/sidecard/native_multi_route/{__init__.py,contract.py,build.py,adapter.py,README.md}`, `tools/sidecard/multi_route_probe.py`, `tests/ut/sidecard/test_multi_route.py`; authenticated external source staging and prepare-only build.

### T8: Develop an isolated Ascend DSA capture adapter

- **depends_on**: [T1, T3]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/npu_dsa_decode_graph.py`, `tools/sidecard/npu_dsa_probe.py`, `tests/ut/sidecard/test_npu_dsa_decode_graph.py`.
- **description**: Adapt the fixed-math/eager-cache split for TP2 Ascend layers 19/23/27/31. The existing `DsaDecodeGraph` explicitly accepts only unsharded CUDA and remains unchanged. Use persistent one-row buffers and owner-local graph state. Keep dynamic pool/cache decisions eager until capture safety is demonstrated.
- **acceptance_criteria**:
    - No captured stale position, cache length, or request reservation. Both ranks execute compatible capture/collective schedules; reset and cache addresses remain valid.
    - Cover dense/sparse thresholds, every incomplete-tail phase, changed positions, multi-row fallback, partial capture failure, and graph close.
    - Provide a parent probe with `--config` and `--output` that launches the coordinated TP2 pair, plus integration notes for `DecodeGraphs`. Distinguish partial capture from full sparse-attention capture.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_npu_dsa_decode_graph.py tests/ut/sidecard/test_layer_indexer.py tests/ut/sidecard/test_dsa_decode_graph.py`
    - `python -m tools.sidecard.npu_dsa_probe --help`
    - T12 uses a coordinated two-rank real-weight gate; independent rank probes cannot qualify captured HCCL.
- **status**: Completed
- **log**: Partial owner-local graphs capture mHC/QKV preparation and output projection only; dynamic attention/cache/indexer, HCCL boundaries, routing and experts remain eager. Joint pair admission, failure cleanup, persistent cache identity and coordinated real-weight probe provided. Parent verified 64 prescribed portable tests with OMP_NUM_THREADS=1/MKL_NUM_THREADS=1; unchanged indexer exact-tail test fails independently at default CPU thread counts due to 2.384e-7 reduction differences. No indexer tolerance was changed. Worker scoped hooks/help passed. Coordinated device/full-model/capacity/latency gates remain T12.
- **files edited/created**: `tools/sidecard/npu_dsa_decode_graph.py`, `tools/sidecard/npu_dsa_probe.py`, `tests/ut/sidecard/test_npu_dsa_decode_graph.py`.

### T9: Define two-slot DMA ownership and sequencing

- **depends_on**: [T1, T3]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/stage_slots.py`, `tests/ut/sidecard/test_stage_slots.py`, `tools/sidecard/STAGE_SLOTS.md`; existing bridge and `stage_dma.py` are read-only.
- **description**: Implement a pure control/lease model for two complete-state slots with versioned registration. Specify producer-ready, both-rank upload/compute completion, rank-0 return, CUDA suffix-consumption, and retirement transitions. Include a bounded submit/receive interface and drain/quarantine behavior for T16.
- **acceptance_criteria**:
    - Each lease includes session/nonce, request, global sequence, slot generation, position/rows, reservation, and physical rank. Reject stale acknowledgements, replay, overlapping writes, and reordered cache consumption.
    - Do not conflate transfer completion with final consumer completion. An uncertain native operation retains allocation owners. Decode keeps its serialized fallback.
    - Account for two CUDA input/result slots and per-chip HBM/scratch without assuming the blocking bridge becomes nonblocking. Runtime calls remain on their owning context/thread.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_stage_slots.py tests/ut/sidecard/test_stage_dma.py tests/ut/sidecard/test_completion_contract.py`
    - Manually review success, cancellation, worker loss, timeout, and shutdown transition tables with T16's local owner before binding hardware.
- **status**: Completed
- **log**: Pure bounded two-slot lease model and transition tables reviewed by parent. Identity, reset, producer, both-rank upload/compute, rank0 return, CUDA consumer retirement, partial-registration ownership, cancellation/drain and quarantine are explicit; native blocking calls remain owner-local. Worker 142 targeted tests and scoped hooks passed; parent independently included all lease tests in a 224-test trace/transport/math run. Memory formula is 16 MiB CUDA/16 MiB per chip/8 MiB host guard per chip at 4 MiB slots, before scratch. Hardware binding and lifecycle/capacity/speed gates remain T16/T17.
- **files edited/created**: `tools/sidecard/stage_slots.py`, `tools/sidecard/STAGE_SLOTS.md`, `tests/ut/sidecard/test_stage_slots.py`.

### T10: Build completed quality, context, and lifecycle acceptance tests

- **depends_on**: [T1]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/execution_acceptance.py`, `tests/ut/sidecard/test_execution_acceptance.py`, `tests/e2e/nightly/sidecard/test_execution_acceptance.py`, and public synthetic acceptance fixtures under `tests/e2e/nightly/sidecard/fixtures/`.
- **description**: Implement at least 20 scored arithmetic/instruction/code/retrieval/tool cases, complete 32K/128K retrieval, a real 131,008+64 capacity run, and a 100-request mixed lifecycle suite. Accept fixture/run manifests from T1. Keep client content and tools unchanged. Add explicit opt-in hardware execution without modifying shared `conftest.py` in this worker.
- **acceptance_criteria**:
    - CLI accepts `--base-url`, `--fixtures`, `--output`, and `--tier quality|context|stability`. Tests skip without explicit hardware inputs and distinguish skip from pass.
    - A successful HTTP admission or plausible reasoning prefix cannot pass a completed-answer/context gate. The existing `context_validate.py` deliberately disconnects after headers and is insufficient here.
    - Check reasoning/final separation, valid tool arguments, finish reasons, client output reservation of 8,192, over-limit rejection, cancellation, state reset, prefix support/fallback, and completed responses after restart.
    - Worker-loss/restart actions belong to the local hardware owner; the harness records their checks without blind process termination.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_execution_acceptance.py`
    - `python -m tools.sidecard.execution_acceptance --help`
    - `python -m pytest -q --confcutdir=tests/e2e/nightly/sidecard tests/e2e/nightly/sidecard/test_execution_acceptance.py` collects and explicitly skips without exclusive inputs.
- **status**: Completed
- **log**: Implemented scored 20-case quality, fully consumed context/capacity, and 100-request lifecycle acceptance harnesses. Parent verified 43 portable tests passed and three nightly tests explicitly skipped without hardware inputs; worker scoped hooks/help passed. Missing owner recovery/cancellation/prefix/reset evidence remains pending rather than passed. Device qualification is T12/T17. New nightly URL/fixture/output options and packaging the public JSON fixture are T11 integration requirements.
- **files edited/created**: `tools/sidecard/execution_acceptance.py`, `tests/ut/sidecard/test_execution_acceptance.py`, `tests/e2e/nightly/sidecard/test_execution_acceptance.py`, `tests/e2e/nightly/sidecard/fixtures/acceptance-quality.json`.

### T19: Build a full Code teacher-forced state gate

- **depends_on**: [T1, T3]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/execution_model_gate.py`, `tools/sidecard/stage_diagnostics.py`, `tests/ut/sidecard/test_execution_model_gate.py`; existing actors, transport, and cache helpers are read-only.
- **description**: Generalize diagnostic qualification to 2K, 8K, and the original full Code fixture with fixed teacher IDs and complete state comparisons. Reuse the existing prompt/tokenization and CUDA snapshot patterns while specifying authenticated diagnostic hooks for NPU-owned state. Supply hook/protocol notes to T11; do not edit shared actor/transport files here.
- **acceptance_criteria**:
    - CLI accepts `--config`, `--fixtures`, `--output`, optional `--reference` plus `--reference-sha256`, and `--precision-contract`. Reference runs produce teacher IDs, tensor geometry, data/hash manifests, and private tensor references for numerical comparisons.
    - Observe complete deferred boundary fields at prompt chunks, all final recurrent/convolution/KV/indexer pool/tail state after prefill and teacher decode, and all output-vocabulary vectors. Each architecture gets its own qualified reference. Compare full tensors rather than only scalar samples or hashes when arithmetic changes.
    - Diagnostics synchronize/copy only in explicit gate mode, outside performance timing. Stream private references to bounded CPU/disk storage instead of retaining duplicate full-context device caches. They do not alter captured cache addresses or reuse admission-only context tests.
    - Tests reject changed prompts, missing rank/state coverage, unqualified references, altered teacher IDs, bad shapes/dtypes, undeclared bounds, and incomplete chunk/tail observations.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_execution_model_gate.py`
    - `python -m tools.sidecard.execution_model_gate --help`
    - T12 runs the gate only after T11 installs the diagnostic hooks; T17 repeats it for changed pipeline/packing/math combinations.
- **status**: Completed
- **log**: Authenticated teacher-forced gate and bounded private tensor capture helpers implemented offline, grounded in actual deferred/recurrent/convolution/KV/pool/tail fields and both rank owners. Graph selectors require retained capture references observed after replay; missing coverage fails. Parent verified 49 gate tests together with 36 mHC tests (85 passed); worker scoped hooks/help passed. KDA's existing zero length field is preserved and logical positions checked separately. Actual hooks, fresh qualified references and full-model device comparisons remain T11/T12.
- **files edited/created**: `tools/sidecard/execution_model_gate.py`, `tools/sidecard/stage_diagnostics.py`, `tests/ut/sidecard/test_execution_model_gate.py`.

### T11: Integrate decode candidates and native build identities offline

- **depends_on**: [T4, T5, T6, T7, T8, T10, T19]
- **agent_type**: local integration; isolated build, diagnostic and admission-helper workers
- **ownership**: Local owner: shared native registration in `csrc/torch_binding.cpp`, `csrc/torch_binding_meta.cpp`, and build lists only if required; `tools/sidecard/layer_math.py`, `layer_weights.py`, `layer_actor.py`, `layer_decode_graph.py`, `dsa_decode_graph.py`, `npu_dsa_decode_graph.py`, `layer_stage_profile.py`, `stage_dma.py`, `stage_options.py`, `dma_worker.py`, `dma_dispatch.py`, `expert_actor.py`, `expert_decode_graph.py`, `cold_decode_graph.py`, `layer_runtime.py`, and their specific existing unit tests plus `tests/ut/sidecard/test_execution_bindings.py`. Build-helper worker only: new `tools/sidecard/execution_kernel_build.py`, `tests/ut/sidecard/test_execution_kernel_build.py`, private addon source under `csrc/sidecard_execution/`; after that helper is accepted, the same worker owns `tests/e2e/nightly/sidecard/conftest.py`, new `test_execution_kernels.py`, new `tools/sidecard/execution_nightly.py` and its unit test. Diagnostic-helper worker only: new `tools/sidecard/execution_diagnostic_runtime.py`, `tests/ut/sidecard/test_execution_diagnostic_runtime.py`; expert diagnostic extensions are isolated in new `tools/sidecard/expert_diagnostics.py` and its unit test. Admission-helper worker only: new `tools/sidecard/execution_options.py`, `tests/ut/sidecard/test_execution_options.py`. Helpers make no shared runtime/build-list edits or hardware calls.
- **description**: Reconcile schemas/workspaces and bind each candidate behind a default-off runtime JSON option. Keep only declared viable candidates; numerical rejection selects the qualified fallback. Build manifests must authenticate private native sources, libraries, and required operator paths. Make source snapshots include new Python/native inputs and binary manifests, not unrelated live research. After admission review, its worker owns `cold_decode_graph.py` and `tests/ut/sidecard/test_cold_decode_graph.py` solely for matching capture/replay diagnostic hooks; parent retains other shared runtime files. Parent also owns KDA/mHC probe loading/device-ordinal integration and their existing unit tests. The local owner may compile the authenticated private package during an explicit T20 startup hold with its API controller stopped and no timed API subprocess active; resume the controller after compilation quiesces. This CPU-only build lane does not admit or test candidates on devices and does not weaken T12's T20 dependency. Build-helper follow-up ownership includes `csrc/cmake/config.cmake` and `csrc/cmake/scripts/prepare.sh` only to pass an optional authenticated initial CMake cache into the nested preparation build; preserve the default build path.
- **acceptance_criteria**:
    - Every candidate can be enabled independently; invalid combinations fail before allocation. Baseline operation and exact source freeze remain intact with all options off.
    - Native Meta functions match shape/dtype/mutation behavior; preserve graph cache tensors and completion ownership. Do not assume a registered custom operator is already graph-safe.
    - Existing grouped projection/NZ, Sinkhorn, CUDA indexer, and decode graph optimizations remain baseline features. Reuse improvements in the expert-only path where their semantics match.
    - Bind T19 diagnostics only in explicit authenticated gate mode. Add a nightly `--sidecard-execution-manifest` option with exclusive-ownership checks and kernel/state-gate coverage; retain T10's no-input skip behavior.
    - Freeze fresh default-off source/configuration templates with diagnostic hooks. Candidate and same-math socket templates record the selected options; T12 completes authenticated native build references and calls the launcher to create final phase-local `reference/runtime.json`, `candidate/runtime.json`, and `candidate-socket/runtime.json` after compilation. Uncompiled builds must never be admitted to a runtime snapshot. Retain prior snapshots untouched. For arithmetic changes, the socket reference uses the candidate's same math; the default-off model reference separately checks declared numerical differences.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard`
    - `python -m tools.sidecard.execution_kernel_build --help`
    - `python -m pytest -q --confcutdir=tests/e2e/nightly/sidecard tests/e2e/nightly/sidecard/test_execution_kernels.py tests/e2e/nightly/sidecard/test_execution_acceptance.py` collects/skips without an exclusive manifest.
    - Run changed-file pre-commit hooks and inspect source snapshot coverage before handing the build to T12.
- **status**: Completed
- **log**: Offline integration complete; no device qualification is implied. Default-off admission, serial-block and expert diagnostics, actual capture/replay observations, R1 dispatch before padding, KDA/mHC probes, and explicit nightly manifests are integrated. Parent also owns the diagnostic control tag in `tools/sidecard/dma_protocol.py`. The private three-operator OPP/Torch addon and authenticated R1 expert bundle compiled successfully; both receipts report hardware_validated=false. Final frozen runtime source contains 158 files with SHA256 72f7ee9cca0e0154a0b400115e5e1222fd3bfafc497408e4dd37279eeceeb021; 14 configuration templates remain explicitly unqualified. Native source/ABI bytes and executable permissions are retained. Full portable suite passed 2058 tests in 10.04 seconds with single-thread CPU math; four nightly cases skipped without explicit inputs. Scoped hooks passed; isolated full CI retained the same seven failing hooks as pristine HEAD, without scoped diagnostics. Final receipt: `artifacts/sidecard-execution-20261013-215113c/preflight/t11-offline-build-receipt-v2.json`. Card 16409 failed before the first compiler invocation; T12 device gates remain blocked on hardware recovery and T20.
- **files edited/created**: `tools/sidecard/execution_kernel_build.py`, `execution_options.py`, `execution_diagnostic_runtime.py`, `csrc/sidecard_execution/`, shared runtime/probe files and corresponding unit tests listed in ownership.

### T12: Qualify each decode candidate on both chips

- **depends_on**: [T10, T11, T20]
- **agent_type**: local
- **ownership**: New native/device/model/performance decision receipts and private builds; integrated source is read-only. Corrections return to the owning task before a new freeze.
- **description**: Under exclusive ownership, compile isolated operator builds, run component gates, and compare candidate math/state on real weights with T19. Run T10's completed task-quality suite. Test independent candidates before combinations and compare against their same-architecture reference. Collect unprofiled API measurements with T2 using T4's actual alternating launch/cold-cache protocol and update the trace after accepted changes. Restore the best service between experiments as needed.
- **acceptance_criteria**:
    - Per-candidate decisions separate build success, numerical eligibility, task accuracy, component speed, model-forward speed, and API benefit. Neither finite synthetic outputs nor a faster microbenchmark qualifies release.
    - Arithmetic bounds, router/indexer selections, complete recurrent/cache/vocabulary checks, source identities, and both-chip coverage are recorded.
    - Provide `decode-qualified.json` with eligible options, rejections, reasons, and baseline fallback. KDA prefill may proceed if decode math qualified even when decode performance did not improve.
    - Generate each candidate-math socket server receipt with `layer_server --gate-only` before its direct-DMA server starts. Authenticate it through the existing transport-reference options. Preserve exact transport parity; do not bypass startup gates or compare changed arithmetic with an old-math socket hash.
- **validation**:
    - Run each new probe with its pinned NPU interpreter, frozen config, output receipt, and each assigned rank; the DSA gate launches the TP2 pair jointly.
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.execution_model_gate --config "$SIDECARD_PHASE_ROOT/reference/runtime.json" --fixtures "$SIDECARD_PRIVATE_FIXTURES/manifest.json" --output "$SIDECARD_PHASE_ROOT/model-reference.json" --precision-contract "$SIDECARD_RUN_ROOT/precision-contract.json"` uses T11's fresh hooks/default-off freeze to produce a same-architecture reference before the candidate run.
    - Repeat the model gate with the candidate config and authenticated `--reference`/`--reference-sha256`. Run T10's `--tier quality` against its gated server. Native math failures prevent performance promotion.
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.layer_server --config "$SIDECARD_PHASE_ROOT/candidate-socket/runtime.json" --gate-only` produces the candidate's same-math startup reference before a direct-DMA launch.
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.stage_benchmark --config "$SIDECARD_PHASE_ROOT/candidate/runtime.json" --reference "$SIDECARD_PHASE_ROOT/candidate-socket/stage-benchmark.json"` requires a preceding stage benchmark on that fresh same-math socket spec. Archived T4 receipts are historical evidence, not replacements for new diagnostic/transport references.
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --base-url "$SIDECARD_BASE_URL" --fixtures "$SIDECARD_PRIVATE_FIXTURES/manifest.json" --output "$SIDECARD_PHASE_ROOT/$SIDECARD_PAIR_ID-candidate.json" --repeats 1 --output-tokens 256 --pair-id "$SIDECARD_PAIR_ID" --pair-order baseline-first --mode candidate`
    - Collect the actual paired baseline side using T4's lifecycle, then run `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --compare-pairs "$SIDECARD_PHASE_ROOT/api-pairs.json" --output "$SIDECARD_PHASE_ROOT/api-comparison.json"` after at least five pairs per fixture.
- **status**: Blocked
- **log**: T11 offline integration/builds are complete, but assigned card 16409 chips 2/3 are Critical after driver-level heartbeat loss. The original GLM service could not restart and is offline. Device files have no users and DMA references are zero; no reset was performed. The card-only reset exception is pending because PRD R6 prohibits card resets. T20 has one completed baseline pair, not the required repeated window. No candidate numerical, full-model, quality, context, lifecycle, or performance gate has passed. Preserve all device-dependent decisions until healthy hardware and T20 are available.
- **files edited/created**: Hardware and build evidence under `artifacts/sidecard-execution-20261013-215113c/preflight/`.

### T13: Implement chunked native KDA prefill

- **depends_on**: [T12]
- **agent_type**: worker
- **ownership**: New `csrc/attention/sidecard_kda_prefill_v310/`, `tools/sidecard/npu_kda_prefill.py`, `tools/sidecard/kda_prefill_probe.py`, `tests/ut/sidecard/test_npu_kda_prefill.py`.
- **description**: Extend the numerically eligible recurrence with sequential per-prompt carry inside a native chunk. Remove the Python row loop while retaining projection/convolution rounding and exact chunk order. Keep the T5 operator source unchanged and provide a distinct prefill schema for T16 registration.
- **acceptance_criteria**:
    - Partitioned and unpartitioned chunks produce state/output within the predeclared contract; cover 1/4/32 rows, partial chunks, nonzero initial carry, and long repeated carries.
    - No parallel treatment of dependent prompt rows or FP16 carry downgrade. If T12 rejects the precision contract, complete this task with an explicit rejection/fallback receipt and leave prefill on the qualified path; independent scheduling work can continue.
    - Provide `--config`, `--output`, `--rank` probe and workspace/memory notes; multi-row steady-state performance is measured separately from decode.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_npu_kda_prefill.py tests/ut/sidecard/test_prefill_convolution.py`
    - `python -m tools.sidecard.kda_prefill_probe --help`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T14: Develop owner-local shared/expert stream overlap

- **depends_on**: [T12]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/npu_expert_overlap.py`, `tools/sidecard/npu_overlap_probe.py`, `tests/ut/sidecard/test_npu_expert_overlap.py`.
- **description**: Provide a `with_shared` adapter for the existing `TextLayer.finish_forward` extension point. Shared and routed branches use separate native scratch and events, then join before ordered combination and the existing all-reduce. Build against T12's qualified options or baseline fallback; no shared runtime edits.
- **acceptance_criteria**:
    - Explicit stream ownership, input readiness, result readiness, scratch isolation, close, and exception behavior. No concurrent use of the same native workspace or dependence on implicit default-stream ordering.
    - Probe records overlap versus core/DDR contention; if overlap increases wall time or graph incompatibility outweighs savings, retain the serialized path.
    - No overlap of dependent layers or future generated tokens. Probe accepts `--config`, `--output`, and `--rank`.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_npu_expert_overlap.py tests/ut/sidecard/test_native_overlap.py`
    - `python -m tools.sidecard.npu_overlap_probe --help`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T15: Evaluate specialized pair collectives, scheduling and rank imbalance

- **depends_on**: [T12]
- **agent_type**: worker
- **ownership**: New `tools/sidecard/npu_collective_schedule.py`, `tools/sidecard/npu_collective_probe.py`, `tests/ut/sidecard/test_npu_collective_schedule.py`. User-authorized specialized-library follow-up belongs to the local owner: `tools/sidecard/pair_collective.py`, `pair_collective_acl.py`, `pair_collective_build.py`, `PAIR_COLLECTIVES.md`, `csrc/sidecard_collectives/`, corresponding new unit tests and private build receipts.
- **description**: Use the current trace to isolate rank-arrival delays and actual HCCL transit. Evaluate a bounded scheduling/event or capture policy and the user-requested specialized two-peer FP32 SUM library, preserving the two reduction boundaries per layer, identical rank call order, and current expert ownership. The user explicitly authorized isolated library hardware tests on both Duos before T12; this exception does not admit model changes or remove T12's model gates. Report expert imbalance and a proposed mapping contract if a remap is the larger opportunity.
- **acceptance_criteria**:
    - A trace-based decision identifies exposed wait reductions without changing collective arithmetic or omitting either rank. Distinguish already-captured KDA collectives from eager DSA work.
    - Do not move collectives across dependent math, change 144-expert ownership implicitly, or add a competing hardware test. A remap needing broader changes becomes an explicit follow-up task rather than an undocumented edit to the loader.
    - Provide a pure policy test and coordinated pair probe with `--config` and `--output`; skip implementation with a bounded savings/rejection receipt if the measured transit is negligible.
    - For a custom transport, qualify P2P memory, shareable Notify ordering, timeout units, both-rank reuse/shutdown and complete FP32 SUM parity before timing. Compare 1/4/32-row HCCL/native calls and 30-boundary sequences with matched arrivals and injected skew. A compiled library or CPU simulation is not device qualification. Require a stream-aware Torch adapter and fresh model/context gates before serving admission.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_npu_collective_schedule.py tests/ut/sidecard/test_tp_rank_inputs.py`
    - `python -m tools.sidecard.npu_collective_probe --help`
- **status**: Not Completed
- **log**: User expanded this follow-up to N NPU ranks and real two-Duo Qwen Flash-Next comparison. The general-N owner and queued Torch facade support fixed FP16/FP32 SUM; binomial schedules cover arbitrary N, recursive doubling covers power-of-two N. Raw hardware tests passed N2/N3/N4 with full CPU/HCCL finite-pattern parity, 90 distinct-input/output calls per rank and explicit all-rank release. At QFN hidden size 2560, N4 FP16 completed batches measured 80.30 microseconds versus HCCL 142.09; the three-token size measured 80.34 versus 143.44. N3 FP16 measured 107.67 versus 190.73. Initial staged binomial N2 regressed and was replaced by direct I/O plus recursive doubling. These are synthetic microbenchmarks, not QFN token rates. The eager queued Torch gate passed all 14 registered FP16/FP32 geometries on all four ranks, including 30 distinct live input/output calls per geometry, CPU/HCCL exact-byte checks and all-rank release; matched short-prompt QFN TP4/EP4 eager tests completed with MTP2 and 262K configured context: C1 median 19.03 versus HCCL 18.03 tok/s, C4 aggregate 58.52 versus 53.45 (two timed trials each). Every trial proved custom call deltas on all ranks; both worker trees shut down gracefully and all custom owners closed. Outputs were coherent but baseline token IDs varied across repeated trials; broad model quality, long-context input execution and graph admission remain open; no serving/graph admission. Portable sidecard suite passed 5223 tests including the Torch probe, controlled QFN harness, strict config regression and receipt comparison. New public receipts: `artifacts/npu-collective-qfn-20261014-024609/`. Specialized push/SDMA plus READY/CONSUMED passed full synthetic FP32 CPU/HCCL comparisons, 90 changed-input captures per rank and orderly shutdown on BOTH Duos for 1/4/32 rows. Seven alternating matched batches measured approximately 43/50/74–75 microseconds per call versus HCCL 59–64/69–71/86–87; single-call host-drained comparisons showed little benefit. These are completed-batch microbenchmarks, not pure wire transit or GLM token rates. A remote-read vector candidate passed copy/Notify gates but timed out at its first add and left driver calls unresponsive; user rebooted, both Duos recovered, and all three admission layers now reject remote-read before allocation/tensor creation. A local-only custom add passed device FP32 tests but did not improve completed latency across the core-count sweep. Two alternating inboxes/READY objects are the next candidate; independent byte-level scheduling proof and CPU native tests precede hardware execution. The portable suite passed 2517 tests before this ping-pong addition. New receipts are in `artifacts/sidecard-pair-hardware-20261014-014000/`; Torch adapter, timeout-unit resolution, graph/model/context and repeated performance gates remain pending. Original T11 runtime freeze is unchanged.
- **files edited/created**: Specialized-library files and tests above; `tools/sidecard/PAIR_COLLECTIVES.md` records the mechanism, limitations and required hardware comparisons. Original T11 runtime freeze is unchanged.

- **Graph follow-up (2026-10-14)**: Added fixed-address recursive doubling with on-device READY/CONSUMED acknowledgements, optional graph admission, and native C++ ownership leases. Production graph entries call native replay without Python tracking; diagnostics are opt-in. The unobserved N4 gate passed all 14 FP16/FP32 geometries, 28 graphs and 392 released leases per rank; three-SUM graph medians at FP16 7680 elements were 17.86 µs custom versus 26.33 µs HCCL (clone included for baseline out-of-place semantics). The first QFN graph attempt rejected an unregistered breakable-wrapper capture before the custom op was admitted; upstream workers exited and all chips remained healthy. Breakable-wrapper cold ownership integration is implemented with no production replay observer. The final hardware sweep replayed all older captures in reverse and forward order after later captures, with outputs cleared first; all four ranks passed and released their leases. QFN graph integration completed on all four ranks with unobserved replay and clean retirement of four captures/56 leases per rank. Matched short-prompt medians: `[3, 6]` C1 custom 31.09 vs HCCL 30.33 tok/s (+2.53%); `[9, 12]` C4 custom 62.09 vs HCCL 58.50 aggregate tok/s (+6.14%), no eager-fallback warnings. Larger captures pad C1 and reduce it to ~20 tok/s. These are two-trial offline results, not quality/long-context qualification or a live endpoint. Portable suite: 5804 passed; 13 protected source hashes unchanged.

- **Fresh profile and promotion (2026-10-14)**: Four-rank C1 custom-path profiling measured 31.23 tok/s unprofiled median. Each MTP2 round launches 22 native graph segments with 19 eager callbacks. Median recorded device-task union is 63.61 ms of an 85.57-ms round; the remaining 22.09 ms includes dependencies and is not entirely recoverable idle time. FP16 matmul and casts/layouts dominate named kernel costs; HCCL all-gather is only 0.53 summed ms/round, excluding custom SDMA/Notify/Add. The pinned-host draft-position copy passed 120 changing-input hardware checks and three full model trials (median 31.42 tok/s). Promoted at user request as the custom research baseline; throughput remains neutral because ranges overlap. Remote selection: `run-qfn-custom-current.sh` under the QFN artifact root, preserving the previous runtime. Local backport retains CPU reference behavior; 14 position tests and 5814 sidecard tests passed. [Report and profile PNG](artifacts/npu-collective-qfn-20261014-024609/README.md) include provenance and promotion hashes. No 40 tok/s claim and no broad quality qualification.

- **Parallel optimization follow-up (2026-10-14)**: Three isolated agents delivered QSA selection, partitioned W8A16 draft execution, and FP16 HC row-padding candidates (145 targeted CPU tests passed). Parent hardware gates rejected HC padding for latency regression; the draft candidate passed 90 exact synthetic changing-route comparisons and a TP4 short-prompt model trial at 32.71 tok/s median versus earlier 31.42 (+4.1%, sequential measurements). All ranks retired native graph leases cleanly; replay tracking stayed disabled. QSA passed 50 changed-input replays across 13 component cases through 262144 context, with large short-context savings but a packed 2052 bucket regression; model dispatcher/admission integration remains pending. No new serving default or model ID. See [wave report](artifacts/npu-collective-qfn-20261014-024609/optimization-wave.md).

- **Persistent server follow-up (2026-10-14)**: At the user's request, subsequent optimization must use hot reload and leave the server running. Stable `qwen38-flash-next` is live at port 8001 with 262144 context. Two resident switches took 2.89 and 2.80 seconds, retaining all four worker PIDs and tracked weight-storage identities; each rank retired/rebuilt 56 graph leases. Draft partition plus a pre-reset device barrier are installed. Streaming chat and tool calls passed, as did 240 exact real-weight local draft comparisons. Live HTTP C1 medians were 31.56 baseline and 31.69 candidate (neutral); generated sequences were not identical across all trials. QSA dispatcher is implemented/offline tested but still requires full-model boundary qualification. See [resident handoff](artifacts/npu-collective-qfn-20261014-024609/resident-live-v2/README.md). Do not stop/reload the model for further experiments.

- **QSA boundary correction (2026-10-14)**: Target prefix graphs removed twelve eager callbacks, but the live boundary recapture failed on the default stream and exited the engine. The prior configuration was restored under `resident-live-v3`, same model/port/context. A corrected dedicated-stream window now passes private graph and mixed-eager hardware probes on all four ranks with unchanged serving PIDs. Full-model QSA remains unqualified; follow-up work replaces live recapture with paused capture of both variants. See [failure and fix evidence](artifacts/npu-collective-qfn-20261014-024609/qsa-resident-v1/README.md).

- **Cold QSA correction (2026-10-14)**: Capturing general/prefix variants with shared GDN metadata reused derived scratch without recording its producer. Per-capture isolated metadata fixes the observed corruption. The corrected payload passed eight boundary/concurrency/return-to-prefix requests through 8192 input tokens and remains active at the user's request. Four resident PIDs and weights stayed unchanged; five captures/70 leases per rank remained healthy and subsequent hot reloads retired them successfully. Matched C1 HTTP medians 31.55 versus 31.66 tok/s are neutral. The qLens owned-capacity fix is retained. Bounded native decode and zero-cache prefill traces are captured; early TP draft ordering is a separate candidate. Future device-driven work is specified in [distributed graph ledger](tools/sidecard/DISTRIBUTED_GRAPH_LEDGER.md); autonomous device graph launch is unproven.

### T16: Integrate two-slot prefill and final result packing

- **depends_on**: [T9, T13, T14, T15]
- **agent_type**: local
- **ownership**: `tools/sidecard/stage_dma.py`, `stage_options.py`, `layer_actor.py`, `layer_math.py`, `layer_weights.py`, `layer_decode_graph.py`, `layer_runtime.py`, `stage_benchmark.py`, `csrc/torch_binding.cpp`, `csrc/torch_binding_meta.cpp`, and build lists if needed; corresponding existing tests plus new `tests/ut/sidecard/test_stage_prefill_pipeline.py`, `tests/ut/sidecard/test_stage_result_pack.py`, and `tests/e2e/nightly/sidecard/test_stage_pipeline.py`; shared nightly `conftest.py` only after T11.
- **description**: Bind T13–T15 behind independent default-off options and integrate T9 leases. Split complete-block submission/receive so CUDA prefix B can execute while Ascend owns chunk A; serialize conflicting RTX prefix/suffix work and Ascend state mutation. Optimize rank-0 packing into final DMA views and omit unused rank-1 result packing while preserving its completion event.
- **acceptance_criteria**:
    - Two slots have persistent private allocations, versioned identities, bounded control flow, safe drain/quarantine, and no host tensor staging. Blocking bridge calls and native context/thread affinity remain explicit.
    - Test reuse only after both ranks and final CUDA consumer complete; positions, reservation, result-to-suffix association, and last partial chunk remain correct.
    - One-slot and two-slot tests use identical math/geometry to isolate scheduling. Keep ordinary decode and socket fallback unchanged. Complete-block prefix reuse remains disabled unless separately qualified.
    - Include native prefill/buffer sources in frozen manifests; every combination fails closed on an incompatible binary. Added memory fits the 128K reservation.
    - Integrate as three independently frozen/tested checkpoints: A binds native prefill/overlap/collective adapters independently; B adds two-slot scheduling using qualified baseline math; C changes final packing independently. Each has a default-off option, manifest, portable tests, and rollback config. T17 tests each checkpoint separately before combining them.
    - Each checkpoint gets fresh phase-local default-off, candidate, and candidate-math socket snapshots containing the current diagnostics. T17 produces their new state/transport receipts rather than reusing a pre-integration source archive.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard/test_stage_prefill_pipeline.py tests/ut/sidecard/test_stage_result_pack.py tests/ut/sidecard/test_stage_dma.py tests/ut/sidecard/test_layer_block_design.py tests/ut/sidecard/test_layer_runtime.py`
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard`
    - `python -m pytest -q --confcutdir=tests/e2e/nightly/sidecard tests/e2e/nightly/sidecard/test_stage_pipeline.py` collects/skips without an exclusive manifest; its opt-in device run compares one/two slots and changed packing through T19's state gate.
    - Inspect control/event timelines for cancellation, worker loss, constructor failure, shutdown, stale completion, and graph replay before device qualification.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T17: Run complete performance and release acceptance

- **depends_on**: [T10, T16]
- **agent_type**: local
- **ownership**: New final performance/context/quality/lifecycle/memory receipts; `tools/sidecard/layer_server.py` and focused server tests only for acceptance-discovered compatibility fixes, which require a fresh freeze and repeated gates.
- **description**: Under exclusive accelerator ownership, qualify individual prefill/overlap/packing options before the combined candidate. Run matched API comparisons, both-chip/native state gates, all quality cases, completed context, and lifecycle stress. Compare the best complete-block candidate and reusable expert-only improvements against a fresh best-service baseline.
- **acceptance_criteria**:
    - At least five paired cold repetitions and 256 timing outputs for each 2K/8K/full Code fixture, with spread, uncertainty, and zero cached tokens. Record force-length timing separately from normal-EOS quality.
    - Use T4's actual alternating launch protocol and T19's same-architecture full Code/state gate for each independently frozen T16 checkpoint before testing the combination. Cache-enabled warm-prefix traffic is a separate workload.
    - Actually complete the 131,008+64 capacity request and 32K/128K retrieval/state tests on the exact candidate; report quality separately from capacity. No shortened Code input or fewer tools.
    - At least 20 completed/scored quality cases and 100 mixed requests, stable memory, no device faults, valid cancellation/restart, and exact mapping/guard release accounting.
    - A machine-readable release decision applies the PRD's per-fixture targets. Unmet gates retain the qualified baseline and report the remaining bottleneck; no forced promotion.
- **validation**:
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --base-url "$SIDECARD_BASE_URL" --fixtures "$SIDECARD_PRIVATE_FIXTURES/manifest.json" --output "$SIDECARD_PHASE_ROOT/$SIDECARD_PAIR_ID-candidate.json" --repeats 1 --output-tokens 256 --pair-id "$SIDECARD_PAIR_ID" --mode candidate`
    - Repeat the matched baseline side through T4's actual lifecycle. After five or more pairs per fixture: `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.performance_suite --compare-pairs "$SIDECARD_PHASE_ROOT/api-pairs.json" --output "$SIDECARD_PHASE_ROOT/api-final.json"`
    - `"$SIDECARD_CUDA_PYTHON" -m tools.sidecard.execution_acceptance --base-url "$SIDECARD_BASE_URL" --fixtures "$SIDECARD_PRIVATE_FIXTURES/manifest.json" --output "$SIDECARD_RUN_ROOT/quality-final.json" --tier quality`
    - Repeat the same acceptance command with `--tier context` and `--tier stability`, writing distinct receipts. T10's capacity tier must consume the full stream, not merely open it.
    - Use the existing explicit complete-stage nightly wrapper only for same-math socket/DMA parity; changed arithmetic requires its declared numerical gates.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T18: Deliver the measured design and promote the qualified service

- **depends_on**: [T17]
- **agent_type**: local
- **ownership**: New public `artifacts/sidecard-execution-<run-id>/README.md`, `design.png`, reproducible plotting source, and sanitized release/restoration receipts; `tools/sidecard/README.md`, `LAYER_RUNTIME.md`, `EXPERT_RUNTIME.md`, and this plan's execution log/status.
- **description**: Publish the selected architecture, achieved timings, rejected experiments, critical path, memory/context limits, and reproduction commands. Generate a PNG from the measured placement and timings. If T17 passes, launch its exact frozen build and verify readiness/real traffic; otherwise restore and verify the previous service. Preserve rollback inputs and normal mapping ownership.
- **acceptance_criteria**:
    - Delivered source/binary/configuration manifests match the served bytes. Startup gates and a completed real request pass after launch/restoration; model limits still advertise 131,072.
    - Public report distinguishes API/forward rates, cold/cached status, quality/capacity, and targets/results. PNG depicts the executed design; inspect it visually.
    - All required checks are recorded with unrelated failures identified; protected research remains intact. A rejection report cannot claim optimization targets were achieved.
- **validation**:
    - `python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard`
    - Run changed-file hooks, then `bash format.sh ci` in an isolated worktree containing only scoped changes. The prior baseline had seven failed hooks; compare actual diagnostics before labeling failures pre-existing.
    - Check `/health`, `/v1/models`, complete one normal Code/chat request, verify exact manifest hashes and bridge-owner accounting, and inspect the exported PNG.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
| --- | --- | --- |
| 1 | T1 local; T2, T3 workers | Immediately; inspect-only hardware preflight |
| 2 | T4 local; T5, T6, T7 workers | T1–T3 complete; kernels use frozen T1 math/ABI offline while local profiling runs |
| 3 | T20 local; T8, T9, T10 workers | T20 needs T4; offline workers may start after T1/T3 in earlier free slots |
| 4 | T19 worker | T1/T3 complete; fixed precision contract available before device testing |
| 5 | T11 local | T4, T5–T8, T10, T19 complete or candidates have explicit rejection/fallback receipts |
| 6 | T12 local | T10/T20 complete; T11 integrated and frozen; exclusive assigned accelerator ownership |
| 7 | T13, T14, T15 workers | T12 decisions and numerical eligibility available |
| 8 | T16 local | T9 and T13–T15 complete or explicitly rejected with fallback |
| 9 | T17 local | T10 and T16 complete; exclusive assigned accelerator ownership |
| 10 | T18 local | T17 release/rejection decision complete |

Only three worker tasks run concurrently with the parent. A wave is a safe
default schedule, not permission to violate dependencies or the hardware lease.

## Integration Strategy

- T3 owns initial shared tracing edits. T11 is their next shared-file writer
  through T4's dependency; T16 follows T12. No worker silently expands ownership.
- T5–T9, T19, and T13–T15 provide separate modules and explicit schemas/interfaces.
  T11/T16 review integration notes and reconcile native registration, config
  validation, graph lifetimes, and frozen source coverage locally.
- Keep every candidate independently selectable and default off. Test baseline,
  candidate alone, and accepted combinations; never attribute combined gains to
  an unmeasured component.
- Allocate accelerators only from the local integration owner. No worker may
  stop serving, allocate a resident full model, change global OPP/driver state,
  or reset a card. Native API calls remain with their owning process/context.
- If an input, hardware lease, compiler, or numerical contract is blocked,
  record the blocker in the task log, continue disjoint offline work, and leave
  dependent hardware validation pending. Time passing is not qualification.
- If profiling makes a candidate unhelpful, preserve its rejection receipt and
  route integration to the qualified fallback. If all release candidates fail,
  restore the baseline and report unmet targets instead of promoting a slower
  design. A broader expert remap requires a separately scoped follow-up plan.

## Testing Strategy

- Portable tests use the existing sidecard isolation boundary and exercise
  stateful edge cases, not copies of implementation constants. New device
  functionality adds explicit nightly tests with opt-in hardware manifests.
- T12/T17 own real-weight device testing and numerical contracts. Same-math
  transport/scheduling gates require complete state/vocabulary bitwise parity;
  altered arithmetic uses bounds declared in T4 plus model/task checks.
- Production-path event traces are distinct from synchronized component probes.
  Warmups and first-call compilation are excluded from steady-state component
  timing; CPU diagnostic observation stays outside model-forward timing.
- API comparisons use matching prompt IDs, tools, reservations, parsers,
  output budgets, hardware, source identities, and cold-cache state. Report
  uncertainty; add repetitions when the target gain cannot be distinguished.
- Context admission is separate from completed capacity and recall. Recurrent,
  DSA, graph-buffer, slot, or workspace changes trigger exact-build 128K gates.
- Failed DMA, stale completion, missing peer acknowledgement, and uncertain
  quiescence retain owners. Fault injection must not issue unsafe resets or
  destroy allocations with possible in-flight DMA.

## Risks and Mitigations

- **Risk**: FP16 substitutions or different reduction order change later routing.
  **Mitigation**: Preserve FP32 preparation/carry and BF16 residual range;
  declare limits before timing and inspect route/indexer state and final answers.
- **Risk**: Microbenchmark gains disappear inside a graph or behind HCCL wait.
  **Mitigation**: Profile the current joined critical path, qualify complete
  model/API behavior, and reject improvements without serving benefit.
- **Risk**: Two slots overwrite deferred views or outlive a cancelled request.
  **Mitigation**: Versioned slot leases, both-rank completion, explicit suffix
  retirement, bounded drain, quarantine, and state-machine failure tests.
- **Risk**: Multiple streams compete for cores/DDR or use one native scratch.
  **Mitigation**: Independent workspace/event contracts and serial/overlap
  comparisons on both ranks; keep the serial path when contention wins.
- **Risk**: Graphs/slots consume memory needed by 128K state.
  **Mitigation**: Per-chip peak accounting and actual full-capacity completion
  before release; no context reduction to hide an allocation regression.
- **Risk**: Stale baseline source or old fixture behavior produces false claims.
  **Mitigation**: Fresh manifests and API baselines, completed acceptance tests,
  same-build parity references, and exact promoted-source receipts.

## Reviewer Notes

- Local review verified repository paths, existing CLI behavior, shared-file
  sequencing, candidate source ownership, and portable-test isolation.
- One reviewer agent checked the draft plan and repository paths/APIs without
  editing source or touching hardware. Its five actionable findings were applied:
  actual alternating cold-service comparisons, a produced full Code/state gate
  (T19), T10's quality-harness dependency for T12, assigned nightly wrappers/options,
  and independently frozen integration checkpoints in T16.
- Dependency/wave validation is repeated after these revisions. Planning and
  review do not change the pending implementation or hardware-validation status.
- The reviewer rechecked all 19 dependencies/waves and identified a stale
  diagnostic reference. T11/T12 now freeze and use fresh phase-local references;
  T16/T17 repeat this for each checkpoint. Local startup-gate review also requires
  a same-math candidate socket receipt so arithmetic experiments do not weaken
  bitwise DMA transport qualification.
- Execution scheduling refinement: T5–T7's isolated offline kernels require
  T1's frozen ABI/math inputs, not completion of slow model reloads. They now
  overlap T4; T11 still depends on T4's measured priorities and precision contract.
  T20 retains the full repeated-baseline requirement before T12 qualification.
  No worker can allocate hardware or enter the frozen profiling source.
- Further offline scheduling refinement: T8/T9/T19 need the frozen ABI, tracing
  interface and predeclared precision contract, which are available during T4.
  Their isolated implementation can overlap model reloads; T11 still waits for
  completed T4 profiling/restoration before binding any candidate. T16 retains
  measured budgets and all numerical, capacity and performance gates.
