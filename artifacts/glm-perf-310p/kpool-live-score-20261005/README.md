# GLM graph decode: score only live pooled keys

## Scope and status

Opt-in `ascend_glm_live_kpool_score`, for packed GLM's existing MTP full-graph
path. After the initial no-server study, the user authorized taking over the
NPUs and stopping Qwen. A matched resident comparison completed on all four
NPUs. The user selected approximately 300K context for the final server.

The current fixed graph selector gathers every configured pooled key for
every row, zeros invalid keys, expands keys to FP32, materializes all head
logits, applies ReLU and head weights, and reduces them. At 311,040 configured
tokens the loop processes 77,760 pooled keys even for a 256-token request.
The old constructor's stale maximum was not the cause: the page table already
bounded the range. Merely clamping that constructor bound did not help.

`GlmKpoolScoreV310` reads live positions and cumulative request ends on device.
It gathers only completed key tiles directly into local storage, uses Cube
matmul with FP32 accumulation, and fuses ReLU, FP32 head weighting, and head
reduction. It writes a fixed-shape score array with unused entries set to
`-inf`; existing per-row top-k and token expansion remain. Score initialization
is a separate stream-ordered fill to avoid cross-core write races.

Shared cache backing is passed as a flat storage alias with explicit offsets
and page/row strides. The entire strided cache is never made contiguous.
Replay can change positions, pages, queries, head weights, and request ends
without a device-to-host length read. Unsupported shapes retain the old path.

The query Hadamard transform retains BF16 rounding, then converts to FP16 for
Cube. BF16 values outside FP16's range or near its subnormal floor are not
exactly representable. The changed accumulation order is also not bitwise
identical. The feature remains opt-in; the serving qualification below covers
this checkpoint and launch configuration.

## Serving comparison (2026-10-05)

Same four resident workers and weight-storage addresses, CPU affinity,
MTP1, full decode graph sizes `[2,8]`, native mHC, 640-token prefill chunks,
prefix caching, sampling, and 311,040 configured context. The shared top-k
padding correction is present in both modes. Only the selector changed.

| Measurement | Baseline | Live selector | Change |
| --- | ---: | ---: | ---: |
| c1, 256 output tokens | 3.532 tok/s | 4.872 tok/s | +37.9% |
| c4, 256 output tokens each | 6.201 aggregate tok/s | 10.794 aggregate tok/s | +74.1% |
| Strict quality | 17/20 | 17/20 | Same three misses |
| Tool call | Pass | Pass | — |

All five short-suite responses reached 256 tokens. Known misses remained
`instr_reverse`, `instr_first`, and `code_slice`. The candidate retrieved the
8K key correctly: cold TTFT 89.842 s; exact-prefix repeat 7.096 s with 7,680
cached prompt tokens. The user ended the remaining baseline checks and asked
to switch immediately, so the baseline 8K request was canceled and no second
baseline/repeat timing round was run. The throughput gain is from one matched
pair, not an ABBA study or an uncertainty estimate.

Recapture took four seconds and reduced recorded graph allocation from
0.60 to 0.44 GiB. Every worker acknowledged the same candidate and retained
its PID and weight-storage digest. See `native-switch-receipts.json`,
`resident640-results.jsonl`, `native640-results.jsonl`, and
`native640-summary.json`.

Configured 311,040 tokens is **not a tested full-length prompt**. This change
addresses graph decode; it does not accelerate the separate prefill selector.
The final launcher enables live scoring explicitly, retains 640-token chunks,
and disables resident development endpoints for normal serving on port 8001.
Decode throughput and prefill latency remain equally important priorities.
The separate 1,280-token prefill experiment remains pending.

## Hardware gate

- CANN 9.1.0 kernel and supplemental PyTorch binding compiled on Ascend 310P.
- 19 hardware regressions passed, including score parity through 77,760 live
  pools, shared strided storage, invalid pages, zero/negative positions,
  exact ties, graph replay with changing metadata, and the serving wrapper's
  rotation/casts under capture.
- Finite score parity: `rtol=3e-5`, `atol=1e-4` against a CPU FP32 reference.
- 75 targeted CPU tests passed: 18 scorer/selection tests and 57 mHC tests.
- A broader local import-based suite could not load `torch_npu`; its runtime
  dependency checks were retried on the Ascend host: **31 existing
  regressions passed**, separately from the hardware performance measurements.

Initial hardware testing found scalar writes racing vector output masking in
partial tiles. The kernel now uses a vector bit mask. It also found a separate
existing selector bug: at large score widths, 310P top-k can return arbitrary
indices for `-inf` padding, including duplicate in-range IDs. Checking only
`index < completed_pools` admits those entries. `select_kpool_groups` now
checks the result rank and both index bounds. This reproduced without loading
the new operator; see `probe-padding.log`. The regression requires exactly
64 selected pools / 256 unique tokens for a short request at all three widths.

## Measurement method

`tools/glm_perf/benchmark_kpool_live_score_310.py` captures the old and new
selector bodies independently on one 310P, alternates timing order, and records
nine samples of ten graph replays. Both include score construction, top-k,
and token expansion; both use the padding fix. The common query rotation and
KV writes are outside this measurement. These are synthetic operator/selector
results, not model throughput, acceptance, cold TTFT, or full-context serving
validation. A two-row graph corresponds to one verifier request with MTP1;
eight rows model four such requests.

| Graph rows | Live tokens | Configured tokens | Baseline ms | Candidate ms |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 256 | 32,768 | 1.697 | 0.302 |
| 2 | 256 | 131,072 | 5.375 | 0.358 |
| 2 | 256 | 311,040 | 12.308 | 0.486 |
| 8 | 256 | 32,768 | 6.602 | 1.128 |
| 8 | 256 | 131,072 | 21.206 | 1.360 |
| 8 | 256 | 311,040 | 48.955 | 1.874 |
| 2 | 8,192 | 32,768 | 1.690 | 0.320 |
| 2 | 8,192 | 131,072 | 5.418 | 0.375 |
| 2 | 8,192 | 311,040 | 12.341 | 0.508 |
| 2 | 131,072 | 131,072 | 5.695 | 0.762 |
| 2 | 131,072 | 311,040 | 12.737 | 0.899 |
| 2 | 311,040 | 311,040 | 13.467 | 1.361 |

All 12 final cases produced exactly matching expanded indices, including
padding and order, after the shared padding correction. This finite test
does not establish general bitwise score parity. At 311K configured capacity,
two-row capture allocation peaked at 121,150,464 bytes for the baseline and
3,975,680 bytes for the candidate.

The final timing sweep ended at 10:54:30 UTC. An unrelated four-worker
server group started at 10:54:22 UTC, overlapping the final eight seconds.
Those workers were left untouched. Treat the last long-window timings as
provisional; the earlier sweep (ending before that group started) measured
13.414 → 1.322 ms for two rows at a fully live 311K-equivalent window. The
short-request results and their large gain also reproduce across sweeps.

Final measurements are in `operator-qualified-results.json`. Initial results
are retained separately and must not be used as selection-correctness evidence
because they predate the padding fix.

## Next performance work

Leave the validated server available for user testing. Further hardware
experiments must be coordinated around that use.

1. Trace the current decode path to identify remaining expert memory traffic,
   selector/top-k overhead, KDA launches, and MTP verification cost. Select
   work by its contribution to end-to-end c1 and aggregate c4 throughput.
2. Test 1,280-token actual prefill chunks with native mHC. The mHC study
   already measured 5.86% lower matched cold 8K TTFT at 640-token chunks.
3. Measure cold/warm TTFT alongside decode throughput, acceptance, quality,
   and memory for each candidate. Earlier grouped-projection/KDA profiles are
   directional, not measurements of the current MTP server.

## Applying the three optimization questions

| Question | Concrete GLM target | Evidence / next decision |
| --- | --- | --- |
| Faster operators | Native Cube pooled scoring | Matched serving: +37.9% c1 and +74.1% aggregate c4 |
| More fusion | Gather, matmul, ReLU, weighting and reduction on chip | Avoids full gathered-key and head-logit intermediates in device RAM |
| Remove unnecessary work | Skip unused configured cache; rotate all query rows together | Kernel reads live lengths during graph replay |
| More fusion / fewer passes | Native mHC post plus FP16 round | Matched cold TTFT improved 5.86%; larger prefill chunk test pending |
| Amortize expert work | Larger actual expert batches after workspace reduction | Prior 4x640 versus 2560 projection study supports testing; no predicted serving speedup |
| Reduce expert memory traffic | Tile-local unpack and Cube consumption across W2/W3/W4 | Earlier L1 candidate regressed; require complete pipeline reuse and realistic NZ benchmarks |
| Reduce stage overhead | Revisit KDA launches and scratch writes | Reprofile current server first; preserve stage/event correctness on 310P |

Busy devices do not establish that useful arithmetic saturates the hardware.
The next trace should distinguish vector/Cube activity, memory traffic,
communication, and launch gaps. Do not re-run the already-flat decode-table
candidate without a new architectural change.

## Reproduction

- Sources: `csrc/attention/glm_kpool_score_v310/` and
  `vllm_ascend/models/glm5next/ops/kpool_native.py`.
- `build-only.sh` builds an isolated OPP package without launching a server.
- `bindings.cpp` supplies an experimental supplemental binding; do not load it
  alongside a full extension that already registers the operator.
- `test-kernel.sh` uses one NPU and requires hardware authorization.
- Source changes remain on shared main. The remote CANN tree is a build input,
  not a second development branch. Existing unrelated dirty edits are retained.
