# Device-side Mamba commit: private graph qualification

This report records the earlier private qualification. The subsequent
[live integration report](mamba-commit-live-results.md) supersedes its serving
status: actual model binding and state-copy verification now pass, while CPU
bookkeeping remains and the serving A/B shows no clear speedup.

The new AICore kernel computes aligned acceptance decisions and copies the
selected convolution/recurrent states without observing those decisions on the
host. A grouped entry processes all 72 state banks in one native launch, using
eight cores across disjoint banks and preserving request order within each bank.
It passed private graph tests on all four Ascend 310P NPUs. **It is not installed
in the serving model, and these results are not an end-to-end token-rate gain.**

The service remains `qwen38-flash-next` at `http://192.168.53.187:8001/v1`, with
workers 1638646/1638647/1638648/1638649. No restart or model recapture occurred.
One read-only candidate preparation briefly drained inference to inspect shapes;
it was never applied. The previous acceptance-reuse candidate remains active.
The final chat smoke request returned `ready`; all four NPUs were healthy at
60/60/64/63°C, with only the original serving workers remaining on the devices.

## Captured component measurements

| Private 72-bank case | Host-driven copy dispatch | One grouped graph launch |
| --- | --- | --- |
| No alignment boundary, no state copies | 24.4–26.3 µs | 33.6–35.3 µs |
| Two request rows require copies | 22.22–22.62 ms | 0.758–0.763 ms |

Ranges are per-device medians across four alternating-order repetitions of ten
calls. Host dispatch follows the existing request-first loop: it decides once
per request and skips all banks when no copy is needed. CPU acceptance counts
are already available; D2H and live-model metadata lookup are excluded. Captured
timings exclude input staging, correctness checks and D2H. Tests ran in isolated
processes alongside the resident service; they are component wall times, not
exclusive-device kernel latency or serving benchmarks.

The boundary-copy component is roughly 29 times faster in this test. The no-copy
case adds roughly 9 µs, so an end-to-end gain depends on actual boundary frequency,
input staging, stream ordering and contention. Do not multiply this component
ratio into the model's tok/s or claim that the next-token host boundary is gone.

Live inspection found 36 FP16 SD convolution banks shaped `[64,5,2560]` and
36 FP32 recurrent banks shaped `[64,12,128,128]` per rank, across three Mamba
groups, with block size 128. The group test uses the same inner shapes and dtypes
but **four private slots per bank**, versus 64 in serving. This bounds diagnostic
memory to approximately 112 MiB and does not qualify the live slot/archive path.

## Correctness and capture evidence

- Tensor-only planner: 16 changing-input graph replays on NPU 0 exactly match
  an independent scalar oracle, without changing input addresses.
- Individual state kernel: 160 eager/graph checks per device, 640 total, covering
  small/large convolution and temporal banks. Convolution includes overlapping
  copies spanning multiple 16-KiB tiles.
- Grouped kernel: 576 bank comparisons per device, 2,304 total. Each comparison
  checks the complete bank and acceptance/error receipt against a CPU oracle.
  Raw FP16/FP32 bit patterns include non-finite values; comparisons use integer
  views, not floating-point tolerances.
- Changing acceptance, physical block mappings, shared request destinations,
  no-copy rows, and invalid metadata/table indices were tested. Bounds failures
  produce diagnostic receipts and skip that row's copy; serving must never
  ignore such receipts or treat partial completion as successful commit.
- Cold group admission rejects overlapping banks and table/state aliasing.
  The resource retains every pointer target and descriptor across graph replay.
- Scoped host regressions passed: 216 tests. Scoped repository hooks passed;
  the full repository suite and secret-scan hook were not run.

The native kernel copies SD convolution suffixes forward. A suffix's source
address is above its destination within the same bank slot; completing each tile
before the next preserves the fallback's clone/copy behavior. Temporal copies
select the accepted speculative block. Separate banks are disjoint, so cores may
process them independently. Dim-first convolution and unaligned layouts remain
unsupported and require fallback.

## Remaining model integration

1. Bind all live banks through a retained worker-owned resource and qualify their
   64-slot layout without overwriting serving state during warmup/capture.
2. Stage the **compact postprocess mappings** from
   `_prefix_mamba_postprocess_tables`, preserving request order. Scheduler global
   block IDs are not indices into these compact 64-slot banks.
3. Order the sampled acceptance producer, metadata/table staging, commit launch,
   and next model consumer on explicit streams/events. Current CPU count reuse
   can supply diagnostic comparison, but does not eliminate host sampling.
4. Compare actual model commits with the retained fallback at alignment boundaries,
   including C1–C4 and prefix/archive reuse, before any performance promotion.
5. Measure unprofiled C1/C4 and cold prefill. Group replay is still host-submitted;
   device continuation and the distributed DMA ledger remain separate work.

## Artifacts and reproduction

Code is in `tools/qwen4exp/mamba_commit.py`, `native_mamba_commit.cpp`,
`mamba_commit_integer.h`, and `build_mamba_commit.py`. The three
`benchmark_mamba_*_310.py` modules provide bounded standalone gates.

Evidence under `iteration-closure-v1/` includes:

- `commit-plan-graph-v1.jsonl`
- `state-commit-graph-v3-rank0.jsonl` through `rank3.jsonl`
- `group-commit-v4-final-rank0.jsonl` through `rank3.jsonl`
- `mamba-commit-build-v4.json` with frozen source/binary/bridge hashes
- `live-commit-layout-rank*.json` and `layout-inspection.json`

The earlier `group-commit-v4-rank0.jsonl` is superseded for timing: its host
baseline unnecessarily recalculated decisions per bank. The final harness and
regression test preserve the fallback's request-first no-copy behavior. Build
v3 validates the individual entry; build v4 adds the grouped entry. Native
artifacts remain on the hardware host under `mamba-commit-build-v4`.
The exact compiled C++ sources are also preserved in
`iteration-closure-v1/mamba-commit-build-v4-sources/`; subsequent repository
formatting only changed whitespace and line wrapping in the working sources.
