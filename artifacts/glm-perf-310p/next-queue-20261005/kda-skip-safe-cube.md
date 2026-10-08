# Tenth experiment: remove the safe-gate KDA Cube-score launch

## Source finding

GLM's KDA prefill uses `safe_gate=True`. The 310P host currently launches prepare
stages `0,6,7,8` for both gate modes. Stage 6 reaches `ComputeScores310P`, whose
first condition returns whenever `SAFE_GATE` is true. Stage 7 instead computes
the safe-gate scores through `ComputeRawAqkAkkVector310P` before the solve.

`kda-skip-safe-cube.patch` adds the opt-in **host compiler** definition
`GLM_KDA_SKIP_SAFE_SCORE_CUBE`. Within the existing 310P split-stage branch, it
skips stage 6 only for safe gating. This removes its executor node, dependency
token allocation, and physical kernel launch. It changes no score arithmetic.
The other gate mode and other hardware branches retain their existing paths.

The patch is an experiment artifact; it has not been applied to shared source.
`kda-skip-safe-cube-base.json` guards the host source and both device files that
establish the no-op premise. Re-audit if those hashes change.

## Dependencies preserved

| Path | Prepare stages | Remaining core stages |
| --- | --- | --- |
| Baseline | 0 → 6 → 7 → 8 | 1 → 4 → 2 → 3 → 5 |
| Safe-gate candidate | 0 → 7 → 8 | 1 → 4 → 2 → 3 → 5 |
| Other gate mode | 0 → 6 → 7 → 8 | 1 → 4 → 2 → 3 → 5 |

Skipping happens before `launchStage`, so stage 7 consumes stage 0's dependency
token. The existing explicit FP32-score cast/copy barrier before stage 1 is
retained. This is nine to eight **core-stage** launches per KDA call; gate
cumsum, casts, copies and other executor operations are additional work.
There is no stage fusion, recurrent-state change or reduced precision.

The physical launch saves overhead, not a score matrix multiplication: that
multiplication already does no work in this mode. No NPU or serving speedup
has been measured. Larger expert batches and eliminating repeated weight
expansion remain higher-impact opportunities.

## CPU checks

`tests/ut/glm_perf/test_kda_skip_safe_cube.py` passes **4 tests**:

- Flag absent: preprocessing the host source with includes stripped produces
  identical tokens to baseline.
- Actual extracted host launch lambda and stage loops, compiled against a mock
  executor with the flag off and on: stage order, dependency tokens and the
  cast barrier match expectations for both gate modes. Every stage allocation
  failure and null kernel result terminates before any later stage is launched.
- Actual extracted `ComputeScores310P` compiled with counted Cube calls: safe
  gating makes zero calls for empty, partial and full chunks; the other gate
  mode retains full-chunk Cube work for supported widths.

These tests do not validate native CANN graph ordering or device output parity.
Scratch allocations remain unchanged; removing potentially unused score scratch
needs a separate consumer/workspace audit.

## Native gate when hardware is available

1. Apply-check against the intended source snapshot, then rebuild the host
   op-api with the definition. A device-kernel-only define cannot enable this
   host scheduling change. Keep the device binaries and all other flags fixed.
2. Require exact attention-output and final-state parity on identical inputs:
   dense and variable-length batches, empty sequences, chunk tails, optional
   initial state, both safe-gate modes, and repeated execution.
3. Exercise sequential long prefills across the earlier stall boundary and
   subsequent MTP decode; verify no hangs or state divergence.
4. Measure isolated KDA and cold 8K / greater-than-20K TTFT. Retain c1/c4 checks,
   the same context limit and scheduler batch. Promote only with a repeatable
   benefit; do not infer a model speedup from the launch count.

The host library needs a qualified load path: a fresh process or an independently
versioned native operator integrated into the resident harness. Do not overwrite
a loaded OPP or assume a Python method replacement updates its host library.

```bash
git apply --check artifacts/glm-perf-310p/next-queue-20261005/kda-skip-safe-cube.patch
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest --noconftest -q \
  tests/ut/glm_perf/test_kda_skip_safe_cube.py \
  tests/ut/glm_perf/test_next_optimization_queue.py
```

No NPUs, remote hosts or running servers were accessed for this experiment.
