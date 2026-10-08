# Ninth experiment: prepare KDA gate constants once per head task

## Concrete redundant work

In `csrc/attention/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp`, `ProcessChunk`
calls `ApplyGate` once per token. For a safe gate, that function currently:

- reloads the same head's FP32 bias vector;
- evaluates `ExpScalar(ReadFloat(aLog_, hv))`;
- fills the same vector of ones before the sigmoid division.

Those operands are immutable during the kernel call. `ExpScalar` itself uses
vector setup, an exponential, barriers, and a vector-to-scalar event before
reading the result. This repeats for every token even though the head is fixed
through the task's token/chunk loops.

The native patch `kda-prepare-head.patch` introduces the opt-in compiler flag
`GLM_KDA_GATE_PREPARE_HEAD`. It performs all three preparations once per task,
then reuses the operands. The row's gate sigmoid, multiplication order, FP32
accumulation and accumulator reset at every chunk boundary remain unchanged.
It preserves the existing per-row gate-value exponential; only the redundant
scalar exponential of the fixed head parameter is hoisted.

## Expected operation counts

With GLM's variable-length path, a task covers one sequence/head across all its
chunks. For a single 640-token sequence with 16 local heads:

| Work | Existing | Candidate |
| --- | ---: | ---: |
| Scalar exponential of A_log | 10,240 | 16 |
| Bias-vector loads | 10,240 | 16 |
| Vector-of-ones setup | 10,240 | 16 |

These counts assume safe gating with both A_log and bias present. For multiple
sequences the candidate count is `nonempty_sequences * heads`. The dense path
assigns tasks per chunk/head, so its 640-token case with chunk size 64 prepares
160 times instead of 10,240. The optimization does not alter task ownership or
try to share UB across tasks/cores. Empty sequences perform no preparation.

The extra bias buffer is 256 FP32 elements, **1 KiB per active core**, using the
existing maximum gate-row width. No persistent device-memory cache or context
capacity change is introduced. This is distinct from the queued decode
gate/beta fusion: it optimizes the existing prefill gate-cumsum kernel.

The counts are established from source and CPU simulation, not measured NPU
time. They do not imply a 640x KDA or model speedup. Other gate arithmetic,
weight expansion and KDA stages remain.

## CPU checks completed

`tests/ut/glm_perf/test_kda_prepare_head.py`:

1. Applies the patch only to a temporary copy after checking the recorded source
   hash. With the flag absent, C++ preprocessing produces the same kernel tokens
   as the baseline.
2. Compiles and runs the actual extracted `ApplyGate`, `ProcessTask`,
   `ProcessChunk` and new `PrepareHead` methods in a small CPU AscendC arithmetic
   simulation. Baseline/candidate FP32 outputs match bitwise. Cases include
   640x16x128, partial chunks, zero-length sequences, changing heads, absent
   optional operands, dense/variable-length scheduling, and safe-gate disabled.
   Counters verify the expected exponential and bias-load reductions.

**Both tests pass.** This is host C++ simulation, not native Ascend compilation.
It does not validate device event ordering, vector exponential approximations,
or UB resource allocation.

## Remaining gates and deployment

The patch has not been applied to shared source. `kda-prepare-head-base.json`
records its base and compiler flag. Validate against the intended build:

```bash
git apply --check artifacts/glm-perf-310p/next-queue-20261005/kda-prepare-head.patch
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest --noconftest -q \
  tests/ut/glm_perf/test_kda_prepare_head.py
```

Next steps, only with hardware available:

- Build the candidate in the qualified native build environment. First verify
  resource usage and native numerical/event correctness against the original
  gate-cumsum operator, including repeated replay, sequence tails and optional
  operands. Require exact FP32 gate-cumsum outputs.
- Compare full KDA attention outputs and recurrent carries on identical inputs.
  Preserve all stage boundaries and MTP state handling.
- Measure gate-cumsum time first, then cold 8K and greater-than-20K TTFT with
  unchanged context, batching, decoder and other candidates. Retain c1/c4 checks
  after prefill to catch state or memory regressions.
- This changes code reached through the existing native KDA operator. It needs
  a qualified package at launch or explicit versioned native integration; a
  Python method swap alone cannot apply it. Do not overwrite a loaded OPP.

No NPU use, native compilation, live patching, or server changes occurred.
