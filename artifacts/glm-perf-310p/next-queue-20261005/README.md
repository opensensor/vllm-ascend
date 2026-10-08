# GLM next five experiments — 2026-10-05

Hardware follow-up: [qualification and comparison record](hardware/README.md)
tracks the subsequently authorized NPU run. The preparation notes below describe
the earlier CPU-only staging phase.

Latest: [completed-pool prefill](completed-pools/README.md) removes redundant
compression and state-write preparation. It passes 82 CPU tests, 24 isolated
NPU parity cases and seven serving checks; the candidate is running on 8001.
Cold 8K TTFT is 88.2 seconds, a modest change from earlier roughly 90-second
runs, rather than the large factor seen in isolated writer timing.
The machine-readable queue records the first ten candidates' measured outcomes
so flat or rejected experiments are not mistaken for untested work.

Follow-up: a [sixth candidate](extra-memory-traffic.md) now stages FP16 expert
output reordering before FP32 weighting. The machine-readable queue includes it;
the original five below retain their independent gates.

A [seventh candidate](indexer-projection.md) stages shared-input indexer
projection fusion with target/MTP preparation before resident graph capture.

An [eighth candidate](direct-route-tokens.md) removes expanded route token IDs
and their gather in a private GLM dispatch path, preserving the shared Qwen code.

A [ninth candidate](kda-prepare-head.md) hoists repeated scalar exponentials,
bias loads and constant-vector setup out of KDA prefill's token loop.

A [tenth candidate](kda-skip-safe-cube.md) removes a no-op Cube-score launch
in the 310P safe-gate path while retaining the dependency chain.

Qwen owns the NPUs during preparation. This batch made no server requests,
device calls, restarts, worker patches, or native library loads. All candidates
are experimental. No speedup is claimed; native compilation and hardware gates
are pending. The native source must compile before reserving hardware time.

## Queue and readiness

| Experiment | Target | Prepared | Remaining before serving comparison |
| --- | --- | --- | --- |
| Larger actual expert batches | Cold prefill; amortize expert unpacking | Checked 640/1280/2560 scheduler and route-cap profiles | Build matching route cap; memory fit; runner reallocation/launch |
| Adaptive expert teams | Prefill; parallelize small expert groups without starving hot experts | C++ ownership helper; opt-in integration patch; CPU coverage test | Native build; W2/W3/W4 parity and mixed-routing timing |
| Batched KDA Q/K normalization | Decode; one norm and one input assembly | Reversible Python resident candidate; CPU output/state tests | NPU parity at graph rows 2/8; recapture; c1/c4 |
| Batched selector bookkeeping | Decode at large configured context | Reversible Python resident candidate; CPU exact-index tests | NPU exact indices, ties and padding; recapture; c1/c4 |
| Fused KDA gate/beta preparation | Decode; remove intermediate writes and launches | AscendC source, versioned wrapper, build preparation and numerical gate | Native compile; exact numerical gate; graph/state tests |

Print machine-readable profiles without accessing devices:

```bash
python -m tools.glm_perf.optimization_queue
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest --noconftest -q \
  tests/ut/glm_perf/test_next_optimization_queue.py
```

CPU result: **24 passed**. The C++ test checks each active expert's output tile
is owned once, for 1–8 cores, invalid and valid lane counts, empty groups, mixed
groups and hot experts. This establishes ownership, not NPU kernel correctness.

## 1. Larger actual expert batches

Keep the expert kernel schedule fixed for this comparison. Change both:

| Scheduler tokens | Compiled `GLM_W2_GROUPED_MAX_ROUTES` | HF override `ascend_glm_grouped_max_routes` |
| --- | --- | --- |
| 640 control | 6144 | 6144 |
| 1280 first candidate | 10240 | 10240 |
| 2560 conditional candidate | 20480 | 20480 |

Top-8 routing needs eight route slots per token. The current 6144 route cap
would split a 1280-token expert batch into 768 and 512. The queue rejects that
configuration. Verify actual iteration context-token counts, not just launcher
arguments. Keep the scheduler, Python chunk limit, and installed native cap in
agreement. `serve-candidate.sh` accepts the grouped route cap as argument 16;
do not leave it blank for this test.

This needs a runner allocation change and possibly a launch; the resident
Python harness cannot resize those buffers. Keep configured context at 311040,
MTP1, full graphs `[2,8]`, max-seqs 4, CPU affinity, prefix-cache policy and all
other qualified switches unchanged. Try 1280 first. Admit 2560 only if measured
workspace and KV memory fit at the same context. Do not silently reduce context.

Previous projection batching results already show reuse potential. A previous
1280 trial also changed resident W3 dispatch and hurt c4; it does not isolate
this experiment. The new comparison must isolate scheduler/route capacity and
include decode after a long prefill.

## 2. Adaptive expert teams

`adaptive-expert-teams.patch` adds an opt-in compile flag:
`GLM_W2_GROUPED_ADAPTIVE_EXPERT_LANES=2` or `=4`. It is mutually exclusive with
the earlier fixed-team flag. The patch includes its new header and records the
current source hash in `patch-base.json`. It has **not** been applied to the
shared source tree. Check/rebase it against the build snapshot before compiling:

```bash
git apply --check artifacts/glm-perf-310p/next-queue-20261005/adaptive-expert-teams.patch
```

Groups with at least 128 expert rows retain all cores. Small groups are assigned
by their active small-group ordinal; empty groups leave no holes. Calls with at
most 64 total routes retain the original all-core schedule, including qualified
MTP1 decode. Physical-core workspace indexing is unchanged. Counts are read
from existing device boundaries; no host count transfer is added.

Test baseline, lanes 2 and lanes 4 on identical routing, including mixed
1/127/128/129/640-row experts, empty experts and a single hot expert. Verify
both gate/up and down, W2/W3/W4, NZ layout, and actual serving route sizes.
Uniform-only gains are insufficient: fixed two-core teams previously regressed
the single-hot-expert case badly. Hold route capacity and resident/GM dispatch
constant when comparing this change.

## 3–4. Python resident candidates

Candidate source files:

- `tools/glm_perf/resident_candidates/kda_input_preparation.py`
- `tools/glm_perf/resident_candidates/kpool_decode_epilogue.py`

The Q/K candidate concatenates the two strided inputs along their leading
batch dimension, invokes the existing native normalization once, and takes
contiguous views. It replaces only the five preparation assignments in a
private copy of `_run_recurrent`, checking their AST first. The rest of the
current recurrent function is retained, including accepted-token slot tables
and the validation that previously protected against MTP repetition. A source
change to the preparation block causes candidate preparation to fail closed.
This candidate affects recurrent decode, not chunked KDA prefill.

The selector candidate keeps one `[1, capacity]` top-k per row with the same
full capacity. It batches rank masking, expansion, tail construction and output
copy. It retains the rank mask needed when top-k padding aliases valid pool
indices. Unsupported scorer paths call the original method. CPU checks include
ties, negative padded positions, pool boundaries and capacity 77760 (311040/4).

Use the existing resident harness with each file's `replacements()` factory.
Synchronize the helper modules under `tools/glm_perf` to the worker import path
before preparing. Apply each candidate independently, while paused, with graph
recapture and unchanged worker/weight-storage receipts. Roll back through the
harness with an empty candidate. Failed capture must leave the server paused.

## 5. Fused gate and beta

`kda-gate-beta.cpp` computes FP32 bounded gate sigmoid and FP16 beta sigmoid in
one launch. It uses load-time cached FP32 scale/bias and writes only final
outputs. It does not change recurrence arithmetic. Supported initial geometry:
1–8 decode rows, heads divisible by 16, head dimension 128, contiguous FP16 or
FP32 raw operands. BF16, strided inputs, stale cached weight identities and
unsupported shapes retain reference preparation. Constants and tilings are
created when the resource is prepared, outside capture.

Generate a fresh CPU build directory on the qualified host:

```bash
python artifacts/glm-perf-310p/next-queue-20261005/prepare-native-build.py \
  --output /home/matteius/experiments/glm-kda-gate-beta-v1-build
bash /home/matteius/experiments/glm-kda-gate-beta-v1-build/build.sh
```

These commands are **staged, not executed**. The generated build uses the
existing CANN 9.1.0 and qualified venv paths; check them before compilation.
It registers a new `glm_kda_prepare_v1` namespace and refuses to overwrite
compiled outputs. Use `GateBeta` from `tools/glm_perf/kda_gate_beta_native.py`
with the actual local head count and layer lower bound, then run
`validate-kda-gate-beta.py:validate`. That gate requires exact outputs; failures
need arithmetic investigation, not an automatic tolerance relaxation.

Only after successful compilation and isolated qualification, generate a
hash-verified native manifest for the existing harness. The resource key is
`kda_gate_beta_v1`; the independent candidate is
`tools/glm_perf/resident_candidates/kda_gate_beta.py`. No load manifest is
provided with unbuilt artifacts. After isolated parity, qualify graph replay,
recurrent carries and MTP accept/reject transitions before serving tests.

## Hardware order and decision rule

1. Run cheap isolated correctness gates first. Reject failures before recapture.
2. Test the two Python decode candidates independently with resident weights.
3. Test native gate/beta after its build and numerical gate pass.
4. Test adaptive teams in isolation, then larger expert batches with matching
   native capacity. Bundle launch-required builds to minimize reloads.
5. Combine only winners and repeat the serving comparison once.

For decode candidates, use paired c1/c4 256-token runs in both orders and report
MTP acceptance alongside useful output tok/s. A first-window improvement alone
is insufficient: the previous query-rotation gain did not repeat. Do not repeat
that isolated rotation trial as a new optimization.

For prefill candidates, use matched cold 8K and a prompt beyond the former
approximately 16.6K stall boundary (at least 20K), recording actual prompt tokens,
TTFT, peak memory and following c1/c4 decode. Use distinct prompt content to
avoid accidental prefix hits; verify zero cached tokens in timing output.
Retain the existing quality cases and MTP repetition checks for arithmetic
changes. Configured 311040-token context remains **not full-length validated**.

No candidate is promoted from CPU results or isolated operator timing alone.
