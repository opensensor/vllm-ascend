# Offline follow-up: make Qwen streaming competitive

Keep the image-enabled baseline running while the user tests. This plan needs no
live-server mutation or NPU execution. Preserve v5 and its failed hardware gate;
create a new ABI, contract and bundle for subsequent candidates. The immediate
target is to remove regressions in the complete expert path before integrating
additional GDN, shared-expert or communication changes.

## Evidence boundary

The measured 2,560-token expert partial is 158 ms versus 45 ms, with exact output.
Source inspection explains plausible overheads but does not assign measured time
or heat to individual instructions. The reference OPP is retained compiled code;
confirm its actual dispatched specialization in a later isolated trace rather
than assuming a source template equals the running binary.

Source anchors:

- [Candidate projection](../../../../tools/qwen4exp/qwen_streaming_projection.h):
  `Run`, `Consume`, `StageMetadata`, `Process`, `PrepareExpert`.
- [Operand staging](../../../../tools/qwen4exp/qwen_streaming_operands.h): `Produce`.
- [Memory contract](../../../../tools/qwen4exp/streaming_memory.py): `regions`.
- [Reference schedule](../../../../csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel/native_int4_schedule.h):
  `ProductPair`, `Project`, `CorrectRows`, boundary cache.
- [Reference dispatch](../../../../csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel/qwen_w4_a8_int4_matmul_v310.cpp):
  `RunSchedule` and projection-width selection.

## 1. Remove vector instruction inflation first

**depends_on**: []

The candidate issues three banks times eight N16 strips: 24 short metadata casts
for each G128 group and row tile at N128. The ordinary reference template issues
one contiguous cast of all three banks per group; its optional cached variant
also has short casts. The candidate then repeats each of eight correction
operations per live row: 128 calls at M16, plus the weight-sum scaling call.

Convert all metadata once per expert/output tile into an FP32 cache. Preserve
FP16 input conversion and the multiply-by-eight weight-sum operation exactly.
Consume strided metadata directly without repacking every group. Choose between
row-oriented correction for sparse tails and N16-strip-oriented correction for
larger row counts. For M32/N160, strip-oriented correction needs approximately
80 arithmetic calls per group versus 256 row-oriented calls. These are source
instruction counts, not cycle or bandwidth estimates.

Preserve ascending G128 accumulation and every FP32 correction operation order.
The paired Cube result interleaves low/high limbs by strip: flattening it as if
it were the reference's separate dense matrices would be incorrect. Add CPU
indexing fixtures for every element, partial rows, odd tails, both limbs and
changing inputs. Keep synchronization between dependent vector stages.

## 2. Widen tiles without carrying unused epilogue scratch

**depends_on**: [1]

The fixed M16/N128 candidate uses more row tiles than the bulk reference M32
schedule. At gate/up N1280, ten N128 output tiles distribute unevenly across
eight cores: two cores handle two tiles, six handle one. N160 yields eight gate/up
tiles and sixteen down N2560 tiles. The current reference source selects N160
when width divides eight times 160; it also supports sparse M16 and dense M128
specializations, so verify dispatch for each workload before comparison.

Separate projection-only scratch from future on-core activation scratch. The
current full contract reserves a 32 KiB quantizer and persistent gate storage even
though the native projection does not use them. Keep a distinct full fused-stage
contract for future work; do not remove buffers from a stage that actually needs
those lifetimes.

Current static UB formula, with two slots and MAX_GROUPS=20:

`UB_full = 40*M*N + 384*M + 144*N + 33536` bytes.

Projection-only removes `32768 + 2*M*N` bytes. An FP32 metadata cache replacing
the FP16 cache adds `120*N` bytes; eliminating two per-group metadata stage
buffers removes `24*N` bytes. A third activation-metadata slot adds `64*M` bytes.

| Shape and scratch policy | UB bytes | Fits usable 253,952 bytes? |
| --- | --- | --- |
| M16/N128, current full contract | 140032 | Yes |
| M32/N128, current full contract | 228096 | Yes |
| M32/N160, current full contract | 273664 | No |
| M32/N160, projection only | 230656 | Yes |
| M32/N160, projection only, FP32 metadata without stage buffers | 246016 | Yes |
| Previous row plus third activation-metadata slot | 248064 | Yes, 5888 bytes remain |

The limit already reserves 8 KiB of nominal 256 KiB UB for SDK scratch. This
arithmetic establishes static feasibility only; it does not prove buffer
lifetimes, compiler hidden scratch or execution correctness. Retaining the old
metadata stage buffers would exceed this proposed budget.

For M32/N160, two-slot L1 activation plus resident weights needs 212,992 bytes;
L0A needs 8,192, L0B 20,480, and CO1 40,960 bytes, all below current modeled limits.
Any additional physical activation/L0 slots need a separately recomputed budget.

M32 requires actual loader/index changes. The current operand helper asserts
M equals BLOCK and loads only one M block per limb. Implement the explicit M/BLOCK
loop, update the limb and M-block offsets and the activation byte-index helper,
and test them independently. Changing only constants would silently break data
layout. Generate a new contract hash and entrypoint version before compilation.

## 3. Recover operand prefetch overlap with proven ownership

**depends_on**: [1, 2]

The candidate completes `Produce` and metadata staging before `IssueCube`, then
consumes the prior product and reads back the current one. It overlaps Cube with
vector work, but does not prefetch the next operands under Cube. The reference's
`ProductPair` already loads the next weight group before its Cube wait. Both
paths already retain an expert/output weight tile in L1 across row batches.

Extend the event/lifetime model first. The two-slot activation metadata aliases
producer j+1 with consumer j-1; simply moving production earlier races. Consider
separate producer/consumer metadata ownership or a third activation-metadata
slot, then prove all UB/L1/L0A/L0B producer releases independently. CO1 remains
single-owner until readback completes. Event IDs cannot alias an outstanding
generation. Preserve startup, drain, expert changes and tail behavior.

Demonstrate legal producer j+1 / Cube j / consumer j-1 traces with delayed,
out-of-order CPU event fixtures. Never replace this proof with blanket barrier
removal. A later hardware trace must establish actual overlap, not just buffers
that could permit it.

## 4. Cache control metadata and reduce window launches

**depends_on**: [2]

`Process` scalar-loads every expert end on every core for every projection window.
Use the already reserved 256-byte ENDS arena as a 32-boundary DMA cache, as the
reference does. This is NPU scalar GM traffic, not a device-to-host `.item()`
transfer. Preserve monotonic boundary validation and zero peer output.

The candidate down projection uses three windows `[8, 8, 4]` at N128, three
finalizers and three copies into the full output. The largest routed FP16 buffer
falls from 125 MiB to 50 MiB, but total routed write/read payload remains and
extra full-output copy traffic survives. N160 permits two eight-tile windows of
1,280 columns. Consider a single full-width window only for small-row shapes when
its complete rank memory envelope is proven; keep bounded windows for bulk loads.

Record allocations and logical payload separately from profiler bus traffic.
Benchmark window-count changes independently from arithmetic changes.

## 5. Isolate and repair WY correctness before fusion

**depends_on**: []

Keep the reference WY path in every streaming performance experiment. The failed
native WY gate needs saved exact Q/K/V, cumulative gates, beta, W/U, chunk states,
outputs and initial/final recurrent states. Run each path from independently
cloned initial states. Find the first divergence using a small FP64 CPU reference
and stage-level checks. The blocked inverse/matmul candidate and forward-
substitution reference have different reduction orders; this is a hypothesis,
not a demonstrated explanation for the failure.

Add adversarial gates, normalized Q/K, near-zero FP16 outputs, cancellation,
chunk boundaries and TP-local/unsharded shapes. Require downstream output and
final-state gates even when component tolerances pass. Do not relax acceptance
thresholds to hide the observed discrepancy.

## Later hardware gates

The user's testing owns the current service. No further NPU diagnostics are
part of this offline follow-up. When testing resumes, keep candidates in
isolated versioned bundles and run, in order:

1. CPU contract/layout/event tests, scoped lint and contained host-only build.
2. Raw kernel parity and changed-input graph stress, including zero/tail rows.
3. Real-weight expert ABBA at 128/2560 tokens, measuring stages and full expert
   path separately. Stop if the complete path still regresses.
4. Profiler attribution: dispatched tile shape, instruction counts, DMA payload,
   pipe utilization, barrier waits and per-core imbalance. Capture profiling
   outside performance timings and bind traces to source/binary identities.
5. All-bank/layer/rank real-activation parity and complete model gates: images,
   shared expert, EP/HCCL, MTP, graph replay, prefix CoW and cancellation.
6. Sustained mixed-service and thermal qualification under the existing
   94C hold / all-core 85C resume / independent 96C cutoff. Include hold time.

A faster isolated projection is insufficient for promotion. Require the original
T9 service and quality criteria, retained capacity, and no repeatable decode or
mixed-service regression. Keep the baseline if a defensible overall gain is not
established. Do not predict a speed multiplier or thermal improvement from this
static plan.

## Execution update

The October 9 follow-up implements steps 1–4 in a new v2 projection contract and
adds step 5 snapshot/oracle diagnostics while retaining reference WY. The original
hardware WY failure remains unresolved. Qwen six-chip padding, load/state and
vocabulary alignment are staged separately. See the
[implementation report](../../../qwen38-streaming-followup-20261009/REPORT.md)
and its validation boundary. No candidate has been promoted or installed live.
