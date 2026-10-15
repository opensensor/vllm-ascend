# Fresh decode projections and host gaps

Status: this document preserves the read-only profile analysis and experiment
design. The subsequent [private GDN projection gate](gdn-projection-private-gate.md)
passed on all four ranks. It remains outside the serving model: its modest
component gain does not justify whole-model duplicate weights without further
memory and end-to-end qualification.

Remote artifact root is
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609`.
The latest rank-0 trace is
`qsa-resident-v1/native-timeline-decode-hc-v1/decode/rank0/decode_rank0_579046_20261014132229718_ascend_pt/ASCEND_PROFILER_OUTPUT`.
Evidence uses `kernel_details.csv`, `op_statistic.csv`, `step_trace_time.csv`
and `trace_view.json`. These are profiled durations, not serving throughput.

## Device exposure and attribution

The trace contains eight MTP2 iterations and 80 graph submissions. Kernel-type
sums are MatMulV2 164.248 ms, TransData 38.762 ms, target W4 gate/up 37.698 ms,
and QuantBatchMatmulV3 21.723 ms. These sums can overlap other work.

| Logical FP16 projection | Calls | Kernel sum | Attribution |
| --- | ---: | ---: | --- |
| 10240→320 | 824 | 34.063 ms | HC low-rank down |
| 320→10240 | 824 | 35.072 ms | HC low-rank up |
| 2560→2560, three rows | 304 | 24.248 ms | Primarily GDN QKV; 288 calls expected from 36 layers × 8 iterations |
| 2560→1536 | 512 | 23.418 ms | GDN Z plus QSA Q/output-gate; mixed bucket |
| 1536→2560 | 400 | 19.984 ms | GDN and QSA output projections; mixed bucket |
| 2560→512 | 512 | 9.623 ms | Router/index-query shapes overlap; do not attribute all to attention |

The largest activation conversions are `[3,2560]` FP16 ND→NZ at
2,752 calls / 11.062 ms and `[3,10240]` at 1,624 / 6.290 ms.
HC up-input `[3,320]` conversion is 824 / 3.182 ms.
Small HC injection weights `[4,10240]` are converted 800 times / 2.235 ms:
the pinned formatter intentionally excludes weights with a dimension below 16.
This static-weight conversion is real but not the largest opportunity.

The LM-head shape is local vocabulary 62080: QuantBatchMatmulV3 has eight
three-row calls / 6.734 ms and sixteen one-row calls / 13.493 ms. These match
target verification plus two draft steps per iteration. The remaining eight
12800-output quantized projections total 1.496 ms and are a separate shape.
The LM head is already dynamic W8A8; this trace does not justify another
precision change or treating the head as an FP16 projection problem.

Pinned `vllm_ascend/models/qwen4_exp/model.py` SHA256:
`1d94ecc195979c01f8b2e61e5ef56c3ce8e6509fd77e4eee64bd625e417592d7`.
Relevant lines:

- 464–468: `_linear` preserves checkpoint FP16 operands on NPU.
- 505–523: persistent NZ weight preparation excludes dimensions below 16.
- 593–611: HC projections, gated mean and injection.
- 750–760: GDN QKV/Z/BA/output projection shapes.
- 1090–1091: GDN QKV and BA consume the same `block_input`.
- 1199–1203: Z consumes that same input after recurrent work; its projection
  has no dependency on recurrent output. Norm/gating/output projection do.
- 1742–1752: QSA separate Q/K/V/gate/index projections.

Shape attribution is an inference from these call sites and layer counts, not
a module-labelled native trace. Preserve mixed-bucket caveats when estimating
how much work a candidate actually covers.

## Host gaps are smaller than the profiler's Free category

The full step summary reports 671.533 ms stage, 453.505 ms computing,
3.522 ms nonoverlapped communication and 214.507 ms Free. Free is not equivalent
to a Python scheduler leaving the device completely idle.

For steady analysis, sort exact `AscendCL@aclmdlRIExecuteAsync` events, group
them ten at a time, and take the interval from group 2's first submission to
group 8's first submission. This excludes first-iteration profiler startup and
the final incomplete boundary. The six intervals are
84.228/82.599/84.834/83.689/81.369/85.607 ms.
Union device intervals across **all** streams, rather than summing overlap:

| Measurement | Six steady intervals |
| --- | ---: |
| Interval span | 502.326 ms |
| AI_CORE/AI_CPU execution union | 371.716 ms |
| Compute or DMA union | 393.768 ms |
| Any recorded device-task union, including waits/control | 446.759 ms |
| No recorded device task | 55.566 ms |
| Host stream-synchronization union | 385.285 ms |
| Host stream sync overlapping compute/DMA | 341.965 ms |
| Host event-synchronization union | 2.715 ms |
| Graph submission API execution union | 0.415 ms |

Thus roughly 11.1% of these intervals has no recorded device task, while
88.8% of host stream synchronization overlaps compute/DMA. Moving or removing
a host wait cannot automatically remove its underlying device dependency.
The 53 ms difference between compute/DMA and any-task unions contains recorded
control/wait work, not proven idle scheduler time.

Largest no-task gaps are approximately 0.79–1.13 ms. They repeatedly overlap
`Event::synchronize`, four host-to-device copies, nine slices and nine
`as_strided` calls. Additional 0.56–0.63 ms gaps overlap copies, slices,
`lift_fresh` and stream synchronization with timeout. This points toward
bounded metadata/input preparation and copy batching as a host-side follow-up.
It does not establish whether each gap belongs to PLE input preparation,
draft eager work or scheduler bookkeeping: the trace has operator names but
no Python function-stack category. One rank also cannot separate cross-rank
dependencies or quantify profiler overhead. Do not report all 214.507 ms as
recoverable scheduler delay or derive a tok/s prediction from it.

The earlier
[overlap audit](qsa-resident-v1/native-timeline-v1/decode/decode-overlap-audit.md)
found a similar 56.997 ms with no recorded task over six intervals. Current
evidence does not show a new host-gap regression.

## Next distinct projection gate: GDN QKV/Z sharing

HC low-rank projection work remains largest, but already-screened approaches
should not be recycled as new findings:

- [HC projection padding](profile-c1/hc-projection-layout/README.md) passed its
  first real module's byte gates but regressed complete decode by 10–15% in
  component timing; no promotion.
- [October 5 screening](../qwen38-decode-next-20261005/EXPERIMENT_RESULTS.md)
  found grouped HC projection slower and no useful ordinary-layout gain.
  Dynamic `npu_format_cast` also hit an uncapturable Identity path.
- [October 10 audit](../qwen38-cast-layout-20261010/README.md) records real-weight
  differences from joined/padded HC injection and warns that repeated baseline
  projection runs themselves need stability checks.

A distinct, not previously located experiment is to concatenate independent
GDN input projections. Start with QKV `[2560,2560]` plus Z `[1536,2560]`, whose
combined output width 4096 remains 16-aligned. Compare a second arm including
BA `[24,2560]` only if the first passes. Leave convolution, recurrent state,
RMS, sigmoid, output projection and collectives untouched.

This changes two or three GEMMs sharing one input into one GEMM. It does not
reduce weight bytes or total dot products. CANN can select a different tiling
or accumulation order at the wider output shape, so unchanged FP16 dtypes do
**not** establish exact numerical parity. Do not silently accept tolerances.

Private gate design:

1. Borrow one real resident GDN module's original weights read-only. Materialize
   a separate concatenated FP16 weight, then format it NZ before timing. Start
   one module at a time. QKV+Z adds 20 MiB per module; all 36 GDN modules would
   add 720 MiB per rank before alignment/allocator overhead. QKV+Z+BA adds
   about 724 MiB across 36 modules. Persistent whole-model duplication needs
   a separate memory admission decision.
2. Use changing private inputs at 1/3/6 tokens, multiple scales, signed zeros
   and rounding-sensitive cases. First check baseline repeat stability, then
   compare each QKV/Z/BA output byte against the separate original projections.
   Check every rank and representative modules before broadening to all 36.
3. Include output splitting and required materialization in timing. A slice
   of `[T,4096]` has row stride 4096; the old QKV tensor's stride is 2560 and Z's
   is 1536. Consumers may require contiguous input or otherwise insert copies.
   A GEMM-only timing that omits these costs is not representative.
4. Capture private graphs using separate persistent input/output storage and
   retained packed weights. Replay changed inputs, compare bytes again, and
   synchronize/reset only owned graphs before releasing buffers. Do not put
   dynamic weight concatenation or format conversion inside graph capture.
5. Preserve old graph/model weight lifetimes. A later model binding must keep
   projected Z alive until its existing downstream consumer, avoid an
   instance scratch cache with stale producer dependencies, and preserve all
   recurrent-state and collective ordering. The initial private gate touches
   no model state.
6. Require a benefit for the complete projection/split/materialization group
   before proposing a guarded model patch. Any byte mismatch rejects the exact
   arm; tolerance-only equality is insufficient for promotion. Follow with
   unprofiled controlled decode/acceptance measurements, not a projected tok/s.
