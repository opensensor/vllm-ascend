# T6: bounded eager MoE communication schedule

Implemented immutable `SchedulePolicy` and `SchedulePlan`, pre-admission all-rank
receipt validation, `ScheduledMoE`, and an explicitly invoked NPU adapter. The
default chunk size is the reference's 2,560 tokens. Optional smaller chunks are
separate candidates with their own numerical/performance gates.

The schedule computes routing once for the batch and invokes the complete grouped
local expert callback once per fixed chunk. FP32 TP-sharded shared output is added
to local routed output before reduction. Replicated shared output is added only
after reduction completion. The `none` policy performs routed reduction alone.
These placements match the baseline whole-MoE addition/reduction order; changing
shared GEMM chunk geometry can still change real hardware rounding and must pass
independent quality/parity gates.

Each invocation owns at most two local reduction scratch slots and one fixed full
FP32 output. Completed chunks are copied into their final output views after the
completion dependency, before local slot reuse. There is no retained list of all
chunk outputs and no concatenation. Tensor slots are not stored on 48 separate
MoE modules between calls. The explicitly supplied stream context may persist;
tensor allocation is per invocation and queued allocator ownership guards release.

Scratch allocation is not a claim that total memory consists of only two buffers.
The full output remains necessary. An out-of-place all-reduce can also retain up
to two returned buffers, and current local/shared/combine tensors and T5 kernel
scratch remain live. With default 2,560-token scheduler batches there is only one
2,560-token chunk, so this default communication schedule does not by itself create
compute/communication overlap. Smaller gated profiles can create multiple chunks
and more collectives. The local-slot copy and final-output copy add logical D2D
traffic; whether the complete architecture gains overall must be measured.

Saved admission receipts require all four ranks exactly once, worker identities,
one execution namespace, exact generation and the complete plan hash. The immutable
plan includes chunk size, in-flight bound, shared placement, TP, maximum input
capacity and source/arithmetic/reference identities. This validation occurs before
the first route, allocation or collective, without new RPCs per forward. Identical
future input row counts across TP ranks remain the existing engine scheduler
contract: saved static receipts do not inspect future tensor shapes or contents.

The NPU adapter is lazy and explicit. It records producer readiness, allocator
ownership of the local tensor on the communication stream, completion events,
and reduced-result ownership on the main stream. Main-stream completion waits
precede output copies and slot rewriting. It imports no device runtime at module
import. It requires an explicitly owned stream context; it creates no global
streams or persistent per-layer tensor slots. Capture/decode fallback remains the
outer candidate's responsibility.

On errors, no new collectives or result publication occur. A compute/copy failure
can drain known completions once. An uncertain submission or first failed wait
stops all further drain work, preserves unresolved status and poisons the invocation
owner. This does not operate server recovery, resume dispatch or confirm asynchronous
HCCL errors: the outer controller must hold service and establish safe completion.

Validation: 47 CPU tests passed, including independent deterministic four-rank
reduction arithmetic, all shared placements, full/tail/zero rows, identical
collective sequences, one/two in-flight bounds, slot reuse, rank mismatches before
work, and enqueue/wait/local/store/shared failures. Scoped Ruff and formatting pass.
No NPU adapter execution, device allocation, native loading, server query or
runtime mutation occurred. Hardware overlap, all-rank ordering, numerical quality,
thermal behavior and end-to-end performance remain pending T9.

## T8 API

Construct `SchedulePlan(policy, generation, source_sha256, arithmetic_sha256,
reference_sha256, max_input_tokens=2560, tp_size=4)` and obtain saved rank receipts
containing `rank`, `pid`, `execution_namespace`, `generation`, and `plan_sha256`.
Instantiate `ScheduledMoE` with explicit route/local/shared, reduction submission,
completion wait, ordered store and allocation callbacks. `run(inputs)` returns
FP32; the outer candidate performs the baseline final output cast.

For device execution, `npu_streaming_prefill(module, inputs, plan, rank_receipts,
route=..., local=..., context=DeviceScheduleContext(owned_deferred_stream, ledger))`
uses the admitted complete T3/T4/T5 local callback and the module's existing shared
expert/reduction functions. This entry point must not run before real hardware
evidence admission. No measured speed or temperature improvement is asserted.
