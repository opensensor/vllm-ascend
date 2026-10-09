# T5: builtin activation handoff and bounded finalization

T5 implements a complete offline Python pipeline using the T4 native column
projection seam. It preserves the selected builtin FP16 nonlinear arithmetic
and existing CANN-v2 finalizer boundaries. It does not implement on-core
SwiGLU, native hidden-packing fusion, or device-side route reduction. No NPU
was opened and no server was queried or changed.

## Concrete pipeline and interfaces

`run_streaming_epilogue(projected, dispatch, weights, group_ends, down_bank,
*, activation, pack, columns, finalize, complete, plan=None)` consumes
FP16 projected rows with 1,280 columns. `activation(projected)` is the
selected builtin FP16 SwiGLU callback and must produce FP16 rows with 640
columns; an FP32 replacement is rejected. This preserves the explicit FP16
projection and nonlinear boundaries.

`pack(hidden, group_ends)` is called exactly once. Its result is two INT8
packed limbs of shape `[physical_rows, 320]` plus FP32 scale and sum banks
of shape `[physical_rows, 5, 8]`. The callback may use device ends to pack
only the local prefix if that implementation has separately passed its gates.
The existing baseline quantizer packs all physical rows. T5 never silently
changes the quantizer, its extent or rounding. Inactive capacity must remain
unread by down through the unchanged device ends.

`columns(down_bank, prepared, group_ends, first_tile, tile_count)` calls the
T4 projection window while preserving the complete resident weight bank,
its full output stride and the complete hidden axis. The default immutable
`WindowPlan` splits 2,560 output columns into tiles 0–7, 8–15 and 16–19,
or windows of 1,024, 1,024 and 512 columns. The last window uses only four
of eight output owners; extra launches and reduced core occupancy are explicit
tradeoffs, not presumed savings. Sparse down/reduce remains a separate path.

`finalize(routed_window, dispatch, original_weights)` invokes the unchanged
selected baseline finalizer separately for each window. Dispatch and original
weights are passed by identity; no new sort, route order, scale precision or
addition order is introduced here. CANN-v2's FP16 combined output is widened
by that callback to FP32. Each contiguous returned window is copied to its
original column interval in the complete token output.

`complete(stage, tensor)` establishes the required dependency on the existing
stream before a consumer is submitted or references are released. Same-stream
FIFO ordering or a device event is sufficient; this callback must raise on
failure. The orchestration itself adds no host synchronization, device value
reads or CPU route decisions. Python return and recorded epochs do not prove
that asynchronous native tasks have completed. T6 owns dependencies across
streams. No partial `PipelineResult` is returned on failure.

The immutable result reports output, window ownership, dependency epochs and
logical buffer bounds. Unknown backend scratch is explicitly unmeasured. Each
routed/finalized window reference is removed before the next window callback,
preventing two routed tensors from coexisting during callback evaluation.

`qwen_streaming_epilogue.h` supplies matching host/device-safe column bounds,
byte/index mappings and gate/up ownership columns. It does not launch a kernel
or replace the builtin nonlinear function. Gate and up column owners can differ;
the full projected-row launch/stream boundary remains necessary. On-core paired
nonlinear fusion is deferred until ownership and arithmetic can be proven.

## Surviving GM boundaries and logical costs

The pipeline retains projected FP16 gate/up, builtin FP16 hidden activation,
packed hidden limbs and scale/sum banks, one FP16 routed down window, native
finalizer casts/combined FP16 output, and the complete FP32 token output.
`epilogue_buffer_bounds` includes known inverse/scales conversion storage and
both finalizer output dtypes; builtin/CANN internal scratch, allocator reserve,
weights, attention, state and rank resources belong to separate admission bounds.
The input is conservatively retained throughout this estimate.

At maximum physical capacity, 25,600 rows and 2,560 tokens:

| Boundary | Logical allocation bytes |
| --- | ---: |
| FP16 projected gate/up | 65,536,000 |
| FP16 builtin hidden activation | 32,768,000 |
| Packed hidden limbs | 16,384,000 |
| FP32 hidden scale/sum lanes | 8,192,000 |
| Largest FP16 routed window | 52,428,800 |
| FP32 finalized window | 10,485,760 |
| FP16 finalizer output window | 5,242,880 |
| FP16 route scales and INT32 inverse | 153,600 |
| Complete FP32 token output | 26,214,400 |

The routed-output allocation bound falls from 131,072,000 bytes for the full
width to 52,428,800 bytes for the largest window, a difference of 78,643,200
bytes (75 MiB). This is one tensor bound, not measured whole-rank peak memory.
The total worst-case down writes and finalizer reads remain 131,072,000 bytes
each. The new column store adds 52,428,800 logical read/write bytes. Three
projection/finalizer calls replace one full-width call, and known finalizer
casts are repeated three times. Saved `logical-costs.json` exposes these costs
without a predicted speed multiplier or physical bus measurement.

## Offline validation and limits

`OMP_NUM_THREADS=2 python -m pytest --noconftest -q
 tests/ut/qwen38_1m/test_streaming_epilogue.py`: 41 passed.

The tests run the complete projected FP16 → FP16 activation → packed limbs
→ integer group dots → FP16 down rows → stable weighted combination path.
They compare every output column and independently generated NumPy packed
bytes/scales/sums for window sizes 1, 4 and 8, including full, skewed,
all-peer, shrinking/growing and empty routes. Additional tests cover exact
8/8/4 ownership, cancellation-sensitive route addition, metadata/shape/dtype
failures, no consumer submission after a failed dependency, logical maximum
capacity and CPU compilation of native column contracts.

The CPU activation/finalizer callbacks are declared surrogates of their dtype
boundaries, not hardware substitutes or claims of CANN numerical parity. The
pack reference verifies baseline quantizer bytes independently; native builtin
arithmetic, CANN finalizer windows, kernel stream ordering, graph inputs,
real-weight quality, image processing and thermal/service behavior require T9.
