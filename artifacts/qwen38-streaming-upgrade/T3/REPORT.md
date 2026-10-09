# T3: token operands and grouped producer

This is an offline candidate. No device runtime was imported, no NPU was opened,
and no server was queried or changed. Hardware ordering, arithmetic on real
weights, latency, bus traffic and temperatures remain unqualified.

## Implemented interface

`tools/qwen4exp/streaming_operands.py` provides
`prepare_grouped_operands(inputs, weights, ids, *, pack, dispatch, gather,
num_local_experts, expert_offset=0)`. It packs once per token batch and uses the
injected stable device dispatcher and local-prefix gather. Physical capacity is
bounded by 2,560 tokens times top-k 10, or 25,600 rows. Shape/dtype checks read
host tensor metadata; they never read device route values. Empty token batches
skip all three callbacks. Nonempty all-peer batches still quantize the token
batch; their device gather performs zero active-row copies.

The gather callback preserves sorted dispatch order, including its inverse map
and route weights. Only the device-counted local prefix may be read later.
Unwritten peer capacity is deliberately undefined, so the native projection must
use the same ends and zero inactive outputs. Sparse device-routed decode remains
on its existing path. Default W8A16 MTP host dispatch remains unresolved under a
separate arithmetic contract; this candidate does not substitute W8A8.

`tools/qwen4exp/qwen_streaming_operands.h` supplies concrete
`qwen_streaming::OperandProducer`. `Init` takes low/high and scale/sum global
views; packed UB, metadata UB, L1 activation, L0A activation, L0B weight and
resident L1 weight views; and K. Each local view already starts at its matching
T2 arena region. The helper allocates no storage and loads no new weight bank.
`Produce(slot, row, group, liveRows)` stages one G128 into one of two slots.

Packed UB/L1 uses `[slot, limb, G128/K64, M, K64/2]`; L0A uses
`[slot, limb, M/16, G128/K64, 16, K64/2]`. The resident weight tile uses
`[N/16, groups, G128/K64, 16, K64/2]`, and L0B uses
`[slot, G128/K64, N/16, 16, K64/2]`. Both limbs, group order and packed byte
order match the existing native ABI. Padding is cleared every invocation, even
when the previous live tile was larger. A zero-live call clears operands without
issuing zero-length GM copies; normal all-peer dispatch skips projection entirely.

The helper pairs M_MTE1/V_MTE2 before writes, V_MTE2 after vector zeroing,
MTE2_V for load completion, V_MTE3/MTE3_MTE1 for activation staging,
MTE3_MTE2 for packed UB release, MTE1_MTE3 for L1 release and MTE1_M for Cube
readiness. Every event is paired within the call. T4 owns product buffers, CO1,
readback and correction; it must acquire a slot only after prior Cube and vector
users complete. This is a concrete producer with explicit fences, not proof of
hardware overlap. Resident full-tile weights and cached metadata are T4-owned
and remain reusable across row tiles.

## Transfer comparison

`operand_transfer_cost` includes both packed limbs, FP32 scale/sum lanes,
repeated reads by every output tile, compact gather read/write payloads and
common quantizer output writes. It reports fixed-capacity storage separately
from executed active-prefix payloads. The checked examples are saved in
`logical-costs.json`.

For production K2560, one packed row occupies 3,840 logical bytes. At 25,600
local rows and ten output tiles, indexed projection operand reads are 983,040,000
bytes; compact gather adds 196,608,000 bytes before those same projection reads.
This does not predict which path is faster: indexed access may be scattered,
compact packing may improve projection access, and caches, weight loads,
collectives, quantizer input reads and arithmetic are outside this operand model.
No backend is selected from logical bytes alone. Native v1 integrates compact
local operands because that ABI already has an existing reference layout; the
architecture remains a candidate until the complete service gates pass.

## Validation

`OMP_NUM_THREADS=2 python -m pytest --noconftest -q
 tests/ut/qwen38_1m/test_streaming_operands.py`: 31 passed.

Tests compile and execute the actual producer against CPU pipe/layout stubs for
100 calls across all 20 groups, both slots, and live rows 16, 1, 15, 0, 16.
Independent address formulas verify exact packed bytes, scales/sums, L1/L0A/L0B
layout, zero padding, untouched opposite slots and matched event tokens. Python
checks cover quantization count, stable route order, empty/skewed/all-peer and
shrinking/growing routes, 25,600 physical-row capacity, callback violations and
absence of device-to-host controls. An independent T2 lifetime test rejects
premature slot reuse. CPU stubs do not model asynchronous hardware pipes or
quantizer numerical arithmetic; injected pack inherits the baseline unchanged.

Hardware compile/order, native projection parity, image behavior, MTP quality,
graph replay, mixed-service behavior and thermal gates remain deferred to T9.

## Host compiler follow-up

The coordinator's first host-only dav-2002 build found that four constexpr
address helpers lacked `__aicore__` annotations. All four now carry that
annotation; the CPU regression also checks it explicitly because CPU execution
alone cannot detect host/device call-domain mistakes. No layout or arithmetic
changed. The original receipt is preserved; the follow-up receipt is
`validation-aicore.json`, with new source hashes and 31 passing tests.

This correction does not itself claim a successful CANN build or hardware
execution; the coordinator owns complete-kernel compilation and its receipt.
