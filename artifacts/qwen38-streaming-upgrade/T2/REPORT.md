# T2: Qwen streaming memory and ownership contract

The proposed G128 projection fits the installed dav-2002 local-memory limits
under the frozen byte contract. This is an offline storage and event-lifetime
proof. It does not qualify arithmetic on real devices, prove actual overlap, or
establish lower temperature or faster serving.

## Accepted topology

The fixed tile is M16/N128, with G128 and the original K0=64 packed layout.
Eight blocks own output tiles `block_id + k*8`. Production gate/up is N1280/K2560;
down is N2560/K640. Two slots hold both activation limbs, raw INT32 products,
FP32 correction products, FP32 activation scale/sum banks and metadata stages.
A single CO1 slot stores the paired 32-row low/high Cube product. Every G128
contributes to one FP32 accumulator in the original order before the final FP16
projection boundary. Weight and three FP16 metadata banks remain resident for an
expert/output tile across M tiles.

| Space | Used bytes per core | Available bytes |
| --- | ---: | ---: |
| UB | 140032 | 253952 |
| L1 | 167936 | 1048576 |
| L0A | 4096 | 65536 |
| L0B | 16384 | 65536 |
| L0C | 16384 | 262144 |

The UB limit deducts the SDK's 8192-byte reservation from 262144 bytes. The model
counts a dedicated 32768-byte quantizer arena and a persistent 4096-byte FP16
gate even while their fused epilogue is deferred. The generated header includes
all byte offsets, sizes, slot sizes and static capacity assertions. The JSON
contains matching layouts, arithmetic boundaries, event rules and an identity
hash. SDK source snapshots and their hashes are separate provenance inputs;
this ABI does not depend on a superseded T1 workload manifest.

The paired product layout is `[N/16,2*M,16]`; low and high rows occupy adjacent
M-row spans inside each output strip. The accumulator is `[N/16,M,16]`. Sparse
prefill tiles use a separately bounded row loop over 1–15 live rows; bulk has
16 live rows. Existing device-routed decode remains separate. Routed peer output
must be explicitly zeroed so replay cannot expose stale output.

The corrected operand address contract is `[slot,limb,GROUP/K0,M,K0/2]` in
packed UB and L1, and `[slot,limb,M/16,GROUP/K0,16,K0/2]` in L0A. The original
receipt's row/group layout descriptions were byte equivalent but gave the
wrong address order. `contract-v1.json` preserves that receipt;
`contract-v2.json` and the current `contract.json` contain the corrected layouts.
ABI constants, offsets and capacities are unchanged.

## Lifetime proof and limits

The CPU model keeps Cube group j and vector correction of j−1 simultaneously
live in disjoint buffers. It requires completed CO1 readback before the next Cube
issue and retains each slot's activation and weight metadata through correction.
Event direction/ID pairs carry a generation, permit one outstanding signal,
require exactly one matching wait, and cannot authorize a release preceding
the acquisition. Startup, all 1–20 group counts, row tails and final drain are
checked. UB→L1 MTE3 fences, L1→L0 staging, Cube operand consumption, readback and
FP16 store completion have separate ownership boundaries.

Full three-stage overlap is not established by this proof. Loading all metadata
for j+1 into slot `(j+1)%2` while j−1 still consumes that slot would violate the
accepted lifetime model. Packed operand staging can be split from metadata
staging because their storage lifetimes differ; a later producer implementation
must supply a corresponding trace before claiming that additional overlap.
The accepted implementation target is Cube j alongside consumption j−1.

The qualified `cann_builtin_fp16` nonlinear path retains its projected GM
boundary. Any FP32 SwiGLU fusion is a distinct unqualified candidate. The packed
hidden handoff remains in GM because down requires columns from multiple cores.
Independent storage does not replace required vector ordering fences.

The MoE GM arena estimator retains gathered FP16 input, sort scratch, projected
gate/up, FP16 nonlinear output, packed input/hidden limbs and scale/sum banks,
routed FP16 output and combined FP32 output. Its aligned regions conservatively
coexist for the whole call. It describes logical allocation bytes, not measured
memory traffic. External inputs, shared experts and attention belong to the
separate activation budget; framework overhead belongs to the runtime reserve.

Whole-rank admission requires explicit values for checkpoint residency, old and
new native resources, KV cache, Mamba primary/archive, activations, route workspace,
graphs, HCCL, shadow comparison, verification scratch and runtime reserve. Missing,
unknown, negative or overcommitted budgets fail before allocation. These values
are currently unknown; no claim that the complete model fits follows from the
per-core table. Host spill capacity cannot be credited to the NPU budget.

Rejected alternatives include retaining every G128 product (327680 bytes before
other UB scratch), reusing CO1 before readback finishes, prematurely reusing a
correction slot, borrowing GLM's K32 numerical contract, and removing the GM
handoff without changing column ownership.

## Validation

- `OMP_NUM_THREADS=2 python -m pytest --noconftest -q tests/ut/qwen38_1m/test_streaming_memory.py`:
  122 passed. This includes a host g++ compile of the generated header and complete
  bijective checks for both product limbs.
- `ruff check tools/qwen4exp/streaming_memory.py tests/ut/qwen38_1m/test_streaming_memory.py`:
  passed. Both Python files were formatted with Ruff.
- Contract JSON equals the generated value, header equals the generated text,
  and all archived SDK hashes agree.
- No server operation, device runtime import, NPU open, native device-library
  load, kernel launch or hardware allocation occurred.

Compiler target validation, kernel parity, device event legality under execution,
whole-rank admission, quality, images, graph replay, service latency and sustained
thermal results remain pending.
