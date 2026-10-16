# Two-slot complete-stage ownership contract

`stage_slots.py` is an offline control model for T9. It does not allocate memory,
call a bridge, launch a kernel, or establish a performance improvement. The
existing `stage_dma.py` serialized transport is unchanged. T16 must bind and
hardware-qualify this model behind a default-off option before it is a usable
pipeline. Ordinary decode remains serialized.

## Registration and lease identity

Create one `Registration` for a fresh session with an unpredictable nonce, the
exact checkpoint SHA-256, and two distinct opaque mapping-set identities. Each
mapping identity must refer to a private, persistent CUDA input/result allocation
pair and the corresponding input/result/guard allocations on each chip. Freeze
that association for the session; reallocation requires a fresh session and
nonce. These strings are allocation identities, not pointer values or ownership
of actual handles.

The registration fixes protocol `glm-complete-stage-slots-v1`, schema 1, bridge
UAPI 5, physical chips 2/3, two slots, at most 32 rows, and the full 131,072-token
service context. Per-request reservations can be smaller. Slot capacity defaults
to 4 MiB and cannot exceed the qualified 8 MiB region bound. This does not admit
larger row counts or change complete-state field precision. Every submission's
payload byte count must equal `stage_wire.descriptors(rows)` for FP32 residuals.
T16 must also validate the full canonical field metadata before packing.

Authenticate each worker with the existing socket credential checks. Call
`offer_mapping(rank)` **before** any descriptor can cross to that rank, so the
controller retains owners even if the ready acknowledgement is lost. Complete
descriptor export, import, and native persistent mapping registration before
accepting `register(rank, registration.packet(rank))`. A lost acknowledgement
after export is uncertain ownership and requires quarantine unless an
authenticated, fully quiescent mapping-release acknowledgement can be obtained.
The registration packet is an exact match, including rank,
physical chip, mapping order, version, and nonce. Boolean values are not integer
identities. Packet equality is validation, not cryptographic authentication.

Each immutable `Submission` carries session, nonce, request, monotonically
increasing request generation, global sequence, slot index/generation, mapping
identity, first position, rows, fixed request reservation, payload size, and mode.
`submission.for_rank(rank)` adds the physical rank/chip to every worker envelope.
A reused request name cannot authenticate an older request generation; reused
slots cannot authenticate an older slot generation. `RankLease.from_packet`
rejects noncanonical envelopes. `accept(phase, packet)` then matches every field
against the currently retained submission before changing state. Unknown phases,
replayed acknowledgements, altered reservations, and wrong physical chips fail.

## Ordered transitions

`StageSlots` belongs to one coordinator thread. It stores at most two submissions
and no unbounded history or control queue. `submit` and `receive` return
immediately or raise `WouldBlock`; neither spins or performs a device wait.

| Transition | Required completed evidence / ownership | Result |
| --- | --- | --- |
| `offer_mapping(rank)` | Before descriptor export to that authenticated rank | Rank owes a mapping-release acknowledgement even if registration never completes |
| `begin_request(request, reservation)` | Both registrations; no previous active request | New request generation, position zero |
| `reset_started(request, generation, owner)` | Named CUDA/rank runtime owner idle, no leased slots | Cache reset owns that runtime until completion |
| `reset_complete(request, generation, owner)` | Actual CUDA or named rank cache reset has completed | Submission enabled only after all three owners acknowledge |
| `submit(first_position=..., rows=..., payload_bytes=...)` | Free credit, exact next position and reservation, full reset quorum | One immutable slot lease; advances submission position/sequence |
| `prefix_started(submission)` | CUDA idle, prior prefixes ready, slot not written before | CUDA producer owns this slot |
| `producer_ready(submission)` | Source event completed for packing and all prior prefix writes | CUDA prefix ownership ends; input DMA may start |
| `upload_started(rank_lease)` | Producer ready; named native owner idle | Named rank owns an in-flight input transfer |
| `upload_complete(rank_lease)` | Blocking bridge returned successfully with its completion guarantee | Named rank upload is complete |
| `compute_started(rank_lease)` | Both uploads complete, next Ascend cache sequence, named owner idle | Named rank owns stage computation |
| `compute_complete(rank_lease)` | All stage/collective/packing work represented by that rank's completion event has completed | Both-rank quorum advances Ascend cache order |
| `return_started(rank0_lease)` | Both ranks complete; rank 0 idle | Rank 0 owns result DMA into this slot's CUDA result buffer |
| `return_complete(rank0_lease)` | Blocking result DMA returned successfully | Result exists, but is not yet a retired or CUDA-visible consumer |
| `receive()` | Next unclaimed suffix sequence has completed result DMA | Returns the ordered submission once |
| `cuda_visible(submission)` | Existing CUDA allocation owner's visibility flush completed | Suffix may read result views |
| `suffix_started(submission)` | Result received/visible and CUDA prefix/suffix owner idle | CUDA suffix owns all returned views |
| `retire(submission)` | CUDA suffix completion event and every downstream result-view consumer have completed | Slot credit returns; CUDA suffix cache order advances |

Call the start transition **before** the corresponding operation. Call the
complete transition only after its documented completion guarantee. A kernel
launch, event record, socket send, or transfer acknowledgement is not evidence
that every device consumer has finished. Include deferred residual/post/comb
views, graph buffers, logits consumers that retain views, and any other alias in
the final CUDA consumer event. If a graph retains slot-backed addresses after
that event, those addresses need a longer lease or persistent private copies.

Prefix writes, Ascend cache mutation, and CUDA suffix retirement have separate
monotonic sequence counters. CUDA prefix B may execute while Ascend owns A.
Ascend B may mutate its caches only after **both** ranks complete A. Results may
return into their separate allocations out of order, but `receive` cannot hand B
to the suffix before A retires. CUDA prefix and suffix operations cannot overlap
in this conservative model; T16 must use one ordered CUDA execution domain.
Decode admits only one outstanding submission, including across a prefill/decode
mode transition. It is not a speculative decode implementation.

Rank 1 never returns result payload to CUDA, but its event remains required for
every quorum and lease retirement. The initial budget reserves its result buffer
conservatively. Removing rank-1 packing/allocation is a separate T16 checkpoint,
with separate numerical, lifetime, and memory receipts.

## Cancellation, failure, and shutdown

| Situation | Required behavior | Allocation ownership |
| --- | --- | --- |
| Successful request | Retire every consumer; `close_request()`; require all cache resets for the next request | Persistent allocations retained for reuse |
| Client cancellation with healthy workers | `begin_drain(now=monotonic_time, timeout=..., cancelled=True)`; stop new submissions and drain existing operations | Retain every lease until its result and consumer are safe |
| Cancelled result with no suffix launched | After both-rank compute, result DMA, ordered `receive`, and visibility flush, use `discard_cancelled` | Retires the unconsumed result in order; no suffix launch required |
| Cancel after suffix launched | Wait for its actual consumer completion, then `retire` | Never discard an in-flight consumer |
| Healthy cancel fully drained | `resume_after_cancel(now)` before timeout, with no mapping released; next request must reset all caches | Same registered owners can remain alive |
| Worker loss, uncertain native operation, socket loss after export, or invalid guard | `quarantine(reason)`; publish `snapshot()` receipt | Retain all owners, leases, exports, guards and mappings; no automatic reset/free/kill |
| Bounded drain timeout | `poll_drain(now)` raises `Quarantined` if slots or mapping releases remain unresolved | Retains diagnostic identities and operation ownership |
| Normal shutdown | Stop submissions with `begin_drain`; finish consumers, then close each rank's native mappings/imports and verify its guard | CUDA exports and allocations remain alive through both-rank release acknowledgements |
| Both ranks safely unmapped | `mapping_released(rank, registration.packet(rank), guard_poison_exact=True)` for each offered rank, including incomplete registrations | `snapshot()['owners_releasable']` permits owner cleanup |
| Final owner cleanup | Free owners only after the previous condition, record `release_owners`, then complete existing owner-release/worker-exit handshake | Session closed |

The timeout is finite, positive, at most 300 seconds, and cannot be extended by
replaying `begin_drain`. The integration loop must poll with monotonic timestamps
and apply the same deadline to socket waits; this pure model has no background
timer and cannot interrupt a blocking native call. A timed-out native operation
must remain owned/quarantined on its original runtime thread. Never release
resources because a Python future timed out or a client disconnected.

Cancellation also drains any started cache-reset operations. It conservatively
finishes already submitted chunks, including a slot
whose producer has not started. It does not claim a native abort primitive. If
those operations cannot finish, the fixed timeout quarantines the session. The
adapter may suppress user output while draining. No new request may reset or
reuse the Ascend/CUDA caches until the previous leases retire. Once any mapping
is released, cancellation cannot resume and shutdown must finish.

Malformed/stale packets raise without changing live ownership. If a bad packet
also means native quiescence can no longer be established, the adapter must
explicitly quarantine. Snapshot diagnostics remain available during quarantine;
all operational transitions are disabled. A process restart needs fresh mapping
identities and nonce, not recovery of the old session's leases.

## Additional memory reservation

For slot count `S = 2` and capacity `C`, each owner reserves both input and result:

- CUDA device bytes: `2 * S * C + CUDA scratch`.
- HBM bytes **per chip**: `2 * S * C + chip scratch`.
- Host lifetime-guard bytes per chip: `S * C`, separate from HBM.
- Total host guard bytes across both chips: `2 * S * C`.

At 4 MiB this is 16 MiB CUDA, 16 MiB HBM per chip, and 8 MiB host guard per chip,
before scratch. Relative to an otherwise identical one-slot design the buffer
increment is 8 MiB CUDA and 8 MiB HBM per chip, plus 4 MiB guard per chip. Guard
storage is not tensor staging. `memory_requirements` reports this formula and
labels it unvalidated. Add weights, the exact 131K cache reservation, graph
pools, collectives, allocator/runtime reservations, scratch peaks, and measured
fragmentation to each device's real high-water mark. Event/descriptor resources
and pinned-host allocation limits need separate runtime accounting. Capacity
qualification must use the exact release build; shrinking context is not a
remedy for insufficient headroom.

## T16 adapter boundary and validation

T16 owns actual buffers, exported descriptors, authenticated sockets, native
contexts/threads, events, and cleanup. Use persistent per-slot views. Run every
CUDA runtime/driver call on its existing CUDA owner context/thread and every
Ascend call on its existing rank owner. Marshal small completion records to the
single control owner; this module must never be used to move native calls onto
an arbitrary executor. Each rank's blocking upload, compute completion wait,
and result return remain sequential operations on that owner. No concurrent
calls are admitted for a busy native rank.

A minimal scheduling iteration is: reserve A, produce A, dispatch A to both
ranks; reserve/produce B while A computes; receive/flush A, run/retire suffix A;
let Ascend mutate B only after the A quorum. Continue with at most two credits.
Worker start/complete envelopes must preserve the exact submission. The
controller cannot infer an event completed from elapsed wall time.

Bind one-slot and two-slot configurations with identical math and geometry for
the scheduling comparison. Keep socket fallback, decode, and complete-block
prefix-reuse policy unchanged. Parent integration must manually review the
success, cancellation, worker-loss, timeout, and shutdown tables above before
binding hardware, then run repeated/changed-input parity, exact per-chunk/final
cache state gates, guard/mapping accounting, cancellation/restart tests, 128K
capacity checks, and matched full-Code API measurements. Portable state-machine
tests establish control invariants only; they do not qualify DMA or model speed.

Validation:

```bash
python -m pytest -q --confcutdir=tests/ut/sidecard \
  tests/ut/sidecard/test_stage_slots.py \
  tests/ut/sidecard/test_stage_dma.py \
  tests/ut/sidecard/test_completion_contract.py
```
