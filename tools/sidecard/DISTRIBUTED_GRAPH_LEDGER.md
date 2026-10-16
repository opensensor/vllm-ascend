# Distributed graph ledger: future design

**Proposal only; not implemented or hardware-qualified.** The aim is to move
dependency handoffs between admitted graph segments onto peer DMA and device
notifications. A ledger here is a bounded set of execution descriptors and
completion records. Trusted ranks execute a CPU-admitted dependency graph;
blockchain, replicated consensus, and a global polling barrier are unnecessary.

## What exists and what remains unknown

The [current collective](NPU_COLLECTIVES.md) specializes fixed-shape FP16/FP32
SUM. Its [exchange ABI](../../csrc/sidecard_collectives/npu_exchange.h) permits
peer writes into imported inboxes; arithmetic reads local addresses. The
[fixed handshake](../../csrc/sidecard_collectives/npu_exchange.cpp) enqueues:

```text
PUSH payload → record peer.READY → wait/reset local.READY
             → consume local inbox → record peer.CONSUMED
             → wait/reset local.CONSUMED → permit slot reuse
```

This is qualified for the recorded fixed collective graph geometries. It is
not a general message queue: fixed mode uses slot zero and binary notifications,
without a wire sequence identity. Ordinary alternating slots depend on host
state and cannot simply advance during graph replay. Existing ownership also
requires one serialized stream per owner; concurrent reuse of its scratch is
not authorized.

Unobserved replay already removes Python collective observers. The runtime
still launches native graphs and executes remaining host work. Device waits can
order work that has already been submitted. **An Ascend device's ability to
launch the next native graph, conditionally select a graph, or sustain an
autonomous token loop has not been established.** A peer READY notification
alone does not launch a graph. The first implementation should use finite,
CPU-submitted graph chains; eliminating per-token launch admission is a later
capability gate. Cross-vendor CUDA launch/control requires separate proof.

The next stage is a finite, pre-submitted
`compute → publish sequence → peer Notify wait → consume → acknowledge` window.
A small local AiCore bookkeeping operation could validate the received identity,
update bounded slot state, and publish completion through peer DMA. That would
move bookkeeping onto the relevant NPU without claiming it can launch arbitrary
work. The validation-to-consumer dependency and failure suppression still need
proof. An NPU-level DMA monitor that polls while idle is a separate hypothesis:
an available runtime/firmware mechanism, cache visibility and noninterference
with producers have not been demonstrated. An AiCore spin loop is not evidence
of such a monitor and could occupy resources required for progress. Prefer
ordered Notify waits for the initial experiment.

## Local ownership and descriptor contract

Each rank has one writer for its local execution state. Each incoming edge has
a separate local inbox whose sole payload/header writer is its admitted peer.
Acknowledgments occupy separate records with the opposite writer. Ranks never
race to append to a shared global log and never vector-load a peer address.

| Field | Required meaning |
| --- | --- |
| Session epoch | Fresh identity after a fully retired session; rejects stale peers |
| Generation | Immutable graph, model, cache-layout and collective-plan generation |
| Edge and sequence | Admitted producer/consumer pair and monotonic edge-local work identity |
| Slot | `sequence mod window`, reused only after matching consumption |
| Operation and shape | Graph/segment ID, dtype, dimensions, byte count, layout and dependency schema |
| Storage binding | Pre-admitted local offsets and bounds; never an arbitrary executable pointer |
| Lease binding | Cold registry identity retaining graphs, mappings, Notify objects and workspace |
| Completion | Matching epoch/generation/sequence plus completed, cancelled or failed status |

READY follows the payload and descriptor writes on the producer's ordered
stream. The consumer waits, checks its **local** descriptor, and permits only
the expected admitted operation. Consumption acknowledgment follows the last
local reader, including asynchronous consumers on other streams joined by an
explicit event. The producer cannot overwrite that slot or re-record its READY
until the matching acknowledgment has been consumed/reset.

These ordering and validation rules are requirements, not guarantees supplied
by writing a header. Device visibility, descriptor publication, validation before
consumer access, and safe notification reuse need native gates. A sequence
number does not fix an incorrectly reset binary Notify. Begin with window one
and the existing handshake; a larger window needs separate per-slot storage,
notification ownership and a proven graph-address schedule. Never wrap an epoch
or sequence counter: drain and create a new session first.

## CPU admission, device execution and retirement

CPU control remains responsible for membership, capabilities, authenticated
schedule identity, supported shapes, memory budgets, captured storage, and
native leases before submission. It admits a finite dependency DAG and a bounded
amount of future work. Local streams execute that plan using peer-write
dependencies; each edge waits only for its own producer/credit. No new global
barrier is required between model layers, although the model's SUM still needs
all scheduled contributions. Preserve its existing reduction order and math.

Changing batch shape, graph variant, weights, KV layout or admission policy
requires a new generation. A ledger entry never transfers ownership or proves
that a graph has stopped referring to memory. Retirement retains the existing
[native lease contract](../../csrc/sidecard_collectives/npu_group.h): stop
admission, drain queued consumers, reset every affected native graph, then
release leases. Close imports only after the existing all-rank drain exchange;
release exports after the import-closure exchange. These cold lifecycle
exchanges remain necessary even without hot global barriers.

## Finite progress, backpressure and failure

An empty credit window stops the producer before overwrite. Dependencies must
be acyclic within an admitted window, with known initial credits; an unsubmitted
consumer or a circular stream wait must not hold the only execution resource
needed by its producer. Do not assume persistent spinning kernels, remote
atomics, or unlimited pending Notify waits are safe on 310P.

Before publication, cancellation can withdraw unsubmitted work. After any
payload or notification is submitted, treat cancellation as pending until a
defined drain boundary; do not skip a consumer and strand its producer. A
future cancelled acknowledgment must itself be ordered and validated. Already
queued waits are not assumed to be interruptible.

Use finite queue windows, bounded SDK waits, and an independent host watchdog.
A timeout, rank loss, descriptor mismatch or uncertain partial submission stops
new admission and quarantines affected owners. Retain graphs, mappings and
slots; no automatic HCCL fallback, retries, counter reset or allocation reuse.
Unrelated work may continue only with separately owned resources and no failed
dependency. A watchdog detects lack of progress; it does not prove device drain
or authorize freeing a failed rank's imported memory.

## Gates and QFN integration order

The required end state is zero eager callbacks in admitted model execution,
including the boundary between token iterations. Finite host-submitted chains
are the first capability test, not the final architecture. Native conditional
continuation, device-side acceptance and state commit, and graph-compatible PLE
lookup remain implementation and hardware-qualification tasks. Treat missing
support in today's implementation as an open engineering question rather than
proof of a hardware limitation. Direct DMA is the intended dependency transport;
whether the resulting complete design is fastest must be measured.

1. Use tiny private tensors and fresh owners to prove native graph producer →
   peer write → READY → local consumer ordering, with changing values and peer
   skew. Check results and explicit clean retirement on every rank. Do not use
   serving KV, graph scratch, or the previously unsafe direct-read path.
2. Reuse one slot repeatedly with distinguishable sequences. Exercise delayed
   consumption, duplicate/stale identities, full-window backpressure and reset
   ordering. Host-side fault injection first; device timeout/failure probes
   require a bounded recovery design before hardware execution.
3. Submit a finite chain using the smallest graph/event budget. Demonstrate
   no overwrite, correct validation, bounded memory and balanced leases across
   multiple generations. Compare against the existing stream schedule, reporting
   CPU submissions, device time and transferred bytes separately.
4. Gate larger windows and independent tensors using disjoint owners before
   introducing cross-stream overlap. Measure whether added notifications cost
   more than the host handoffs they replace.
5. Only then investigate a documented native device launch/conditional mechanism.
   If unavailable, retain CPU submission of finite chains and report that limit
   explicitly. No persistent device-spin scheduler is assumed.

For QFN, DMA dependencies address only part of the execution path. The qualified
baseline still has a host PLE callback; QSA graph admission and general-context
selection have separate constraints. Sampling and MTP acceptance determine the
next token/accepted count. KV page allocation, causal positions, Mamba state and
rejected speculative updates must agree before another step executes. Moving
these into an admitted device chain needs explicit completion and commit rules;
the ledger alone cannot remove those dependencies or make autoregressive
tokens independent. Prefill tiles, whose input tokens are already known, offer
a bounded earlier test. Preserve selected-group ordering, cache ownership and
existing math while measuring any end-to-end gain.

## October 14 graph closure inventory

The [fresh four-rank profile](../../artifacts/npu-collective-qfn-20261014-024609/graph-refresh-results.md)
records these remaining boundaries in the restored QFN service. MTP uses two
draft steps per iteration. All listed host dependencies are removal targets.

| Boundary | Current behavior | Required graph/ledger behavior |
| --- | --- | --- |
| Target PLE | Host table lookup plus eager device projection, gating and convolution | Owned lookup/staging buffers; captured device arithmetic; then graph-compatible lookup transport |
| Target general QSA | Twelve eager selection callbacks | Admitted capacity variants, safe padding, exact ordering, device positions |
| Draft positions | One host position-copy callback per draft step | Positions derived from retained device metadata |
| Draft QSA | One selection callback per draft step | Captured selection with stable metadata ownership |
| Draft experts | IDs read to host; Python grouping and expert dispatch | Device routing into owned slots with unchanged W8A16 arithmetic |
| Iteration transition | Sampling/acceptance, copies, metadata and host scheduling | Device acceptance and next-input construction; ordered KV/recurrent commit and peer completion |

Prefix iterations have seven eager callbacks and ten native graph launches;
general iterations have nineteen callbacks and twenty-two launches. The
iteration transition adds work outside those counted callbacks. A count of zero
callbacks alone therefore cannot qualify autonomous execution.

For each closure milestone, require changing-input replay parity, correct
speculative rejection, mixed-request isolation, stable graph addresses, balanced
resource retirement, and unprofiled accepted-token performance. For the final
chain, verify that the CPU neither reads token decisions nor submits dependent
per-token work during an admitted execution window. Streaming output may observe
completed records without gating device progress. Measure task-free intervals,
notification waits, rank skew, and graph/task-pool consumption across repeated
windows; moving a wait from the CPU into a device queue is not a speedup.

The [iteration-boundary follow-up](../../artifacts/npu-collective-qfn-20261014-024609/iteration-closure-results.md)
retains one acceptance observation for the CPU Mamba commit fallback, removing
its duplicate D2H count readback with neutral measured throughput. The new
[tensor commit planner](../qwen4exp/mamba_commit_plan.py) now passes changing-input
NPU graph replay. A [grouped native state-commit kernel](../../artifacts/npu-collective-qfn-20261014-024609/mamba-commit-graph-results.md)
also passes private graph tests on all four NPUs, combining 72 state banks in
one launch. The [live integration](../../artifacts/npu-collective-qfn-20261014-024609/mamba-commit-live-results.md)
now binds and verifies the actual model banks. It retains CPU sampling
observation, admission, bookkeeping, metadata staging, and graph submission;
the kernel runs downstream of that host dependency. The serving A/B shows no
clear speedup. Device acceptance, next-input construction, and DMA dependency
integration remain required before claiming an autonomous model graph.

The subsequent [device transition bridge](../../artifacts/npu-collective-qfn-20261014-024609/device-transition-results.md)
now computes acceptance, commits the actual 72 model banks, and supplies next
input IDs from device graphs. The corrected live input hook passes boundary,
concurrency and request-turnover checks. Its serving A/B remains neutral because
the synchronous engine and host PLE path still gate dependent iteration.
The authenticated engine control extension now reaches the engine process,
allowing drained scheduler updates; installing it does not enable asynchronous
scheduling or complete the distributed ledger.

The [device PLE history follow-up](../../artifacts/npu-collective-qfn-20261014-024609/device-ple-history-results.md)
now advances rejection-safe history and publishes next query IDs on device.
The actual host-paged lookup consumes a checked mirror of that packet. A
blocking D2H mirror, host lookup, and the synchronous scheduler remain; this
qualification must not be treated as permission to bypass async/cache guards.

The subsequent [queued scheduling integration](../../artifacts/npu-collective-qfn-20261014-024609/queued-scheduling-results.md)
now overlaps engine scheduling with worker execution, overlaps output observation
with drafting, and consumes unchanged-batch draft packets on device. Early PLE
readback and pinned compact-table staging remove additional blocking transfers.
The retained service measures 32.056 C1 and 69.179 aggregate C4 HTTP tokens/s,
including prefill and API overhead. CPU metadata, host PLE lookup, draft callbacks,
request lifecycle, and graph submission remain; the autonomous-loop contract is
still incomplete. The linked results distinguish hardware validation from
untested cache-pressure and full-context cases.
