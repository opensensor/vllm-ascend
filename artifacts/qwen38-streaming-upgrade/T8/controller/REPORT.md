# T8: guarded resident admission and transaction seams

The controller and worker transaction interfaces are implemented and checked
with synthetic evidence and a fake four-rank executor. The default controller
performs a dry run with zero RPC calls. This work does not operate a service or
qualify the streaming candidate for live loading.

## Admission

`admit(manifest, evidence, bundle_root=..., reference_git_root=...,
artifact_roots=..., plan=..., plan_receipts=..., memory_receipts=...,
configuration_sha256=..., resources_sha256=...)` requires complete T1 live
manifest, gate, native-component and workload evidence. Current offline receipts
fail this entry point. It checks actual committed reference blobs, the builder's
complete compiled-bundle schema and real source/binary/bridge/build-receipt bytes.
The logical contract, configuration, native resource inventory and original
reference identity are separately bound; this avoids circular candidate hashes.
Host compilation must include the verified Landlock/seccomp containment receipt,
its actual helper binary/source, policy and six command attestations. Their log
bytes remain bound after admission. Uncontained compilation receipts are rejected.

Checkpoint and serving-bundle receipts must contain actual file inventories.
Every hardware gate, native-component gate, workload trial and budget receipt
needs a local content-addressed artifact. Fresh image bytes are verified too.
External Git source links require expanded byte inventories under their pinned
commits; unresolved content fails admission. The full pinned external blob list
must match, and current bytes must equal the pinned blobs. Nested Git source links
need their own expanded inventory in a future schema; they are not independently
qualified by this implementation.

Four plan receipts must agree on plan, generation, namespace, candidate, worker
PID and physical weight-storage identity. Four explicit memory budgets must cover
old/new resources, caches, activations, routes, graphs, HCCL, comparison and
verification scratch. Route workspace has a conservative baseline lower bound
for ten routes per chunk. Activation workspace counts the full FP32 output,
two local slots, out-of-place reduction outputs, local/shared/add scratch and
full-batch router arrays. Resolved builtin scratch is required and tied to actual
budget artifacts. These are logical allocation bounds, not traffic measurements.

The immutable `Admission` exposes hashes and `plan_receipts`, which returns fresh
decoded copies. `require_execution(...)` verifies its seal, configuration/resource
identities, optional plan identity and all admitted file bytes again before use.
Hashes bind evidence content; they do not authenticate hardware claims.

## Controller and worker contracts

`ResidentController(client, journal_dir, transaction_id, live=False,
scheduler_state=None)` uses an injected client and authoritative scheduler probe.
`execute(admission, payload)` validates the payload against admission. Live use
requires idle requests/queue, complete rank identity, healthy preserved graphs,
no pending model execution, no outstanding collective, current sensors and an
exclusive maintenance lease. An existing pause belongs to its owner and cannot
be appropriated.

The controller fsyncs an exclusive, append-only intent before pausing, then uses
`clear_cache=false`. It does not cancel requests, reset caches, recapture graphs,
restart workers, reload weights or automatically retry a mutation. All ranks must
prepare the same admitted candidate and acknowledge the installed generation
before an owned resume. Resume requires every core at or below 85°C; switching
is refused during a thermal hold, with missing sensors or at 94°C and above.

`WorkerTransaction(snapshot=..., prepare=..., apply=..., restore=...)` provides
`status`, `prepare`, `apply` and `restore`. Its preparation callback is metadata
only and returns `WorkerPreparation(admission, apply_token, restore_token)`.
The undo token exists before mutation. Only the apply callback may load native
resources or install dispatch. Physical identity/storage and graph health are
checked before and after operations; failed installation poisons the session.
Restoration occurs only when the invocation may have changed dispatch.
Worker preparation, installation and restoration independently require the
authoritative owned maintenance lease; a direct RPC cannot bypass that lease.

Known-complete partial acknowledgments retain the owned pause and require explicit
recovery. A timeout, uncertain completion, outstanding collective or failed
recovery enters `HALTED`: no further status, collective, restoration or resume is
issued. `acknowledge_external_completion(proof, proof_path=...)` consumes an actual
operator-written artifact without RPC, checks engine/transaction/owned-pause and
four rank identities, then permits explicit recovery. It never reapplies or
resumes a partially installed candidate.

A future live adapter must expose the four `streaming_*` worker RPCs, normalize
integer streaming generations, establish an atomic exclusive scheduler lease,
and supply an authoritative collective-completion envelope. An owner field sent
to the existing pause endpoint is insufficient by itself. The stock response
without completion/ownership proof fails closed. Worker callbacks are injectable
and trusted to obey their metadata/load responsibilities; arbitrary callback
side effects cannot be prevented by this host interface. A worker transaction
owns one transition; subsequent generations require an explicitly prepared
session retaining the then-current dispatch as its undo state.

## Validation

- 37 CPU tests passed, including complete synthetic admission, byte changes,
  missing/stale ranks, pending gates, unknown/undersized memory, scratch ownership,
  idle/active/foreign-pause handling, preparation without native loading, partial
  mutations, timeout/outstanding execution, explicit rollback, rollback failure,
  append-only intent collisions, external completion artifacts, direct worker
  lease enforcement and changed containment-helper bytes.
- Three of these tests exercise expanded external Git sources: valid pinned
  content is admitted, changed source bytes are rejected, and nested Git links
  with unexpanded coverage are rejected.
- Scoped manual repository hooks passed, including Ruff, spelling, forbidden
  import checks and repository structure checks.
- The fake receipts deliberately use synthetic model, image, gate, binary and
  bridge data. Their hardware labels are test inputs, never qualification results.
- No actual HTTP request, server mutation, NPU import/open, device-library load,
  device kernel launch or hardware budget allocation occurred.

Real artifact attestation, live adapter ownership/completion behavior, hardware
gates, complete model memory admission and sustained thermal qualification remain
pending. All containment-helper and command attestation records in these unit
fixtures are synthetic inputs; no compilation or containment execution occurs
in this controller test suite.
