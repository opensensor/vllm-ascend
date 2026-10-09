# Qwen streaming architecture: offline execution

T1–T8 are implemented offline. T9 hardware qualification is deferred by the user.
Nothing is promoted or enabled on the live server. Images remain a required gate;
no vision configuration or context/concurrency capacity was changed.

The candidate composes once-per-token quantization and device-local routing,
resident W4 projection tiles, two bounded Cube/vector slots, builtin FP16
activation, packed hidden handoff, bounded down-output windows, stable CANN-v2
finalization, shared-expert placement and rank-agreed communication. One owner
also binds native FP32 GDN state IO/WY, prefix phase handling and bounded PLE
staging. Any stage failure poisons that owner before later dispatch can continue.

The M16/N128 arena uses 140,032 UB bytes. All G128 corrections/additions retain
baseline order. The proposed three-stage producer/Cube/consumer overlap did not
fit the two-slot metadata ownership proof: this implementation supports the
Cube j/vector j−1 candidate only. Required CO1 readback and phase barriers survive.

Down N2560 uses 8/8/4 windows: the largest routed FP16 result drops from 125 MiB
to 50 MiB at the maximum profile. Writes/reads are not eliminated; extra launches,
casts and full-output copies must be measured. Gate/up and qualified nonlinear GM
boundaries survive. Default 2560-token scheduling produces one chunk at the
reference batch limit, so it does not provide independent communication overlap.
Smaller chunks require new arithmetic/performance gates. Scratch is per invocation,
not persistent across 48 MoE layers.

Sparse decode, graph MoE dispatch, W8A16 MTP, residuals and vision retain their
original paths. QSA graph selection-copy fusion, device accepted-state selection,
on-core nonlinear fusion and cross-layer overlap are not implemented. Source
byte checks do not authenticate already imported code; frozen worker import roots
and real asynchronous ordering remain T9 requirements.

## Validation

The final new suite comprises 518 CPU tests across protocol, memory ownership,
actual native bodies with synchronous SDK stubs, operands, epilogue, scheduler,
layer composition, EP4 composition, bundle and fake resident transactions.
The independent full EP4 reference covers distinct expert partitions, shared
placement, tails, windows and complete output. Native CPU stubs are not hardware
ordering/rounding or real-weight quality evidence.

A broader existing regression group produces 179 passes and four GDN lifecycle
failures because its torch_npu stub lacks float4_e2m1fn_x2. The exact same results
reproduce on clean unchanged baseline 2a2a4e416. Original logs are retained; this
package does not change those production/test files. T7's existing targeted
prefix/GDN/transfer integration group separately passes 150 tests.

Scoped lint and required repository-wide CI results, final source byte identities,
and host-only compiled bundle receipts are recorded in validation.json. Existing
repository-wide failures are reported separately from this package's checks.
No measured service improvement, memory bus reduction, thermal improvement or
whole-model hardware fit is claimed.

## Guarded deployment preparation

The append-only builder freezes complete candidate/plugin/native source bytes,
contract, reference and configuration; compiles coherent native resources and a
unique versioned bridge; and verifies actual bytes and entrypoints. Admission
requires complete matching reference/workload/native/model/image/cache/EP/MTP/
graph/thermal/service evidence and all-rank memory/PID/storage/plan identities.
Missing backend scratch or capacity headroom fails before loading.

An uncontained preliminary CANN build opened driver manager nodes inside SDK
compiler constructors, despite having no ACL initialization or inference calls.
That bundle is quarantined and its no-device-open receipt claim was invalidated.
The final qwen_streaming_v5 host build passes, with 40 device-open attempts
denied and zero successful other-device opens. All source and compiled artifact
bytes match. The builder requires a static Landlock/seccomp helper before compiler work,
denies other device files and driver IPC, closes inherited descriptors, and binds
per-command containment attestations/logs to the build receipt. No server was
operated or inference submitted. Actual driver-access trace evidence is retained
under `T8/host-build/`.

The controller defaults to dry-run and makes no network calls. Its live interface
is injectable and needs authoritative scheduler maintenance ownership and complete
collective-completion proof. Stock server pause/status endpoints alone are
insufficient. WorkerTransaction supplies metadata-only preparation and owned
apply/undo seams; an actual worker/scheduler adapter must be bound and validated
when renewed hardware access is granted. Unknown completion halts all further
RPCs, including status and resume, until out-of-band completion is proven.

Follow [the deferred qualification runbook](T9/RUNBOOK.md) before any hardware
operation. Enforce 94C hold, all-core 85C resume and the independent 96C cutoff.
The current reference and queued configuration deliberately cannot pass admission.
