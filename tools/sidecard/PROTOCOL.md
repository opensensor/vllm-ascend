# Sidecard binary expert protocol, version 1

This is a host-buffer socket implementation of the expert boundary. It has
no CUDA/NPU memory-registration or device-pointer payloads. Limits are configured
per session; the GLM candidate uses width 4096, at most 640 tokens, 5120
routes, 288 expert IDs and eight route slots per token.

## Frame layout

| Segment | Encoding |
| --- | --- |
| Prefix, 16 bytes | `!8sII`: magic `SCARD001`, metadata byte count, body byte count |
| Metadata, at most 2048 bytes | UTF-8 JSON with the exact keys below |
| Request body | Token IDs, route triples, FP16 hidden rows |
| Reply body | Route triples, FP32 raw expert output rows |

Prefix lengths use network byte order. All body integers and floats use
little endian. Token IDs are UINT32. A route occupies three UINT32 values:
original token index, top-k slot index and global expert ID. The request's
token-ID list maps its compact hidden rows to original batch indices. Each
requested token appears once per owner even when several routes use it.
The reply carries one full-width output row per route in its route-list order.

Metadata keys are `kind`, `placement_sha256`, `checkpoint_sha256`,
`execution_id`, `layer_id`, `owner`, `width`, `rows`, `routes`, and
`payload_sha256`. `kind` is `request` or `reply`; placement hashes cover
canonical sorted JSON for `ExpertPlacement.to_dict()`. Checkpoint hashes must
identify the complete verified weight manifest supplied by the caller. The
offline smoke hashes its synthetic weights; a header-only checkpoint-plan
hash is insufficient as a production weight identity. Execution and owner
strings are bounded to 256 UTF-8 bytes each.

Every row and route count must be positive and within the configured limits.
The declared body length must equal the precise length derived from shape,
kind and route count. Unknown metadata keys, wrong session identities,
duplicate token/slot routes, orphan token rows, invalid IDs, nonfinite tensors,
truncated frames, extra bytes and payload checksum mismatch are rejected.
Finite values outside the wire dtype range are also rejected during encoding.
Maximum frame length is derived from the configured limits before receiving
its body; a peer cannot request an arbitrary allocation through the prefix.

## Arithmetic and ownership

Requests contain no weights or cache state. Router scoring, correction bias,
normalization, route weights and shared-expert output remain at the
coordinator. The worker checks each global expert against its layer's
placement and computes one complete owner batch. Replies are unweighted.

The coordinator validates exact route coverage and restores original route
slot order before FP32 multiplication and addition. It applies routed scaling
once and adds shared output once. Worker completion order cannot change this
reduction. FP16 input transmission is part of the candidate's numerical
contract and must also be applied to the serial comparison. Agreement with
CUDA GLM or the existing Ascend reducer remains a real-input validation gate.

## Lifetime and failure

`LocalWorkerPool` owns every child process and socket. Workers keep their
expert banks resident across passes; the production backend will
need explicit uploaded-bank lifetime and copy-completion fences. CPU decoded
packets own their tuple storage and cannot be overwritten by a later packet.

Before accepting requests, a worker calls `ExpertBackend.prepare()` and checks
the returned `(layer, global expert)` keys against its complete placement.
The canonical CPU backend verifies its partition and loads bounded fixture
weights at this step. A production backend must also finish uploads and
completion fences before returning. Worker startup has a separate deadline.

The startup frame uses the same 16-byte prefix with magic `READY001`, a
bounded metadata length, and zero body bytes. Metadata contains `status`,
`owner`, `placement_sha256` and `checkpoint_sha256`. Status is `ready` or
`failed`; a failed frame also contains `error`, limited to 128 characters.
The coordinator validates the owner/session identity, surfaces preparation
failures and requires acknowledgement from every worker before dispatch.
Duplicate JSON control fields are rejected in both startup and expert frames.

Each socket permits one outstanding request. The coordinator snapshots input
rows and generates a fresh pass identity. A monotonic deadline bounds send
and all fragmented receive operations. Malformed replies, dead peers and
timeouts close the session; late data cannot be accepted on a reused stream.
No partial expert result is returned. Closing a pool cancels the entire pool
and reaps its children; per-request cancellation, concurrent scheduled passes,
reconnection and device graphs are future integration work.

Workers are spawned from trusted local configuration. Multiprocessing startup
serializes that configuration; request/reply packets never use pickle.
Canonical backend specifications contain paths and identities, with no
prepared weight bank; loading occurs inside each child.
The current transport is a local Unix socket pair with ordinary CPU buffers.
Its correctness tests do not measure PCIe bandwidth, pinned-buffer behavior,
accelerator overlap or serving throughput.

The joined service optionally uses `tensor_wire.TensorPacket`, with immutable
contiguous tensor bytes and empty scalar tuple fields. Its encoder emits the
same `SCARD001` frame as the reference encoder. NumPy checks finite values
without per-element Python float conversion. CUDA receives raw FP16 input
buffers and returns FP16 expert output, expanded to the required FP32 reply.
The resident daemon can accept sequential new peers after clean EOF without
reloading weights; malformed packets and backend failures still close it.
This does not add overlapping exchanges or cross-vendor device pointers.

`hardware_probe` is a separate bounded diagnostic byte channel (`SCOPY001`).
It has exercised explicit synchronous device copies on the joined host;
it does not carry these expert requests or qualify this protocol's device
buffer integration. `native_expert_probe` separately checks existing native
single-expert arithmetic. Neither probe implements a production expert worker.

`ExternalWorkerPool` now carries these packets between separately selected
Python interpreters. Its bounded selected-expert backends authenticate native
checkpoint shards, upload resident banks and finish fences before acknowledging
readiness. The Ascend backend gathers one row per route and uses unit-weight
top-k=1 native execution; the CUDA correctness backend returns matching raw
rows. A CUDA coordinator reducer validates coverage and restores original
route slots before unfused FP32 arithmetic. No shape or wire schema changed.

All copies and expert operations in this gate are synchronous. Backend service
totals include their host tensor conversions and device work, but exclude
packet decoding/encoding and coordinator reduction. Workers print these
bounded totals when closing. The external pool waits for graceful exits and
reports stuck PIDs instead of terminating accelerator processes. Full bank
capacity, pinned buffers, graphs and joined model scheduling remain pending.
