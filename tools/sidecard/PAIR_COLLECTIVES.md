# Specialized Ascend pair collectives

The N-rank follow-up now has separate schedule/owner/native/Torch components in
`npu_collective_*` and `csrc/sidecard_collectives/npu_*`. It supports FP16/FP32 SUM
with binomial schedules for arbitrary N and recursive doubling for power-of-two
groups. Two-, three- and four-rank raw device tests passed; larger groups and
Torch/model admission require separate validation. See the
[hardware results](../../artifacts/npu-collective-qfn-20261014-024609/README.md).
The pair-specific contract below remains the original GLM research interface.

The target is faster FP32 SUM for the two chips on one Atlas 300I Duo. Decode
uses 16 KiB per reduction; 4-row and 32-row prefill use 64 KiB and 512 KiB.
Complete Ascend layers 18–32 require two reductions per layer, or 30 per token.
The expert-only architecture has no HCCL expert reduction to replace.

This is an experimental standalone library. Push/SDMA plus READY/CONSUMED
passed synthetic FP32 tests on both Duos using the installed CANN 9.1 SDK.
Decode-sized completed batches measured about 43 microseconds per call versus
59–64 for raw HCCL. Single-call measurements showed a much smaller difference;
these are not model token rates. See the
[hardware evidence](../../artifacts/sidecard-pair-hardware-20261014-014000/README.md).
There is no Torch adapter or production admission; GLM remains offline.

A direct remote-memory vector-read candidate timed out on its first reduction
and left the driver unresponsive. The user recovered the host by rebooting.
The probe, Python backend and native C API reject that mode before resource
allocation or tensor creation. Successful imports and SDMA copies did not
establish that imported memory was safe for vector reads.

## Mechanism

Each rank owns persistent FP32 input, receive-inbox and output buffers. It
exports the inbox and two Notify objects with a peer bare-device-PID allowlist.
The peer imports those objects. Each Notify is waited on by its creator and
recorded by its importing peer.

One native call queues this sequence on the owner's dedicated stream:

1. Copy the local partial directly into the peer's imported inbox.
2. Record the peer's READY notification after the copy.
3. Wait for and reset the local READY notification.
4. Add the local input and received partial into a separate FP32 output.
5. Record the peer's CONSUMED notification after the reduction.
6. Wait for and reset the local CONSUMED notification before reusing the lane.

The CONSUMED round trip prevents a faster rank from overwriting the peer's
inbox while its previous reduction still reads it. It also prevents successive
binary notifications from coalescing. This adds a handshake whose actual cost
must be compared with HCCL; specialization does not guarantee a win.

The data plane has no host barrier, host payload copy, polling loop or
per-reduction allocation by this library. It still dispatches SDK operations
from the host. CANN creates a fresh ACLNN executor per reduction; its overhead
and internal allocations must be measured. Tensor descriptors and up to 1 MiB
of external workspace per rank persist. A later fused add/communication kernel
is justified only if profiling identifies this submission/reduction work as
material. A private local-only FP32 kernel removes ACLNN planning, but hardware
tests found no useful completed-latency gain; it remains an optional research
comparator. Peer import success alone does not prove an HCCS route or bandwidth.

The optional `handshake="ping-pong"` candidate uses two distinct inboxes and
the existing two Notify objects as alternating READY slots. Each call queues
copy, record READY, wait/reset READY and local ACLNN add. Before rank A can
reuse slot `s % 2` at `s+2`, it must receive rank B's READY at `s+1`. B records
that only after its wait/reset and local add at `s` finish. This acknowledges
both inbox consumption and notification reset without a separate CONSUMED
round trip. The proof requires all producers, copies, adds and consumers to
execute on one in-order owner stream. Slot parity persists across requests
and drains. See [the policy and proof](pair_pingpong.py) and its adversarial
byte-level simulation; those tests alone do not establish device correctness.
The local-kernel callback and ping-pong mode cannot currently be combined.

## Supported contract

- Exactly two ranks; fixed contiguous FP32 shape `[rows, 4096]`, rows 1–32.
- Separate sessions/lanes for different row counts; fresh 256-bit session nonce.
- Peers agree on handshake and authenticated reducer identity before import.
- Identical fixed order: attention SUM then MLP SUM for each layer 18–32.
- Monotonic sequence numbers continue across requests. No partial-session reset.
- One owning process, thread and initialized SDK context per endpoint.
- Input producers and output consumers enqueue on the same dedicated stream.
- At most 64 host submissions before an explicit bounded stream drain.
- No graph capture, dynamic collective order, communicator resizing or automatic
  fallback during an uncertain transfer.

The native implementation is in `csrc/sidecard_collectives/pair_sum.cpp`;
`pair_collective.py` controls identity, ordering, state and shutdown.
`pair_collective_acl.py` owns SDK resources and calls the native C ABI.
The native handle retains its descriptors while work is outstanding.

Direct SDK submission must not bypass a Torch NPU task queue. The pair-specific API has no Torch adapter; the separate N-rank API has an
eager-only queued Torch adapter pending device/model qualification. GLM integration requires a native stream-aware adapter at the
existing two reduction seams, validated against the unchanged HCCL path. It
must preserve conversions, expert ownership, FP32 combination, recurrent state,
and both reduction boundaries. Do not add a serving option before qualification.

## Setup and teardown

The caller initializes and verifies its assigned device context. This library
does not initialize/reset devices, change driver configuration, or mutate the
environment. Exchange each owner's SDK bare-device PID through the existing
private control channel before preparing resources. Then:

```python
config = PairConfig(session=fresh_shared_nonce, rows=1)
endpoint = PairEndpoint(config, rank, authenticated_acl_backend)
local_export = endpoint.prepare(peer_bare_pid)
local_connected = endpoint.connect(peer_export)
endpoint.activate(peer_connected)

# Native producers/consumers use authenticated_acl_backend.stream.
endpoint.submit(sequence, BOUNDARIES[sequence % len(BOUNDARIES)])

local_drained = endpoint.drain()
local_imports_closed = endpoint.release_imports(peer_drained)
endpoint.close(peer_imports_closed)
```

Exported keys and mapping receipts are private capabilities; do not publish
them or raw device addresses. Peers verify the same protocol, session, fixed
schedule, geometry and opposite ranks before activation. Final shutdown first
drains both streams at the same sequence count, then closes both peers'
imports, then frees owned exports. A partial native failure quarantines the
endpoint and retains every owner. There is no destructor that guesses it is
safe to unmap memory. Device loss requires an external recovery decision.

The installed SDK header and CANN Python documentation specify milliseconds
for Notify waits, while some C documentation specifies seconds. The prototype
therefore exposes a nonzero raw SDK timeout, without claiming a wall-clock
duration. Its separate stream-drain timeout is 30,000 milliseconds. Returning
from a failed drain does not cancel device work or permit memory release.
Resolve the exact Notify timeout behavior in a bounded capability gate before
using the library in a model process.

## Hardware gate and benchmark

The user temporarily allocated both Duos, physical chips 0–3, for the isolated
library follow-up. Each probe names exactly one two-chip pair. The local owner
stops the recorded model service before using its chips and records recovery
and ownership; there are no competing accelerator probes or automatic resets.
The original GLM release allocation remains chips 2/3 plus the RTX.

1. Authenticate source, SDK, libraries, device mapping and driver identities.
   Probe IPC memory and Notify support on the actual pair. Exercise P2P inbox
   copies and remote Notify completion separately, then qualify their ordering
   together. Verify timeout units, copy visibility and graph behavior independently.
2. Run matching 1/4/32-row inputs through HCCL FP32 SUM and this path on both
   ranks. Compare complete output bytes, including cancellation-sensitive values,
   signed zeros, repeated/random inputs and the existing nonfinite policy.
   No precision tolerance may be relaxed to promote a transport change.
3. Stress many reuse cycles with delayed peer submission, changed input values,
   stale receipts and orderly shutdown. Test failures without freeing uncertain
   mappings or forcing device resets. Measure both ranks, not just the faster one.
4. Compare warmed raw HCCL and specialized native entry points on the same SDK
   streams/buffer geometry. Keep initialization, upload, CPU readback and output
   comparison outside the timed interval. Report host submission, matched
   arrival spread, completed batch latency, p50/p95 and effective bytes separately.
   Include 0/50/250/1000-microsecond injected arrival skew, isolated calls and
   30-boundary sequences. Profiling must observe both ranks in the same run.
5. Alternate at least five matched baseline/candidate runs and retain all raw
   timings. Do not subtract timestamps from unqualified device clock domains or
   treat rank waiting as physical transfer time. The previous profile's negative
   NPU events prohibit using that clock source without fresh validation.
6. Integrate only an eligible candidate; compare all handoffs, complete model
   state, logits and selectors with the fresh reference. Run unchanged Code,
   tool, lifecycle and completed 128K gates before choosing the serving path.

The optimization target is exposed latency on the joined model's critical path.
If reduction replacement saves 100 microseconds at each of 30 boundaries, that
is only 3 ms per token before other effects. If waiting dominates because a peer
has more expert work, balance/scheduling may offer more than a transport rewrite.

## Build and portable checks

The private builder snapshots both native sources and hashes the installed
headers, linked libraries, compiler, output and log. It runs only the compiler;
it does not load the output or initialize an accelerator. Outputs must be fresh.

```bash
python -m tools.sidecard.pair_collective_build \
  --sdk /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux \
  --output /private/fresh-pair-build

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  --confcutdir=tests/ut/sidecard \
  tests/ut/sidecard/test_pair*.py
```

`--hccl-reference` builds the isolated raw-HCCL comparison shim.
`python -m tools.sidecard.pair_add_build --sdk <architecture-root> --output
<fresh-directory>` builds the optional local-add runtime and fixed 1/4/32-row
device binaries. Its ACLRTC subprocess initializes ACL without selecting a
device; use a healthy driver. Both builders preserve failed-build receipts
and install nothing into shared OPP directories. The probe authenticates all
libraries and optional device code before launching owners.

The native CPU stub tests compile the actual C++ shim and check ABI, operation
ordering, FP32 buffer handling and each submission failure. The independent
stream simulator explores delayed/interleaved peers, inbox reuse and binary
notification coalescing. These are offline tests, not HCCS or NPU validation.

## Primary API references

- [IPC memory and P2P allocation requirements](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1070.html)
- [Notify export, PID allowlist and creator-side waits](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1052.html)
- [Notify import flags and hardware support](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1054.html)
- [IPC events exclude older Atlas inference products](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2092.html)
- [Ordinary events cannot wait across devices](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0036.html)
- [ACLNN executor is obtained for each add invocation](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0032.html)
