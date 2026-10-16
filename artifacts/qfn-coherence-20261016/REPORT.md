# TP6 coherence: draft KV block-size mismatch

## Root cause

Later MTP draft steps used 128-token addressing against a cache and block table
with 64-token pages. All six live `AscendEagleProposer` instances reported
`has_gdn=True`, `kernel_block_size=128`, and `block_size=64`. The former came
from the first advertised backend capability, rather than the selected page
size. The old GDN branch selected it for both the block-table column and the
physical slot stride.

The cache contained 26,884 pages, or 1,720,576 token slots. Address arithmetic
from two damaged recurrent states identified draft value-cache writes at
physical pages 31,460 and 35,204, outside that allocation. FP16 write patterns
appeared inside FP32 recurrent states; the values matched across the TP ranks
that replicate each KV head. An older cached prefix contained NaNs. A newly
created prefix lost 4,048–4,066 bytes of layer 16 state on each rank during the
third mixed prefill/decode round, despite not being recomputed.

`attn_update_stack_num_spec_norm` now consistently uses `self.block_size`.
This fork's upstream `initialize_attn_backend` already sets that field from
the selected kernel block sizes. The correction runs before graph replay and
fills the same persistent slot tensors, so existing graphs can consume it.

## Client compaction isolation

Identical direct API requests with one user message reproduced `Thisimosimos…`
from the old 14,336-token cached prefix. These requests bypassed Kilo and did
not compact a conversation or force a minimum generation length. Changing only
`cache_salt` restored coherent generation. The limited local Kilo metadata
audit is recorded in [client-audit.json](client-audit.json); it is not an audit
of every possible client. The reproduced failure is server-side memory
corruption and does not require conversation compaction.

## Validation and deployment

- Host regressions: **156 passed**, including 64 new metadata cases.
- Private NPU metadata gate: **64 cases on each of six NPUs**, all passed.
- Six mixed rounds paired a 768-token decode on a 14,444-token prompt with
  a fresh approximately 4,340-token prefill, followed by a cached replay.
  All responses remained coherent. All **13 checkpoint snapshots per rank**
  were finite and byte-identical; no recurrent-state bytes changed.
- The old corrupted prefix still repeated after code replacement, confirming
  the need to invalidate existing damage. A coordinated scheduler/worker cache
  reset removed it while preserving worker PIDs, weight storage, dispatch,
  graph generations, and native owners.
- The exact same prompt then generated 768 coherent tokens cold and on two
  successive cache hits. Server decode throughput was **34.3, 33.0, and
  32.7 tok/s**, respectively. Cold prefill was 475.6 effective tok/s; each hot
  replay reused 14,336 tokens and computed 108. These are correctness checks,
  not a speedup claim.
- No registered or packed weight storage intersected the possible draft KV
  overrun ranges in the live allocation audit. Diagnostic wrappers were
  removed; private watcher buffers were released. Shared-host prefill PLE,
  baseline pacing, loaded weights, and the existing decode graphs were retained.
- Full local suite collection was attempted but stopped at the missing
  `modelscope` dependency in `tests/e2e/conftest.py`. This is not a full-suite
  pass; real-weight service validation was performed on the six NPUs.

The production change also persists in the active launch runtime at
`/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/runtime/` on
`192.168.53.187`. Raw replay responses, checkpoint tensors, graph dumps,
controller receipts, and scripts remain under
`/srv/ai/storage/artifacts/qfn-coherence-20261016/`.

The checked-in [validation summary](validation-summary.json) records request
counts and timings, per-rank checks, and SHA256 values of the raw evidence.
Output need not be bit-identical across different batching shapes; the
checkpoint comparison is byte-for-byte, including the recurrent state.
These gates validate this real-weight TP6 service, not a broad model accuracy
benchmark or other hardware configurations.

## Reproduction and maintenance

Run the host regressions without importing the NPU test harness:

```bash
python -m pytest --noconftest -q \
  tests/ut/spec_decode/test_draft_cache_block_size.py \
  tests/ut/spec_decode/test_autoregressive_draft_metadata.py \
  tests/ut/qwen38_1m/test_mtp_draft_capture_geometry.py \
  tests/ut/qwen38_1m/test_draft_step_graph_identity.py \
  tests/ut/qwen38_1m/test_resident_reset.py \
  tests/ut/_310p/test_prefix_mamba_state.py
```

The new regression executes the full production metadata method with selected
64/128-token pages, GDN enabled/disabled, eager/full graph modes, ordinary/MRoPE
positions, first/later draft steps, constant/advancing positions, high physical
page IDs, context-limit masking, and padding. Its `exercise_slot_mapping`
helper also ran against private tensors on every live NPU, without writing a
model KV cache. The original method fails the 64-token GDN case.

The existing graph-address regression needed its missing optional collective
observer stub restored; no production graph-dispatch change was made.

Repository-wide `bash format.sh ci` was attempted. Existing unrelated lint,
spelling, C++ formatting, and Markdown failures prevent a whole-repository
pass. Formatter changes outside this fix were discarded in the isolated
worktree. Changed-file checks pass. Fork Actions remain disabled.

A failed diagnostic RPC exposed the executor's existing early-error response
backlog behavior. The queues were realigned with per-rank nonce acknowledgments
before subsequent tests. That separate executor behavior is not changed here.
