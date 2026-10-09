# Streaming protocol schema version 1

All hashes use lowercase SHA256 hex. `canonical_sha256` encodes UTF-8 JSON with
sorted keys, compact separators, preserved Unicode and rejected NaN/Infinity.
`file_sha256` hashes exact file bytes. The manifest identity excludes only its own
`manifest_sha256` field. Source identity covers `{assets, gitlinks}`: asset paths
and exact blob SHA256 plus separately identified external Git commits. Revision
names are resolved to full commits; committed bytes are used even with dirty files.

The reference contains arithmetic policy and `arithmetic_sha256`, fixed comparison
capacity, historical runtime/provenance, live state explicitly unknown, pending
checkpoint and coherent serving bundle identity, archived evidence, staged native
host-build receipt, thermal policy and workload matrix. Image bytes and token IDs
are never inferred from filenames, old message logs or token counts.

An optional T8 candidate binding contains nonempty `assets`, `binaries`, `bridges`
lists of `{path, sha256}`, `contract_sha256`, and `native_components`. Its
`candidate_sha256` hashes all candidate fields except itself. T8 must verify the
actual bytes and complete inclusion; this structural validator cannot infer which
files a runtime will import. It does not load native code.

Saved evidence carries `schema_version`, `manifest_sha256`, `source_sha256`,
`arithmetic_sha256`, `checkpoint_sha256`, `serving_bundle_sha256`,
`candidate_sha256`, `rank_receipts`, `trials`, and `gates`. All six identities are
required even if an offline reference has pending/null checkpoint, serving bundle,
or candidate identities. Rank receipts repeat these fields, require ranks 0–3
exactly once, positive integer PID, and one common `execution_namespace`.

A trial binds `workload_id`, `cache_policy`, `token_ids_sha256`,
`conversation_sha256`, `image_sha256`, `max_new_tokens`, `mtp_tokens`,
`submitted_requests`, `computed_prompt_tokens`, and `cached_prompt_tokens`.
Computed plus cached must equal a frozen token list's length. Cold means zero
cached prompt tokens; warm requires nonzero reuse and the identical frozen cold
control. Live trials additionally require `hardware_validated: true`.

Each gate has status `pending`, `failed`, or `passed`. Passed gates require scope
`hardware`, a saved `artifact_sha256`, ranks `[0,1,2,3]`, and the exact
`manifest_sha256`. Required live gates are kernel parity, real weights, quality,
image, cache/CoW, EP, MTP, graph replay, sustained thermal and service performance.
Each candidate native component also needs a corresponding
`native_component_evidence` entry with hardware scope, passed status, full ranks,
artifact hash, and matching `candidate_sha256`.

`validate_manifest(..., require_live=False)` admits only internally consistent
offline preparation; `root=` additionally verifies committed blob and external
commit identities. `validate_evidence(..., require_live=False)` validates supplied
rank/trial/gate receipts, without implying complete coverage or qualification.
`require_live=True` refuses unresolved inputs, missing checkpoint/serving/candidate
identities, incomplete workload coverage, and pending/failed hardware gates. The
controller must additionally verify actual bytes, timings, sustained thermal and
recovery/maintenance ownership before it mutates dispatch. A structurally valid
receipt alone is never permission to touch a service.

## Payload revision 2

The final reference is `reference-v2.json`. The original `reference.json` remains
unchanged and valid as a historical payload-revision-1 receipt. Schema version 1
is retained; `payload_revision: 2` adds frozen authored request bodies.

`request_payloads` stores each concrete conversation and rendered request once.
Workloads reference these through `request_payload_ref` and
`conversation: {payload_ref: ...}`. Their `conversation_sha256` hashes the actual
catalog conversation, while `rendered_request_sha256` hashes the canonical JSON
request body, including messages, tool schema, generation limit and decoding
settings. Warm controls must bind identical rendered request hashes. Revision-2
trial receipts additionally require the matching `rendered_request_sha256`.

Model ID and tokenizer ID/revision remain unresolved; image content/URL remains
pending. These requests therefore describe reproducible input templates, not yet
executable qualified requests. `target_prompt_tokens` is a requested benchmark
size; `actual_prompt_tokens` is null until genuine tokenizer IDs are frozen. No
character or paragraph count is asserted to equal a token count. Future actual
prompt counts must equal the frozen token-ID list length. Filling unresolved model,
tokenizer or image values changes the manifest and invalidates old gate evidence.

Tool input contains a concrete function schema and deterministic response fixture.
Mixed input defines prefill arrival after generated token 16 of an ongoing decode;
failure to reach that trigger is inconclusive, not replaced with a wall-clock delay.
Fixtures are authored benchmark inputs, not recovered historical conversations.
